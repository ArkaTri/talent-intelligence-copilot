"""Supervisor — merangkai guardrail dan ketiga agent dengan LangGraph.

ALUR:
  query
    ↓
  [guardrail_pre]  ─ ditolak ─→ [END] jawaban penolakan
    ↓ lolos
  [route]          ─ tentukan agent, ekstrak filter
    ↓
  [retrieval] / [evaluator] / [analytics]
    ↓
  [guardrail_post] ─ verifikasi citation
    ↓
  [END] AgentResponse

KENAPA StateGraph, BUKAN create_react_agent:

Contoh kelas memakai create_react_agent, di mana agent memilih tool
sendiri. Tiga alasan menyimpang:

1. GUARDRAIL TIDAK BOLEH OPSIONAL
   Sebagai tool, agent bisa memutuskan tidak memakainya — dan justru
   pada query bermasalah. Sebagai node, jalurnya dipaksa.

2. ALUR DETERMINISTIK, TIDAK BUTUH ITERASI
   Sistem ini: routing (1 panggilan) → agent (1-2) → selesai. ReAct
   unggul saat agent perlu mencoba-lihat-coba lagi; di sini kebebasan
   itu hanya menambah risiko looping tanpa manfaat.

3. BIAYA TERPREDIKSI
   ReAct mengirim ulang seluruh riwayat di setiap iterasi. Dengan
   payload chunk resume yang besar, itu berlipat cepat. StateGraph
   memakai jumlah panggilan tetap.

Requirement capstone mewajibkan LangChain Framework (LangChain,
LangGraph, Langfuse). StateGraph adalah LangGraph — requirement
terpenuhi.

CHAT HISTORY:
Disimpan di state dan dipakai node route untuk menyelesaikan referensi
("yang tadi nomor 2 itu, apa pendidikannya?"). Dibatasi 6 giliran
terakhir agar payload routing tidak membengkak.
"""

import json
import os
from typing import Annotated, Literal, TypedDict
import re

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.agents.analytics_agent import AnalyticsAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.retrieval_agent import run_retrieval
from src.agents.schemas import (
    AgentResponse,
    AnalyticsResult,
    EvaluationResult,
    Evidence,
    GuardrailVerdict,
    RouteDecision,
    UsageRecord,
    UsageSummary,
)
from src.guardrails.citation_verifier import verify
from src.guardrails.query_filter import check_query
from src.retrieval.vector_store import SearchHit, VectorStore

load_dotenv()

MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-4.1-nano")

# Giliran percakapan yang dikirim ke router. Lebih dari ini membuat
# payload routing membengkak tanpa menambah akurasi resolusi referensi.
HISTORY_TURNS = 6


ROUTER_PROMPT = """You route recruiter questions to the right handler.

HANDLERS:

"retrieval" — find candidates matching criteria
  "siapa yang punya pengalaman audit?", "cari kandidat banking"

"evaluator" — score candidates against a job description
  "nilai kandidat ini untuk role X", "bandingkan 3 kandidat teratas"
  Use when the question states REQUIREMENTS to judge against.

"analytics" — count or aggregate
  "berapa banyak kandidat yang menyebut AWS?", "berapa yang punya PMP?"

"profile" — show everything about ONE specific candidate
  "info lengkap tentang kandidat tersebut", "detail kandidat 11183737",
  "apa pendidikan kandidat yang tadi?"
  Fill target_resume_ids with ACTUAL numeric IDs. If the user names an ID,
  use it. If they say "kandidat tersebut" or "yang tadi", scan the CHAT
  HISTORY for citations like [12345678#3] and extract 12345678.
  Never output a placeholder — if no ID can be found, return an empty list
  and route to "retrieval" instead.

"refuse" — outside the system's scope
  Questions not about candidate screening at all.

Return JSON:
{
  "agent": "retrieval" | "evaluator" | "analytics" | "profile" | "refuse",
  "reasoning": "<max 15 words>",
  "category_filter": ["CATEGORY"],
  "section_filter": [],
  "search_query": "<optimized English search phrase>",
   "target_resume_ids": []
}

CATEGORY FILTER — only when the user NAMES an industry or role category.
Valid values: HR, DESIGNER, INFORMATION-TECHNOLOGY, TEACHER, ADVOCATE,
BUSINESS-DEVELOPMENT, HEALTHCARE, FITNESS, AGRICULTURE, BPO, SALES,
CONSULTANT, DIGITAL-MEDIA, AUTOMOBILE, CHEF, FINANCE, APPAREL,
ENGINEERING, ACCOUNTANT, CONSTRUCTION, PUBLIC-RELATIONS, BANKING,
ARTS, AVIATION.
Leave EMPTY if the user did not name one. Testing showed forcing a filter
on general queries narrows results without benefit.

SECTION FILTER — leave EMPTY almost always. Testing showed embeddings
already route correctly: "proficient in Excel" hits skills 10/10,
"bachelor degree" hits education 10/10. Only set it if the user explicitly
asks about one section.

SEARCH_QUERY — rewrite for semantic search, in English, focused on the
substance. "siapa kandidat finance yang pernah memimpin audit?" becomes
"led financial audit engagements".

If chat history is given, resolve references ("kandidat nomor 2", "yang tadi")
into explicit search terms.

No markdown."""


class GraphState(TypedDict):
    """State yang mengalir antar node.

    LangGraph mengoper dict ini dari node ke node. Setiap node menerima
    state penuh dan mengembalikan dict berisi field yang diubah —
    LangGraph menggabungkannya, bukan mengganti seluruh state.
    """
    query: str
    history: list[dict]

    guardrail: GuardrailVerdict | None
    route: RouteDecision | None

    answer: str
    evidence: list[Evidence]
    retrieved: list[SearchHit]
    evaluation: EvaluationResult | None
    analytics: AnalyticsResult | None

    citation_ok: bool
    citation_issues: list[str]
    usage: list[UsageRecord]

def _as_list(value) -> list[str]:
    """Normalisasi field yang seharusnya list.

    Model kadang mengembalikan string tunggal ("FINANCE") alih-alih list,
    dan list comprehension akan mengiterasinya per karakter — filter jadi
    [F,I,N,A,N,C,E], nol hasil TANPA error.

    Juga membuang placeholder ("<id>", "..."), karena model sesekali
    menyalin contoh format dari prompt alih-alih mengisinya.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [
        v for v in value
        if isinstance(v, str) and v.strip()
        and not v.startswith("<") and v != "..."
    ]

class Supervisor:
    def __init__(self, vs: VectorStore | None = None,
                 client: OpenAI | None = None):
        self.vs = vs or VectorStore()
        self._oa = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = EvaluatorAgent(vs=self.vs, client=self._oa)
        self.analytics = AnalyticsAgent(client=self._oa)
        self.graph = self._build()

    # ── NODES ────────────────────────────────────────────────────────

    def _node_guardrail_pre(self, state: GraphState) -> dict:
        """Pre-check. Node pertama — tidak ada jalan memutarinya."""
        verdict, usage = check_query(state["query"], client=self._oa)
        out = {"guardrail": verdict, "usage": state["usage"] + usage}

        if not verdict.allowed:
            parts = [verdict.explanation or "Pertanyaan ini tidak dapat diproses."]
            if verdict.suggested_rephrase:
                parts.append(f"\nSaran: {verdict.suggested_rephrase}")
            out["answer"] = "\n".join(parts)

        return out

    def _node_route(self, state: GraphState) -> dict:
        """Tentukan agent dan ekstrak filter dari maksud pengguna.

        Filter diekstrak di sini, bukan di agent, karena supervisor yang
        membaca maksud. Agent hanya menerima instruksi.
        """
        history_text = ""
        if state["history"]:
            recent = state["history"][-HISTORY_TURNS:]
            parts = []
            for m in recent:
                content = m["content"]
                # Citation biasanya muncul di tengah/akhir jawaban, jauh
                # setelah karakter ke-200. Memotong di depan membuat router
                # tidak pernah melihat ID kandidat — dan follow-up seperti
                # "kandidat tersebut" jadi tidak bisa diselesaikan.
                # Ambil awal DAN semua citation yang ada.
                ids = re.findall(r"\[\d+#\d+\]", content)
                snippet = content[:200]
                if ids:
                    snippet += "  [citations: " + " ".join(dict.fromkeys(ids)) + "]"
                parts.append(f"{m['role']}: {snippet}")
            history_text = "\n\nCHAT HISTORY:\n" + "\n".join(parts)

        resp = self._oa.chat.completions.create(
            model=MODEL_ROUTER,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": state["query"] + history_text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        u = resp.usage
        usage = state["usage"] + [UsageRecord(
            component="routing", model=MODEL_ROUTER,
            input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
        )]

        try:
            d = json.loads(resp.choices[0].message.content)
            route = RouteDecision(
                agent=d.get("agent", "retrieval"),
                reasoning=str(d.get("reasoning", ""))[:100],
                category_filter=_as_list(d.get("category_filter")),
                section_filter=_as_list(d.get("section_filter")),
                search_query=str(d.get("search_query", state["query"])),
                target_resume_ids=_as_list(d.get("target_resume_ids")),
            )
        except Exception:
            # Fallback ke retrieval: jalur paling umum dan paling aman.
            # Routing gagal tidak boleh mematikan sistem.
            route = RouteDecision(
                agent="retrieval",
                reasoning="routing gagal, fallback ke retrieval",
                search_query=state["query"],
            )

            # Model kadang menulis placeholder atau menyertakan nomor chunk
            # ("27914096#6") padahal yang dibutuhkan hanya resume_id.
            # Bersihkan: ambil bagian sebelum "#", buang yang bukan digit.
        route.target_resume_ids = [
            rid.split("#")[0].strip()
            for rid in route.target_resume_ids
            if rid.split("#")[0].strip().isdigit()
        ]

        if route.agent == "profile" and not route.target_resume_ids:
            for m in reversed(state["history"]):
                found = re.findall(r"\[(\d+)#\d+\]", m.get("content", ""))
                if found:
                    route.target_resume_ids = [found[0]]
                    break

        return {"route": route, "usage": usage}
    def _node_retrieval(self, state: GraphState) -> dict:
        r = state["route"]

        # Ambil kandidat mentah terpisah agar citation_verifier punya
        # daftar sumber yang sah untuk dicocokkan.
        retrieved = self.vs.search(
            r.search_query or state["query"],
            top_k=20, fetch=20,
            category=r.category_filter or None,
            section_type=r.section_filter or None,
            dedupe_by_resume=True,
        )

        answer, evidence, usage = run_retrieval(
            state["query"],
            search_query=r.search_query,
            category_filter=r.category_filter or None,
            section_filter=r.section_filter or None,
            vs=self.vs,
            client=self._oa,
        )

        return {
            "answer": answer,
            "evidence": evidence,
            "retrieved": retrieved,
            "usage": state["usage"] + usage,
        }

    def _node_evaluator(self, state: GraphState) -> dict:
        r = state["route"]
        result, usage = self.evaluator.run(
            requirements=state["query"],
            search_query=r.search_query,
            category_filter=r.category_filter or None,
        )

        if not result:
            return {
                "answer": "Tidak dapat mengevaluasi kandidat untuk kriteria tersebut.",
                "usage": state["usage"] + usage,
            }

        # Ringkasan teks untuk UI; objek terstruktur tetap dibawa terpisah
        lines = [f"Kriteria: {result.job_context}\n"]
        for c in result.ranked():
            lines.append(f"[{c.resume_id}] skor {c.overall_score}/10 — {c.summary}")
            if c.gaps:
                lines.append(f"  Gap: {'; '.join(c.gaps)}")

        evidence = [e for c in result.candidates for e in c.evidence]

        # Kumpulkan chunk kandidat sebagai sumber sah untuk verifikasi
        retrieved: list[SearchHit] = []
        for c in result.candidates:
            retrieved.extend(self.vs.get_resume_chunks(c.resume_id))

        return {
            "answer": "\n".join(lines),
            "evaluation": result,
            "evidence": evidence,
            "retrieved": retrieved,
            "usage": state["usage"] + usage,
        }

    def _node_analytics(self, state: GraphState) -> dict:
        result, usage = self.analytics.run(state["query"])

        if not result.answerable:
            answer = result.refusal_reason or "Pertanyaan ini tidak dapat dijawab."
        else:
            parts = [f"Ditemukan {result.count} resume."]
            if result.per_term:
                parts.append("Per istilah: " + ", ".join(
                    f"{t} ({n})" for t, n in result.per_term.items()))
            if result.breakdown:
                top = list(result.breakdown.items())[:5]
                parts.append("Sebaran kategori: " + ", ".join(
                    f"{c} ({n})" for c, n in top))
            if result.caveat:
                parts.append(f"\nCatatan: {result.caveat}")
            answer = "\n".join(parts)

        return {
            "answer": answer,
            "analytics": result,
            "usage": state["usage"] + usage,
        }
    
    def _node_profile(self, state: GraphState) -> dict:
        """Tampilkan profil lengkap satu kandidat.

        Memakai get_resume_chunks (scroll, bukan vector search) — nol biaya
        API karena tidak perlu embedding query. Seluruh chunk resume diambil,
        bukan hanya yang cocok dengan pencarian.
        """
        rid = (state["route"].target_resume_ids or [None])[0]
        if not rid:
            return {"answer": "Sebutkan ID kandidat yang ingin dilihat, "
                              "misalnya: detail kandidat 11183737."}

        chunks = self.vs.get_resume_chunks(rid)
        if not chunks:
            return {"answer": f"Kandidat {rid} tidak ditemukan di korpus."}

        lines = [f"**Profil lengkap kandidat [{rid}]** "
                 f"· kategori {chunks[0].category} · {len(chunks)} bagian\n"]
        for c in chunks:
            lines.append(f"**{c.citation()}** · {c.section_type}")
            lines.append(c.text.strip())
            lines.append("")

        evidence = [
            Evidence(resume_id=c.resume_id, chunk_index=c.chunk_index,
                     quote=c.text.strip()[:200], section=c.section_type)
            for c in chunks
        ]

        return {
            "answer": "\n".join(lines),
            "evidence": evidence,
            "retrieved": chunks,
        }
    
    def _node_refuse(self, state: GraphState) -> dict:
        return {
            "answer": (
                "Sistem ini khusus untuk menelusuri dan menilai kandidat "
                "dari korpus resume. Pertanyaan tersebut di luar cakupannya."
            )
        }

    def _node_guardrail_post(self, state: GraphState) -> dict:
        """Post-check citation.

        Analytics dilewati karena tidak menghasilkan klaim tentang
        kandidat individual — outputnya angka agregat.
        """
        if state.get("analytics") is not None:
            return {"citation_ok": True, "citation_issues": []}

        if not state.get("retrieved"):
            return {"citation_ok": True, "citation_issues": []}

        ok, issues = verify(
            state["answer"], state.get("evidence", []), state["retrieved"]
        )
        return {"citation_ok": ok, "citation_issues": issues}

    # ── ROUTING EDGES ────────────────────────────────────────────────

    def _after_guardrail(self, state: GraphState) -> Literal["route", "end"]:
        return "route" if state["guardrail"].allowed else "end"

    def _after_route(self, state: GraphState) -> str:
        return state["route"].agent

    # ── GRAPH ────────────────────────────────────────────────────────

    def _build(self):
        g = StateGraph(GraphState)

        g.add_node("guardrail_pre", self._node_guardrail_pre)
        g.add_node("route", self._node_route)
        g.add_node("retrieval", self._node_retrieval)
        g.add_node("evaluator", self._node_evaluator)
        g.add_node("analytics", self._node_analytics)
        g.add_node("profile", self._node_profile)
        g.add_node("refuse", self._node_refuse)
        g.add_node("guardrail_post", self._node_guardrail_post)

        g.set_entry_point("guardrail_pre")

        # Query yang ditolak guardrail langsung berakhir — tidak menyentuh
        # agent mana pun, tidak ada biaya retrieval.
        g.add_conditional_edges("guardrail_pre", self._after_guardrail,
                                {"route": "route", "end": END})

        g.add_conditional_edges("route", self._after_route, {
            "retrieval": "retrieval",
            "evaluator": "evaluator",
            "analytics": "analytics",
            "profile": "profile",
            "refuse": "refuse",
        })

        for node in ("retrieval", "evaluator", "analytics", "profile", "refuse"):
            g.add_edge(node, "guardrail_post")

        g.add_edge("guardrail_post", END)

        return g.compile()

    # ── PUBLIC API ───────────────────────────────────────────────────

    def ask(self, query: str, history: list[dict] | None = None) -> AgentResponse:
        """Jalankan satu query melalui graph."""
        init: GraphState = {
            "query": query,
            "history": history or [],
            "guardrail": None,
            "route": None,
            "answer": "",
            "evidence": [],
            "retrieved": [],
            "evaluation": None,
            "analytics": None,
            "citation_ok": True,
            "citation_issues": [],
            "usage": [],
        }

        # recursion_limit rendah karena alurnya deterministik — maksimal
        # 4 node per query. Nilai ini berfungsi sebagai alarm kebakaran:
        # kalau tercapai, ada bug arsitektur, bukan limit terlalu kecil.
        final = self.graph.invoke(init, config={"recursion_limit": 10})

        return AgentResponse(
            answer=final["answer"],
            route=final.get("route"),
            evidence=final.get("evidence", []),
            evaluation=final.get("evaluation"),
            analytics=final.get("analytics"),
            guardrail=final.get("guardrail"),
            usage=UsageSummary(records=final.get("usage", [])),
            citation_ok=final.get("citation_ok", True),
            citation_issues=final.get("citation_issues", []),
        )


if __name__ == "__main__":
    sup = Supervisor()

    tests = [
        "Siapa kandidat dengan pengalaman audit keuangan paling kuat?",
        "Berapa banyak kandidat yang menyebut AWS?",
        "Nilai kandidat untuk posisi Financial Audit Manager: butuh 5 tahun "
        "pengalaman audit, SOX compliance, dan sertifikasi CPA",
        "Cari kandidat laki-laki di bidang banking",
        "Apa resep rendang yang enak?",
         "Berikan detail lengkap kandidat 27914096",
    ]

    grand_total = 0.0

    for q in tests:
        print("=" * 74)
        print(f"Q: {q}")
        print("=" * 74)

        r = sup.ask(q)

        if r.route:
            print(f"route: {r.route.agent}  |  {r.route.reasoning}")
            if r.route.category_filter:
                print(f"       filter: {r.route.category_filter}")
        print()
        print(r.answer[:900])

        if r.evidence:
            print(f"\nbukti: {len(r.evidence)} — "
                  f"{', '.join(e.citation() for e in r.evidence[:5])}")

        mark = "OK" if r.citation_ok else "BERMASALAH"
        print(f"\ncitation: {mark}")
        for i in r.citation_issues[:3]:
            print(f"  - {i}")

        cost = r.usage.total_cost_usd
        grand_total += cost
        print(f"\nusage: in={r.usage.total_input:,} out={r.usage.total_output:,} "
              f"${cost:.6f}")
        for rec in r.usage.per_component():
            print(f"  {rec['component']:10s} {rec['model']:14s} ${rec['cost_usd']:.6f}")
        print()

    print("=" * 74)
    print(f"Total {len(tests)} query: ${grand_total:.6f}")
    print(f"Rata-rata: ${grand_total / len(tests):.6f} "
          f"→ ${grand_total / len(tests) * 1000:.2f} per 1.000 query")