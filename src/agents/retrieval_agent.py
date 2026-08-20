"""Retrieval agent — mencari kandidat dan mengembalikan bukti terkutip.

POSISI DALAM SISTEM:
  supervisor → RETRIEVAL AGENT → vector_store → reranker → Evidence

Agent ini menangani pertanyaan "cari kandidat yang...". Untuk pertanyaan
agregat ("berapa banyak...") ada analytics_agent, untuk perbandingan
terhadap job description ada evaluator_agent.

KENAPA JAWABAN DIHASILKAN DI SINI, BUKAN DI SUPERVISOR:
Model jawaban menyerap 78,3% biaya per query (lihat schemas.py). Menaruh
generasi jawaban di agent yang punya konteks lengkap — bukan di supervisor
yang harus mengoper konteks — menghindari duplikasi payload.

CITATION ENFORCEMENT:
Prompt meminta setiap klaim menyertakan [resume_id#chunk]. Tapi prompt
saja tidak cukup — verifikasi struktural dilakukan di guardrails/
citation_verifier.py. Di sini kita menyediakan bahannya: setiap chunk
yang dikirim ke LLM diberi label citation yang persis.

MODEL: gpt-4.1-mini (MODEL_ANSWER)
Kandidat ablation vs gpt-5.4-mini. Uji Tahap 5 menemukan gpt-5.4-mini
menyisipkan karakter non-Latin (Georgia, Bengali) saat menjawab dalam
Bahasa Indonesia — perlu diuji sistematis di evaluation harness sebelum
diputuskan.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.schemas import Evidence, UsageRecord
from src.retrieval.reranker import RankedHit, rerank
from src.retrieval.vector_store import VectorStore

load_dotenv()

MODEL_ANSWER = os.getenv("MODEL_ANSWER", "gpt-4.1-mini")

# Ambil 20 kandidat mentah, rerank jadi 5. Fetch > top_k diperlukan
# karena dedup membuang sebagian, dan reranker butuh kandidat berlebih
# untuk bisa menaikkan yang layak dari luar top-5 vektor.
FETCH_N = 20
TOP_K = 5


SYSTEM_PROMPT = """You are a talent screening assistant. You answer recruiter questions using ONLY the resume excerpts provided.

RULES — these are not suggestions:

1. EVERY factual claim about a candidate MUST cite its source as [resume_id#chunk].
   Write: "Candidate [11759079#3] led inventory test work at multiple audit clients."
   Never: "One candidate has audit experience."

2. Cite once per claim, at the point of the claim. Do not repeat the same
   citation at the end of a sentence that already contains it.

3. If the excerpts do not support an answer, say so plainly. Do not fill gaps
   with plausible-sounding detail.

4. Distinguish evidence from keywords. A candidate listing "Sarbanes-Oxley,
   audit controls" as skills is weaker than one describing concrete engagements.
   Say which is which.

5. Never infer or comment on age, gender, ethnicity, nationality, marital
   status, or any protected attribute — even if the excerpt mentions it.

6. Be concise. Lead with the strongest candidates. Note real gaps.

Answer in the same language as the question."""


def _format_context(ranked: list[RankedHit]) -> str:
    """Susun chunk jadi konteks untuk LLM.

    Label citation ditulis persis seperti format yang diminta di prompt.
    Kalau formatnya berbeda antara konteks dan instruksi, model akan
    mengarang format sendiri dan citation_verifier gagal mencocokkan.
    """
    parts = []
    for r in ranked:
        h = r.hit
        parts.append(
            f"{h.citation()} category={h.category} | section={h.section_type}\n"
            f"{h.text.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _to_evidence(ranked: list[RankedHit]) -> list[Evidence]:
    """Konversi hasil rerank jadi objek Evidence.

    Quote dipotong 200 karakter — cukup untuk verifikasi manual, tidak
    membebani payload UI. Teks lengkap tetap bisa diambil lewat
    vector_store.get_resume_chunks kalau pengguna ingin melihat konteks.
    """
    return [
        Evidence(
            resume_id=r.hit.resume_id,
            chunk_index=r.hit.chunk_index,
            quote=r.hit.text.strip()[:200],
            section=r.hit.section_type,
        )
        for r in ranked
    ]


def run_retrieval(
    query: str,
    search_query: str | None = None,
    category_filter: list[str] | None = None,
    section_filter: list[str] | None = None,
    top_k: int = TOP_K,
    fetch: int = FETCH_N,
    vs: VectorStore | None = None,
    client: OpenAI | None = None,
    model: str = MODEL_ANSWER,
) -> tuple[str, list[Evidence], list[UsageRecord]]:
    """Cari kandidat dan hasilkan jawaban terkutip.

    `query` adalah pertanyaan asli pengguna — dipakai untuk menghasilkan
    jawaban. `search_query` adalah versi yang dioptimalkan supervisor
    untuk pencarian; kalau None, dipakai query asli.

    Pemisahan ini penting: "siapa saja kandidat finance yang pernah
    memimpin audit?" adalah pertanyaan yang baik untuk LLM, tapi query
    pencarian yang lebih baik adalah "led financial audit engagements".

    Returns (jawaban, bukti, catatan usage).
    """
    vs = vs or VectorStore()
    oa = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    usage: list[UsageRecord] = []

    # Ambil kandidat mentah. dedupe aktif agar reranker menilai kandidat
    # unik, bukan beberapa potongan dari orang yang sama.
    candidates = vs.search(
        search_query or query,
        top_k=fetch,
        fetch=fetch,
        category=category_filter or None,
        section_type=section_filter or None,
        dedupe_by_resume=True,
    )

    if not candidates:
        return (
            "Tidak ada kandidat yang cocok dengan kriteria tersebut di korpus.",
            [],
            usage,
        )

    ranked, rerank_stats = rerank(query, candidates, top_k=top_k, client=oa)
    usage.append(UsageRecord(
        component="rerank",
        model=rerank_stats["model"],
        input_tokens=rerank_stats["input_tokens"],
        output_tokens=rerank_stats["output_tokens"],
    ))

    resp = oa.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"QUESTION: {query}\n\nRESUME EXCERPTS:\n\n{_format_context(ranked)}"},
        ],
        # temperature rendah: jawaban harus konsisten dan terikat sumber,
        # bukan bervariasi. Query yang sama harus memberi jawaban serupa.
        temperature=0.2,
    )

    u = resp.usage
    usage.append(UsageRecord(
        component="answer",
        model=model,
        input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens,
    ))

    return resp.choices[0].message.content, _to_evidence(ranked), usage


if __name__ == "__main__":
    from src.agents.schemas import UsageSummary

    vs = VectorStore()

    tests = [
        ("Siapa kandidat dengan pengalaman audit keuangan yang paling kuat?", None, None),
        ("Cari kandidat banking yang punya pengalaman risk management",
         "banking risk management experience", ["BANKING"]),
        ("Adakah kandidat yang pernah memimpin tim besar?",
         "led large team operational management", None),
    ]

    for q, sq, cat in tests:
        print("=" * 74)
        print(f"Q: {q}")
        if cat:
            print(f"   filter: {cat}")
        print("=" * 74)

        answer, evidence, usage = run_retrieval(q, search_query=sq, category_filter=cat, vs=vs)

        print(f"\n{answer}\n")

        print("BUKTI:")
        for e in evidence:
            print(f"  {e.citation()} {e.section:14s} {e.quote[:75]}...")

        summary = UsageSummary(records=usage)
        print(f"\n  usage: in={summary.total_input:,} out={summary.total_output:,} "
              f"${summary.total_cost_usd:.6f}")
        for r in summary.per_component():
            print(f"    {r['component']:10s} {r['model']:14s} ${r['cost_usd']:.6f}")
        print()