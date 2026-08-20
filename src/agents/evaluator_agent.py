"""Evaluator agent — menilai kandidat terhadap job description.

BEDA DARI RETRIEVAL AGENT:
  retrieval  → "siapa yang punya pengalaman audit?"  → prosa + bukti
  evaluator  → "nilai 3 kandidat ini untuk role X"   → objek terstruktur

Output berupa Pydantic, bukan prosa, karena hasilnya harus bisa
diurutkan, difilter, dan ditampilkan berdampingan di UI. Prosa tidak bisa.

DUA MODE:
1. Query menyebut kandidat spesifik → ambil profil lengkap via
   get_resume_chunks (nol biaya API, tidak perlu embedding)
2. Query hanya menyebut kriteria → cari dulu via retrieval, baru nilai

STRUCTURED OUTPUT VIA PYDANTIC:
Skema memaksa model menyertakan evidence untuk setiap skill match.
Model yang lupa akan gagal validasi — citation enforcement di level
struktur, bukan sekadar instruksi prompt.

Validasi bahwa status "strong" harus punya evidence TIDAK dipasang
sebagai validator Pydantic, karena model kadang benar-benar tidak
menemukan bukti untuk skill "missing" — dan itu informasi yang valid.
Pemeriksaan itu dilakukan di guardrails/citation_verifier.py.

MODEL: gpt-4.1-mini (MODEL_ANSWER)
Butuh kepatuhan skema yang tinggi. Kandidat ablation vs gpt-5.4-mini.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.schemas import (
    CandidateEvaluation,
    Evidence,
    EvaluationResult,
    SkillMatch,
    UsageRecord,
)
from src.retrieval.vector_store import SearchHit, VectorStore

load_dotenv()

MODEL_ANSWER = os.getenv("MODEL_ANSWER", "gpt-4.1-mini")

# Batas kandidat yang dinilai sekaligus. Lebih dari 3 membuat payload
# membengkak (tiap kandidat bisa 5-10 chunk) dan kualitas penilaian
# menurun karena model harus membagi perhatian.
MAX_CANDIDATES = 3

# Batas chunk per kandidat. Resume rata-rata 9 chunk; 8 sudah mencakup
# hampir seluruh profil tanpa membuat payload berlebihan.
MAX_CHUNKS_PER_CANDIDATE = 8


SYSTEM_PROMPT = """You evaluate job candidates against stated requirements, using ONLY the resume excerpts provided.

Return a JSON object in exactly this shape:
{
  "job_context": "<the criteria you evaluated against, one sentence>",
  "candidates": [
    {
      "resume_id": "<id>",
      "overall_score": <0-10>,
      "summary": "<max 2 sentences>",
      "skill_matches": [
        {
          "skill": "<requirement>",
          "status": "strong" | "partial" | "missing",
          "evidence": [
            {"resume_id": "<id>", "chunk_index": <n>,
             "quote": "<exact text from the excerpt, max 200 chars>",
             "section": "<section name>"}
          ]
        }
      ],
      "gaps": ["<what is missing or unclear>"],
      "evidence": [
        {"resume_id": "<id>", "chunk_index": <n>, "quote": "<exact text>",
         "section": "<section>"}
      ]
    }
  ]
}

RULES:

1. Every quote must be COPIED EXACTLY from an excerpt. Never paraphrase
   inside a quote field. If you cannot find supporting text, use status
   "missing" with an empty evidence array.

2. Score on evidence strength, not keyword presence. A candidate describing
   concrete engagements scores higher than one listing the same words as skills.

3. Gaps must be specific. "Lacks cloud experience" is useful.
   "Could be stronger" is not.

4. Never infer or comment on age, gender, ethnicity, nationality, marital
   status, or any protected attribute.

5. LANGUAGE: detect the language of the REQUIREMENTS text and write
   "job_context", "summary", and "gaps" in that language. If requirements
   are in Indonesian, these fields must be Indonesian. Skill names stay
   as written in the requirements.

No markdown. No text outside the JSON object."""


def _format_candidate(resume_id: str, chunks: list[SearchHit]) -> str:
    """Susun profil satu kandidat jadi teks untuk prompt.

    Chunk sudah terurut by chunk_index dari get_resume_chunks, jadi
    urutan section mengikuti urutan asli di resume.
    """
    parts = [f"=== CANDIDATE {resume_id} (category: {chunks[0].category}) ==="]
    for c in chunks[:MAX_CHUNKS_PER_CANDIDATE]:
        parts.append(f"{c.citation()} [{c.section_type}]\n{c.text.strip()}")
    return "\n\n".join(parts)


def _parse_evaluation(content: str, valid_ids: set[str]) -> EvaluationResult | None:
    """Parse dan validasi respons LLM.

    valid_ids dipakai membuang kandidat yang tidak ada di input —
    model sesekali mengarang resume_id. Ini lapisan pertama pertahanan
    terhadap halusinasi sumber; verifikasi kutipan dilakukan terpisah
    di citation_verifier.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    candidates = []
    for c in data.get("candidates", []):
        rid = str(c.get("resume_id", ""))
        if rid not in valid_ids:
            continue

        skill_matches = []
        for sm in c.get("skill_matches", []):
            evs = [
                Evidence(
                    resume_id=str(e.get("resume_id", rid)),
                    chunk_index=int(e.get("chunk_index", 0)),
                    quote=str(e.get("quote", ""))[:200],
                    section=str(e.get("section", "unknown")),
                )
                for e in sm.get("evidence", [])
                if isinstance(e, dict) and e.get("quote")
            ]
            skill_matches.append(SkillMatch(
                skill=str(sm.get("skill", "")),
                status=sm.get("status", "missing"),
                evidence=evs,
            ))

        top_evidence = [
            Evidence(
                resume_id=str(e.get("resume_id", rid)),
                chunk_index=int(e.get("chunk_index", 0)),
                quote=str(e.get("quote", ""))[:200],
                section=str(e.get("section", "unknown")),
            )
            for e in c.get("evidence", [])
            if isinstance(e, dict) and e.get("quote")
        ]

        try:
            candidates.append(CandidateEvaluation(
                resume_id=rid,
                overall_score=int(c.get("overall_score", 0)),
                summary=str(c.get("summary", "")),
                skill_matches=skill_matches,
                gaps=[str(g) for g in c.get("gaps", [])],
                evidence=top_evidence,
            ))
        except Exception:
            # Kandidat yang gagal validasi Pydantic dilewati, bukan
            # membatalkan seluruh hasil — satu kandidat cacat tidak
            # boleh menghapus penilaian kandidat lain.
            continue

    if not candidates:
        return None

    return EvaluationResult(
        job_context=str(data.get("job_context", "")),
        candidates=candidates,
    )


class EvaluatorAgent:
    def __init__(self, vs: VectorStore | None = None,
                 client: OpenAI | None = None):
        self.vs = vs or VectorStore()
        self._oa = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _gather(self, resume_ids: list[str]) -> dict[str, list[SearchHit]]:
        """Ambil profil lengkap tiap kandidat.

        get_resume_chunks memakai scroll, bukan vector search — nol biaya
        API karena tidak perlu embedding query.
        """
        out = {}
        for rid in resume_ids[:MAX_CANDIDATES]:
            chunks = self.vs.get_resume_chunks(rid)
            if chunks:
                out[rid] = chunks
        return out

    def run(
        self,
        requirements: str,
        resume_ids: list[str] | None = None,
        search_query: str | None = None,
        category_filter: list[str] | None = None,
        model: str = MODEL_ANSWER,
    ) -> tuple[EvaluationResult | None, list[UsageRecord]]:
        """Nilai kandidat terhadap requirements.

        Kalau resume_ids diberikan, kandidat itu yang dinilai.
        Kalau tidak, cari dulu berdasarkan search_query.
        """
        usage: list[UsageRecord] = []

        if not resume_ids:
            hits = self.vs.search(
                search_query or requirements,
                top_k=MAX_CANDIDATES,
                fetch=20,
                category=category_filter or None,
                dedupe_by_resume=True,
            )
            resume_ids = [h.resume_id for h in hits]

        profiles = self._gather(resume_ids)
        if not profiles:
            return None, usage

        context = "\n\n".join(_format_candidate(rid, ch)
                              for rid, ch in profiles.items())

        resp = self._oa.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"REQUIREMENTS:\n{requirements}\n\n{context}"},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        u = resp.usage
        usage.append(UsageRecord(
            component="evaluate", model=model,
            input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
        ))

        result = _parse_evaluation(resp.choices[0].message.content,
                                   set(profiles.keys()))
        return result, usage


if __name__ == "__main__":
    from src.agents.schemas import UsageSummary

    agent = EvaluatorAgent()

    reqs = (
        "Financial Audit Manager: minimal 5 tahun pengalaman audit, "
        "penguasaan SOX compliance, pengalaman memimpin tim audit, "
        "sertifikasi CPA."
    )

    print("=" * 74)
    print("REQUIREMENTS:")
    print(reqs)
    print("=" * 74)

    result, usage = agent.run(
        requirements=reqs,
        search_query="financial audit SOX compliance team leadership CPA",
    )

    if not result:
        print("Evaluasi gagal.")
    else:
        print(f"\nKonteks: {result.job_context}\n")

        for c in result.ranked():
            print("-" * 74)
            print(f"[{c.resume_id}]  skor {c.overall_score}/10")
            print(f"  {c.summary}\n")

            for sm in c.skill_matches:
                mark = {"strong": "++", "partial": " ~", "missing": " -"}.get(sm.status, "?")
                print(f"  {mark} {sm.skill}")
                for e in sm.evidence[:1]:
                    print(f"       {e.citation()} \"{e.quote[:90]}...\"")

            if c.gaps:
                print(f"\n  Gap:")
                for g in c.gaps:
                    print(f"    - {g}")
            print()

        s = UsageSummary(records=usage)
        print("-" * 74)
        print(f"usage: in={s.total_input:,} out={s.total_output:,} "
              f"${s.total_cost_usd:.6f}")