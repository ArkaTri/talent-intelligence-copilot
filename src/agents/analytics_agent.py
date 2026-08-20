"""Analytics agent — menjawab pertanyaan agregat dari ISI TEKS.

KENAPA BUKAN TEXT-TO-SQL:
Rencana awal memakai SQL terhadap metadata. Dibatalkan setelah verifikasi
(notebooks/check_analytics.py) menunjukkan label kategori tidak tepercaya.

Pertanyaan uji "berapa kandidat dengan kemampuan cloud?":
  label INFORMATION-TECHNOLOGY : 120 resume
  teks menyebut AWS/Azure      :  29 resume
  irisan                       :   9 resume
  cloud tapi bukan label IT    :  20 (69% terlewat)
  label IT tanpa cloud         : 111 (93% false positive)

Kandidat AWS tersebar di CONSULTANT (8), ADVOCATE (5), AGRICULTURE (4),
CONSTRUCTION, BANKING. Label bukan proksi untuk kemampuan teknis.

DUA TUGAS LLM DI SINI:

1. EKSPANSI ISTILAH
   MatchText mencocokkan kata, bukan makna. "Kemampuan cloud" harus
   diterjemahkan jadi ["AWS", "Azure", "GCP", "cloud computing"] —
   kandidat yang menulis "Amazon Web Services" tanpa singkatan tidak
   tertangkap query "AWS".

2. MENOLAK PERTANYAAN YANG TIDAK BISA DIJAWAB DENGAN BENAR
   Pertanyaan berbasis label ("berapa kandidat IT?") ditolak dengan
   penjelasan. Menolak adalah fitur — sistem yang tahu batas
   pengetahuannya lebih bernilai daripada yang selalu menjawab.

MODEL: gpt-4.1-nano
Tugasnya klasifikasi + ekstraksi istilah, tidak butuh penalaran dalam.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

from src.agents.schemas import AnalyticsResult, UsageRecord

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-4.1-nano")

# Batas scroll. 22.866 chunk total, jadi 10.000 sudah jauh melebihi
# hasil realistis untuk istilah spesifik (AWS hanya 47 chunk).
SCROLL_LIMIT = 10_000
SCROLL_BATCH = 256


PLANNER_PROMPT = """You convert recruiter questions into text-search terms over a resume corpus.

The corpus has 2,481 resumes split into 22,866 text chunks. Search is EXACT
WORD MATCHING — not semantic. So you must list every surface form a candidate
might have written.

Return JSON:
{
  "answerable": true|false,
  "refusal_reason": "<why not, if answerable is false>",
  "terms": ["term1", "term2", ...],
  "caveat": "<limitation the user should know, or null>"
}

ANSWERABLE — questions about what resumes SAY.

CRITICAL: search tokenizes on whitespace and matches ANY word. A term like
"Project Management Professional" will match every resume mentioning
"project management" — hundreds of false positives.

So: use SHORT, DISTINCTIVE terms. One or two words maximum. Prefer acronyms
and unique compounds over generic phrases.

  "how many mention AWS"       -> ["AWS", "Azure", "GCP"]
  "who has PMP certification"  -> ["PMP"]
  "candidates with Six Sigma"  -> ["Sigma"]
  "cloud skills"               -> ["AWS", "Azure", "GCP", "Kubernetes"]

Never use terms containing common words like: project, management,
professional, experience, business, service, development.

NOT ANSWERABLE — questions that depend on the dataset's category labels:
  "how many IT candidates"
  "which category has the most resumes"
  "average experience of finance candidates"

Why: the category labels in this dataset are unreliable. A resume titled
"IT COMPLIANCE AUDITOR" is labelled APPAREL. A resume about loan processing
is labelled CHEF. Counting by label produced 93% false positives in testing.
Refuse these, explain the label problem, and suggest a text-based alternative.

Also refuse questions about protected attributes (age, gender, ethnicity,
nationality, marital status).

Keep terms to at most 6. No markdown."""


def _plan(question: str, client: OpenAI, model: str) -> tuple[dict, UsageRecord]:
    """Minta LLM mengekstrak istilah pencarian atau menolak pertanyaan."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        # JSON mode: sama seperti di reranker, mencegah model menghasilkan
        # format bebas yang gagal di-parse.
        response_format={"type": "json_object"},
    )

    u = resp.usage
    usage = UsageRecord(
        component="analytics", model=model,
        input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
    )

    try:
        plan = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        # Fallback konservatif: kalau planner gagal, tolak daripada
        # menghasilkan angka yang tidak bisa dipertanggungjawabkan.
        plan = {
            "answerable": False,
            "refusal_reason": "Tidak dapat memproses pertanyaan ini.",
            "terms": [],
        }

    return plan, usage


class AnalyticsAgent:
    def __init__(self, collection: str = COLLECTION,
                 client: OpenAI | None = None,
                 qdrant: QdrantClient | None = None):
        self.collection = collection
        self._oa = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._qd = qdrant or QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )

    def _resumes_matching(self, term: str) -> tuple[set[str], dict[str, int]]:
        """Cari resume_id unik yang memuat `term`, plus distribusi kategori.

        Menghitung resume, bukan chunk — satu resume bisa punya beberapa
        chunk yang cocok, dan menghitung chunk akan melebih-lebihkan
        jumlah kandidat.
        """
        ids: set[str] = set()
        by_cat: dict[str, int] = {}
        offset = None

        while True:
            points, offset = self._qd.scroll(
                self.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="text", match=MatchText(text=term))
                ]),
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=["resume_id", "category"],
                with_vectors=False,
            )
            for p in points:
                rid = p.payload["resume_id"]
                if rid not in ids:
                    ids.add(rid)
                    cat = p.payload["category"]
                    by_cat[cat] = by_cat.get(cat, 0) + 1

            if offset is None or len(ids) > SCROLL_LIMIT:
                break

        return ids, by_cat

    def run(self, question: str,
            model: str = MODEL_ROUTER) -> tuple[AnalyticsResult, list[UsageRecord]]:
        """Jawab pertanyaan agregat, atau tolak dengan penjelasan."""
        plan, usage_rec = _plan(question, self._oa, model)
        usage = [usage_rec]

        if not plan.get("answerable", False):
            return AnalyticsResult(
                question=question,
                answerable=False,
                refusal_reason=plan.get("refusal_reason",
                                        "Pertanyaan ini tidak dapat dijawab "
                                        "secara akurat dari data yang tersedia."),
            ), usage

        terms = [t for t in plan.get("terms", []) if isinstance(t, str)][:6]

        # MatchText melakukan tokenisasi — frasa multi-kata pecah jadi
        # kata terpisah dan dicocokkan dengan OR. "Project Management
        # Professional" menangkap setiap resume yang menyebut "project
        # management" (ratusan), bukan hanya pemegang sertifikasi PMP.
        # Frasa >2 kata karena itu dibuang.
        terms = [t for t in terms if len(t.split()) <= 2]
        if not terms:
            return AnalyticsResult(
                question=question,
                answerable=False,
                refusal_reason="Tidak ada istilah pencarian yang dapat diekstrak.",
            ), usage

        # Union across terms: kandidat yang menyebut salah satu varian
        # dihitung sekali. Menjumlahkan per-term akan menghitung ganda
        # kandidat yang menyebut "AWS" DAN "Amazon Web Services".
        all_ids: set[str] = set()
        merged_cat: dict[str, int] = {}
        per_term: dict[str, int] = {}

        for t in terms:
            ids, by_cat = self._resumes_matching(t)
            per_term[t] = len(ids)
            new = ids - all_ids
            all_ids |= ids
            # Distribusi hanya dihitung untuk resume yang belum tercatat,
            # agar total breakdown = total count.
            for cat, _ in by_cat.items():
                pass
            for rid in new:
                pass

        # Hitung ulang distribusi kategori dari himpunan final agar konsisten
        merged_cat = self._category_distribution(all_ids, terms)

        return AnalyticsResult(
            question=question,
            answerable=True,
            count=len(all_ids),
            unit="resumes",
            terms_searched=terms,
            per_term=per_term,
            breakdown=dict(sorted(merged_cat.items(),
                                  key=lambda x: x[1], reverse=True)[:10]),
            caveat=plan.get("caveat") or (
                "Pencarian berbasis kecocokan kata persis, bukan makna. "
                "Kandidat yang menuliskan istilah dengan bentuk lain mungkin "
                "tidak tertangkap."
            ),
        ), usage

    def _category_distribution(self, resume_ids: set[str],
                               terms: list[str]) -> dict[str, int]:
        """Hitung distribusi kategori untuk himpunan resume final.

        Dilakukan terpisah agar total breakdown selalu sama dengan count —
        kalau dihitung per-term, resume yang cocok beberapa term akan
        terhitung ganda.
        """
        dist: dict[str, int] = {}
        seen: set[str] = set()

        for t in terms:
            offset = None
            while True:
                points, offset = self._qd.scroll(
                    self.collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="text", match=MatchText(text=t))
                    ]),
                    limit=SCROLL_BATCH,
                    offset=offset,
                    with_payload=["resume_id", "category"],
                    with_vectors=False,
                )
                for p in points:
                    rid = p.payload["resume_id"]
                    if rid in resume_ids and rid not in seen:
                        seen.add(rid)
                        cat = p.payload["category"]
                        dist[cat] = dist.get(cat, 0) + 1
                if offset is None:
                    break

        return dist


if __name__ == "__main__":
    from src.agents.schemas import UsageSummary

    agent = AnalyticsAgent()

    questions = [
        "Berapa banyak kandidat yang menyebut kemampuan cloud?",
        "Berapa kandidat yang punya sertifikasi PMP?",
        "Berapa banyak kandidat IT di korpus ini?",          # harus ditolak
        "Kategori mana yang paling banyak resumenya?",        # harus ditolak
        "Berapa kandidat laki-laki yang punya pengalaman audit?",  # harus ditolak
    ]

    for q in questions:
        print("=" * 74)
        print(f"Q: {q}")
        print("=" * 74)

        result, usage = agent.run(q)
        s = UsageSummary(records=usage)

        if not result.answerable:
            print(f"  DITOLAK: {result.refusal_reason}")
        else:
            print(f"  Jumlah   : {result.count} {result.unit}")
            print(f"  Per term :")
            for t, n in result.per_term.items():
                print(f"    {t:24s} {n:4,} resume")
            print(f"  Sebaran  :")
            for cat, n in list(result.breakdown.items())[:8]:
                print(f"    {cat:24s} {n:4,}")
            if result.caveat:
                print(f"  Catatan  : {result.caveat}")

        print(f"\n  usage: ${s.total_cost_usd:.6f}\n")