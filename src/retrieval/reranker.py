"""Reranking kandidat dengan LLM.

Alur: vector_store mengembalikan 20 kandidat "cukup mirip" → reranker
membaca isinya dan menilai mana yang benar-benar menjawab query → 5 teratas.

KENAPA RERANKER DIPERLUKAN — bukti dari check_02_retrieval.py:

Query "financial audit and compliance" mengembalikan skor 0.6014–0.6328.
Semuanya "cukup mirip", tapi isinya berbeda kualitas:

  peringkat 3: "Information System Audit and Control Association (ISACA)
                Sarbanes-Oxley  Project risk and controls"
                → daftar sertifikasi, bukan bukti pengalaman

  peringkat 4: "...at multiple audit clients, including leading the sales
                and inventory test work of an international company"
                → bukti pengalaman nyata

Embedding tidak bisa membedakan keduanya — jarak vektornya hampir sama.
Reranker membaca dan menilai.

KENAPA CHUNK PENUH, BUKAN TRUNCATED:

Guideline awal menyarankan truncate ke 150 token untuk hemat biaya. Itu
dibuat saat model reranker belum ditentukan. Setelah verifikasi, reranker
memakai gpt-4.1-nano ($0.10/$0.40 per 1M) — 12x lebih murah dari asumsi.

  chunk penuh  : ~5.400 tok input → $0.00060/query
  truncated    : ~3.400 tok input → $0.00040/query
  selisih 2.000 query development: $0.40

$0.40 tidak layak dibeli dengan risiko kehilangan sinyal pembeda, yang
justru ada di detail — kata kerja, konteks, angka pencapaian.

Truncation dicatat sebagai tuas pertama kalau biaya membengkak.

MODEL: gpt-4.1-nano
Reranking dipanggil di setiap query terhadap 20 kandidat — volume
tertinggi di sistem. Model reasoning DIHINDARI: uji Tahap 4 menunjukkan
gpt-5-nano membakar 100 token reasoning untuk prompt "Say OK", dan itu
ditagih sebagai output (4x tarif input).
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.vector_store import SearchHit

load_dotenv()

MODEL_RERANK = os.getenv("MODEL_RERANK", "gpt-4.1-nano")

# Batas atas payload per kandidat. Chunk normal ~250 token, jadi batas ini
# tidak memotong apa pun sekarang. Fungsinya jaga-jaga kalau ablation
# mengubah chunk_size dan payload membengkak tanpa disadari.
MAX_CHARS_PER_CANDIDATE = 1400

PRICE = {"in": 0.10, "out": 0.40}   # USD per 1M token, gpt-4.1-nano


SYSTEM_PROMPT = """You rank resume excerpts by how well they answer a recruiter's query.

Judge on EVIDENCE, not keyword overlap:
- Concrete experience, achievements with numbers, specific responsibilities → HIGH
- Bare skill lists, certification names, generic phrases → LOW

A candidate listing "Sarbanes-Oxley, audit controls" as keywords is WEAKER
than one describing "led inventory test work across multiple audit clients."

Return ONLY a JSON object in exactly this shape:
{"ranking": [{"id": <candidate number>, "score": <0-10>, "why": "<max 12 words>"}]}

The "ranking" array must be ordered best first and include every candidate.
Do NOT prefix items with [1], [2] — that is the input format, not the output."""


@dataclass
class RankedHit:
    """SearchHit + penilaian reranker.

    vector_score dan rerank_score disimpan terpisah agar bisa dianalisis
    di evaluation harness — seberapa sering reranker mengubah urutan.
    """
    hit: SearchHit
    rerank_score: float
    reason: str
    original_rank: int

    @property
    def vector_score(self) -> float:
        return self.hit.score


def _build_payload(hits: list[SearchHit]) -> str:
    """Susun kandidat jadi teks untuk prompt.

    Metadata (category, section) disertakan karena membawa konteks yang
    tidak selalu tersirat dari teks — reranker jadi tahu "ini bagian
    Experience dari kandidat BANKING", bukan sekadar potongan kalimat.
    """
    parts = []
    for i, h in enumerate(hits, 1):
        text = h.text[:MAX_CHARS_PER_CANDIDATE].strip()
        parts.append(
            f"[{i}] category={h.category} | section={h.section_type}\n{text}"
        )
    return "\n\n".join(parts)


def _parse_response(content: str, n: int, debug: bool = False) -> list[dict]:
    """Parse JSON dari respons LLM.

    LLM kadang membungkus JSON dalam markdown fence meski diminta tidak.
    Fence dibuang sebelum parsing.
    """
    txt = content.strip()
    if txt.startswith("```"):
        txt = "\n".join(txt.split("\n")[1:-1]).strip()

    try:
        data = json.loads(txt)
    except json.JSONDecodeError as e:
        if debug:
            # Respons mentah dicetak agar penyebab kegagalan bisa
            # didiagnosis — tanpa ini, fallback menyembunyikan masalah.
            print(f"\n  [parse gagal: {e}]")
            print(f"  raw (400 char pertama):\n  {content[:400]}\n")
        return []

        # JSON mode mengembalikan objek. Array polos tetap diterima agar
    # kompatibel kalau response_format dimatikan saat eksperimen.
    items = data.get("ranking", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    # Validasi: id harus dalam rentang kandidat yang dikirim.
    # LLM sesekali mengarang nomor di luar rentang.
    return [d for d in items if isinstance(d, dict)
            and isinstance(d.get("id"), int) and 1 <= d["id"] <= n]

def rerank(
    query: str,
    hits: list[SearchHit],
    top_k: int = 5,
    model: str = MODEL_RERANK,
    client: OpenAI | None = None,
    debug: bool = False,
) -> tuple[list[RankedHit], dict]:
    """Rerank kandidat. Returns (hasil, usage stats).

    Kalau parsing gagal, fallback ke urutan vector search asli — sistem
    tetap jalan meski reranker bermasalah. Kegagalan reranker tidak boleh
    mematikan retrieval.
    """
    if not hits:
        return [], {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    oa = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = oa.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"QUERY: {query}\n\nCANDIDATES:\n{_build_payload(hits)}"},
        ],
        # temperature rendah: penilaian relevansi harus konsisten,
        # bukan kreatif. Query yang sama harus memberi urutan yang sama.
        temperature=0,
        # JSON mode: memaksa output JSON valid di level decoding, bukan
        # sekadar instruksi prompt. Tanpa ini, model sempat meniru format
        # payload input ("[1] {...}") dan parsing gagal.
        response_format={"type": "json_object"},
    )

    u = resp.usage
    stats = {
        "component": "rerank",
        "model": model,
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "cost_usd": (u.prompt_tokens * PRICE["in"]
                     + u.completion_tokens * PRICE["out"]) / 1_000_000,
    }

    parsed = _parse_response(resp.choices[0].message.content, len(hits), debug=debug)

    if not parsed:
        # Fallback: pertahankan urutan vector search
        stats["fallback"] = True
        return [
            RankedHit(hit=h, rerank_score=0.0, reason="rerank failed",
                      original_rank=i + 1)
            for i, h in enumerate(hits[:top_k])
        ], stats

    ranked = [
        RankedHit(
            hit=hits[d["id"] - 1],
            rerank_score=float(d.get("score", 0)),
            reason=str(d.get("why", ""))[:80],
            original_rank=d["id"],
        )
        for d in parsed
    ]

    # LLM diminta mengurutkan, tapi urutkan ulang berdasarkan skor untuk
    # berjaga — kadang urutan array tidak konsisten dengan skor yang diberi.
    ranked.sort(key=lambda r: r.rerank_score, reverse=True)

    return ranked[:top_k], stats


if __name__ == "__main__":
    from src.retrieval.vector_store import VectorStore

    vs = VectorStore()

    queries = [
        "financial audit and compliance experience",
        "managed a large team and improved operational efficiency",
        "python machine learning deployment",
    ]

    total_cost = 0.0

    for q in queries:
        print("=" * 74)
        print(f"QUERY: {q}")
        print("=" * 74)

        # fetch 20, dedup aktif — reranker menilai kandidat unik
        candidates = vs.search(q, top_k=20, fetch=20)
        ranked, stats = rerank(q, candidates, top_k=5, debug=True)
        total_cost += stats["cost_usd"]

        print("\nSEBELUM rerank (urutan vector):")
        for i, h in enumerate(candidates[:5], 1):
            print(f"  {i}. [{h.score:.4f}] {h.citation()} {h.category:20s} "
                  f"{h.text[:70].strip()}")

        print("\nSESUDAH rerank:")
        for i, r in enumerate(ranked, 1):
            moved = r.original_rank - i
            arrow = f"↑{moved}" if moved > 0 else (f"↓{-moved}" if moved < 0 else " =")
            print(f"  {i}. [{r.rerank_score:4.1f}] {arrow:>4s} "
                  f"{r.hit.citation()} {r.hit.category:20s} {r.reason}")
            print(f"        {r.hit.text[:70].strip()}")

        print(f"\n  tokens: in={stats['input_tokens']:,} "
              f"out={stats['output_tokens']:,}  ${stats['cost_usd']:.6f}")
        print()

    print("=" * 74)
    print(f"Total {len(queries)} query: ${total_cost:.6f}")
    print(f"Estimasi per 1.000 query: ${total_cost / len(queries) * 1000:.4f}")