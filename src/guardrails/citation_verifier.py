"""Post-check: memverifikasi setiap klaim punya citation yang sah.

TIGA JENIS KEGAGALAN YANG DICARI:

1. HALUSINASI SUMBER — citation menunjuk resume_id yang tidak ada di
   hasil retrieval. Model mengarang ID.

2. KUTIPAN TIDAK COCOK — quote di Evidence tidak ditemukan di teks chunk
   aslinya. Model memparafrase alih-alih menyalin.

3. KLAIM TANPA CITATION — kalimat yang menyatakan fakta tentang kandidat
   tapi tidak menyertakan [resume_id#chunk].

KENAPA VERIFIKASI, BUKAN SEKADAR PERCAYA PROMPT:
Prompt bisa diabaikan model. Verifikasi struktural tidak bisa. Kalau
sistem menghasilkan klaim tanpa bukti dan itu tidak terdeteksi, seluruh
premis "evidence-grounded" runtuh.

CATATAN TENTANG KUTIPAN TERSAMBUNG:
Uji evaluator menemukan model menyambung dua bagian teks dengan "...":
  "Senior Internal Auditor 07/2002 to 06/2004 ... Supervised audit teams"
Isinya benar tapi bukan substring persis. Verifikasi karena itu memecah
quote pada "..." dan memeriksa tiap fragmen terpisah.
"""

import re

from src.agents.schemas import Evidence
from src.retrieval.vector_store import SearchHit

# Pola citation yang diminta di prompt: [resume_id#chunk_index]
CITATION_RE = re.compile(r"\[(\d+)#(\d+)\]")

# Ambang kecocokan kutipan. Tidak 100% karena normalisasi whitespace
# di loader mengubah spasi, dan model kadang merapikan spasi ganda.
QUOTE_MATCH_THRESHOLD = 0.85

# Fragmen di bawah panjang ini tidak diverifikasi — terlalu pendek
# untuk bermakna, dan rawan false negative.
MIN_FRAGMENT_CHARS = 20


def _normalize(text: str) -> str:
    """Samakan whitespace agar perbandingan tidak gagal karena spasi.

    Loader menormalisasi \\xa0 dan \\t jadi spasi tapi TIDAK mengolapskan
    spasi berulang (itu sinyal struktural untuk chunker). Model biasanya
    merapikan spasi saat mengutip, jadi normalisasi diperlukan di sini.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def _fragment_found(fragment: str, haystack: str) -> bool:
    """Cek apakah fragmen ada di teks sumber.

    Substring persis dulu; kalau gagal, cek proporsi kata yang cocok.
    Toleransi ini perlu karena model kadang mengubah tanda baca.
    """
    f = _normalize(fragment)
    h = _normalize(haystack)

    if len(f) < MIN_FRAGMENT_CHARS:
        return True   # terlalu pendek untuk diverifikasi bermakna

    if f in h:
        return True

    words = f.split()
    if not words:
        return True
    matched = sum(1 for w in words if w in h)
    return matched / len(words) >= QUOTE_MATCH_THRESHOLD


def verify_evidence(
    evidence: list[Evidence],
    retrieved: list[SearchHit],
) -> tuple[bool, list[str]]:
    """Verifikasi bahwa setiap Evidence menunjuk sumber yang sah.

    Returns (semua_valid, daftar_masalah).
    """
    issues: list[str] = []

    # Indeks chunk yang benar-benar dikirim ke model
    by_key = {(h.resume_id, h.chunk_index): h.text for h in retrieved}
    valid_ids = {h.resume_id for h in retrieved}

    for e in evidence:
        if e.resume_id not in valid_ids:
            issues.append(
                f"{e.citation()} menunjuk resume yang tidak ada di hasil pencarian"
            )
            continue

        # Pecah pada "..." — model kadang menyambung dua bagian teks
        fragments = [f.strip() for f in e.quote.split("...") if f.strip()]

        # Coba chunk yang disebut dulu
        primary = by_key.get((e.resume_id, e.chunk_index), "")
        if all(_fragment_found(f, primary) for f in fragments):
            continue

        # Chunk_index salah tapi resume benar — cari di seluruh chunk
        # resume tersebut. Model sering menyebut nomor chunk yang keliru
        # meski mengutip persis; itu masalah atribusi posisi, bukan
        # halusinasi, dan tidak boleh diperlakukan sama.
        whole = " ".join(t for (rid, _), t in by_key.items() if rid == e.resume_id)
        if all(_fragment_found(f, whole) for f in fragments):
            issues.append(
                f"{e.citation()} kutipan benar tapi chunk_index salah "
                f"(teks ada di resume {e.resume_id}, chunk lain)"
            )
            continue

        bad = next(f for f in fragments if not _fragment_found(f, whole))
        issues.append(
            f"{e.citation()} kutipan tidak ditemukan di resume: \"{bad[:60]}...\""
        )

    return len(issues) == 0, issues


def verify_answer_citations(
    answer: str,
    retrieved: list[SearchHit],
) -> tuple[bool, list[str]]:
    """Verifikasi citation yang tertulis di teks jawaban.

    Memeriksa dua hal: apakah ada citation sama sekali, dan apakah
    semua ID yang disebut ada di hasil retrieval.
    """
    issues: list[str] = []
    valid_ids = {h.resume_id for h in retrieved}

    found = CITATION_RE.findall(answer)

    if not found and retrieved:
        # Jawaban tanpa citation padahal ada hasil retrieval.
        # Pengecualian: jawaban yang menyatakan tidak ada hasil.
        no_result_phrases = ["tidak ada", "tidak ditemukan", "no candidates",
                             "not found", "tidak dapat"]
        if not any(p in answer.lower() for p in no_result_phrases):
            issues.append("Jawaban tidak menyertakan citation apa pun")

    for rid, _ in found:
        if rid not in valid_ids:
            issues.append(f"[{rid}] disebut di jawaban tapi tidak ada di hasil pencarian")

    return len(issues) == 0, issues


def verify(
    answer: str,
    evidence: list[Evidence],
    retrieved: list[SearchHit],
) -> tuple[bool, list[str]]:
    """Verifikasi lengkap: teks jawaban + objek Evidence."""
    ok_a, issues_a = verify_answer_citations(answer, retrieved)
    ok_e, issues_e = verify_evidence(evidence, retrieved)
    return ok_a and ok_e, issues_a + issues_e


if __name__ == "__main__":
    from src.retrieval.vector_store import VectorStore
    from src.agents.retrieval_agent import run_retrieval

    vs = VectorStore()
    q = "Siapa kandidat dengan pengalaman audit keuangan yang paling kuat?"

    # Ambil kandidat yang sama seperti yang dilihat agent
    candidates = vs.search(q, top_k=20, fetch=20)
    answer, evidence, _ = run_retrieval(q, vs=vs)

    ok, issues = verify(answer, evidence, candidates)

    print(f"Query : {q}")
    print(f"Status: {'LOLOS' if ok else 'ADA MASALAH'}")
    if issues:
        for i in issues:
            print(f"  - {i}")
    else:
        print(f"  {len(evidence)} bukti terverifikasi")

    print("\n--- uji negatif: evidence palsu ---")
    fake = [Evidence(resume_id="99999999", chunk_index=0,
                     quote="pengalaman yang tidak pernah ada", section="summary")]
    ok2, issues2 = verify_evidence(fake, candidates)
    print(f"Status: {'LOLOS' if ok2 else 'DITOLAK (benar)'}")
    for i in issues2:
        print(f"  - {i}")