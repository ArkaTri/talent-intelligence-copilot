"""Pecah resume jadi chunk untuk embedding.

Strategi: HYBRID (section-aware dengan fallback).

Alasan hybrid, bukan section-aware murni — dari 01_data_exploration.ipynb:
    Skills      93.6%     Summary          54.1%
    Education   77.4%     Highlights       35.2%
    Experience  69.2%     Accomplishments  30.8%

Coverage terlalu rendah untuk mengandalkan section-aware saja. Sekitar
30% dokumen tidak punya cukup header dan harus lewat splitter generik.

Header dideteksi lewat POLA SPASI, bukan newline. Resume hasil flattening
HTML kehilangan struktur baris — satu dokumen 5.442 karakter hanya punya
2 newline. Runs of spaces adalah satu-satunya sinyal struktural tersisa.
"""

import re
from dataclasses import dataclass, asdict

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_HEADERS_FOR_SECTION_MODE = 2
MIN_CHUNK_CHARS = 100

# Urutan penting: varian panjang ("Professional Summary") harus dicoba
# sebelum yang pendek ("Summary"), kalau tidak akan ter-match sebagian.
SECTION_HEADERS = [
    "Professional Summary",
    "Core Qualifications",
    "Accomplishments",
    "Certifications",
    "Highlights",
    "Experience",
    "Education",
    "Summary",
    "Skills",
]

# Peta ke section_type yang disimpan di payload Qdrant.
# Beberapa header berbeda dipetakan ke tipe yang sama — "Summary" dan
# "Professional Summary" secara semantik identik.
HEADER_TO_TYPE = {
    "professional summary": "summary",
    "summary": "summary",
    "core qualifications": "skills",
    "skills": "skills",
    "highlights": "skills",
    "accomplishments": "accomplishments",
    "certifications": "certifications",
    "experience": "experience",
    "education": "education",
}


@dataclass
class Chunk:
    resume_id: str
    category: str
    section_type: str
    chunk_index: int
    text: str
    char_length: int
    chunking_method: str   # "section" | "fallback" — untuk ablation


def find_headers(text: str) -> list[tuple[int, int, str]]:
    """Cari posisi semua header dalam teks.

    Returns list of (start, end, header_name), terurut berdasarkan posisi.

    Pola: minimal 2 spasi, header, minimal 2 spasi. Word boundary mencegah
    "Experience" di "5 years of experience" ikut ter-match — tapi syarat
    2 spasi di kedua sisi sudah menyaring sebagian besar false positive.
    """
    found = []
    for header in SECTION_HEADERS:
        pattern = rf"\s{{2,}}({re.escape(header)})\s{{2,}}"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append((m.start(), m.end(), header))

    found.sort(key=lambda x: x[0])

    # Buang header yang tumpang tindih. Terjadi karena "Summary" bisa
    # match di dalam rentang "Professional Summary" yang sudah ketemu.
    deduped = []
    last_end = -1
    for start, end, name in found:
        if start >= last_end:
            deduped.append((start, end, name))
            last_end = end
    return deduped


def split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """Pecah teks panjang jadi potongan berukuran `size` dengan overlap.

    Pemotongan diusahakan di batas spasi. Jendela pencarian diperlebar
    dari 100 ke 200 karakter karena teks ini punya spasi berulang —
    dengan jendela sempit, rfind kadang tidak menemukan batas dan
    memotong paksa di tengah kata ("cquisitions" dari "acquisitions").
    """
    if len(text) <= size:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = start + size

        if end < len(text):
            space = text.rfind(" ", start + size - 200, end)
            # Fallback: kalau tetap tidak ketemu, cari spasi PERTAMA
            # setelah batas. Lebih baik chunk sedikit lebih panjang
            # daripada kata terpotong.
            if space <= start:
                nxt = text.find(" ", end)
                space = nxt if nxt != -1 and nxt - end < 100 else end
            end = space

        parts.append(text[start:end].strip())

        if end >= len(text):
            break
        start = end - overlap

    return [p for p in parts if p]


def chunk_resume(
    resume_id: str,
    text: str,
    category: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Pecah satu resume jadi list Chunk.

    Jalur section: kalau ≥2 header terdeteksi, potong per section.
    Jalur fallback: kalau kurang, pakai splitter generik dengan
    section_type "unknown".

    Overlap HANYA berlaku saat memecah section yang terlalu panjang,
    tidak antar-section. Menyambung akhir Education dengan awal Skills
    tidak menambah konteks — hanya mencemari batas semantik yang justru
    ingin dipertahankan.
    """
    headers = find_headers(text)
    chunks: list[Chunk] = []
    idx = 0

    if len(headers) >= MIN_HEADERS_FOR_SECTION_MODE:
        method = "section"

        # Teks sebelum header pertama — biasanya nama & jabatan.
        preamble = text[: headers[0][0]].strip()
        if len(preamble) > 50:
            for piece in split_long_text(preamble, size, overlap):
                chunks.append(Chunk(resume_id, category, "header_info",
                                    idx, piece, len(piece), method))
                idx += 1

        for i, (start, end, name) in enumerate(headers):
            section_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            body = text[end:section_end].strip()

            if len(body) < 20:
                continue

            stype = HEADER_TO_TYPE.get(name.lower(), "other")

            for piece in split_long_text(body, size, overlap):
                chunks.append(Chunk(resume_id, category, stype,
                                    idx, piece, len(piece), method))
                idx += 1
    else:
        method = "fallback"
        for piece in split_long_text(text.strip(), size, overlap):
            chunks.append(Chunk(resume_id, category, "unknown",
                                idx, piece, len(piece), method))
            idx += 1

    # Buang chunk terlalu pendek — tidak cukup informasi untuk
    # dinilai relevansinya, tapi tetap memakan slot di hasil retrieval.
    chunks = [c for c in chunks if c.char_length >= MIN_CHUNK_CHARS]

    # Reindex setelah filtering agar chunk_index tetap berurutan
    for i, c in enumerate(chunks):
        c.chunk_index = i
        
    return chunks

def chunk_dataframe(df, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Chunk seluruh DataFrame. Returns (list[dict], stats dict)."""
    all_chunks = []
    n_section = 0
    n_fallback = 0

    for row in df.itertuples(index=False):
        cs = chunk_resume(str(row.ID), row.Resume_str, row.Category, size, overlap)
        all_chunks.extend(asdict(c) for c in cs)
        if cs and cs[0].chunking_method == "section":
            n_section += 1
        else:
            n_fallback += 1

    stats = {
        "total_docs": len(df),
        "docs_section": n_section,
        "docs_fallback": n_fallback,
        "total_chunks": len(all_chunks),
        "avg_chunks_per_doc": len(all_chunks) / max(len(df), 1),
    }
    return all_chunks, stats


if __name__ == "__main__":
    from src.ingestion.loader import load_resumes

    df = load_resumes()
    chunks, stats = chunk_dataframe(df)

    print()
    for k, v in stats.items():
        print(f"{k:22s} {v:,.2f}" if isinstance(v, float) else f"{k:22s} {v:,}")

    print()
    print("Distribusi section_type:")
    from collections import Counter
    for t, n in Counter(c["section_type"] for c in chunks).most_common():
        print(f"  {t:16s} {n:6,}")

    lens = [c["char_length"] for c in chunks]
    print()
    print(f"Panjang chunk: min {min(lens)}, median {sorted(lens)[len(lens)//2]}, max {max(lens)}")