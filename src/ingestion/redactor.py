"""Redaksi PII sebelum embedding.

Dijalankan SETELAH chunking, SEBELUM embedding.

Kenapa di ingestion, bukan query time:
- Nama yang ikut ter-embed memengaruhi vektor. Dua resume dengan nama
  mirip bisa jadi berdekatan di ruang vektor tanpa alasan substantif.
- Redaksi di query time mengharuskan aplikasi Streamlit memuat model
  NER, membebani memori deployment yang terbatas.

Strategi bertingkat:
- Regex (email, telepon, URL) → SELURUH chunk.
- NER (nama orang) → HANYA 300 karakter pertama dari chunk pertama
  tiap dokumen.

KENAPA LINGKUP NER SESEMPIT ITU:
STATUS NER: DIMATIKAN

Tiga iterasi diuji dan semuanya gagal:
1. NER pada seluruh chunk "header_info" — "Lawson", "Kronos" (software
   HRIS) jadi [NAME]; "Peak volume" jadi "[NAME] volume"; kata kerja
   "Wrote", "Overseen" jadi [NAME].
2. Ditambah filter allowlist + kata jabatan — turun dari 1.698 ke 1.551
   redaksi, rasio kesalahan tidak membaik.
3. Dipersempit ke 300 karakter pertama chunk pertama — turun ke 392,
   tapi sampel manual 5 kasus menunjukkan NOL yang benar-benar nama:
   "Applicant Screening", "Visio", "Outlook", "HR Recruiter" ikut
   teredaksi, dan "Benefits" terpotong jadi "[NAME]fits".

Penyebab: spaCy en_core_web_sm dilatih pada teks berstruktur kalimat.
Data ini hasil flattening HTML — tanpa tanda baca, penuh frasa Title
Case. Model kehilangan sinyal gramatikal dan menebak dari kapitalisasi.

Keputusan: redaksi nama tidak dilakukan. Komponen yang merusak
informasi teknis (Outlook, Visio) demi menangkap segelintir nama
adalah trade-off yang buruk untuk sistem screening.

Didokumentasikan di README sebagai keterbatasan yang diakui.

Penyebabnya: spaCy en_core_web_sm dilatih pada teks berstruktur kalimat
normal. Data ini hasil flattening HTML — tanpa tanda baca kalimat, penuh
frasa Title Case. Model kehilangan sinyal gramatikal dan menebak dari
kapitalisasi saja.

Ditemukan juga bahwa section_type "header_info" tidak selalu berisi
identitas: resume yang header pertamanya muncul jauh di dalam dokumen
membuat konten Experience ikut terklasifikasi header_info.

Membatasi ke 300 karakter pertama chunk pertama mengembalikan asumsi
awal — nama kandidat ada di baris pertama dokumen.

KETERBATASAN YANG DIAKUI:
PII redaction terbatas pada pola deterministik.
NER (spaCy en_core_web_sm) diuji dan ditolak. Pada teks hasil flattening HTML tanpa struktur kalimat, model salah menandai istilah teknis sebagai PERSON — "Outlook", "Visio", "Applicant Screening", bahkan memotong kata di tengah ("Benefits" → "[NAME]fits"). Dari 392 chunk yang terkena redaksi nama, sampel manual menunjukkan mayoritas false positive.
Redaksi nama kandidat karenanya tidak dilakukan. Ini keterbatasan yang diakui, bukan fitur yang terlewat.

Nama kandidat, alamat jalan, dan kode pos tidak teredaksi. Contoh nyata dari korpus: blok "Anthony Nelson / 88 Malard Drive / Clarksville, TN 93002" lolos sepenuhnya, sementara nomor requisition di baris berikutnya sempat salah teredaksi sebelum guard ditambahkan.
"""

import re
from functools import lru_cache

NER_HEAD_CHARS = 300

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)[\s.-]?)?"
    r"\d{3,4}[\s.-]?\d{3,4}"
    r"(?:[\s.-]?\d{2,4})?"
)
URL = re.compile(r"\b(?:https?://|www\.)[^\s]+", re.IGNORECASE)

# Rentang tahun (1998-1999, 2015 - Present) cocok dengan pola telepon.
# Dilindungi sebelum PHONE dijalankan, dikembalikan setelahnya.
YEAR_RANGE = re.compile(
    r"\b(19|20)\d{2}\s*[-–]\s*((19|20)\d{2}|Present|Current)\b",
    re.IGNORECASE,
)

NER_ALLOWLIST = {
    "oracle", "adobe", "watson", "bloomberg", "morgan", "chase",
    "salesforce", "sap", "aws", "azure", "lawson", "kronos",
    "paychex", "adp", "peoplesoft", "workday", "taleo",
    "company name", "city", "state",
}

_HEADER_WORDS = {
    "summary", "highlights", "accomplishments", "experience", "education",
    "skills", "certifications", "qualifications", "profile", "objective",
    "achievements", "skill", "highlight",
}

_ROLE_WORDS = {
    "manager", "director", "analyst", "engineer", "associate",
    "administrator", "specialist", "coordinator", "assistant",
    "supervisor", "officer", "consultant", "executive", "lead",
}

# ISSN/ISBN dan nomor publikasi lain berpola mirip telepon.
ISSN = re.compile(r"\bISSN\s*:?\s*[\d-]+\b", re.IGNORECASE)

# Rentang kuantitas (100-200 applicants, 400-500 attendees) berpola
# mirip telepon. Nomor telepon asli umumnya 7+ digit atau berkurung.
QTY_RANGE = re.compile(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b(?!\d)")
@lru_cache(maxsize=1)
def _get_nlp():
    """Muat model spaCy sekali saja.

    Tanpa cache, model dimuat ulang di setiap panggilan (~1.5 detik),
    yang akan mendominasi total waktu proses.
    """
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

# ID dokumen (114760BR, Advert ID# 224278) berpola angka panjang
# tapi bukan telepon. Ditandai oleh huruf yang menempel atau
# kata kunci ID di depannya.
DOC_ID = re.compile(
    r"\b(?:(?:advert|req|requisition|job|posting|ref)\s*(?:id)?\s*#?\s*\d+"
    r"|\d{4,}[A-Z]{2,})\b",
    re.IGNORECASE,
)


def _is_likely_name(ent_text: str) -> bool:
    """Filter entitas PERSON yang kemungkinan besar bukan nama orang."""
    t = ent_text.lower().strip()

    if t in NER_ALLOWLIST:
        return False
    if len(t) < 3:
        return False

    words = t.split()

    # Nama orang jarang lebih dari 4 kata
    if len(words) > 4:
        return False

    # Mengandung kata section header atau jabatan → bukan nama
    if any(w in _HEADER_WORDS or w in _ROLE_WORDS for w in words):
        return False

    return True


def redact_patterns(text: str) -> tuple[str, int]:
    """Redaksi email, telepon, URL. Returns (teks, jumlah redaksi)."""
    n = 0

    text, c = EMAIL.subn("[EMAIL]", text)
    n += c

    text, c = URL.subn("[URL]", text)
    n += c

    # Lindungi rentang tahun sebelum regex telepon dijalankan
    protected: list[str] = []

    def _stash(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    text = YEAR_RANGE.sub(_stash, text)
    text = ISSN.sub(_stash, text)
    text = QTY_RANGE.sub(_stash, text)
    text = DOC_ID.sub(_stash, text)

    # Telepon paling akhir: pola angkanya cukup longgar untuk cocok
    # dengan digit di dalam URL atau email kalau dijalankan lebih dulu.
    text, c = PHONE.subn("[PHONE]", text)
    n += c

    for i, val in enumerate(protected):
        text = text.replace(f"\x00{i}\x00", val)

    return text, n


def redact_names(text: str) -> tuple[str, int]:
    """Redaksi entitas PERSON dengan NER. Returns (teks, jumlah redaksi)."""
    doc = _get_nlp()(text)
    n = 0

    # Iterasi terbalik agar penggantian tidak menggeser offset
    # entitas yang belum diproses.
    for ent in reversed(doc.ents):
        if ent.label_ != "PERSON":
            continue
        if not _is_likely_name(ent.text):
            continue

        text = text[: ent.start_char] + "[NAME]" + text[ent.end_char :]
        n += 1

    return text, n


def redact_names_in_head(text: str, head: int = NER_HEAD_CHARS) -> tuple[str, int]:
    """Jalankan NER hanya pada `head` karakter pertama."""
    if len(text) <= head:
        return redact_names(text)

    front, back = text[:head], text[head:]
    front, n = redact_names(front)
    return front + back, n


def _should_run_ner(chunk: dict) -> bool:
    """NER DIMATIKAN — lihat docstring modul untuk alasannya.

    Fungsi dipertahankan (bukan dihapus) agar jalur NER mudah
    diaktifkan kembali kalau nanti dicoba model yang lebih akurat
    seperti en_core_web_trf.
    """
    return False


def redact_chunk(chunk: dict) -> dict:
    """Redaksi satu chunk. Mengembalikan dict baru, tidak memodifikasi input."""
    text, n_pattern = redact_patterns(chunk["text"])

    n_name = 0
    if _should_run_ner(chunk):
        text, n_name = redact_names_in_head(text)

    out = dict(chunk)
    out["text"] = text
    out["char_length"] = len(text)
    out["n_redactions"] = n_pattern + n_name
    return out


def redact_chunks(chunks: list[dict]) -> tuple[list[dict], dict]:
    """Redaksi seluruh chunk. Returns (chunks, stats)."""
    out = []
    total_pattern = 0
    total_name = 0
    n_touched = 0

    for c in chunks:
        new = redact_chunk(c)
        np_ = new["n_redactions"]

        total_pattern += np_
        if np_ > 0:
            n_touched += 1

        out.append(new)

    # Hitung ulang pemisahan pattern vs name untuk statistik
    total_name = sum(
        1 for o, c in zip(out, chunks)
        if _should_run_ner(c) and "[NAME]" in o["text"]
    )

    stats = {
        "total_chunks": len(chunks),
        "chunks_redacted": n_touched,
        "pct_redacted": n_touched / max(len(chunks), 1) * 100,
        "total_redactions": total_pattern,
        "chunks_with_name": total_name,
        "ner_scope_chunks": sum(1 for c in chunks if _should_run_ner(c)),
    }
    return out, stats


if __name__ == "__main__":
    from src.ingestion.loader import load_resumes
    from src.ingestion.chunker import chunk_dataframe

    df = load_resumes()
    chunks, _ = chunk_dataframe(df)

    print(f"\nMeredaksi {len(chunks):,} chunk...")
    redacted, stats = redact_chunks(chunks)

    print()
    for k, v in stats.items():
        print(f"{k:22s} {v:,.1f}" if isinstance(v, float) else f"{k:22s} {v:,}")

    print("\nContoh chunk yang teredaksi:")
    shown = 0
    for orig, red in zip(chunks, redacted):
        if orig["text"] == red["text"]:
            continue
        i = next(
            (k for k in range(min(len(orig["text"]), len(red["text"])))
             if orig["text"][k] != red["text"][k]),
            0,
        )
        a, b = max(0, i - 80), i + 120
        print(f"\n--- {red['resume_id']} / {red['section_type']} ---")
        print("SEBELUM: ...", orig["text"][a:b])
        print("SESUDAH: ...", red["text"][a:b])
        shown += 1
        if shown >= 5:
            break