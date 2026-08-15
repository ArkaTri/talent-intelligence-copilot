"""Muat dan bersihkan Resume dataset.

Keputusan cleaning berasal dari notebooks/01_data_exploration.ipynb:
- 1 baris <500 karakter (min 21 char) → drop
- 2 duplikat Resume_str → drop, pertahankan yang pertama
- \\xa0 (8.877), \\t (7.257), \\u2028 (9) → normalisasi ke spasi
- Korpus final: 2.481 dokumen
"""

import re
from pathlib import Path

import pandas as pd

MIN_CHARS = 500
UNUSUAL_CHARS = r"[\xa0\t\u2028\u2029\r]"


def normalize_whitespace(text: str) -> str:
    """Ganti karakter tak biasa jadi spasi biasa.

    PENTING: fungsi ini TIDAK mengolapskan spasi berulang.

    Eksplorasi data menunjukkan resume hasil scraping ini kehilangan
    struktur baris — satu dokumen 5.442 karakter hanya punya 2 newline.
    Header ("Summary", "Experience") dipisahkan dari teks sekitarnya
    oleh runs of spaces, bukan newline.

    Artinya spasi berulang adalah SATU-SATUNYA sinyal struktural yang
    tersisa. Menjalankan re.sub(r"\\s+", " ", text) di sini akan
    menghancurkan kemampuan chunker mendeteksi header.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(UNUSUAL_CHARS, " ", text)


def load_resumes(path: str | Path = "data/raw/Resume.csv") -> pd.DataFrame:
    """Muat CSV, bersihkan, kembalikan DataFrame siap chunking."""
    df = pd.read_csv(path)
    before = len(df)

    # Resume_html dibuang: isinya konten sama dengan Resume_str tapi
    # dibungkus markup. Embedding markup = bayar token untuk noise.
    df = df[["ID", "Resume_str", "Category"]].copy()

    df["Resume_str"] = df["Resume_str"].apply(normalize_whitespace)

    # Ambang 500 dari eksplorasi: hanya 1 baris di bawahnya (21 karakter),
    # jelas artefak scraping. Median korpus 5.886 — ambang ini konservatif.
    too_short = df["Resume_str"].str.len() < MIN_CHARS
    df = df[~too_short]

    # Duplikat dicek pada teks, bukan ID — ID semuanya unik (0 duplikat),
    # tapi ada 2 resume dengan isi identik.
    dup = df["Resume_str"].duplicated()
    df = df[~dup]

    df = df.reset_index(drop=True)

    print(f"Loaded {before} → {len(df)} dokumen "
          f"(drop {too_short.sum()} pendek, {dup.sum()} duplikat)")

    return df


if __name__ == "__main__":
    df = load_resumes()
    print(df.shape)
    print(df["Category"].nunique(), "kategori")
    print(f"Rata-rata {df['Resume_str'].str.len().mean():.0f} karakter")