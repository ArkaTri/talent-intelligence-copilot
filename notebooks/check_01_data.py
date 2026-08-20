"""Ringkasan verifikasi data — jalankan dari root project:
   python notebooks/checkpoint.py
"""
import re
from collections import Counter
import pandas as pd

df = pd.read_csv("data/raw/Resume.csv")

print("=" * 60)
print("SHAPE & CLEANING")
print("=" * 60)
lengths = df["Resume_str"].str.len()
print(f"shape            : {df.shape}")
print(f"mean chars       : {lengths.mean():.0f}")
print(f"median chars     : {lengths.median():.0f}")
print(f"min / max        : {lengths.min()} / {lengths.max()}")
print(f"< 500 chars      : {(lengths < 500).sum()}")
print(f"dup Resume_str   : {df['Resume_str'].duplicated().sum()}")
print(f"dup ID           : {df['ID'].duplicated().sum()}")
print(f"null Resume_str  : {df['Resume_str'].isna().sum()}")

print()
print("=" * 60)
print("HEADER COVERAGE — pola spasi (bukan substring polos)")
print("=" * 60)
headers = ["Summary", "Highlights", "Accomplishments", "Experience",
           "Education", "Skills", "Professional Summary",
           "Core Qualifications", "Certifications"]

norm = df["Resume_str"].fillna("").str.replace(r"[\xa0\t\u2028\u2029]", " ", regex=True)

for h in headers:
    pat = rf"\s{{2,}}{re.escape(h)}\s{{2,}}"
    n = norm.str.contains(pat, case=False, regex=True).sum()
    print(f"{h:22s} {n:5d}  ({n/len(df)*100:5.1f}%)")

print()
print("=" * 60)
print("KARAKTER TAK BIASA")
print("=" * 60)
sus = Counter()
for t in df["Resume_str"].dropna():
    sus.update(re.findall(r"[\u2028\u2029\r\t\xa0]", t))
for ch, n in sus.most_common():
    print(f"  {repr(ch):10s} {n:,}")

print()
print("=" * 60)
print("BIAYA EMBEDDING")
print("=" * 60)
est = lengths.sum() / 4
print(f"est tokens       : {est/1e6:.2f}M")
print(f"biaya 1x index   : ${est/1e6*0.02:.4f}")