"""Verifikasi retrieval — jalankan dari root project:
   python notebooks/checkpoint.py

Menjawab 4 pertanyaan sebelum menulis vector_store.py:
1. Apakah pencarian semantik akurat?
2. Apakah payload filter bekerja di skala 22.866 titik?
3. Apakah hybrid lebih baik dari vector polos?  ← klaim utama project
4. Apakah section filter berguna?

Biaya: ~30 query embedding ≈ $0.000006
"""

import os
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qd = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)


def embed_query(text):
    return oa.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def search(query, k=5, category=None, section_type=None):
    conds = []
    if category:
        conds.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if section_type:
        conds.append(FieldCondition(key="section_type",
                                    match=MatchValue(value=section_type)))
    return qd.query_points(
        COLLECTION,
        query=embed_query(query),
        limit=k,
        query_filter=Filter(must=conds) if conds else None,
    ).points


def show(hits, width=130):
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"  {i}. [{h.score:.4f}] {p['category']:22s} "
              f"{p['section_type']:14s} id={p['resume_id']}")
        print(f"     {p['text'][:width].strip()}")


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ── SETUP ────────────────────────────────────────────────────────────
info = qd.get_collection(COLLECTION)
header("COLLECTION")
print(f"name       : {COLLECTION}")
print(f"points     : {info.points_count:,}")
print(f"dimensi    : {info.config.params.vectors.size}")


# ── Q1: AKURASI PENCARIAN SEMANTIK ───────────────────────────────────
header("Q1 — AKURASI PENCARIAN SEMANTIK")

for q in [
    "financial audit and compliance experience",
    "kitchen management and menu development",
    "python machine learning deployment",
]:
    print(f"\nQUERY: {q}")
    show(search(q, k=5))


# ── Q2: PAYLOAD FILTERING ────────────────────────────────────────────
header("Q2 — PAYLOAD FILTERING DI SKALA PENUH")

q = "audit and financial reporting"
print(f"QUERY: {q}\n")

print("-- tanpa filter --")
show(search(q, k=5), width=90)

print("\n-- filter category=BANKING --")
show(search(q, k=5, category="BANKING"), width=90)

print("\n-- filter category=CHEF (sengaja tidak relevan) --")
show(search(q, k=5, category="CHEF"), width=90)


# ── Q3: HYBRID vs VECTOR POLOS ───────────────────────────────────────
header("Q3 — HYBRID vs VECTOR POLOS  [klaim utama project]")

cases = [
    ("experienced banking professional with risk management", "BANKING"),
    ("registered nurse patient care experience", "HEALTHCARE"),
    ("civil construction project supervision", "CONSTRUCTION"),
    ("digital marketing campaign management", "DIGITAL-MEDIA"),
]

print(f"{'query':47s} {'no filter':>10s} {'filtered':>10s}")
print("-" * 70)

for q, cat in cases:
    plain = search(q, k=5)
    filt = search(q, k=5, category=cat)
    p = sum(1 for h in plain if h.payload["category"] == cat)
    f = sum(1 for h in filt if h.payload["category"] == cat)
    print(f"{q[:45]:47s} {p}/5{'':>7s} {f}/5")

print("\nDistribusi kategori pada top-10 tanpa filter:")
for q, cat in cases:
    dist = Counter(h.payload["category"] for h in search(q, k=10))
    print(f"\n  {q[:55]}")
    print(f"    target : {cat}")
    print(f"    hasil  : {dict(dist)}")


# ── Q4: SECTION FILTERING ────────────────────────────────────────────
header("Q4 — SECTION FILTERING")

q = "managed a team of 20 people across multiple departments"
print(f"QUERY: {q}\n")

print("-- tanpa section filter --")
show(search(q, k=5), width=110)

print("\n-- section_type=experience --")
show(search(q, k=5, section_type="experience"), width=110)

print("\nDistribusi section_type pada top-10:")
for q in [
    "managed a team across departments",
    "proficient in Excel and SAP",
    "bachelor degree in business administration",
]:
    dist = Counter(h.payload["section_type"] for h in search(q, k=10))
    print(f"\n  {q}")
    print(f"    {dict(dist)}")


print("\n" + "=" * 72)
print("SELESAI")
print("=" * 72)