"""Verifikasi kelayakan analytics berbasis teks — jalankan dari root:
   python notebooks/check_03_analytics.py

KONTEKS:
Rencana awal (guideline Bagian 2) memakai Text-to-SQL untuk pertanyaan
agregat. Dibatalkan setelah ditemukan label kategori dataset kotor —
"IT COMPLIANCE AUDITOR" berlabel APPAREL, loan processing berlabel CHEF.
Menghitung berdasarkan label akan menghasilkan angka yang salah dengan
percaya diri.

Penggantinya: hitung dari ISI TEKS, bukan label.

Yang diverifikasi:
1. Apakah payload index TEXT bisa dibuat di field `text`?
2. Apakah pencocokan full-text cukup cepat di 22.866 chunk?
3. Seberapa besar selisih hitungan berbasis label vs berbasis teks?
   (angka ini yang membuktikan keputusan membatalkan SQL benar)
"""

import os
import time
from collections import Counter

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    PayloadSchemaType,
)

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")

qd = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)


def header(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def count_matching(text_query: str = None, category: str = None) -> int:
    """Hitung chunk yang cocok, tanpa mengambil datanya.

    count() jauh lebih murah daripada scroll — Qdrant hanya menghitung
    di sisi server, tidak mengirim payload.
    """
    conds = []
    if text_query:
        conds.append(FieldCondition(key="text", match=MatchText(text=text_query)))
    if category:
        conds.append(FieldCondition(key="category", match=MatchValue(value=category)))

    return qd.count(
        COLLECTION,
        count_filter=Filter(must=conds) if conds else None,
        exact=True,
    ).count


def unique_resumes(text_query: str, limit: int = 5000) -> set:
    """Ambil resume_id unik yang chunk-nya memuat teks tertentu.

    Dipakai karena satu resume bisa punya beberapa chunk yang cocok —
    menghitung chunk akan melebih-lebihkan jumlah kandidat.
    """
    ids = set()
    offset = None
    while True:
        points, offset = qd.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="text", match=MatchText(text=text_query))
            ]),
            limit=256,
            offset=offset,
            with_payload=["resume_id"],
            with_vectors=False,
        )
        ids.update(p.payload["resume_id"] for p in points)
        if offset is None or len(ids) > limit:
            break
    return ids


# ── 1. SETUP PAYLOAD INDEX TEXT ──────────────────────────────────────
header("1 — PAYLOAD INDEX untuk full-text")

info = qd.get_collection(COLLECTION)
print(f"points: {info.points_count:,}")

existing = info.payload_schema or {}
print(f"index terpasang: {list(existing.keys())}")

if "text" not in existing:
    print("\nMembuat index TEXT untuk field 'text'...")
    t0 = time.time()
    # TEXT berbeda dari KEYWORD: KEYWORD mencocokkan nilai persis,
    # TEXT melakukan tokenisasi sehingga bisa mencari kata di dalam teks.
    qd.create_payload_index(
        collection_name=COLLECTION,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT,
    )
    print(f"selesai dalam {time.time() - t0:.1f}s")
else:
    print("\nindex 'text' sudah ada")


# ── 2. KECEPATAN PENCOCOKAN ──────────────────────────────────────────
header("2 — KECEPATAN PENCOCOKAN FULL-TEXT")

terms = ["AWS", "Python", "Sarbanes-Oxley", "PMP", "Six Sigma", "Kubernetes"]

print(f"{'term':20s} {'chunks':>8s} {'waktu':>8s}")
print("-" * 40)
for t in terms:
    t0 = time.time()
    n = count_matching(text_query=t)
    print(f"{t:20s} {n:8,} {time.time() - t0:7.2f}s")


# ── 3. LABEL vs TEKS — bukti label tidak tepercaya ───────────────────
header("3 — HITUNGAN BERBASIS LABEL vs BERBASIS TEKS")

print("Pertanyaan: berapa kandidat yang punya kemampuan cloud (AWS/Azure/GCP)?\n")

# Cara LAMA (Text-to-SQL berbasis label): hitung kategori IT
label_it = qd.count(
    COLLECTION,
    count_filter=Filter(must=[
        FieldCondition(key="category",
                       match=MatchValue(value="INFORMATION-TECHNOLOGY"))
    ]),
    exact=True,
).count

it_resumes = set()
offset = None
while True:
    points, offset = qd.scroll(
        COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="category",
                           match=MatchValue(value="INFORMATION-TECHNOLOGY"))
        ]),
        limit=256, offset=offset,
        with_payload=["resume_id"], with_vectors=False,
    )
    it_resumes.update(p.payload["resume_id"] for p in points)
    if offset is None:
        break

print(f"  [label]  kategori INFORMATION-TECHNOLOGY : {len(it_resumes)} resume")

# Cara BARU (berbasis teks): cari yang benar-benar menyebut cloud
aws = unique_resumes("AWS")
azure = unique_resumes("Azure")
cloud_resumes = aws | azure

print(f"  [teks]   menyebut AWS atau Azure         : {len(cloud_resumes)} resume")

overlap = it_resumes & cloud_resumes
print(f"\n  irisan keduanya                        : {len(overlap)} resume")
print(f"  punya cloud tapi BUKAN label IT        : {len(cloud_resumes - it_resumes)} resume")
print(f"  label IT tapi tidak menyebut cloud     : {len(it_resumes - cloud_resumes)} resume")


# ── 4. DISTRIBUSI KATEGORI PADA HASIL BERBASIS TEKS ──────────────────
header("4 — KE MANA KANDIDAT CLOUD SEBENARNYA TERSEBAR?")

dist = Counter()
offset = None
while True:
    points, offset = qd.scroll(
        COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="text", match=MatchText(text="AWS"))
        ]),
        limit=256, offset=offset,
        with_payload=["resume_id", "category"], with_vectors=False,
    )
    for p in points:
        dist[p.payload["category"]] += 1
    if offset is None:
        break

print("Distribusi kategori chunk yang menyebut 'AWS':\n")
for cat, n in dist.most_common(10):
    print(f"  {cat:24s} {n:5,}")


print("\n" + "=" * 72)
print("SELESAI")
print("=" * 72)