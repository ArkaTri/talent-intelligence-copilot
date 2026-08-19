"""Embed chunk dan upsert ke Qdrant Cloud.

Pipeline: loader → chunker → redactor → INDEXER → Qdrant

Dua hal yang wajib benar di sini:

1. PAYLOAD INDEX
   Qdrant menolak filtering pada field yang belum di-index. Errornya:
   "Index required but not found for 'category' of type [keyword]".
   Vektor ter-index otomatis; payload TIDAK. Harus dideklarasikan
   eksplisit. Kalau terlewat, kegagalan baru muncul saat agent mencoba
   memfilter — di tengah demo.

2. EMBEDDING CACHE
   22.866 chunk = 229 panggilan API = 4-8 menit sekali jalan. Selama
   membangun retrieval dan agent, pipeline ini dijalankan ulang belasan
   kali dengan chunking yang sama. Cache memotong waktu tunggu itu
   jadi detik.

   Kunci cache = hash(teks + nama model). Nama model WAJIB masuk kunci —
   tanpa itu, mengganti ke text-embedding-3-large akan mengambil vektor
   1536-dim dari cache untuk model 3072-dim.
"""

import hashlib
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")

CACHE_PATH = Path("cache/embeddings.parquet")
BATCH_SIZE = 100        # batas aman request OpenAI
UPSERT_BATCH = 256      # batas aman payload Qdrant

# Field yang akan difilter oleh agent. WAJIB di-index — lihat docstring.
INDEXED_FIELDS = ["category", "section_type", "resume_id", "chunking_method"]


def _cache_key(text: str, model: str) -> str:
    """Hash teks + model. Model masuk kunci agar cache tidak tertukar
    antar model dengan dimensi berbeda."""
    return hashlib.sha256(f"{model}||{text}".encode()).hexdigest()


def load_cache() -> dict[str, list[float]]:
    """Muat cache dari disk.

    File cache bisa kosong atau rusak — misalnya kalau proses terhenti
    di tengah penulisan. Cache yang tidak terbaca diperlakukan sebagai
    cache kosong, bukan error: kehilangan cache hanya berarti embed
    ulang, tidak ada data yang benar-benar hilang.
    """
    if not CACHE_PATH.exists() or CACHE_PATH.stat().st_size == 0:
        return {}
    try:
        df = pd.read_parquet(CACHE_PATH)
        return dict(zip(df["key"], df["vector"]))
    except Exception as e:
        print(f"  cache tidak terbaca ({e.__class__.__name__}), diabaikan")
        return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"key": list(cache.keys()), "vector": list(cache.values())}) \
      .to_parquet(CACHE_PATH, index=False)


def embed_texts(
    texts: list[str],
    client: OpenAI,
    model: str = EMBED_MODEL,
    use_cache: bool = True,
) -> tuple[list[list[float]], dict]:
    """Embed list teks, memanfaatkan cache. Returns (vectors, stats)."""
    cache = load_cache() if use_cache else {}
    keys = [_cache_key(t, model) for t in texts]

    # Kumpulkan yang belum ada di cache. Dedup: chunk dengan teks
    # identik hanya perlu di-embed sekali.
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    missing_unique = list({keys[i]: texts[i] for i in missing_idx}.items())

    n_hit = len(texts) - len(missing_idx)
    n_api = 0

    if missing_unique:
        print(f"  cache hit {n_hit:,} / {len(texts):,} — "
              f"embed {len(missing_unique):,} teks baru")

        for start in range(0, len(missing_unique), BATCH_SIZE):
            batch = missing_unique[start : start + BATCH_SIZE]
            resp = client.embeddings.create(
                model=model, input=[t for _, t in batch]
            )
            for (k, _), item in zip(batch, resp.data):
                cache[k] = item.embedding
            n_api += resp.usage.total_tokens

            done = min(start + BATCH_SIZE, len(missing_unique))
            print(f"  {done:,}/{len(missing_unique):,}", end="\r")

        print()
        if use_cache:
            save_cache(cache)
    else:
        print(f"  cache hit {n_hit:,} / {len(texts):,} — tidak ada API call")

    stats = {
        "cache_hits": n_hit,
        "api_tokens": n_api,
        "api_cost_usd": n_api / 1_000_000 * 0.02,
    }
    return [cache[k] for k in keys], stats


def setup_collection(client: QdrantClient, dim: int, recreate: bool = False):
    """Buat collection + payload index.

    recreate=True menghapus collection lama. Dipakai saat ablation
    dengan konfigurasi chunking berbeda — data lama harus dibuang
    agar hasil retrieval tidak tercampur.
    """
    exists = client.collection_exists(COLLECTION)

    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False
        print(f"  collection '{COLLECTION}' dihapus")

    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"  collection '{COLLECTION}' dibuat (dim={dim})")

        # Payload index — tanpa ini filtering gagal saat query
        for field in INDEXED_FIELDS:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print(f"  payload index: {', '.join(INDEXED_FIELDS)}")


def index_chunks(chunks: list[dict], recreate: bool = False,
                 skip_upsert: bool = False) -> dict:
    """Embed seluruh chunk dan upsert ke Qdrant."""
    oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qd = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )

    t0 = time.time()

    print("\n[1/3] Embedding")
    texts = [c["text"] for c in chunks]
    vectors, embed_stats = embed_texts(texts, oa)

    print("\n[2/3] Setup collection")
    setup_collection(qd, dim=len(vectors[0]), recreate=recreate)

    print("\n[3/3] Upsert ke Qdrant")
    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={
                "resume_id": c["resume_id"],
                "category": c["category"],
                "section_type": c["section_type"],
                "chunk_index": c["chunk_index"],
                "chunking_method": c["chunking_method"],
                "char_length": c["char_length"],
                "text": c["text"],
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ]

    if skip_upsert:
        print("  dilewati (--skip-upsert)")
    else:
        for start in range(0, len(points), UPSERT_BATCH):
            qd.upsert(COLLECTION, points=points[start : start + UPSERT_BATCH])
            done = min(start + UPSERT_BATCH, len(points))
            print(f"  {done:,}/{len(points):,}", end="\r")
        print()

    info = qd.get_collection(COLLECTION)

    return {
        **embed_stats,
        "chunks_indexed": len(points),
        "vector_dim": len(vectors[0]),
        "points_in_collection": info.points_count,
        "elapsed_sec": time.time() - t0,
    }


if __name__ == "__main__":
    import sys

    from src.ingestion.loader import load_resumes
    from src.ingestion.chunker import chunk_dataframe
    from src.ingestion.redactor import redact_chunks

    recreate = "--recreate" in sys.argv
    skip_upsert = "--skip-upsert" in sys.argv

    df = load_resumes()
    chunks, _ = chunk_dataframe(df)
    chunks, _ = redact_chunks(chunks)

    stats = index_chunks(chunks, recreate=recreate, skip_upsert=skip_upsert)

    print()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:22s} {v:,.4f}")
        else:
            print(f"{k:22s} {v:,}")