"""Antarmuka pencarian ke Qdrant.

Semua keputusan desain di modul ini berasal dari verifikasi empiris di
notebooks/check_02_retrieval.py terhadap 22.866 chunk. Rangkuman bukti:

1. CATEGORY FILTER — ada, opsional, TIDAK otomatis
   Tanpa filter, top-5 berisi rata-rata 3,25/5 dari kategori target.
   Query "registered nurse patient care" mengembalikan 3 ADVOCATE dari
   10 hasil — bahasa advokasi secara semantik dekat dengan patient care,
   dan vector search tidak punya cara membedakan.
   Filter menegakkan constraint yang tidak bisa dijamin embedding.
   Tapi hanya berguna kalau pengguna menyebut kategori; memaksakannya
   pada query umum justru mempersempit tanpa alasan.

2. SECTION FILTER — ada, TIDAK default
   Embedding sudah merutekan dengan benar tanpa bantuan:
     "proficient in Excel and SAP"       → skills 10/10
     "bachelor degree in business admin" → education 10/10
   Filter di kasus itu redundan.
   Lebih buruk: "managed a team across departments" tersebar ke
   experience (4) dan accomplishments (3) — dan isi accomplishments
   sama relevannya. Memfilter ke experience membuang hasil valid.

3. TANPA THRESHOLD SKOR
   Rentang skor sangat bergantung domain:
     kitchen management : 0.6325 – 0.6965
     financial audit    : 0.6014 – 0.6328
     python ML          : 0.3839 – 0.4131
   Hasil TERBAIK query Python (0.41) masih di bawah hasil TERBURUK
   query chef (0.63), padahal isinya jelas relevan.
   Threshold apa pun akan mematikan seluruh domain teknologi.

4. DEDUP PER RESUME_ID — default aktif
   Pada uji section_type=experience, resume 49777184 mengisi peringkat
   1 dan 3 sekaligus. Untuk screening, satu kandidat memakan dua slot
   dari lima adalah kerugian — recruiter butuh keragaman kandidat.

5. RETRIEVE BANYAK → RERANK → SEDIKIT
   Dedup dan reranking butuh kandidat berlebih untuk bekerja. Default
   fetch 20 untuk menghasilkan 5.

CATATAN PENTING — LABEL KATEGORI TIDAK TEPERCAYA:
Ditemukan resume "IT COMPLIANCE AUDITOR" berlabel APPAREL, dan resume
tentang loan applications berlabel CHEF. Label dataset ini kotor.
Konsekuensinya: jangan pakai label sebagai ground truth di evaluation
harness — relevansi harus diverifikasi dari isi teks.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

DEFAULT_FETCH = 20      # kandidat mentah sebelum dedup/rerank
DEFAULT_TOP_K = 5       # hasil akhir


@dataclass
class SearchHit:
    """Satu hasil pencarian.

    resume_id dan text dipisah eksplisit karena keduanya wajib ada di
    setiap klaim yang dihasilkan agent — citation enforcement.
    """
    resume_id: str
    category: str
    section_type: str
    chunk_index: int
    chunking_method: str
    text: str
    score: float

    def citation(self) -> str:
        return f"[{self.resume_id}#{self.chunk_index}]"


class VectorStore:
    """Wrapper Qdrant + embedding.

    Dibuat sebagai class agar koneksi dan client OpenAI dipakai ulang
    antar panggilan — retrieval agent akan memanggil search() berkali-kali
    dalam satu sesi.
    """

    def __init__(self, collection: str = COLLECTION):
        self.collection = collection
        self._oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._qd = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )

    def embed(self, text: str) -> list[float]:
        return self._oa.embeddings.create(
            model=EMBED_MODEL, input=[text]
        ).data[0].embedding

    def _build_filter(
        self,
        category: str | list[str] | None,
        section_type: str | list[str] | None,
        resume_id: str | None,
    ) -> Filter | None:
        """Susun payload filter. None kalau tidak ada kondisi.

        MatchAny dipakai untuk list agar agent bisa memfilter beberapa
        kategori sekaligus — mis. query finance yang relevan untuk
        BANKING, FINANCE, dan ACCOUNTANT.
        """
        conds = []

        if category:
            if isinstance(category, str):
                conds.append(FieldCondition(
                    key="category", match=MatchValue(value=category)))
            else:
                conds.append(FieldCondition(
                    key="category", match=MatchAny(any=category)))

        if section_type:
            if isinstance(section_type, str):
                conds.append(FieldCondition(
                    key="section_type", match=MatchValue(value=section_type)))
            else:
                conds.append(FieldCondition(
                    key="section_type", match=MatchAny(any=section_type)))

        if resume_id:
            conds.append(FieldCondition(
                key="resume_id", match=MatchValue(value=resume_id)))

        return Filter(must=conds) if conds else None

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fetch: int = DEFAULT_FETCH,
        category: str | list[str] | None = None,
        section_type: str | list[str] | None = None,
        dedupe_by_resume: bool = True,
    ) -> list[SearchHit]:
        """Cari chunk relevan.

        Ambil `fetch` kandidat, dedup, kembalikan `top_k` teratas.
        Tanpa threshold skor — lihat docstring modul poin 3.

        dedupe_by_resume=False berguna saat agent ingin melihat beberapa
        bagian dari satu kandidat, mis. saat menyusun ringkasan profil.
        """
        raw = self._qd.query_points(
            self.collection,
            query=self.embed(query),
            limit=fetch,
            query_filter=self._build_filter(category, section_type, None),
        ).points

        hits = [
            SearchHit(
                resume_id=p.payload["resume_id"],
                category=p.payload["category"],
                section_type=p.payload["section_type"],
                chunk_index=p.payload["chunk_index"],
                chunking_method=p.payload["chunking_method"],
                text=p.payload["text"],
                score=p.score,
            )
            for p in raw
        ]

        if dedupe_by_resume:
            seen = set()
            deduped = []
            for h in hits:
                if h.resume_id in seen:
                    continue
                seen.add(h.resume_id)
                deduped.append(h)
            hits = deduped

        return hits[:top_k]

    def get_resume_chunks(self, resume_id: str, limit: int = 50) -> list[SearchHit]:
        """Ambil seluruh chunk satu resume, terurut.

        Dipakai saat agent perlu konteks lengkap satu kandidat —
        misalnya untuk perbandingan atau ringkasan profil.
        Scroll, bukan vector search: tidak ada query semantik di sini.
        """
        points, _ = self._qd.scroll(
            self.collection,
            scroll_filter=self._build_filter(None, None, resume_id),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        hits = [
            SearchHit(
                resume_id=p.payload["resume_id"],
                category=p.payload["category"],
                section_type=p.payload["section_type"],
                chunk_index=p.payload["chunk_index"],
                chunking_method=p.payload["chunking_method"],
                text=p.payload["text"],
                score=0.0,   # tidak ada skor: ini bukan hasil pencarian
            )
            for p in points
        ]
        return sorted(hits, key=lambda h: h.chunk_index)

    def stats(self) -> dict:
        info = self._qd.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points": info.points_count,
            "dim": info.config.params.vectors.size,
        }


if __name__ == "__main__":
    vs = VectorStore()
    print(vs.stats())

    print("\n=== dedup aktif (default) ===")
    for h in vs.search("managed a team across departments",
                       section_type="experience"):
        print(f"[{h.score:.4f}] {h.citation()} {h.category:22s} "
              f"{h.text[:80].strip()}")

    print("\n=== dedup mati ===")
    for h in vs.search("managed a team across departments",
                       section_type="experience", dedupe_by_resume=False):
        print(f"[{h.score:.4f}] {h.citation()} {h.category:22s} "
              f"{h.text[:80].strip()}")

    print("\n=== multi-category (BANKING + FINANCE + ACCOUNTANT) ===")
    for h in vs.search("audit and financial reporting",
                       category=["BANKING", "FINANCE", "ACCOUNTANT"]):
        print(f"[{h.score:.4f}] {h.citation()} {h.category:22s} "
              f"{h.text[:80].strip()}")

    print("\n=== seluruh chunk satu resume ===")
    for h in vs.get_resume_chunks("27914096")[:5]:
        print(f"{h.citation()} {h.section_type:14s} {h.text[:70].strip()}")