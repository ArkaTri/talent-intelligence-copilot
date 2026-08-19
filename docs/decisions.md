# Catatan Keputusan Teknis

## PII Redaction — NER ditolak

**Keputusan:** redaksi dibatasi ke pola deterministik (email, URL, telepon).
NER tidak dipakai.

**Alasan:** spaCy `en_core_web_sm` diuji 3 iterasi. Pada teks hasil
flattening HTML tanpa struktur kalimat, model menebak dari kapitalisasi
dan salah menandai "Outlook", "Visio", "Applicant Screening" sebagai
PERSON. "Benefits" bahkan terpotong jadi "[NAME]fits".

**Keterbatasan yang diterima:** nama dan alamat di badan teks lolos.
Contoh dari korpus: "Anthony Nelson / 88 Malard Drive / Clarksville, TN
93002". Redaksi URL secara tidak langsung menangkap sebagian nama yang
tertanam di URL profil (`linkedin.com/in/e-april-bradford`).

**Hasil:** 348 chunk (1,5%), 521 redaksi, nol false positive di sampel.

---

## Chunking — hybrid, bukan section-aware murni

**Keputusan:** section-aware bila ≥2 header terdeteksi, fallback ke
RecursiveCharacterTextSplitter bila kurang.

**Alasan:** coverage header jauh di bawah perkiraan awal — Experience
69,2%, Education 77,4%, Skills 93,6%. Pengukuran awal (99%) keliru
karena memakai substring polos.

**Hasil:** 96,6% dokumen lewat jalur section, 3,4% fallback.
22.866 chunk, median 990 karakter.

## Embedding & Indexing

**Keputusan:** cache embedding ke `cache/embeddings.parquet`, kunci =
hash(teks + nama model).

**Alasan:** 22.866 chunk butuh 229 panggilan API. Dari Indonesia,
sekali jalan 17 menit. Pipeline dijalankan ulang belasan kali selama
membangun retrieval dan agent — dengan chunking yang sama.

Nama model masuk kunci hash agar cache tidak tertukar saat mengganti
ke text-embedding-3-large (3072 dim vs 1536).

**Hasil:** run kedua 0 API call, 1.042 → 349 detik.

**Temuan:** setelah cache aktif, bottleneck bergeser ke upsert Qdrant
(90 batch × round trip Sydney), bukan embedding. Ditangani dengan
flag `--skip-upsert`.

**Biaya aktual:** $0,0661 untuk 3,3M token — lebih rendah dari estimasi
$0,116 karena rasio karakter-per-token teks resume lebih padat dari
asumsi 4:1.

**Payload index:** category, section_type, resume_id, chunking_method.
Wajib eksplisit — Qdrant menolak filtering pada field tak ter-index.