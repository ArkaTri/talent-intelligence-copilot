# 🔍 Talent Intelligence Copilot

**Evidence-grounded candidate screening assistant.** Multi-agent RAG over 2,481 resumes — every claim traceable to a source quote.

🔗 **[Live demo](https://talent-intelligence-copilot-qbyk7qm8nzktyaizmp9ks6.streamlit.app)** · Capstone Project Module 3 — Purwadhika AI Engineering

---

## Latar belakang

BPS mencatat **1.033.182 lulusan perguruan tinggi menganggur** per Mei 2026 — 14,31% dari 7,22 juta pengangguran nasional. Proporsinya naik konsisten: 7,98% (2022) → 10,3% (2023) → 11,28% (2024) → 12,12% (2025) → 14,31% (2026), sementara pengangguran total justru turun.

Di sisi lain, data operasional Jobstreet by SEEK (Maret 2026) menunjukkan satu lowongan rata-rata menerima **500–600 lamaran**, bisa ribuan untuk posisi umum di perusahaan besar.

Satu masalah, dua arah:

- **Perusahaan** — tim HR kewalahan. Riset eye-tracking menunjukkan recruiter menghabiskan ~7 detik untuk pass pertama sebuah CV. Bukan kelalaian, tapi konsekuensi volume.
- **Kandidat** — banyak gugur di seleksi otomatis sebelum dibaca manusia. Kemnaker menyebut *skill mismatch* sebagai penyebab utama, tapi mismatch sering terjadi karena pencocokan berbasis **keyword**, bukan berbasis **bukti pengalaman**.

Sistem ini menyentuh titik pertemuannya: menemukan kandidat relevan dari tumpukan, dengan bukti yang bisa ditelusuri.

**Batasan yang ditetapkan sejak awal:** decision-support, bukan decision-making. Tidak ada auto-reject.

---

## Arsitektur

```
query
  ↓
[guardrail_pre]  ── ditolak ──→ END (nol token)
  ↓ lolos
[route]          ── supervisor menentukan agent + ekstrak filter
  ↓
[retrieval] / [evaluator] / [analytics] / [profile] / [refuse]
  ↓
[guardrail_post] ── verifikasi setiap citation
  ↓
END
```

### Pipeline RAG

**Ingestion** (sekali di awal):
```
Resume.csv (2.484)
  → loader.py     normalisasi karakter, drop 3 baris → 2.481 dokumen
  → chunker.py    hybrid section-aware → 22.866 chunk
  → redactor.py   redaksi PII (email, URL, telepon)
  → indexer.py    embed (text-embedding-3-small, 1536-dim) → Qdrant Cloud
```

**Retrieval** (setiap query):
```
query → embed → Qdrant vector search (+ payload filter)
      → dedup per resume_id → LLM rerank (20 → 5)
      → LLM menyusun jawaban dari chunk terpilih
      → citation verifier
```

Jawaban tidak berasal dari pengetahuan bawaan model — berasal dari 22.866 chunk yang ter-index. Karena setiap chunk membawa `resume_id` di payload, citation bisa dipaksa secara struktural.

### Stack

| Komponen | Teknologi |
|---|---|
| Orchestration | LangGraph StateGraph |
| Vector DB | Qdrant Cloud (GCP Sydney) |
| Embedding | OpenAI `text-embedding-3-small` |
| Routing & rerank | `gpt-4.1-nano` |
| Answer & evaluation | `gpt-4.1-mini` |
| UI | Streamlit Community Cloud |
| Structured output | Pydantic |

---

## Yang bisa ditanyakan

| Jalur | Contoh | Biaya |
|---|---|---|
| **Retrieval** | "carikan kandidat dengan pengalaman audit keuangan" | $0.0014 |
| **Profile** | "berikan detail dari kandidat tersebut" | $0.00011 |
| **Analytics** | "berapa kandidat yang menyebut AWS?" | $0.00017 |
| **Evaluator** | "nilai kandidat untuk posisi X: butuh A, B, C" | $0.0043 |
| **Ditolak** | "cari kandidat laki-laki" | $0.00 |

**Rata-rata $1,20 per 1.000 query** untuk campuran realistis.

---

## Keputusan teknis

Hampir semua keputusan di project ini **berubah setelah diukur**. Empat yang paling menentukan:

### 1. Text-to-SQL dibatalkan — label dataset tidak tepercaya

Rencana awal memakai Text-to-SQL (Session 16 silabus) untuk pertanyaan agregat. Dibatalkan setelah verifikasi.

Pertanyaan uji: *"berapa kandidat dengan kemampuan cloud?"*

| Metode | Hasil |
|---|---|
| Label `category = INFORMATION-TECHNOLOGY` | 120 resume |
| Teks menyebut AWS/Azure | 29 resume |
| Irisan | 9 resume |
| **Punya cloud tapi bukan label IT** | **20 (69% terlewat)** |
| **Label IT tanpa menyebut cloud** | **111 (93% false positive)** |

Penyebabnya: label kotor. Resume berjudul "IT COMPLIANCE AUDITOR" dilabeli **APPAREL**. Resume tentang loan processing dilabeli **CHEF**. Kandidat yang menyebut AWS tersebar di CONSULTANT (8), ADVOCATE (5), AGRICULTURE (4), CONSTRUCTION, BANKING.

**Penggantinya:** Qdrant `MatchText` dengan payload index bertipe TEXT. Dan untuk pertanyaan yang memang bergantung label, sistem **menolak menjawab** dengan penjelasan.

### 2. Coverage header 99% → 69% setelah diukur ulang

Pengukuran awal memakai `str.contains("Experience")` — menghitung kata di mana pun, termasuk di kalimat "5 years of experience". Setelah diukur dengan pola header yang benar (`\s{2,}Header\s{2,}` setelah normalisasi):

| Section | Substring polos | Pola header |
|---|---|---|
| Experience | 97,8% | **69,2%** |
| Education | 99,1% | **77,4%** |
| Skills | 99,0% | **93,6%** |

Temuan kedua: dokumen 5.442 karakter hanya punya **2 newline**. Struktur baris hilang karena flattening HTML — `text.split("\n")` akan menghasilkan nol header terdeteksi. Kegagalan senyap.

**Solusi:** chunker hybrid — deteksi header lewat pola spasi berulang; kalau ≥2 header ditemukan split per section, kalau kurang fallback ke splitter generik. Hasil: 96,6% dokumen lewat jalur section.

### 3. LLM reranking, bukan cross-encoder

Vector search mengembalikan kandidat yang "cukup mirip" — tapi mirip ≠ relevan. Query "financial audit" menghasilkan skor 0,60–0,63 untuk lima teratas, padahal isinya beda kualitas:

- Peringkat 3: `"ISACA, Sarbanes-Oxley, project risk and controls"` — daftar sertifikasi
- Peringkat 4: `"led inventory test work at multiple audit clients"` — bukti pengalaman

Embedding tidak bisa membedakannya. Reranker membaca dan menilai.

**Dampak terukur:** dari 3 query uji, **6 kandidat dari luar top-5 vektor** masuk hasil akhir.

Cross-encoder lebih akurat, tapi butuh ~400MB memori — berisiko melebihi batas Streamlit Cloud free tier. Dipilih LLM-based reranking karena **project yang tidak bisa di-deploy bernilai nol**.

Konsekuensinya terlihat di `requirements.txt`: `sentence-transformers` dan `spacy` hanya ada di `requirements-dev.txt`.

### 4. NER untuk PII redaction ditolak

spaCy `en_core_web_sm` diuji tiga iterasi, semuanya gagal:

- "Outlook", "Visio" (software) terdeteksi sebagai PERSON
- "Applicant Screening" (istilah HR) terdeteksi sebagai PERSON
- Kata "Benefits" terpotong jadi `[NAME]fits`

Penyebabnya: spaCy dilatih pada teks berstruktur kalimat. Data ini penuh frasa Title Case tanpa tanda baca — model kehilangan sinyal gramatikal dan menebak dari kapitalisasi.

Dimatikan. Redaksi dibatasi ke pola deterministik: email, URL, telepon (dengan guard untuk rentang tahun, ISSN, rentang kuantitas, dan ID dokumen).

Komponen yang merusak informasi teknis demi menangkap segelintir nama adalah trade-off yang buruk.

---

## Fitur di atas standar minimum

| Fitur | Kenapa |
|---|---|
| **Guardrail sebagai node, bukan tool** | Sebagai tool, agent bisa melewatinya — justru pada query bermasalah |
| **Citation verifier** | Memverifikasi kutipan ke sumbernya; membedakan halusinasi dari chunk_index keliru |
| **Hybrid retrieval** | Category filter menaikkan precision 3,25/5 → 5/5 saat kategori disebut |
| **Dedup per resume_id** | Tanpa dedup, satu resume mengisi 3 dari 5 slot |
| **Structured output (Pydantic)** | Evaluator menghasilkan objek tervalidasi — bisa diurutkan dan difilter |
| **Token panel per komponen** | Agregasi lintas model menyesatkan: 5.000 tok nano ≠ 5.000 tok mini |
| **Session cap ganda** | Batas query (dipahami pengguna) + batas biaya (pengaman sebenarnya) |
| **Embedding cache** | Re-index 1.042s → 12,6s; menghemat waktu iterasi, bukan uang |

### Guardrail

Dua lapis: regex (nol biaya, pola eksplisit) lalu LLM (proxy halus).

Terbukti menangkap *"fresh graduate yang punya digital native mindset"* sebagai **age proxy**, dengan saran perumusan ulang yang job-related. Query yang tertangkap regex berhenti dengan **nol token** — tidak menyentuh routing maupun agent.

Dasar: EU AI Act mengklasifikasikan employment screening sebagai high-risk; NYC Local Law 144 mewajibkan bias audit tahunan. Sistem ini tidak mengklaim compliance — hanya menunjukkan kesadaran risikonya.

---

## Menjalankan secara lokal

```bash
git clone https://github.com/ArkaTri/talent-intelligence-copilot.git
cd talent-intelligence-copilot

python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env      # isi kredensial
```

Butuh: OpenAI API key, Qdrant Cloud cluster (free tier cukup).

```bash
# Ingestion (sekali)
python -m src.ingestion.indexer --recreate

# Jalankan aplikasi
streamlit run app/main.py
```

Flag indexer:
- `--recreate` — hapus collection dan buat ulang (wajib saat mengubah konfigurasi chunking)
- `--skip-upsert` — verifikasi pipeline tanpa menulis ke Qdrant

---

## Struktur

```
├── app/main.py                  Streamlit UI
├── src/
│   ├── ingestion/               loader → chunker → redactor → indexer
│   ├── retrieval/               vector_store, reranker
│   ├── agents/                  supervisor, retrieval, evaluator, analytics, schemas
│   └── guardrails/              query_filter, citation_verifier
├── notebooks/                   eksplorasi data + script verifikasi
└── docs/decisions.md            catatan keputusan teknis lengkap
```

`docs/decisions.md` memuat setiap keputusan beserta angka dan alasannya — termasuk yang tidak muat di README ini.

---

## Keterbatasan yang diakui

**Struktur teks hilang.** Kolom `Resume_str` adalah HTML yang diratakan. Relasi antar elemen hilang:

```
"Company Name  March 2003  to  Current  Finance Manager  City , State"
```

Di HTML sumber ini adalah tabel — perusahaan di kiri, tanggal di kanan, jabatan di baris berikutnya. Relasi dibawa oleh tag, bukan urutan kata. Akibatnya pertanyaan seperti "berapa lama di posisi terakhir?" sulit dijawab akurat.

Kolom `Resume_html` menyimpan struktur lengkap dan bisa di-parse ulang — **pengembangan lanjutan paling berdampak**, tidak sempat dikerjakan dalam batas waktu.

**Nama kandidat tidak ada di dataset.** Livecareer.com menghapusnya sebelum publikasi; slot nama diisi job title. Terlihat di HTML sumber: `<span id="...FNAM1"> </span>` kosong, `<span id="...LNAM1">FINANCE MANAGER</span>`. `resume_id` adalah satu-satunya identifier yang mungkin.

**Redaksi PII tidak lengkap.** Nama dan alamat di badan teks lolos. Contoh dari korpus: blok "Anthony Nelson / 88 Malard Drive / Clarksville, TN 93002". Redaksi URL secara tidak langsung menangkap sebagian nama yang tertanam di URL profil.

**Evaluation harness belum dijalankan.** Sudah dirancang — golden set 25–30 query, recall@k, MRR, faithfulness dengan LLM-as-judge memakai model yang lebih kuat dari generator (menghindari self-preference bias). Belum sempat dieksekusi.

**chunk_index kadang keliru di jalur evaluator.** Kutipannya benar dan terverifikasi ada di resume yang tepat, tapi nomor chunk-nya meleset — navigasi citation jadi tidak selalu akurat. Muncul saat evaluator memproses beberapa profil sekaligus (~5.400 token input).

**Dataset US-centric.** Hasil scraping livecareer.com; tidak merepresentasikan pasar kerja Indonesia atau Singapura.

---

## Pengembangan lanjutan

1. Parse ulang dari `Resume_html` — mengembalikan struktur jabatan/periode/perusahaan
2. Jalankan evaluation harness, laporkan ablation chunking dan model
3. Uji NER berbasis transformer (`en_core_web_trf`) untuk redaksi
4. Ablation payload reranker: chunk penuh vs truncated

---

## Catatan

Project ini dibangun sebagai capstone bootcamp, bukan sistem produksi. Yang dipertahankan bukan klaim kesempurnaan, melainkan **jejak keputusan** — setiap pilihan teknis di sini punya alasan yang bisa ditelusuri ke pengukuran, termasuk keputusan untuk membatalkan komponen yang tidak bekerja.