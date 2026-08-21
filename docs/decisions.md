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

## Retrieval Design

**Keputusan:**
1. Category filter — ada, opsional, tidak otomatis
2. Section filter — ada, tidak default
3. Tanpa threshold skor
4. Dedup per resume_id — default aktif
5. Fetch 20 → dedup → top 5

**Bukti (notebooks/check_02_retrieval.py, 22.866 chunk):**

*Category filter berguna, tapi bukan karena "meningkatkan relevansi".*
Tanpa filter, top-5 berisi rata-rata 3,25/5 dari kategori target.
Query "registered nurse patient care" mengembalikan 3 ADVOCATE dari 10
hasil — bahasa advokasi dekat secara semantik dengan patient care.
Filter menegakkan constraint yang tidak bisa dijamin embedding.
Hanya dipakai saat pengguna menyebut kategori.

*Section filter sering redundan atau merugikan.*
Embedding sudah merutekan dengan benar tanpa bantuan:
  "proficient in Excel and SAP"       → skills 10/10
  "bachelor degree in business admin" → education 10/10
Dan "managed a team across departments" tersebar ke experience (4) dan
accomplishments (3) — isi accomplishments sama relevannya. Memfilter ke
experience membuang hasil valid.

*Threshold skor akan mematikan seluruh domain.*
  kitchen management : 0.6325 – 0.6965
  financial audit    : 0.6014 – 0.6328
  python ML          : 0.3839 – 0.4131
Hasil TERBAIK query Python (0.41) di bawah hasil TERBURUK query chef
(0.63), padahal isinya jelas relevan.

*Dedup diperlukan.*
Tanpa dedup, resume 49777184 mengisi 3 dari 5 slot (chunk #6, #7, #8).
Untuk screening, recruiter butuh keragaman kandidat.

---

## Label kategori dataset tidak tepercaya

**Temuan:** resume "IT COMPLIANCE AUDITOR — 15 years in Information
Technology" berlabel APPAREL. Resume tentang "evaluate, or process loan
applications" berlabel CHEF.

**Konsekuensi:**
1. Sebagian "kegagalan" retrieval mungkin sebenarnya label salah, bukan
   pencarian salah.
2. Golden set TIDAK boleh memakai label kategori sebagai ground truth.
   Relevansi harus diverifikasi dari isi teks secara manual.
3. Metrik precision-by-category akan understate performa sistem.

Dicatat di README sebagai keterbatasan dataset.

## Reranking

**Keputusan:** LLM-based reranking dengan gpt-4.1-nano, chunk penuh
(bukan truncated), JSON mode wajib.

**Kenapa reranker perlu — bukti:**
Query "financial audit" mengembalikan skor vektor 0.6014–0.6328.
Semuanya "cukup mirip", tapi isinya berbeda kualitas. Reranker
memisahkan bukti pengalaman dari daftar keyword:

  ↑3  "led inventory test work across audit clients"          9.0
  ↓3  "IT compliance — less focus on financial specifics"     6.0

Konsisten lintas domain. Query "python ML deployment":
  ↓3  "Lists Python and ML skills but no deployment details"  2.0
kandidat peringkat 2 versi vektor turun ke 5.

**Dampak terukur:** 6 kandidat dari luar top-5 vektor masuk hasil akhir
di 3 query uji. Tanpa reranker, kandidat itu tidak terlihat sama sekali.

## Analytics — Text-to-SQL dibatalkan

**Keputusan:** analytics agent menghitung dari ISI TEKS lewat Qdrant
MatchText, bukan dari label kategori lewat SQL.

**Bukti (notebooks/check_analytics.py):**

Pertanyaan uji: "berapa kandidat dengan kemampuan cloud?"

  [label]  kategori INFORMATION-TECHNOLOGY : 120 resume
  [teks]   menyebut AWS atau Azure         :  29 resume
  irisan                                   :   9 resume
  cloud tapi BUKAN label IT                :  20 resume  (69% terlewat)
  label IT tanpa cloud                     : 111 resume  (93% false positive)

Distribusi chunk yang menyebut "AWS":
  INFORMATION-TECHNOLOGY 17, CONSULTANT 8, ADVOCATE 5, AGRICULTURE 4,
  DIGITAL-MEDIA 3, DESIGNER 2, ENGINEERING 2, CONSTRUCTION 2, BANKING 2

Label kategori bukan proksi untuk kemampuan teknis. Text-to-SQL berbasis
label akan menghasilkan angka yang salah dengan percaya diri — dan untuk
sistem yang nilai jualnya evidence-grounded, itu kontradiksi.

**Konsekuensi:** Text-to-SQL (Module 3 Session 16) tidak dipakai. Ini
melepas satu kesempatan menunjukkan materi kelas, tapi memakai teknik
pada data yang cacat adalah demonstrasi tanpa penilaian.

**Payload index TEXT** ditambahkan untuk field `text` (3,9 detik).
Berbeda dari KEYWORD: TEXT melakukan tokenisasi sehingga bisa mencari
kata di dalam teks.

**Kecepatan:** 0,13–0,33 detik per term di 22.866 chunk. Aman untuk UI.

**Batasan:** MatchText mencocokkan kata, bukan makna. "Amazon Web
Services" tanpa singkatan tidak tertangkap oleh query "AWS". Agent perlu
mengirim beberapa varian istilah.

**Batasan kedua:** angka hasil selalu kecil (AWS 47 chunk, Kubernetes 2)
karena dataset didominasi profesi non-teknis. Itu benar, bukan kegagalan
pencarian — perlu disampaikan agar tidak disalahartikan.

## Komposisi biaya aktual (dari schemas.py)

| Komponen | Model | Biaya | Porsi |
|---|---|---|---|
| routing | gpt-4.1-nano | $0.000094 | 4,2% |
| rerank | gpt-4.1-nano | $0.000388 | 17,5% |
| answer | gpt-4.1-mini | $0.001740 | 78,3% |
| **total** | | **$0.002222** | |

**$2,22 per 1.000 query.**

Mengoreksi proyeksi guideline (2% / 8% / 90%). Reranking dua kali lebih
mahal dari perkiraan karena keputusan memakai chunk penuh, bukan
truncated — konsekuensi yang bisa dilacak ke keputusan spesifik.

Prioritas ablation tidak berubah: model jawaban (78,3%) lebih berdampak
daripada payload reranker (17,5%). Tapi jaraknya lebih sempit dari
perkiraan, jadi truncation tetap layak diuji kalau ada waktu.

## Biaya aktual retrieval agent (3 query uji)

| Query | Total | rerank | answer |
|---|---|---|---|
| audit keuangan | $0.001298 | $0.000405 | $0.000893 |
| banking risk | $0.001149 | $0.000449 | $0.000700 |
| memimpin tim | $0.001138 | $0.000345 | $0.000793 |
| **rata-rata** | **$0.001195** | ~35% | ~65% |

**$1,20 per 1.000 query** — setengah dari simulasi schemas.py ($2,22).
Penyebab: dedup memangkas kandidat sebelum reranking, payload lebih kecil
dari asumsi 20 chunk penuh.

Komposisi juga bergeser: rerank ~35% (simulasi 17,5%), answer ~65%
(simulasi 78,3%). Jarak keduanya menyempit — ablation payload reranker
jadi lebih layak dipertimbangkan.

**Kualitas jawaban:** citation lengkap di semua klaim, perbandingan
berbasis bukti, angka konkret terekstrak, ketidakpastian dinyatakan
eksplisit ("tidak disebutkan ukuran pasti tim"). Bahasa Indonesia bersih
tanpa karakter non-Latin — kontras dengan temuan gpt-5.4-mini di Tahap 5.

## Analytics agent — exact matching juga punya mode kegagalan

**Temuan:** `MatchText` Qdrant melakukan tokenisasi. Frasa multi-kata
pecah jadi kata terpisah dan dicocokkan dengan OR.

Istilah "Project Management Professional" menangkap **310 resume** —
setiap resume yang menyebut "project management". Setelah frasa >2 kata
dibuang, PMP turun ke **30 resume**.

Kasus serupa: istilah generik "cloud" menghasilkan 91 resume. Setelah
diganti istilah spesifik (AWS, Azure, GCP, Kubernetes): **32 resume**,
konsisten dengan verifikasi awal (AWS+Azure = 29).

**Perbaikan:**
1. Filter `len(t.split()) <= 2` — buang frasa multi-kata
2. Prompt planner melarang kata umum: project, management, professional,
   experience, business, service, development
3. Field `per_term` dilaporkan agar istilah terlalu generik terlihat,
   bukan tersembunyi di angka total

**Pelajaran:** setiap metode retrieval punya mode kegagalan sendiri.
Vector search gagal membedakan bukti dari keyword (→ reranker).
Label kategori gagal karena kotor (→ text matching).
Text matching gagal karena tokenisasi (→ filter istilah).

**Penolakan bekerja:** 3 pertanyaan uji ditolak dengan alasan spesifik —
dua karena bergantung label, satu karena atribut terproteksi.

**Biaya:** ~$0.00007 per query. Pencarian Qdrant gratis, hanya ekstraksi
istilah yang memakai LLM.

## Evaluator agent — structured output

**Keputusan:** output Pydantic tervalidasi, bukan prosa. Memungkinkan UI
mengurutkan kandidat, memfilter berdasarkan gap, menampilkan berdampingan.

**Yang terbukti bekerja:**
- Status "missing" konsisten tanpa evidence — model tidak mengarang bukti
- Kutipan disalin persis; artefak teks asli terbawa ("andadvanced",
  spasi hilang di sumber) — bukti tidak diparafrase
- Gap spesifik dan bisa ditindaklanjuti
- Skor koheren dengan jumlah kriteria terpenuhi

**Biaya:** $0.004686 per evaluasi 3 kandidat — 3,7x retrieval agent
($0.001278). Penyebabnya output (1.729 token), bukan input. Structured
output dengan evidence per skill memang verbose. Sepadan dengan
kualitasnya, tapi perlu diperhitungkan untuk session cap di Streamlit.

**Masalah bahasa:** prompt sistem berbahasa Inggris membuat model
menghasilkan summary/gaps dalam Inggris meski requirements Indonesia.
Diperbaiki dengan instruksi deteksi bahasa eksplisit.

**Validasi berlapis:**
1. JSON mode — output pasti JSON valid
2. `_parse_evaluation` membuang resume_id yang tidak ada di input
   (lapisan pertama terhadap halusinasi sumber)
3. Pydantic — field wajib tidak boleh kosong
4. Kandidat yang gagal validasi dilewati, tidak membatalkan seluruh hasil

## Guardrails

**Keputusan:** guardrail sebagai NODE di StateGraph, bukan tool.

**Alasan:** kalau guardrail jadi tool yang agent putuskan sendiri kapan
dipakai, model bisa melewatinya — dan justru pada query bermasalah.
Sebagai node, jalurnya dipaksa.

### Pre-check (query_filter)

Dua lapis: regex → LLM. Regex menangani pola eksplisit dengan biaya nol;
LLM hanya dipanggil untuk yang lolos.

Terbukti: "fresh graduate yang punya digital native mindset" ditolak
sebagai age proxy — tidak mungkin ditangkap regex. Saran perumusan
ulangnya tepat: "kandidat dengan pengalaman dan kemampuan dalam bidang
digital dan teknologi terbaru".

Biaya: $0.000000 (tertangkap regex) sampai $0.000057 (perlu LLM).

Regex awal punya negative lookahead `(?!.*\bmanager\b)` yang terlalu
luas — "kandidat laki-laki untuk posisi manager" lolos regex dan baru
ditangkap LLM. Lookahead dihapus.

### Post-check (citation_verifier)

Tiga jenis kegagalan yang dicari: halusinasi sumber, kutipan tidak cocok,
klaim tanpa citation.

Verifikasi kutipan memecah pada "..." karena uji evaluator menemukan
model menyambung dua bagian teks:
  "Senior Internal Auditor 07/2002 to 06/2004 ... Supervised audit teams"
Isinya benar tapi bukan substring persis.

Ambang kecocokan 0.85, bukan 100%, karena loader menormalisasi whitespace
sementara model merapikan spasi saat mengutip.

Uji: 5 bukti asli lolos, evidence palsu (resume_id 99999999) ditolak.

**Dasar regulasi:** EU AI Act mengklasifikasikan employment screening
sebagai high-risk; NYC Local Law 144 mewajibkan bias audit tahunan.
Sistem ini tidak mengklaim compliance — hanya menunjukkan kesadaran.

## Citation verifier — membedakan chunk_index salah dari halusinasi

**Kasus:** evaluator mengutip "Delivered quality service to top tier
clients within the PWC..." dengan citation [26530575#3].

Pemeriksaan awal (preview 300 karakter) menyimpulkan halusinasi — SALAH.
Frasa itu ada di chunk #2 pada posisi karakter ke-770, di luar jangkauan
preview. Model mengutip PERSIS; hanya chunk_index yang keliru.

**Bug di verifier:** fallback "cari di chunk lain resume yang sama" hanya
jalan kalau chunk_index tidak ditemukan sama sekali. Di jalur evaluator,
`retrieved` berisi semua chunk resume — jadi #3 ada, verifikasi berhenti
di situ tanpa mencoba chunk lain.

**Perbaikan:** kalau kutipan tidak cocok di chunk yang disebut, cari di
seluruh chunk resume sebelum menyatakan gagal. Dua jenis masalah
dibedakan:
- chunk_index salah → kutipan valid, atribusi posisi keliru (peringatan)
- kutipan tidak ada → halusinasi (kegagalan)

**Pelajaran metodologis:** verifikasi berbasis preview terpotong
menghasilkan kesimpulan yang salah. Periksa dengan `.find()` dan cetak
posisi, bukan potongan awal.

## Identifier kandidat — resume_id, bukan nama

**Fakta:** dataset tidak memuat nama kandidat sama sekali. Livecareer.com
menghapusnya sebelum publikasi. Terlihat di struktur HTML sumber:

  <span class="field" id="...FNAM1"> </span>          <- first name kosong
  <span class="field" id="...LNAM1"> FINANCE MANAGER</span>  <- diisi jabatan

Pola sama di seluruh korpus: "Company Name", "City, State" juga placeholder.

**Konsekuensi:** resume_id adalah satu-satunya identifier yang mungkin.
Ini bukan keputusan desain untuk mengurangi bias — meski efeknya sejalan
dengan praktik blind screening yang direkomendasikan.

**Untuk README:** sebutkan sebagai karakteristik dataset, bukan sebagai
fitur sistem. Mengklaim penyembunyian yang disengaja padahal datanya
memang tidak ada akan terbantah kalau ada yang memeriksa sumbernya.

## Keterbatasan struktur teks — tidak diperbaiki, didokumentasikan

**Masalah:** kolom `Resume_str` adalah HTML yang diratakan tanpa
mempertahankan struktur. Relasi antar-elemen hilang:

  "Company Name  March 2003  to  Current  Finance Manager  City , State"

Di HTML sumber, ini tabel: perusahaan di kiri, tanggal di kanan, jabatan
di baris berikutnya. Relasi dibawa oleh tag dan CSS, bukan urutan kata.
Setelah flattening, tidak ada penanda bahwa "March 2003 to Current" adalah
periode kerja di "Company Name", atau bahwa "Finance Manager" jabatannya.

**Konsekuensi ke kemampuan sistem:**
- Pertanyaan "berapa lama di posisi terakhir?" sulit dijawab akurat —
  model harus menebak pasangan tanggal-jabatan dari urutan kata
- Kutipan model kadang menyambung bagian terpisah dengan "..." karena
  berusaha merekonstruksi struktur yang hilang

**Kenapa tidak diperbaiki:** kolom `Resume_html` menyimpan struktur
lengkap dan bisa di-parse ulang. Estimasi 3-4 jam (parser + penyesuaian
chunker + re-index + uji ulang). Dengan sisa waktu di bawah 3 hari,
risikonya melebihi manfaatnya — chunker harus dirombak karena deteksi
header berbasis spasi tidak berlaku pada teks berstruktur.

Dicatat sebagai pengembangan lanjutan di README.

**Yang diperbaiki:** pemotongan chunk di tengah kata. Jendela pencarian
batas spasi diperlebar 100 → 200 karakter, dengan fallback mencari spasi
pertama setelah batas. Sebelumnya menghasilkan chunk seperti
"cquisitions, and then prepare..." dari kata "acquisitions".