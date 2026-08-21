"""Skema Pydantic untuk output agent dan pelacakan token.

DUA FUNGSI MODUL INI:

1. STRUCTURED OUTPUT
   Evaluator agent mengembalikan objek tervalidasi, bukan prosa bebas.
   Prosa tidak bisa difilter, diurutkan, atau ditampilkan sebagai tabel —
   objek bisa. Ini yang membuat sistem composable, bukan sekadar chatbot.

2. USAGE TRACKING
   Soal capstone mewajibkan aplikasi menampilkan informasi penggunaan
   token. Contoh main.py dari Purwadhika menjumlahkan token dari satu
   agent. Sistem ini punya 3-4 pemanggilan LLM per query dengan model
   berbeda:

     routing   gpt-4.1-nano   $0.10 / $0.40 per 1M
     rerank    gpt-4.1-nano   $0.10 / $0.40
     answer    gpt-4.1-mini   $0.40 / $1.60
     judge     gpt-5.4        $2.50 / $15.00  (offline, evaluasi saja)

   Menjumlahkan token mentah lintas model MENYESATKAN — 5.000 token di
   nano tidak setara 5.000 token di mini. Agregasi harus per-model.

   Karena itu setiap agent mengembalikan UsageRecord, bukan hanya
   jawaban. Kalau tidak dibakukan sekarang, panel token di Streamlit
   harus menangani tiga format berbeda.

CITATION ENFORCEMENT:
Setiap klaim tentang kandidat wajib menyertakan resume_id. Itu dijadikan
FIELD WAJIB di skema, bukan sekadar instruksi prompt — model yang lupa
menyertakan citation akan gagal validasi Pydantic, bukan diam-diam lolos.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Tarif per 1 juta token. Diverifikasi dari halaman pricing OpenAI.
# Dipakai untuk menghitung biaya per komponen di panel usage.
MODEL_PRICING = {
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4o-mini":  {"in": 0.15, "out": 0.60},
    "gpt-5.4-nano": {"in": 0.20, "out": 1.25},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.50},
    "gpt-5.4":      {"in": 2.50, "out": 15.00},
}

USD_TO_IDR = 16_500


# ── USAGE TRACKING ───────────────────────────────────────────────────

class UsageRecord(BaseModel):
    """Catatan penggunaan token satu pemanggilan LLM.

    component dipisah dari model karena satu model bisa dipakai beberapa
    komponen (routing dan rerank sama-sama nano). Untuk analisis biaya,
    yang menarik adalah "berapa mahal reranking", bukan "berapa mahal nano".
    """
    component: Literal["routing", "rerank", "answer", "evaluate", "analytics", "guardrail"]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        p = MODEL_PRICING.get(self.model)
        if not p:
            return 0.0
        return (self.input_tokens * p["in"]
                + self.output_tokens * p["out"]) / 1_000_000

    @property
    def cost_idr(self) -> float:
        return self.cost_usd * USD_TO_IDR


class UsageSummary(BaseModel):
    """Agregat usage satu query — untuk ditampilkan di panel Streamlit."""
    records: list[UsageRecord] = Field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    @property
    def total_input(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def per_component(self) -> list[dict]:
        """Breakdown per komponen untuk tabel di UI.

        Ini yang membuat temuan "model jawaban menyerap ~90% biaya"
        terlihat sendiri tanpa perlu dijelaskan.
        """
        return [
            {
                "component": r.component,
                "model": r.model,
                "in": r.input_tokens,
                "out": r.output_tokens,
                "cost_usd": r.cost_usd,
            }
            for r in self.records
        ]


# ── EVIDENCE & CITATION ──────────────────────────────────────────────

class Evidence(BaseModel):
    """Satu potong bukti yang mendukung sebuah klaim.

    resume_id dan quote WAJIB. Klaim tanpa keduanya tidak bisa
    ditelusuri, dan sistem ini tidak boleh menghasilkan klaim seperti itu.
    """
    resume_id: str = Field(description="ID resume sumber")
    chunk_index: int = Field(default=0, description="Posisi chunk dalam resume")
    quote: str = Field(description="Kutipan persis dari resume, maksimal 200 karakter")
    section: str = Field(default="unknown", description="Bagian resume asal kutipan")

    def citation(self) -> str:
        return f"[{self.resume_id}#{self.chunk_index}]"


# ── EVALUATOR OUTPUT ─────────────────────────────────────────────────

class SkillMatch(BaseModel):
    """Satu skill dari job description dan status pemenuhannya."""
    skill: str
    status: Literal["strong", "partial", "missing"]
    evidence: list[Evidence] = Field(default_factory=list)

    # Validasi silang tidak dipasang sebagai validator Pydantic karena
    # model kadang benar-benar tidak menemukan bukti untuk skill "missing" —
    # dan itu justru informasi yang valid. Pemeriksaan bahwa status
    # "strong" harus punya evidence dilakukan di citation_verifier.


class CandidateEvaluation(BaseModel):
    """Hasil evaluasi satu kandidat terhadap job description.

    Structured, bukan prosa: memungkinkan UI mengurutkan kandidat,
    memfilter berdasarkan gap, dan menampilkan perbandingan berdampingan.
    """
    resume_id: str
    category: str = "unknown"
    overall_score: int = Field(ge=0, le=10, description="Kecocokan keseluruhan 0-10")
    summary: str = Field(description="Ringkasan penilaian, maksimal 2 kalimat")
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="Kekurangan yang teridentifikasi")
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def cited_resume_ids(self) -> set[str]:
        """Semua resume_id yang dikutip — dipakai citation_verifier
        untuk memastikan tidak ada kutipan dari resume di luar hasil
        retrieval (halusinasi sumber)."""
        ids = {e.resume_id for e in self.evidence}
        for sm in self.skill_matches:
            ids.update(e.resume_id for e in sm.evidence)
        return ids


class EvaluationResult(BaseModel):
    """Output evaluator agent — bisa berisi beberapa kandidat."""
    job_context: str = Field(description="Kriteria yang dipakai menilai")
    candidates: list[CandidateEvaluation] = Field(default_factory=list)

    def ranked(self) -> list[CandidateEvaluation]:
        return sorted(self.candidates, key=lambda c: c.overall_score, reverse=True)


# ── ANALYTICS OUTPUT ─────────────────────────────────────────────────

class AnalyticsResult(BaseModel):
    """Output analytics agent.

    `answerable` dan `refusal_reason` ada karena sebagian pertanyaan
    agregat TIDAK BOLEH dijawab — label kategori dataset terbukti tidak
    tepercaya (69% kandidat cloud terlewat, 93% false positive kalau
    dihitung lewat label).

    Menolak menjawab dengan alasan adalah fitur, bukan kegagalan.
    """
    question: str
    answerable: bool = True
    refusal_reason: str | None = None

    count: int | None = None
    unit: Literal["resumes", "chunks"] | None = None
    terms_searched: list[str] = Field(default_factory=list)
    per_term: dict[str, int] = Field(
        default_factory=dict,
        description="Jumlah resume per istilah — membantu mendeteksi istilah"
                    "terlalu generik yang menangkap false positive",
    )
    breakdown: dict[str, int] = Field(default_factory=dict)
    caveat: str | None = Field(
        default=None,
        description="Batasan hasil, mis. exact match tidak menangkap sinonim",
    )


# ── ROUTING ──────────────────────────────────────────────────────────

class RouteDecision(BaseModel):
    """Keputusan supervisor: agent mana yang menangani query.

    Field filter (category, section_type) diekstrak di sini, bukan di
    retrieval agent, karena supervisor yang membaca maksud pengguna.
    Verifikasi menunjukkan category filter hanya berguna kalau pengguna
    MENYEBUT kategori — memaksakannya pada query umum justru mempersempit
    tanpa alasan.
    """
    agent: Literal["retrieval", "evaluator", "analytics", "profile", "refuse"]
    reasoning: str = Field(description="Alasan singkat pemilihan, maksimal 15 kata")
    category_filter: list[str] = Field(
        default_factory=list,
        description="Kategori yang disebut pengguna. Kosong kalau tidak disebut.",
    )
    section_filter: list[str] = Field(
        default_factory=list,
        description="Section spesifik. Kosongkan kecuali ada alasan jelas — "
                    "embedding sudah merutekan dengan benar tanpa bantuan.",
    )
    search_query: str = Field(description="Query yang dioptimalkan untuk pencarian")
    target_resume_ids: list[str] = Field(
        default_factory=list,
        description="Resume ID spesifik yang diminta, mis. untuk follow-up "
                    "'kandidat tadi' atau 'detail kandidat 11183737'",
    )


# ── GUARDRAIL ────────────────────────────────────────────────────────

class GuardrailVerdict(BaseModel):
    """Hasil pre-check terhadap query pengguna.

    Employment screening diklasifikasikan high-risk oleh EU AI Act, dan
    NYC Local Law 144 mewajibkan bias audit. Sistem ini tidak mengklaim
    compliance — hanya menunjukkan kesadaran risikonya.

    Penolakan harus MENDIDIK, bukan sekadar memblokir: jelaskan kenapa
    kriteria itu bermasalah dan tawarkan alternatif yang job-related.
    """
    allowed: bool
    triggered_rule: str | None = None
    explanation: str | None = Field(
        default=None, description="Kenapa ditolak, dalam bahasa yang jelas"
    )
    suggested_rephrase: str | None = Field(
        default=None, description="Alternatif query yang job-related"
    )


# ── RESPONS AKHIR ────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    """Bentuk seragam yang dikembalikan supervisor ke lapisan UI.

    Satu bentuk untuk semua jalur agent — UI tidak perlu tahu agent mana
    yang menangani. Field yang tidak relevan dibiarkan None.
    """
    answer: str
    route: RouteDecision | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
    analytics: AnalyticsResult | None = None
    guardrail: GuardrailVerdict | None = None
    usage: UsageSummary = Field(default_factory=UsageSummary)

    # Diisi citation_verifier. Kalau ada klaim tanpa citation valid,
    # UI menampilkan peringatan alih-alih menyembunyikannya.
    citation_ok: bool = True
    citation_issues: list[str] = Field(default_factory=list)


if __name__ == "__main__":
    # Uji cepat: pastikan skema valid dan perhitungan biaya benar
    usage = UsageSummary()
    usage.add(UsageRecord(component="routing", model="gpt-4.1-nano",
                          input_tokens=620, output_tokens=80))
    usage.add(UsageRecord(component="rerank", model="gpt-4.1-nano",
                          input_tokens=2344, output_tokens=384))
    usage.add(UsageRecord(component="answer", model="gpt-4.1-mini",
                          input_tokens=2750, output_tokens=400))

    print(f"{'component':12s} {'model':14s} {'in':>7s} {'out':>6s} {'cost':>12s}  {'%':>5s}")
    print("-" * 62)
    total = usage.total_cost_usd
    for r in usage.per_component():
        pct = r["cost_usd"] / total * 100 if total else 0
        print(f"{r['component']:12s} {r['model']:14s} {r['in']:7,} "
              f"{r['out']:6,} ${r['cost_usd']:11.6f}  {pct:4.1f}%")

    print("-" * 62)
    print(f"{'TOTAL':12s} {'':14s} {usage.total_input:7,} "
          f"{usage.total_output:6,} ${total:11.6f}")
    print(f"\nper 1.000 query: ${total * 1000:.4f} "
          f"(~Rp {total * 1000 * USD_TO_IDR:,.0f})")

    ev = Evidence(resume_id="11759079", chunk_index=3,
                  quote="led inventory test work at multiple audit clients",
                  section="experience")
    print(f"\ncitation: {ev.citation()}")

    ar = AnalyticsResult(
        question="Berapa kandidat IT?",
        answerable=False,
        refusal_reason="Label kategori dataset tidak tepercaya — 93% resume "
                       "berlabel IT tidak menyebut kemampuan teknis terkait.",
    )
    print(f"analytics refusal: {ar.refusal_reason}")