"""Pre-check: menolak query berbasis atribut terproteksi.

KENAPA INI ADA:
Employment screening diklasifikasikan HIGH-RISK oleh EU AI Act. NYC Local
Law 144 mewajibkan bias audit tahunan untuk automated employment decision
tools. Sistem ini tidak mengklaim compliance — hanya menunjukkan kesadaran
risikonya.

KENAPA NODE, BUKAN TOOL:
Kalau guardrail jadi tool yang agent putuskan sendiri kapan dipakai, model
bisa melewatinya — dan justru pada query bermasalah, di mana model mungkin
menilai guardrail tidak relevan. Sebagai node di StateGraph, jalurnya
dipaksa: setiap query lewat sini.

DUA LAPIS:
1. Regex — cepat, deterministik, nol biaya. Menangkap pola eksplisit.
2. LLM — menangkap yang lolos regex (proxy halus, bahasa tidak langsung).

Regex dijalankan dulu. Kalau sudah tertangkap, LLM tidak dipanggil —
menghemat biaya dan latensi untuk kasus yang jelas.

PENOLAKAN HARUS MENDIDIK, BUKAN MEMBLOKIR:
Jelaskan kenapa kriteria itu bermasalah, tawarkan alternatif job-related.
Pengguna yang bertanya "kandidat laki-laki" mungkin sebenarnya mencari
sesuatu yang sah dan bisa dirumuskan ulang.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.schemas import GuardrailVerdict, UsageRecord

load_dotenv()

MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-4.1-nano")

# Pola eksplisit. Sengaja konservatif — hanya menangkap yang jelas,
# sisanya diserahkan ke LLM agar tidak menolak query sah.
PROTECTED_PATTERNS = [
    (r"\b(laki-laki|perempuan|pria|wanita|male|female)\b", "gender"),
    (r"\b(muda|tua|young|old|di\s*bawah\s*\d+\s*tahun|under\s*\d+|over\s*\d+\s*years?\s*old)\b",
     "usia"),
    (r"\b(menikah|belum\s*menikah|lajang|married|single|unmarried)\b",
     "status pernikahan"),
    (r"\b(agama|muslim|kristen|katolik|hindu|buddha|religion|religious)\b",
     "agama"),
    (r"\b(pribumi|warga\s*negara|kewarganegaraan|citizenship|nationality|ras|suku)\b",
     "kewarganegaraan atau etnis"),
    (r"\b(hamil|pregnant|disabilitas|disabled|cacat)\b",
     "kondisi fisik atau kehamilan"),
]


LLM_PROMPT = """You screen recruiter queries for discriminatory criteria.

BLOCK queries that filter candidates by protected attributes:
gender, age, race, ethnicity, nationality, religion, marital status,
pregnancy, disability, sexual orientation.

ALLOW queries about job-related criteria:
skills, years of experience, certifications, education, industry,
specific responsibilities, tools used.

Watch for PROXIES — indirect ways of asking the same thing:
  "fresh graduate feel", "digital native", "recent grad only" -> age proxy
  "cultural fit with our young team" -> age proxy
  "someone who can lift heavy things" -> disability proxy (unless the job
    genuinely requires it, in which case allow)

Return JSON:
{
  "allowed": true|false,
  "triggered_rule": "<which attribute, if blocked>",
  "explanation": "<why this is problematic, 1-2 sentences, plain language>",
  "suggested_rephrase": "<a job-related alternative that likely serves the
                          user's actual need>"
}

Write explanation and suggested_rephrase in the SAME LANGUAGE as the query.
When allowed, set the other fields to null."""


def _regex_check(query: str) -> tuple[bool, str | None]:
    """Cek pola eksplisit. Returns (lolos, atribut yang terpicu)."""
    low = query.lower()
    for pattern, attr in PROTECTED_PATTERNS:
        if re.search(pattern, low, flags=re.IGNORECASE):
            return False, attr
    return True, None


def check_query(
    query: str,
    client: OpenAI | None = None,
    model: str = MODEL_ROUTER,
    use_llm: bool = True,
) -> tuple[GuardrailVerdict, list[UsageRecord]]:
    """Periksa query. Returns (verdict, usage).

    use_llm=False mempercepat untuk pengujian, tapi hanya menangkap
    pola eksplisit — proxy halus akan lolos.
    """
    usage: list[UsageRecord] = []

    ok, attr = _regex_check(query)
    if not ok:
        # Tertangkap regex — tidak perlu panggil LLM. Nol biaya.
        return GuardrailVerdict(
            allowed=False,
            triggered_rule=attr,
            explanation=(
                f"Pertanyaan ini menyaring kandidat berdasarkan {attr}, "
                "yang merupakan atribut terlindungi dan tidak boleh dipakai "
                "sebagai kriteria seleksi."
            ),
            suggested_rephrase=(
                "Coba rumuskan ulang berdasarkan kriteria pekerjaan — "
                "keterampilan, lama pengalaman, sertifikasi, atau "
                "tanggung jawab spesifik."
            ),
        ), usage

    if not use_llm:
        return GuardrailVerdict(allowed=True), usage

    oa = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = oa.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    u = resp.usage
    usage.append(UsageRecord(
        component="guardrail", model=model,
        input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
    ))

    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        # Gagal parse → izinkan. Guardrail yang rusak tidak boleh
        # memblokir seluruh sistem; regex sudah menangkap kasus jelas.
        return GuardrailVerdict(allowed=True), usage

    allowed = bool(data.get("allowed", True))

    # Kalau lolos, field penjelasan dikosongkan. LLM kadang mengisinya
    # meski allowed=true, dan explanation itu ikut tampil sebagai jawaban
    # di UI — menimpa jawaban agent yang sebenarnya.
    return GuardrailVerdict(
        allowed=allowed,
        triggered_rule=data.get("triggered_rule") if not allowed else None,
        explanation=data.get("explanation") if not allowed else None,
        suggested_rephrase=data.get("suggested_rephrase") if not allowed else None,
    ), usage


if __name__ == "__main__":
    tests = [
        "Cari kandidat dengan pengalaman audit 5 tahun",
        "Cari kandidat laki-laki untuk posisi manager",
        "Kandidat di bawah 30 tahun yang punya sertifikasi PMP",
        "Saya butuh fresh graduate yang punya digital native mindset",
        "Kandidat yang menguasai AWS dan Kubernetes",
        "Cari kandidat yang belum menikah",
    ]

    for q in tests:
        v, usage = check_query(q)
        mark = "LOLOS " if v.allowed else "DITOLAK"
        cost = sum(r.cost_usd for r in usage)
        print(f"[{mark}] {q}")
        if not v.allowed:
            print(f"          aturan  : {v.triggered_rule}")
            print(f"          alasan  : {v.explanation}")
            print(f"          saran   : {v.suggested_rephrase}")
        print(f"          biaya   : ${cost:.6f}\n")