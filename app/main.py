"""Talent Intelligence Copilot — antarmuka Streamlit.

SECRETS:
Streamlit Cloud tidak membaca .env — secrets dikelola lewat dashboard
dalam format TOML. Kode ini menangani keduanya: st.secrets kalau ada,
os.environ kalau tidak. Variabel di-inject ke os.environ sebelum modul
lain diimpor, karena modul-modul itu membaca lewat os.getenv().

SESSION CAP:
Dua batas berjalan bersamaan.
  - Batas query (20) → yang ditampilkan ke pengguna, memberi ekspektasi
  - Batas biaya ($0.03) → pengaman sebenarnya

Biaya per jalur berbeda 25x: analytics $0.00017, evaluator $0.0043.
Batas jumlah query tidak memberi tahu apa pun tentang eksposur finansial.
20x evaluator = $0.086, 20x analytics = $0.003.

CACHING:
Supervisor dan VectorStore dibuat sekali per sesi lewat @st.cache_resource.
Tanpa itu, Streamlit membuat ulang koneksi Qdrant dan client OpenAI di
setiap interaksi — lambat dan boros.
"""
import os
import sys
import re
from pathlib import Path

# streamlit run menjalankan script dari folder app/, sehingga folder src/
# tidak terlihat oleh Python. Root project ditambahkan ke sys.path agar
# `from src...` berfungsi — sama seperti efek `python -m` dari root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

# ── SECRETS: harus dijalankan SEBELUM import modul src ───────────────
# Modul src membaca kredensial lewat os.getenv() saat diimpor, jadi
# nilai harus sudah ada di environment sebelum import terjadi.
_KEYS = [
    "OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION",
    "MODEL_ROUTER", "MODEL_RERANK", "MODEL_ANSWER", "EMBED_MODEL",
]

for _k in _KEYS:
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        # st.secrets melempar error kalau tidak ada file secrets sama
        # sekali. Di lokal itu wajar — .env sudah menyediakan nilainya.
        pass

from src.agents.supervisor import Supervisor          # noqa: E402
from src.retrieval.vector_store import VectorStore    # noqa: E402

MAX_QUERIES_PER_SESSION = 20
MAX_COST_PER_SESSION_USD = 0.03
HISTORY_TURNS = 6

st.set_page_config(
    page_title="Talent Intelligence Copilot",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_supervisor() -> Supervisor:
    """Buat supervisor sekali, pakai ulang di seluruh sesi.

    cache_resource (bukan cache_data) karena objeknya punya koneksi
    jaringan yang tidak bisa di-serialize.
    """
    return Supervisor(vs=VectorStore())


def init_state() -> None:
    defaults = {
        "messages": [],
        "n_queries": 0,
        "total_cost": 0.0,
        "last_response": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limits_reached() -> tuple[bool, str]:
    if st.session_state.n_queries >= MAX_QUERIES_PER_SESSION:
        return True, (
            f"Batas {MAX_QUERIES_PER_SESSION} pertanyaan per sesi tercapai. "
            "Muat ulang halaman untuk memulai sesi baru."
        )
    if st.session_state.total_cost >= MAX_COST_PER_SESSION_USD:
        return True, (
            "Batas biaya sesi tercapai. Aplikasi ini memakai API berbayar, "
            "dan batas ini melindungi biaya operasional. "
            "Muat ulang halaman untuk memulai sesi baru."
        )
    return False, ""


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Tentang")
        st.caption(
            "Asisten screening kandidat berbasis bukti. Setiap klaim "
            "ditelusuri ke kutipan resume sumbernya."
        )
        st.caption("2.481 resume · 22.866 chunk · Qdrant Cloud")

        st.divider()
        st.markdown("### Penggunaan sesi")

        used = st.session_state.n_queries
        cost = st.session_state.total_cost

        c1, c2 = st.columns(2)
        c1.metric("Pertanyaan", f"{used}/{MAX_QUERIES_PER_SESSION}")
        c2.metric("Biaya", f"${cost:.4f}")

        st.progress(min(cost / MAX_COST_PER_SESSION_USD, 1.0))
        st.caption(f"Batas biaya sesi: ${MAX_COST_PER_SESSION_USD:.2f}")

        st.divider()
        st.markdown("### Yang bisa ditanyakan")
        st.caption(
            "**Mencari** — siapa yang punya pengalaman audit keuangan?\n\n"
            "**Menilai** — nilai kandidat untuk posisi X dengan syarat Y\n\n"
            "**Menghitung** — berapa kandidat yang menyebut AWS?"
        )

        st.divider()
        st.markdown("### Batasan")
        st.caption(
            "Sistem ini **decision-support**, bukan decision-making. "
            "Tidak melakukan auto-reject. Query berbasis atribut "
            "terlindungi (gender, usia, status pernikahan) ditolak."
        )

def _clean_display(text: str) -> str:
    """Rapikan teks untuk ditampilkan.

    Dua hal:
    1. Kolapskan spasi berulang. Loader sengaja mempertahankannya karena
       chunker memakai runs of spaces untuk mendeteksi header — tapi itu
       keputusan untuk pipeline, bukan untuk tampilan.
    2. Escape "$". Teks resume mengandung nominal ("$1,000,000 to
       $4,000,000") dan Streamlit menafsirkan pasangan $ sebagai LaTeX.
    """
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)   # rapikan spasi sebelum tanda baca
    return text.replace("$", "\\$")

def render_response(r) -> None:
    """Tampilkan jawaban beserta bukti, rute, dan usage."""
    st.markdown(_clean_display(r.answer))

    # Peringatan citation ditampilkan, bukan disembunyikan — pengguna
    # berhak tahu klaim mana yang tidak terverifikasi penuh.
    if not r.citation_ok and r.citation_issues:
        with st.expander("⚠️ Catatan verifikasi kutipan", expanded=False):
            for issue in r.citation_issues:
                st.warning(issue)

    if r.guardrail and not r.guardrail.allowed:
        st.info(
            f"Guardrail aktif — aturan: **{r.guardrail.triggered_rule}**",
            icon="🛡️",
        )

    if r.evidence:
        with st.expander(f"📄 Bukti ({len(r.evidence)} kutipan)"):
            for e in r.evidence:
                st.markdown(f"**{e.citation()}** · {e.section}")
                st.caption(f"\"{_clean_display(e.quote)}\"")
                
    if r.evaluation:
        with st.expander("📊 Penilaian terstruktur"):
            for c in r.evaluation.ranked():
                st.markdown(f"**[{c.resume_id}]** — skor {c.overall_score}/10")
                st.caption(c.summary)
                for sm in c.skill_matches:
                    mark = {"strong": "✅", "partial": "🟡", "missing": "❌"}
                    st.markdown(f"{mark.get(sm.status, '?')} {sm.skill}")
                if c.gaps:
                    st.caption("Gap: " + "; ".join(c.gaps))
                st.divider()

    if r.analytics and r.analytics.answerable and r.analytics.breakdown:
        with st.expander("📈 Sebaran kategori"):
            st.bar_chart(r.analytics.breakdown)

    with st.expander("🔧 Detail teknis"):
        if r.route:
            st.markdown(f"**Rute:** `{r.route.agent}` — {r.route.reasoning}")
            if r.route.category_filter:
                st.markdown(f"**Filter kategori:** {r.route.category_filter}")
            if r.route.search_query:
                st.caption(f"Query pencarian: {r.route.search_query}")

        st.markdown("**Penggunaan token**")
        rows = r.usage.per_component()
        if rows:
            st.dataframe(
                [
                    {
                        "Komponen": x["component"],
                        "Model": x["model"],
                        "Token in": f"{x['in']:,}",
                        "Token out": f"{x['out']:,}",
                        "Biaya": f"${x['cost_usd']:.6f}",
                    }
                    for x in rows
                ],
                hide_index=True,
                use_container_width=True,
            )
        st.markdown(
            f"**Total:** {r.usage.total_input:,} in · "
            f"{r.usage.total_output:,} out · "
            f"**${r.usage.total_cost_usd:.6f}**"
        )


def main() -> None:
    init_state()

    st.title("🔍 Talent Intelligence Copilot")
    st.caption(
        "Screening kandidat berbasis bukti — setiap klaim ditelusuri "
        "ke kutipan resume sumbernya."
    )

    render_sidebar()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("response"):
                render_response(m["response"])
            else:
                st.markdown(m["content"])

    blocked, msg = limits_reached()
    if blocked:
        st.warning(msg)
        st.stop()

    prompt = st.chat_input("Tanyakan tentang kandidat…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Spinner penting: Qdrant di Sydney menambah latensi, dan jeda
        # tanpa indikator terasa seperti aplikasi menggantung.
        with st.spinner("Mencari dan memverifikasi…"):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-HISTORY_TURNS:]
                ]
                r = get_supervisor().ask(prompt, history=history)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {type(e).__name__}")
                st.caption(str(e)[:300])
                return

        render_response(r)

    st.session_state.n_queries += 1
    st.session_state.total_cost += r.usage.total_cost_usd
    st.session_state.messages.append({
        "role": "assistant",
        "content": r.answer,
        "response": r,
    })

    # Rerun agar sidebar metrics ikut terbarui
    st.rerun()


if __name__ == "__main__":
    main()