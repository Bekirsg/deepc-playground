"""
DeePC Playground  —  Interactive Data-Enabled Predictive Control Demo
======================================================================
Usage:  streamlit run app.py
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    PRESETS,
    SYSTEMS,
    ClassicMPCSolver,
    DeePCSolver,
    auto_interpret,
    build_hankel,
    compute_metrics,
    generate_reference,
    get_ss_matrices,
    simulate_step,
    simulate_system,
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DeePC Playground",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap');

    /* ── Root variables ── */
    :root {
        --clr-bg:        #0d1117;
        --clr-surface:   #161b22;
        --clr-border:    #30363d;
        --clr-primary:   #2f81f7;
        --clr-primary2:  #1f6feb;
        --clr-green:     #3fb950;
        --clr-orange:    #e3b341;
        --clr-red:       #f85149;
        --clr-purple:    #a371f7;
        --clr-text:      #e6edf3;
        --clr-muted:     #8b949e;
        --font-main:     'IBM Plex Sans', sans-serif;
        --font-mono:     'JetBrains Mono', monospace;
        --font-serif:    'IBM Plex Serif', serif;
        --radius:        8px;
        --shadow:        0 4px 24px rgba(0,0,0,0.4);
    }

    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: var(--font-main) !important;
        color: var(--clr-text) !important;
    }

        /* === FORCE DARK MODE FOR ALL STREAMLIT WIDGETS (KALICI ÇÖZÜM) === */
    [data-testid="stAppViewContainer"] {
        background-color: var(--clr-bg) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--clr-bg) !important;
    }
    .stApp {
        background-color: var(--clr-bg) !important;
    }
    /* Tüm widget container’ları ve içerikleri */
    div[data-testid="stExpander"],
    div[data-baseweb="select"],
    div[data-testid="stNumberInput"],
    div[data-testid="stSlider"],
    div[data-testid="stDataFrame"],
    .stMarkdown > div,
    .stTable,
    .stDataFrame,
    div[data-testid="stMarkdownContainer"] {
        background-color: var(--clr-surface) !important;
        color: var(--clr-text) !important;
        border-color: var(--clr-border) !important;
    }
    /* Input iç yüzeyleri (selectbox, number_input, slider) */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] > div > div,
    div[data-testid="stSlider"] > div > div > div {
        background-color: var(--clr-surface) !important;
        color: var(--clr-text) !important;
    }
    /* Tüm metin ve label’lar */
    label, .stCaption, p, span, div, .stMarkdown {
        color: var(--clr-text) !important;
    }
    /* Sidebar tam koruma */
    [data-testid="stSidebar"] * {
        color: var(--clr-text) !important;
    }
    /* Expander header ve content ekstra */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] div[role="region"] {
        background-color: var(--clr-surface) !important;
        color: var(--clr-text) !important;
    }
    .stApp { background: var(--clr-bg) !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--clr-surface) !important;
        border-right: 1px solid var(--clr-border) !important;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--clr-surface) !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--clr-border) !important;
        padding: 4px !important;
        gap: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--clr-muted) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--clr-primary2) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.25rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: var(--font-main) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        border-radius: 6px !important;
        transition: all 0.18s ease !important;
        border: 1px solid var(--clr-border) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--clr-primary) !important;
        color: #fff !important;
        border-color: var(--clr-primary) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--clr-primary2) !important;
        box-shadow: 0 0 12px rgba(47,129,247,0.4) !important;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] > div > div > div > div {
        background: var(--clr-primary) !important;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--clr-primary); }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--clr-muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: var(--font-mono);
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--clr-text);
        line-height: 1;
    }
    .metric-value.green  { color: var(--clr-green); }
    .metric-value.orange { color: var(--clr-orange); }
    .metric-value.red    { color: var(--clr-red); }
    .metric-value.blue   { color: var(--clr-primary); }

    /* ── Info / callout boxes ── */
    .info-box {
        background: rgba(47,129,247,0.08);
        border-left: 3px solid var(--clr-primary);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 0.75rem 1rem;
        font-size: 0.87rem;
        color: #aac8f5;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: rgba(227,179,65,0.08);
        border-left: 3px solid var(--clr-orange);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 0.75rem 1rem;
        font-size: 0.87rem;
        color: #e3c879;
        margin: 0.5rem 0;
    }
    .success-box {
        background: rgba(63,185,80,0.08);
        border-left: 3px solid var(--clr-green);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 0.75rem 1rem;
        font-size: 0.87rem;
        color: #82d996;
        margin: 0.5rem 0;
    }
    .error-box {
        background: rgba(248,81,73,0.1);
        border-left: 3px solid var(--clr-red);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 0.75rem 1rem;
        font-size: 0.87rem;
        color: #ff9492;
        margin: 0.5rem 0;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--clr-text);
        border-bottom: 1px solid var(--clr-border);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* ── Formula display ── */
    .formula-block {
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius);
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        font-family: var(--font-mono);
        font-size: 0.9rem;
    }

    /* ── Preset buttons ── */
    .preset-grid { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }

    /* ── Status badge ── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 2rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-green  { background: rgba(63,185,80,0.18);  color: #3fb950; }
    .badge-orange { background: rgba(227,179,65,0.18); color: #e3b341; }
    .badge-red    { background: rgba(248,81,73,0.18);  color: #f85149; }
    .badge-blue   { background: rgba(47,129,247,0.18); color: #2f81f7; }

    /* ── Page title hero ── */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--clr-muted);
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .hero-sub code {
        background: rgba(47,129,247,0.12);
        color: var(--clr-primary);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: 0.88em;
    }

    /* ── Comparison table ── */
    .cmp-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .cmp-table th {
        background: var(--clr-surface);
        color: var(--clr-muted);
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.6rem 1rem;
        border-bottom: 1px solid var(--clr-border);
        text-align: left;
    }
    .cmp-table td {
        padding: 0.55rem 1rem;
        border-bottom: 1px solid rgba(48,54,61,0.5);
        font-family: var(--font-mono);
        font-size: 0.88rem;
    }
    .cmp-table tr:last-child td { border-bottom: none; }
    .cmp-table .best { color: var(--clr-green); font-weight: 600; }
    .cmp-table .label-col { font-family: var(--font-main); color: var(--clr-muted); font-size: 0.83rem; }

    /* ── Progress bar styling ── */
    .stProgress > div > div { background: var(--clr-primary) !important; }

    /* ── Expander ── */
    details { border: 1px solid var(--clr-border) !important; border-radius: var(--radius) !important; }
    summary { padding: 0.6rem 1rem !important; font-weight: 500 !important; }

    /* ── Footer ── */
    .deepc-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid var(--clr-border);
        color: var(--clr-muted);
        font-size: 0.78rem;
        text-align: center;
        font-family: var(--font-mono);
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    font_family="IBM Plex Sans, sans-serif",
    font_size=12,
    paper_bgcolor="rgba(22,27,34,0.0)",
    plot_bgcolor="rgba(22,27,34,0.0)",
    margin=dict(t=48, b=24, l=8, r=8),
    legend=dict(
        bgcolor="rgba(22,27,34,0.85)",
        bordercolor="#30363d",
        borderwidth=1,
        font_size=11,
    ),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
)

COLORS = {
    "deepc":   "#2f81f7",   # blue
    "mpc":     "#3fb950",   # green
    "ref":     "#f85149",   # red
    "u_deepc": "#e3b341",   # amber
    "u_mpc":   "#a371f7",   # purple
    "data_y":  "#58a6ff",
    "data_u":  "#ff9a3c",
}


def plotly_defaults(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#21262d")
    return fig


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════

_defaults = dict(
    u_data       = None,
    y_data       = None,
    system_name  = "2. Derece: Kütle-Yay-Sönüm",
    deepc_result = None,
    mpc_result   = None,
    # sidebar params
    N_total   = 300,
    noise     = 0.05,
    T_ini     = 10,
    N_p       = 20,
    Q         = 10.0,
    R         = 0.5,
    lambda_g  = 10.0,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, color: str = "") -> str:
    cls = f"metric-value {color}".strip()
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>"""


def check_params(N_total: int, T_ini: int, N_p: int) -> Optional[str]:
    L   = T_ini + N_p
    col = N_total - L
    if col < 2 * L:
        return (
            f"⚠️ Hankel matrisi için yetersiz sütun ({col}). "
            f"N'yi artırın veya T_ini/N_p'yi azaltın."
        )
    return None


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.25rem;font-weight:700;letter-spacing:-0.02em;'
        'margin-bottom:0.1rem">🎛️ DeePC Playground</div>'
        '<div style="font-size:0.75rem;color:#8b949e;margin-bottom:1.25rem">'
        'v2.0 · Data-Enabled Predictive Control</div>',
        unsafe_allow_html=True,
    )

    # ── Preset Configurations ──────────────────────────────────
    st.markdown("**⚙️ Hızlı Ön Ayar**")
    preset_cols = st.columns(2)
    for i, (pname, pvals) in enumerate(PRESETS.items()):
        col = preset_cols[i % 2]
        if col.button(pname, key=f"preset_{i}", use_container_width=True):
            st.session_state.N_total  = pvals["N_total"]
            st.session_state.noise    = pvals["noise"]
            st.session_state.T_ini    = pvals["T_ini"]
            st.session_state.N_p      = pvals["N_p"]
            st.session_state.Q        = pvals["Q"]
            st.session_state.R        = pvals["R"]
            st.session_state.lambda_g = pvals["lambda_g"]
            st.session_state.u_data   = None   # force re-collection
            st.session_state.deepc_result = None
            st.session_state.mpc_result   = None
            st.toast(f"{pname} yüklendi — {pvals['info']}", icon="✅")
            st.rerun()

    st.divider()

    # ── System selection ───────────────────────────────────────
    st.markdown("**🔬 Sistem Seçimi**")
    sys_choice = st.selectbox(
        "Kontrol Sistemi",
        list(SYSTEMS.keys()),
        index=list(SYSTEMS.keys()).index(st.session_state.system_name),
        label_visibility="collapsed",
    )
    if sys_choice != st.session_state.system_name:
        st.session_state.system_name  = sys_choice
        st.session_state.u_data       = None
        st.session_state.deepc_result = None
        st.session_state.mpc_result   = None

    st.markdown(
        f'<div class="info-box">📍 {SYSTEMS[sys_choice]["desc"]}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Data parameters ────────────────────────────────────────
    with st.expander("📊 1. Veri Toplama", expanded=True):
        st.session_state.N_total = st.slider(
            "Veri Uzunluğu (N)", 100, 800,
            st.session_state.N_total, 50,
            help="Hankel matrisinin sütun sayısını belirler. Daha fazla veri = daha iyi sistem temsili.",
        )
        st.session_state.noise = st.slider(
            "Gürültü Seviyesi (σ)", 0.0, 0.4,
            st.session_state.noise, 0.01,
            help="Ölçüm gürültüsünün standart sapması. Gerçek sensör davranışını simüle eder.",
        )
        st.caption("$y_{ölçüm} = y_{gerçek} + \\mathcal{N}(0, \\sigma^2)$")

    # ── Controller parameters ──────────────────────────────────
    with st.expander("🎯 2. Kontrolcü Ayarları", expanded=True):
        st.session_state.T_ini = st.slider(
            "Geçmiş Ufku — $T_{ini}$", 3, 25,
            st.session_state.T_ini,
            help="Şimdiki durumu tahmin etmek için kullanılan geçmiş adım sayısı. Sistem derecesinden büyük olmalı.",
        )
        st.session_state.N_p = st.slider(
            "Öngörü Ufku — $N_p$", 5, 60,
            st.session_state.N_p,
            help="Kaç adım ilerisi optimize edilsin? Büyük değer daha akıllı ama hesaplama maliyeti artar.",
        )
        st.session_state.Q = st.slider(
            "Hata Ağırlığı — $Q$", 0.1, 60.0,
            st.session_state.Q, 0.5,
            help="$Q \\uparrow$ → daha agresif referans takibi, daha fazla kontrol hareketi",
        )
        st.session_state.R = st.slider(
            "Kontrol Ağırlığı — $R$", 0.0, 15.0,
            st.session_state.R, 0.1,
            help="$R \\uparrow$ → kontrol sinyali yumuşar, enerji tasarrufu sağlanır",
        )
        st.session_state.lambda_g = st.slider(
            "Düzenlileştirme — $\\lambda$", 0.1, 150.0,
            st.session_state.lambda_g, 0.5,
            help="DeePC'nin kalbi: gürültülü veride g vektörüne ceza uygular. Aşırı öğrenmeyi engeller.",
        )

    with st.expander("🔒 Kontrol Kısıtları"):
        u_min_val = st.number_input(
            "u_min", value=SYSTEMS[sys_choice]["u_min"], step=0.5
        )
        u_max_val = st.number_input(
            "u_max", value=SYSTEMS[sys_choice]["u_max"], step=0.5
        )

    with st.expander("⚡ Çözücü Ayarları"):
        solver_choice = st.selectbox("CVXPY Çözücü", ["OSQP", "ECOS", "SCS"], index=0,
                                     help="OSQP genellikle en hızlı seçenektir.")

    # Validity check
    warn = check_params(st.session_state.N_total, st.session_state.T_ini, st.session_state.N_p)
    if warn:
        st.markdown(f'<div class="warn-box">{warn}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div class="deepc-footer">DeePC Playground v2.0<br>'
        'Willems (2005) · Coulson et al. (2019)</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN PAGE HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<div class="hero-title">🎛️ DeePC Playground</div>'
    '<div class="hero-sub">'
    'Veriye Dayalı Öngörülü Kontrol — <code>Data-Enabled Predictive Control (DeePC)</code> · '
    'Coulson, Lygeros & Dörfler · ECC 2019'
    '</div>',
    unsafe_allow_html=True,
)

# Quick-look metrics row (populated after data/sim)
metrics_row = st.empty()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Veri Toplama",
    "🚀 2. DeePC Simülasyonu",
    "⚖️ 3. Karşılaştırma",
    "📖 4. Teori",
    "💾 5. Dışa Aktarma",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.markdown(
        '<div class="section-header">Adım 1 — Sistem Tanılama (Open-Loop Veri Toplama)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Herhangi bir modeli bilmeden kontrol edebilmek için sistemin **giriş–çıkış verilerine** ihtiyacımız var.
        PRBS (Pseudo-Random Binary Sequence) sinyali sistemi tüm frekans bileşenlerinde uyarır;
        bu, Willems'in Lemma'sının geçerliliği için zorunlu olan **"ısrarcı uyarım"** (persistently exciting)
        koşulunu sağlar.
        """,
        unsafe_allow_html=True,
    )

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        run_collect = st.button(
            "▶  Veriyi Topla & Hankel Matrisini Oluştur",
            type="primary", use_container_width=True,
        )
    with col_info:
        st.markdown(
            f'<div class="info-box">Seçili sistem: <strong>{st.session_state.system_name}</strong> &nbsp;|&nbsp; '
            f'N = {st.session_state.N_total} &nbsp;|&nbsp; σ = {st.session_state.noise}</div>',
            unsafe_allow_html=True,
        )

    if run_collect:
        with st.spinner("Sistem PRBS sinyaliyle uyarılıyor..."):
            u, y = simulate_system(
                st.session_state.N_total,
                st.session_state.noise,
                st.session_state.system_name,
            )
        st.session_state.u_data       = u
        st.session_state.y_data       = y
        st.session_state.deepc_result = None
        st.session_state.mpc_result   = None
        st.toast("Veri başarıyla toplandı!", icon="✅")

    if st.session_state.u_data is not None:
        u = st.session_state.u_data
        y = st.session_state.y_data

        # ── Summary metrics ────────────────────────────────────
        L   = st.session_state.T_ini + st.session_state.N_p
        Hu  = build_hankel(u, L)
        Hy  = build_hankel(y, L)
        H   = np.vstack([Hu, Hy])
        rnk = int(np.linalg.matrix_rank(H, tol=1e-8))
        pe_ok = rnk >= min(H.shape) * 0.8

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            metric_card("Veri Noktası", f"{len(u)}", "blue"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card("Hankel Satır", f"{H.shape[0]}", ""),
            unsafe_allow_html=True,
        )
        c3.markdown(
            metric_card("Hankel Sütun", f"{H.shape[1]}", ""),
            unsafe_allow_html=True,
        )
        c4.markdown(
            metric_card("Hankel Rank", f"{rnk}",
                        "green" if pe_ok else "red"),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if pe_ok:
            st.markdown(
                '<div class="success-box">✅ <strong>Yeterli Uyarım (Persistently Exciting):</strong> '
                'Hankel matrisinin rank analizi veri kalitesini onaylıyor. DeePC çalışmaya hazır.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-box">⚠️ <strong>Uyarım Yetersiz!</strong> '
                'Rank beklenenden düşük — N\'yi artırın veya gürültüyü azaltın.</div>',
                unsafe_allow_html=True,
            )

        # ── I/O Time Series ───────────────────────────────────
        fig_io = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.45], vertical_spacing=0.06,
            subplot_titles=("Sistem Çıkışı — y(k)", "PRBS Giriş Sinyali — u(k)"),
        )
        fig_io.add_trace(
            go.Scatter(
                y=y, mode="lines", name="y(k)",
                line=dict(color=COLORS["data_y"], width=1.4),
            ), row=1, col=1,
        )
        fig_io.add_trace(
            go.Scatter(
                y=u, mode="lines", name="u(k)",
                line=dict(color=COLORS["data_u"], width=1.0, shape="hv"),
            ), row=2, col=1,
        )
        plotly_defaults(fig_io, height=380)
        fig_io.update_layout(title_text="Açık Döngü Eğitim Verisi", showlegend=True)
        st.plotly_chart(fig_io, use_container_width=True)

        # ── Hankel Heatmap ─────────────────────────────────────
        with st.expander("🔥 Hankel Matrisi Isı Haritası (ilk 40×40 alt-blok)"):
            n_show = min(40, H.shape[0], H.shape[1])
            H_sub  = H[:n_show, :n_show]
            fig_hm = go.Figure(
                go.Heatmap(
                    z=H_sub, colorscale="RdBu_r", zmid=0,
                    colorbar=dict(len=0.8, thickness=12),
                )
            )
            fig_hm.update_layout(
                title=f"[U_Hankel ; Y_Hankel] — ilk {n_show}×{n_show} blok",
                height=380, **PLOTLY_LAYOUT,
                xaxis_title="Sütun indeksi", yaxis_title="Satır indeksi",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.markdown(
                '<div class="info-box">Her sütun, sistemin olası bir yörüngesini (trajectory) temsil eder. '
                'Renkli yapı, sistemin belirli dinamik kalıplarını yansıtır. '
                'Willems Lemma\'sı: Bu matrisin sütun uzayı, sistemin <em>tüm</em> gelecek yörüngelerini kapsar.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="warn-box">👆 Simülasyona başlamak için önce veri toplama butonuna basın.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — DeePC SIMULATION
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="section-header">Adım 2 — DeePC Kapalı Döngü Simülasyonu</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_run = st.columns([1, 2])

    with col_cfg:
        st.markdown("**Simülasyon Konfigürasyonu**")
        sim_steps  = st.number_input("Simülasyon Adımı", 20, 250, 80, 5)
        ref_type   = st.selectbox(
            "Referans Tipi",
            ["Sabit Adım", "Kare Dalga", "Sinüzoidal", "Ramp"],
            help="Kontrolcünün takip etmesini istediğiniz hedef sinyal tipi",
        )
        ref_value  = st.number_input(
            "Referans Değer (r)", value=SYSTEMS[st.session_state.system_name]["ref"], step=0.25,
        )

        run_deepc = st.button(
            "▶  DeePC Simülasyonunu Başlat",
            type="primary", use_container_width=True,
        )

    with col_run:
        if st.session_state.u_data is None:
            st.markdown(
                '<div class="warn-box">⚠️ Önce <strong>Tab 1</strong>\'den veri toplamanız gerekiyor.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="success-box">✅ Veri hazır — {len(st.session_state.u_data)} örnek | '
                f'Sistem: <strong>{st.session_state.system_name}</strong></div>',
                unsafe_allow_html=True,
            )

    if run_deepc:
        if st.session_state.u_data is None:
            st.error("Önce veri toplayın (Tab 1).")
        else:
            T_ini    = st.session_state.T_ini
            N_p      = st.session_state.N_p
            noise_lv = st.session_state.noise

            # Reference trajectory
            r_traj   = generate_reference(ref_type, ref_value, sim_steps)

            # Build solver
            with st.spinner("DeePC çözücüsü başlatılıyor (Hankel matrisleri oluşturuluyor)..."):
                try:
                    solver = DeePCSolver(
                        st.session_state.u_data,
                        st.session_state.y_data,
                        T_ini    = T_ini,
                        N_p      = N_p,
                        lambda_g = st.session_state.lambda_g,
                        Q        = st.session_state.Q,
                        R        = st.session_state.R,
                        u_min    = u_min_val,
                        u_max    = u_max_val,
                    )
                except ValueError as e:
                    st.markdown(
                        f'<div class="error-box">❌ Hankel oluşturma hatası: {e}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

            # ── Simulation loop ────────────────────────────────
            total = sim_steps + T_ini
            y_sim = np.zeros(total + 2)
            u_sim = np.zeros(total + 2)
            solve_times_d: list[float] = []
            cost_history:  list[float] = []
            failed = False

            prog_bar = st.progress(0, text="Optimizasyon çözülüyor...")
            status_placeholder = st.empty()

            for k in range(T_ini, sim_steps + T_ini):
                u_ini = u_sim[k - T_ini : k]
                y_ini = y_sim[k - T_ini : k]

                # Reference window for this step
                r_win_start = k - T_ini
                r_ref       = r_traj[min(r_win_start, sim_steps - 1) :]
                if len(r_ref) < N_p:
                    r_ref = np.pad(r_ref, (0, N_p - len(r_ref)), "edge")
                r_ref = r_ref[:N_p]

                u_opt, y_pred, cost_val, status, dt = solver.solve(
                    u_ini, y_ini, r_ref, solver_choice
                )

                if u_opt is None:
                    status_placeholder.markdown(
                        f'<div class="error-box">❌ Adım {k-T_ini}: Çözücü başarısız — {status}.<br>'
                        f'Öneri: λ artırın, N_p azaltın veya farklı çözücü deneyin.</div>',
                        unsafe_allow_html=True,
                    )
                    failed = True
                    break

                u_sim[k] = u_opt
                solve_times_d.append(dt)
                if cost_val is not None:
                    cost_history.append(cost_val)

                # Simulate true system (hidden model — only for playground)
                y_sim[k] = simulate_step(
                    y_sim[k-3:k], u_sim[k-3:k], u_opt,
                    st.session_state.system_name, noise_lv,
                )

                prog = (k - T_ini + 1) / sim_steps
                prog_bar.progress(prog, text=f"Adım {k-T_ini+1}/{sim_steps} — Çözüm: {dt:.1f} ms")

            prog_bar.empty()

            if not failed:
                status_placeholder.empty()
                metrics = compute_metrics(y_sim, u_sim, r_traj, T_ini, solve_times_d)
                st.session_state.deepc_result = {
                    "y_sim":        y_sim,
                    "u_sim":        u_sim,
                    "r_traj":       r_traj,
                    "sim_steps":    sim_steps,
                    "T_ini":        T_ini,
                    "ref_value":    ref_value,
                    "ref_type":     ref_type,
                    "metrics":      metrics,
                    "cost_history": cost_history,
                    "solve_times":  solve_times_d,
                    "system_name":  st.session_state.system_name,
                }
                st.toast("DeePC simülasyonu tamamlandı!", icon="🎯")

    # ── Results display ────────────────────────────────────────
    if st.session_state.deepc_result is not None:
        res   = st.session_state.deepc_result
        y_sim = res["y_sim"]
        u_sim = res["u_sim"]
        r_traj= res["r_traj"]
        steps = res["sim_steps"]
        T_ini = res["T_ini"]
        m     = res["metrics"]

        time_ax = np.arange(steps)

        # Metric cards
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (lbl, key, clr) in zip(
            [c1, c2, c3, c4, c5],
            [
                ("ISE",       "ISE",                  "blue"),
                ("IAE",       "IAE",                  ""),
                ("TV",        "TV (Kontrol Eforu)",   ""),
                ("Aşım (%)",  "Aşım (%)",             "green" if m["Aşım (%)"] < 10 else "orange"),
                ("Oturma",    "Oturma Adımı",         ""),
            ],
        ):
            col.markdown(
                metric_card(lbl, str(m.get(key, "—")), clr),
                unsafe_allow_html=True,
            )

        if "Ort. Çözüm (ms)" in m:
            st.caption(f"⚡ Ortalama çözüm süresi: **{m['Ort. Çözüm (ms)']} ms** / adım")

        # Main simulation plot
        fig_sim = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.45, 0.30, 0.25],
            subplot_titles=(
                "Sistem Yanıtı — y(k) vs Referans r(k)",
                "Kontrol Sinyali — u(k)",
                "Optimizasyon Maliyeti — J(k)",
            ),
        )
        fig_sim.add_trace(
            go.Scatter(x=time_ax, y=r_traj, name="Referans r(k)",
                       line=dict(color=COLORS["ref"], dash="dash", width=1.5)),
            row=1, col=1,
        )
        fig_sim.add_trace(
            go.Scatter(x=time_ax, y=y_sim[T_ini:T_ini+steps], name="DeePC y(k)",
                       line=dict(color=COLORS["deepc"], width=2.2)),
            row=1, col=1,
        )
        fig_sim.add_trace(
            go.Scatter(x=time_ax, y=u_sim[T_ini:T_ini+steps], name="u(k)",
                       line=dict(color=COLORS["u_deepc"], width=1.8, shape="hv"),
                       fill="tozeroy", fillcolor="rgba(227,179,65,0.07)"),
            row=2, col=1,
        )
        # Add control bounds
        fig_sim.add_hline(y=u_max_val, line=dict(color="#f85149", dash="dot", width=1),
                          annotation_text="u_max", row=2, col=1)
        fig_sim.add_hline(y=u_min_val, line=dict(color="#f85149", dash="dot", width=1),
                          annotation_text="u_min", row=2, col=1)
        if res["cost_history"]:
            fig_sim.add_trace(
                go.Scatter(
                    x=np.arange(len(res["cost_history"])),
                    y=res["cost_history"],
                    name="Cost J",
                    line=dict(color="#a371f7", width=1.5),
                ),
                row=3, col=1,
            )
        plotly_defaults(fig_sim, height=560)
        fig_sim.update_yaxes(title_text="y(k)", row=1, col=1)
        fig_sim.update_yaxes(title_text="u(k)", row=2, col=1)
        fig_sim.update_yaxes(title_text="J",    row=3, col=1)
        fig_sim.update_xaxes(title_text="Zaman Adımı k", row=3, col=1)
        st.plotly_chart(fig_sim, use_container_width=True)

        # Auto interpretation
        interp = auto_interpret(m)
        st.markdown(
            f'<div style="background:var(--clr-surface);border:1px solid var(--clr-border);'
            f'border-radius:8px;padding:1rem 1.25rem;margin-top:0.5rem">'
            f'<strong style="font-size:0.85rem;color:#8b949e;text-transform:uppercase;'
            f'letter-spacing:0.05em">🤖 Otomatik Yorum</strong><br><br>{interp}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — COMPARISON
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        '<div class="section-header">DeePC vs Klasik MPC — Yan Yana Karşılaştırma</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        'Aynı sistem, aynı referans, aynı gürültü realizasyonu üzerinde iki yaklaşım karşılaştırılır.<br>'
        '• <strong>DeePC:</strong> Model bilgisi <em>yok</em> — sadece geçmiş I/O verisi<br>'
        '• <strong>Klasik MPC:</strong> Tam (A,B,C,D) model bilgisi mevcut — gerçek hayatta elde edilmesi zordur'
        '</div>',
        unsafe_allow_html=True,
    )

    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        cmp_steps  = st.number_input("Simülasyon Adımı", 20, 200, 80, 5, key="cmp_steps")
        cmp_ref    = st.number_input(
            "Referans", value=SYSTEMS[st.session_state.system_name]["ref"],
            step=0.25, key="cmp_ref",
        )
        cmp_rtype  = st.selectbox("Referans Tipi", ["Sabit Adım", "Kare Dalga", "Sinüzoidal"],
                                  key="cmp_rtype")
        run_cmp    = st.button("▶  İkisini Birlikte Çalıştır", type="primary",
                               use_container_width=True)
        if st.session_state.u_data is None:
            st.markdown(
                '<div class="warn-box">Önce Tab 1\'den veri toplayın.</div>',
                unsafe_allow_html=True,
            )

    if run_cmp:
        if st.session_state.u_data is None:
            st.error("Önce veri toplayın!")
        else:
            T_ini   = st.session_state.T_ini
            N_p     = st.session_state.N_p
            noise_lv= st.session_state.noise
            r_traj  = generate_reference(cmp_rtype, cmp_ref, cmp_steps)
            total   = cmp_steps + T_ini

            # Noise seed for fair comparison
            noise_seed = np.random.randint(0, 99999)

            # ── Build DeePC solver ─────────────────────────────
            try:
                deepc_solver = DeePCSolver(
                    st.session_state.u_data,
                    st.session_state.y_data,
                    T_ini=T_ini, N_p=N_p,
                    lambda_g=st.session_state.lambda_g,
                    Q=st.session_state.Q, R=st.session_state.R,
                    u_min=u_min_val, u_max=u_max_val,
                )
            except ValueError as e:
                st.error(f"DeePC Hankel hatası: {e}")
                st.stop()

            # ── Build Classic MPC solver ───────────────────────
            A, B, C, D = get_ss_matrices(st.session_state.system_name)
            mpc_solver = ClassicMPCSolver(
                A, B, C, N_p=N_p,
                Q=st.session_state.Q, R=st.session_state.R,
                u_min=u_min_val, u_max=u_max_val,
            )

            # ── Run both simulators with same noise ────────────
            y_d  = np.zeros(total + 2)
            u_d  = np.zeros(total + 2)
            y_m  = np.zeros(total + 2)
            u_m  = np.zeros(total + 2)
            x_ss = np.zeros(A.shape[0])   # true state for Classic MPC

            times_d: list[float] = []
            times_m: list[float] = []

            prog_cmp = st.progress(0, "Karşılaştırma çalışıyor...")

            np.random.seed(noise_seed)
            noise_seq = np.random.normal(0, noise_lv, total + 2) if noise_lv > 0 else np.zeros(total + 2)

            fail_d = fail_m = False
            for k in range(T_ini, cmp_steps + T_ini):
                r_win = r_traj[max(0, k - T_ini) : max(0, k - T_ini) + N_p]
                if len(r_win) < N_p:
                    r_win = np.pad(r_win, (0, N_p - len(r_win)), "edge")

                # DeePC step
                if not fail_d:
                    u_opt_d, _, _, stat_d, dt_d = deepc_solver.solve(
                        u_d[k-T_ini:k], y_d[k-T_ini:k], r_win, solver_choice
                    )
                    if u_opt_d is None:
                        fail_d = True; u_opt_d = 0.0
                    times_d.append(dt_d)
                    u_d[k] = u_opt_d
                    y_d[k] = simulate_step(y_d[k-3:k], u_d[k-3:k], u_opt_d,
                                           st.session_state.system_name) + noise_seq[k]

                # Classic MPC step
                if not fail_m:
                    u_opt_m, _, stat_m, dt_m = mpc_solver.solve(x_ss, r_win, solver_choice)
                    if u_opt_m is None:
                        fail_m = True; u_opt_m = 0.0
                    times_m.append(dt_m)
                    u_m[k] = u_opt_m
                    x_ss   = A @ x_ss + B.flatten() * u_opt_m
                    y_m[k] = (C @ x_ss).item() + noise_seq[k]

                prog_cmp.progress((k - T_ini + 1) / cmp_steps)

            prog_cmp.empty()

            metrics_d = compute_metrics(y_d, u_d, r_traj, T_ini, times_d)
            metrics_m = compute_metrics(y_m, u_m, r_traj, T_ini, times_m)

            st.session_state.mpc_result   = {
                "y_d": y_d, "u_d": u_d,
                "y_m": y_m, "u_m": u_m,
                "r_traj": r_traj, "sim_steps": cmp_steps, "T_ini": T_ini,
                "metrics_d": metrics_d, "metrics_m": metrics_m,
                "fail_d": fail_d, "fail_m": fail_m,
            }
            st.toast("Karşılaştırma tamamlandı!", icon="⚖️")

    if st.session_state.mpc_result is not None:
        res  = st.session_state.mpc_result
        y_d, u_d = res["y_d"], res["u_d"]
        y_m, u_m = res["y_m"], res["u_m"]
        r_t  = res["r_traj"]
        s    = res["sim_steps"]
        Ti   = res["T_ini"]
        md   = res["metrics_d"]
        mm   = res["metrics_m"]
        tax  = np.arange(s)

        if res["fail_d"]:
            st.markdown(
                '<div class="warn-box">⚠️ DeePC bazı adımlarda çözüm bulamadı. λ artırın veya N azaltın.</div>',
                unsafe_allow_html=True,
            )
        if res["fail_m"]:
            st.markdown(
                '<div class="warn-box">⚠️ Klasik MPC bazı adımlarda çözüm bulamadı.</div>',
                unsafe_allow_html=True,
            )

        # ── Comparison chart ───────────────────────────────────
        fig_cmp = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=("Sistem Yanıtı Karşılaştırması", "Kontrol Sinyali Karşılaştırması"),
        )
        fig_cmp.add_trace(
            go.Scatter(x=tax, y=r_t, name="Referans",
                       line=dict(color=COLORS["ref"], dash="dash", width=1.5)),
            row=1, col=1,
        )
        fig_cmp.add_trace(
            go.Scatter(x=tax, y=y_d[Ti:Ti+s], name="DeePC (model-free)",
                       line=dict(color=COLORS["deepc"], width=2.3)),
            row=1, col=1,
        )
        fig_cmp.add_trace(
            go.Scatter(x=tax, y=y_m[Ti:Ti+s], name="Klasik MPC (model-based)",
                       line=dict(color=COLORS["mpc"], width=2.3, dash="dot")),
            row=1, col=1,
        )
        fig_cmp.add_trace(
            go.Scatter(x=tax, y=u_d[Ti:Ti+s], name="u — DeePC",
                       line=dict(color=COLORS["u_deepc"], width=1.6, shape="hv")),
            row=2, col=1,
        )
        fig_cmp.add_trace(
            go.Scatter(x=tax, y=u_m[Ti:Ti+s], name="u — MPC",
                       line=dict(color=COLORS["u_mpc"], width=1.6, shape="hv", dash="dot")),
            row=2, col=1,
        )
        plotly_defaults(fig_cmp, height=520)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Metrics comparison table ───────────────────────────
        st.markdown(
            '<div class="section-header" style="font-size:1rem;margin-top:0.5rem">Performans Tablosu</div>',
            unsafe_allow_html=True,
        )

        def _best(a, b, lower_better=True):
            """Return 'best' for the better value."""
            if lower_better:
                return ("best", "") if a <= b else ("", "best")
            return ("best", "") if a >= b else ("", "best")

        rows_html = ""
        for key in ["ISE", "IAE", "TV (Kontrol Eforu)", "Aşım (%)", "Oturma Adımı", "Ort. Çözüm (ms)"]:
            vd = md.get(key)
            vm = mm.get(key)
            if vd is None and vm is None:
                continue
            vd_s = f"{vd:.3f}" if isinstance(vd, float) else str(vd) if vd is not None else "—"
            vm_s = f"{vm:.3f}" if isinstance(vm, float) else str(vm) if vm is not None else "—"
            if vd is not None and vm is not None:
                cd, cm = _best(vd, vm, lower_better=True)
            else:
                cd = cm = ""
            rows_html += (
                f'<tr><td class="label-col">{key}</td>'
                f'<td class="{cd}">{vd_s}</td>'
                f'<td class="{cm}">{vm_s}</td></tr>'
            )

        table_html = f"""
        <table class="cmp-table">
          <thead>
            <tr>
              <th>Metrik</th>
              <th>🔵 DeePC <em style="font-weight:400;font-size:0.78rem">(model-free)</em></th>
              <th>🟢 Klasik MPC <em style="font-weight:400;font-size:0.78rem">(model-based)</em></th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("🟩 Yeşil = daha iyi değer (alt satır sayısal iyilik için)")

        # Auto interpretation
        interp_cmp = auto_interpret(md, mm)
        st.markdown(
            f'<div style="background:var(--clr-surface);border:1px solid var(--clr-border);'
            f'border-radius:8px;padding:1rem 1.25rem;margin-top:1rem">'
            f'<strong style="font-size:0.85rem;color:#8b949e;text-transform:uppercase;'
            f'letter-spacing:0.05em">🤖 Karşılaştırma Yorumu</strong><br><br>{interp_cmp}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — THEORY
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        '<div class="section-header">Teorik Arka Plan & Matematiksel Temel</div>',
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        with st.expander("📐 Willems'in Temel Lemma'sı (2005)", expanded=True):
            st.markdown(
                """
                **Temel Soru:** Bir sistemin tüm olası davranışlarını matematiksel model olmadan
                nasıl temsil edebiliriz?

                **Cevap (Willems et al., 2005):**
                Doğrusal ve zamanla değişmeyen (LTI) bir sistemin giriş sinyali
                **ısrarcı biçimde uyarıcı** (*persistently exciting*) olacak şekilde seçilirse,
                o sistemin **tüm** olası $L$ uzunluğundaki girdi-çıktı dizileri,
                yalnızca geçmiş veriden oluşturulan Hankel matrisinin
                **sütun uzayında** ifade edilebilir.
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"""
            \mathcal{H}_L(w) = \begin{bmatrix}
            w(1) & w(2) & \cdots & w(N-L+1) \\
            w(2) & w(3) & \cdots & w(N-L+2) \\
            \vdots & \vdots & \ddots & \vdots \\
            w(L) & w(L+1) & \cdots & w(N)
            \end{bmatrix}
            """)
            st.markdown(
                """
                **Israrcı Uyarım Koşulu:** $u$ dizisi en az $n + L$ derecesinde ısrarcı biçimde
                uyarıcıysa (rank tam ise), Hankel matrisi sistemin tüm yörüngelerini temsil eder.
                """,
            )

        with st.expander("🎯 DeePC Optimizasyon Problemi"):
            st.markdown("Her zaman adımında çözülen karesel program (QP):")
            st.latex(r"""
            \min_{g,\, u_f,\, y_f} \;\;
            \underbrace{Q \|y_f - r\|^2}_{\text{takip hatası}}
            + \underbrace{R \|u_f\|^2}_{\text{kontrol eforu}}
            + \underbrace{\lambda \|g\|^2}_{\text{regülarizasyon}}
            """)
            st.markdown("**Kısıtlar (Willems'in Lemma'sı):**")
            st.latex(r"""
            \begin{bmatrix} U_p \\ Y_p \\ U_f \\ Y_f \end{bmatrix} g
            = \begin{bmatrix} u_{ini} \\ y_{ini} \\ u_f \\ y_f \end{bmatrix},
            \quad u_{min} \leq u_f \leq u_{max}
            """)
            st.markdown(
                r"""
                | Sembol | Açıklama |
                |--------|----------|
                | $g \in \mathbb{R}^{N-L+1}$ | Hankel sütunlarının kombinasyon katsayıları |
                | $U_p,\, Y_p$ | Geçmiş giriş/çıkış Hankel bloğu ($T_{ini}$ satır) |
                | $U_f,\, Y_f$ | Gelecek giriş/çıkış Hankel bloğu ($N_p$ satır) |
                | $u_{ini},\, y_{ini}$ | Son $T_{ini}$ adımın gözlemlenen geçmişi |
                | $\lambda$ | Gürültüye karşı düzenlileştirme katsayısı |
                """
            )

        with st.expander("🔄 Geri Çekilme Ufku (Receding Horizon)"):
            st.markdown(
                """
                DeePC, **Model Predictive Control (MPC)** mimarisiyle çalışır:

                1. Her $k$ adımında $N_p$ adım ilerisi optimize edilir
                2. Optimum kontrol dizisinin **yalnızca ilk elemanı** $u^*(0)$ uygulanır
                3. Bir adım ilerlenir, geçmiş penceresi kaydırılır
                4. Problem baştan çözülür (closed-loop feedback)

                Bu yapı sistemi bozuculara ve model hatalarına karşı **sağlam** kılar.
                """,
            )

    with col_t2:
        st.markdown(
            '<div class="formula-block" style="font-size:0.82rem;line-height:1.7">'
            '<strong style="color:#2f81f7">Algoritma Adımları</strong><br><br>'
            '<span style="color:#8b949e">① Açık Döngü:</span><br>'
            '&nbsp;&nbsp;PRBS sinyali ile sistemi uyar<br>'
            '&nbsp;&nbsp;(u, y) kayıtlarını topla<br><br>'
            '<span style="color:#8b949e">② Hankel:</span><br>'
            '&nbsp;&nbsp;T_ini + N_p derinliğinde<br>'
            '&nbsp;&nbsp;U_p, U_f, Y_p, Y_f oluştur<br><br>'
            '<span style="color:#8b949e">③ Kapalı Döngü (her k için):</span><br>'
            '&nbsp;&nbsp;u_ini, y_ini al<br>'
            '&nbsp;&nbsp;QP problemini çöz<br>'
            '&nbsp;&nbsp;u*(0) uygula → y yeni ölç<br>'
            '&nbsp;&nbsp;Pencereyi kaydır, tekrarla</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            **Neden DeePC önemli?**

            | | Model MPC | DeePC |
            |--|-----------|-------|
            | Model gerekir mi? | ✅ Evet | ❌ Hayır |
            | Hankel matrisi | ❌ | ✅ |
            | Gürültüye dayanıklılık | Orta | λ ile ayarlanabilir |
            | Hesaplama | Orta | QP boyutuna bağlı |
            | Doğrusal olmayan sistemler | Kısıtlı | Kısıtlı |
            """,
        )

        with st.expander("📚 Kaynakça"):
            st.markdown(
                """
                1. **Willems et al.** (2005). *A note on persistency of excitation.*
                   Systems & Control Letters, 54(4).

                2. **Coulson, Lygeros & Dörfler** (2019). *Data-enabled predictive control.*
                   European Control Conference (ECC).

                3. **Markovsky & Dörfler** (2021). *Behavioral systems theory in data-driven
                   analysis, signal processing, and control.*
                   Annual Reviews in Control, 52.

                4. **De Persis & Tesi** (2020). *Formulas for data-driven control.*
                   IEEE Transactions on Automatic Control.
                """,
            )


# ═══════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ═══════════════════════════════════════════════════════════════

with tab5:
    st.markdown(
        '<div class="section-header">💾 Sonuçları Dışa Aktar</div>',
        unsafe_allow_html=True,
    )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("**DeePC Simülasyon Sonuçları**")
        if st.session_state.deepc_result is not None:
            res  = st.session_state.deepc_result
            s    = res["sim_steps"]
            Ti   = res["T_ini"]
            tax  = np.arange(s)

            df_sim = pd.DataFrame({
                "step":       tax,
                "y_deepc":    res["y_sim"][Ti:Ti+s],
                "u_deepc":    res["u_sim"][Ti:Ti+s],
                "reference":  res["r_traj"][:s],
            })

            st.download_button(
                "⬇️ DeePC Sonuçları (CSV)",
                to_csv_bytes(df_sim),
                file_name="deepc_simulation.csv",
                mime="text/csv",
                use_container_width=True,
            )

            df_metrics = pd.DataFrame([res["metrics"]])
            st.download_button(
                "⬇️ DeePC Metrikleri (CSV)",
                to_csv_bytes(df_metrics),
                file_name="deepc_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="warn-box">DeePC simülasyonu henüz çalıştırılmadı (Tab 2).</div>',
                unsafe_allow_html=True,
            )

    with col_e2:
        st.markdown("**Karşılaştırma Sonuçları**")
        if st.session_state.mpc_result is not None:
            res2 = st.session_state.mpc_result
            s2   = res2["sim_steps"]
            Ti2  = res2["T_ini"]
            tax2 = np.arange(s2)

            df_cmp = pd.DataFrame({
                "step":       tax2,
                "y_deepc":    res2["y_d"][Ti2:Ti2+s2],
                "u_deepc":    res2["u_d"][Ti2:Ti2+s2],
                "y_mpc":      res2["y_m"][Ti2:Ti2+s2],
                "u_mpc":      res2["u_m"][Ti2:Ti2+s2],
                "reference":  res2["r_traj"][:s2],
            })
            st.download_button(
                "⬇️ Karşılaştırma Verileri (CSV)",
                to_csv_bytes(df_cmp),
                file_name="comparison_deepc_vs_mpc.csv",
                mime="text/csv",
                use_container_width=True,
            )

            df_met_cmp = pd.DataFrame([
                {"Method": "DeePC", **res2["metrics_d"]},
                {"Method": "Classic MPC", **res2["metrics_m"]},
            ])
            st.download_button(
                "⬇️ Karşılaştırma Metrikleri (CSV)",
                to_csv_bytes(df_met_cmp),
                file_name="metrics_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="warn-box">Karşılaştırma simülasyonu henüz çalıştırılmadı (Tab 3).</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Raw training data export ───────────────────────────────
    st.markdown("**Eğitim Verisi (Hankel Girişi)**")
    if st.session_state.u_data is not None:
        df_raw = pd.DataFrame({
            "step":  np.arange(len(st.session_state.u_data)),
            "u_prbs": st.session_state.u_data,
            "y_noisy": st.session_state.y_data,
        })
        st.download_button(
            "⬇️ Ham I/O Verisi (CSV)",
            to_csv_bytes(df_raw),
            file_name="training_data.csv",
            mime="text/csv",
        )
    else:
        st.markdown(
            '<div class="warn-box">Veri toplanmadı (Tab 1).</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── System info summary ────────────────────────────────────
    st.markdown("**Oturum Özeti**")
    p = SYSTEMS[st.session_state.system_name]
    summary_data = {
        "Parametre": [
            "Sistem", "Transfer Fonk. Pay", "Transfer Fonk. Payda",
            "Veri Uzunluğu (N)", "Gürültü (σ)",
            "T_ini", "N_p", "Q", "R", "λ", "u_min", "u_max",
        ],
        "Değer": [
            st.session_state.system_name,
            str(p["num"]), str(p["den"]),
            st.session_state.N_total, st.session_state.noise,
            st.session_state.T_ini, st.session_state.N_p,
            st.session_state.Q, st.session_state.R, st.session_state.lambda_g,
            u_min_val, u_max_val,
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Parametre Özeti (CSV)",
        to_csv_bytes(pd.DataFrame(summary_data)),
        file_name="session_parameters.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════
# GLOBAL FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="deepc-footer">
    DeePC Playground v2.0 &nbsp;·&nbsp;
    Willems (2005) · Coulson, Lygeros &amp; Dörfler (2019) &nbsp;·&nbsp;
    Açık Kaynak Referans Aracı
    </div>
    """,
    unsafe_allow_html=True,
)
