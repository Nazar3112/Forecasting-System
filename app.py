"""
app.py
======
StreamlitDashboard – Pengendali Utama Antarmuka Pengguna
Sistem Prediksi Tren Penjualan Retail Berbasis Kategori
Model Hibrida Prophet + LightGBM (PT. Indomarco Prismatama)

Desain: Airtable Design System (DESIGN-airtable.md)
  – White canvas, dark-ink editorial, signature surface cards
  – Inter Display typography (pengganti Haas Grotesk)
  – Near-black primary CTA (#181d26), hairline secondary
  – Generous whitespace (96px section rhythm)
  – Stateless In-Memory, tanpa DB, tanpa login
"""

import logging

import pandas as pd
import streamlit as st

from src import (
    APP_TITLE, APP_SUBTITLE, APP_ICON,
    FREQ_OPTIONS, SEASONALITY_MODES,
    DataPreprocessor,
    DataFilterEngine,
    HybridForecastingEngine,
    ModelEvaluator,
    DashboardVisualizer,
)
from src.config import (
    DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON,
    MIN_DATA_POINTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi Halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecast · Indomarco",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# AIRTABLE DESIGN SYSTEM — CSS Injection
# Referensi: DESIGN-airtable.md
# ---------------------------------------------------------------------------
AIRTABLE_CSS = """
<style>
/* ─── Google Font: Inter Display ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..700;1,14..32,300..700&display=swap');

/* ─── Design Tokens ──────────────────────────────────────────────────────── */
:root {
  --ink:              #181d26;
  --body:             #333840;
  --muted:            #41454d;
  --hairline:         #dddddd;
  --canvas:           #ffffff;
  --surface-soft:     #f8fafc;
  --surface-strong:   #e0e2e6;
  --surface-dark:     #181d26;
  --sig-coral:        #aa2d00;
  --sig-forest:       #0a2e0e;
  --sig-cream:        #f5e9d4;
  --sig-peach:        #fcab79;
  --sig-mint:         #a8d8c4;
  --sig-yellow:       #f4d35e;
  --link:             #1b61c9;
  --success:          #006400;
  --r-sm:             6px;
  --r-md:             10px;
  --r-lg:             12px;
}

/* ─── Global Reset ───────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  background-color: var(--canvas) !important;
  color: var(--body) !important;
  -webkit-font-smoothing: antialiased;
}

/* ─── Main Content Area ──────────────────────────────────────────────────── */
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1280px !important;
}

/* ─── Headings (Airtable: weight 400–500, never bold for its own sake) ───── */
h1 {
  font-size: 32px !important;
  font-weight: 500 !important;
  line-height: 1.2 !important;
  color: var(--ink) !important;
  letter-spacing: 0 !important;
  margin-bottom: 4px !important;
}
h2 {
  font-size: 24px !important;
  font-weight: 400 !important;
  line-height: 1.35 !important;
  color: var(--ink) !important;
  letter-spacing: 0.12px !important;
  margin-top: 2rem !important;
}
h3 {
  font-size: 18px !important;
  font-weight: 500 !important;
  line-height: 1.4 !important;
  color: var(--ink) !important;
  margin-top: 1.5rem !important;
}
p, li {
  font-size: 14px !important;
  font-weight: 400 !important;
  line-height: 1.6 !important;
  color: var(--body) !important;
}
small, .caption, [data-testid="stCaptionContainer"] {
  font-size: 13px !important;
  color: var(--muted) !important;
}

/* ─── Horizontal Rule ────────────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--hairline) !important;
  margin: 2rem 0 !important;
}

/* ─── SIDEBAR ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: var(--surface-soft) !important;
  border-right: 1px solid var(--hairline) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 1.5rem 1.25rem 2rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  margin-top: 1.5rem !important;
  margin-bottom: 0.5rem !important;
}
/* Sidebar brand mark */
[data-testid="stSidebar"] .sidebar-brand {
  font-size: 15px !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  margin-bottom: 0.25rem !important;
}

/* ─── PRIMARY BUTTON: near-black pill (Airtable signature) ──────────────── */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
  background-color: var(--ink) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--r-lg) !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  padding: 14px 24px !important;
  letter-spacing: 0 !important;
  box-shadow: 0 1px 3px rgba(24,29,38,0.18), 0 0 0 0px transparent !important;
  transition: background 0.15s ease, box-shadow 0.15s ease !important;
  line-height: 1.4 !important;
}
.stButton > button[kind="primary"]:active,
.stDownloadButton > button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active {
  background-color: #0d1218 !important;
}
.stButton > button[kind="primary"]:disabled,
button[data-testid="baseButton-primary"]:disabled {
  background-color: var(--surface-strong) !important;
  color: var(--muted) !important;
}

/* ─── SECONDARY BUTTON: white canvas + hairline ──────────────────────────── */
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
  background-color: var(--canvas) !important;
  color: var(--ink) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-lg) !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  padding: 14px 24px !important;
}
.stButton > button[kind="secondary"]:active {
  border-color: var(--ink) !important;
}

/* ─── TABS — Airtable editorial nav ─────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--hairline) !important;
  background: transparent !important;
  gap: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
  font-size: 14px !important;
  font-weight: 400 !important;
  color: var(--muted) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 20px !important;
  background: transparent !important;
  border-radius: 0 !important;
  margin-bottom: -1px !important;
  transition: color 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  font-weight: 500 !important;
  color: var(--ink) !important;
  border-bottom: 2px solid var(--ink) !important;
}
[data-testid="stTabs"] [data-testid="stTabPanel"] {
  padding-top: 1.75rem !important;
}

/* ─── METRIC CARDS — demo-grid-card style ────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--canvas) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
  padding: 20px 20px 16px !important;
}
[data-testid="stMetricLabel"] {
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 26px !important;
  font-weight: 400 !important;
  color: var(--ink) !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
  font-size: 13px !important;
  font-weight: 400 !important;
}

/* ─── SELECTBOX & RADIO ───────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] label span {
  font-size: 14px !important;
  color: var(--body) !important;
}
[data-testid="stSelectbox"] > div > div:first-child {
  background: var(--canvas) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-sm) !important;
  padding: 10px 14px !important;
  font-size: 14px !important;
  min-height: 44px !important;
}
[data-testid="stSelectbox"] > div > div:first-child:focus-within {
  border-color: #458fff !important;
  box-shadow: 0 0 0 2px rgba(69,143,255,0.25) !important;
}

/* ─── FILE UPLOADER ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--hairline) !important;
  border-radius: var(--r-md) !important;
  background: var(--surface-soft) !important;
  padding: 1rem !important;
}
[data-testid="stFileUploader"] button {
  background: var(--canvas) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-sm) !important;
  color: var(--ink) !important;
  font-size: 13px !important;
  padding: 8px 16px !important;
}

/* ─── DATAFRAME / TABLE ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] iframe,
[data-testid="dataframe"] {
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
}

/* ─── SLIDER ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
  background: var(--ink) !important;
}
[data-testid="stSlider"] [role="slider"] {
  color: var(--ink) !important;
}

/* ─── ALERTS: st.success / st.warning / st.error / st.info ─────────────── */
[data-testid="stAlert"] {
  border-radius: var(--r-md) !important;
  border-left-width: 3px !important;
  font-size: 14px !important;
}
[data-testid="stAlert"][kind="info"],
div[data-baseweb="notification"][role="alert"]:has(.st-emotion-cache-1wbqy5l) {
  background: #f0f5ff !important;
  border-left-color: #254fad !important;
}
[data-testid="stAlert"][kind="success"] {
  background: #f0faf0 !important;
  border-left-color: var(--success) !important;
}
[data-testid="stAlert"][kind="warning"] {
  background: #fffbeb !important;
  border-left-color: #d97706 !important;
}
[data-testid="stAlert"][kind="error"] {
  background: #fff5f5 !important;
  border-left-color: #cc2200 !important;
}

/* ─── SPINNER ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
  border-top-color: var(--ink) !important;
}

/* ─── SIGNATURE CARD UTILITY CLASSES (injected via st.markdown) ─────────── */

/* Hero band — white editorial top section */
.at-hero {
  background: var(--canvas);
  padding: 2rem 0 1.5rem;
  border-bottom: 1px solid var(--hairline);
  margin-bottom: 2rem;
}
.at-hero-title {
  font-size: 28px;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.2;
  margin: 0 0 4px;
}
.at-hero-sub {
  font-size: 14px;
  font-weight: 400;
  color: var(--muted);
  margin: 0;
}
.at-hero-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: var(--surface-strong);
  color: var(--muted);
  border-radius: 9999px;
  padding: 3px 10px;
  margin-bottom: 12px;
}

/* Signature Coral card */
.at-coral {
  background: var(--sig-coral);
  color: #ffffff;
  border-radius: var(--r-lg);
  padding: 32px 36px;
  margin: 1.5rem 0;
}
.at-coral h3, .at-coral p { color: #ffffff !important; }
.at-coral h3 { font-size: 20px; font-weight: 500; margin: 0 0 6px; }
.at-coral p  { font-size: 13px; opacity: 0.85; margin: 0; }

/* Signature Forest card */
.at-forest {
  background: var(--sig-forest);
  color: #ffffff;
  border-radius: var(--r-lg);
  padding: 32px 36px;
  margin: 1.5rem 0;
}
.at-forest h3, .at-forest p { color: #ffffff !important; }
.at-forest h3 { font-size: 20px; font-weight: 500; margin: 0 0 6px; }
.at-forest p  { font-size: 13px; opacity: 0.85; margin: 0; }

/* Signature Dark card */
.at-dark {
  background: var(--surface-dark);
  color: #ffffff;
  border-radius: var(--r-lg);
  padding: 32px 36px;
  margin: 1.5rem 0;
}
.at-dark h3, .at-dark p { color: #ffffff !important; }
.at-dark h3 { font-size: 20px; font-weight: 500; margin: 0 0 6px; }
.at-dark p  { font-size: 13px; opacity: 0.80; margin: 0; }

/* Cream callout */
.at-cream {
  background: var(--sig-cream);
  border-radius: var(--r-md);
  padding: 24px 28px;
  margin: 1rem 0;
}
.at-cream h4 { font-size: 15px; font-weight: 500; color: var(--ink); margin: 0 0 4px; }
.at-cream p  { font-size: 13px; color: var(--body); margin: 0; }

/* Demo-grid cards (peach / mint) */
.at-stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 1rem 0;
}
.at-stat-card {
  background: var(--canvas);
  border: 1px solid var(--hairline);
  border-radius: var(--r-md);
  padding: 18px 20px 14px;
}
.at-stat-card .label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}
.at-stat-card .value {
  font-size: 22px;
  font-weight: 400;
  color: var(--ink);
  line-height: 1.1;
}
.at-stat-card .delta {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}
.at-stat-card.peach  { background: #fff4ec; border-color: #fcd5b3; }
.at-stat-card.mint   { background: #edf7f3; border-color: #b8e4d4; }
.at-stat-card.cream  { background: var(--sig-cream); border-color: #e8d5b3; }
.at-stat-card.yellow { background: #fffbea; border-color: #f4e08a; }

/* CTA Band Light */
.at-cta-band {
  background: var(--surface-strong);
  border-radius: var(--r-lg);
  padding: 32px 36px;
  margin: 2rem 0 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.at-cta-band h3 {
  font-size: 18px;
  font-weight: 500;
  color: var(--ink);
  margin: 0 0 4px;
}
.at-cta-band p {
  font-size: 13px;
  color: var(--muted);
  margin: 0;
}

/* Step badge (welcome guide) */
.at-step {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 14px 0;
  border-bottom: 1px solid var(--hairline);
}
.at-step:last-child { border-bottom: none; }
.at-step-num {
  flex-shrink: 0;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--ink);
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
}
.at-step-body { flex: 1; }
.at-step-body strong { font-size: 14px; color: var(--ink); }
.at-step-body span   { font-size: 13px; color: var(--muted); display: block; margin-top: 2px; }

/* Section label (above section heading) */
.at-eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}

/* Inline tag/badge */
.at-tag {
  display: inline-block;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.06em;
  padding: 2px 8px;
  border-radius: 9999px;
  background: var(--surface-strong);
  color: var(--muted);
}
.at-tag.green { background: #dcf4dc; color: var(--success); }
.at-tag.blue  { background: #dce8fa; color: #254fad; }
.at-tag.coral { background: #fce8e2; color: #aa2d00; }
.at-tag.amber { background: #fef3c7; color: #92400e; }

/* Caption table */
.at-caption-row {
  font-size: 12px;
  color: var(--muted);
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--hairline);
}

/* ─── SIDEBAR special overrides ──────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #ffffff !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
  background: var(--ink) !important;
}
/* Remove default Streamlit blue accent from radio */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:checked + div {
  color: var(--ink) !important;
}

/* ─── Plotly chart container ─────────────────────────────────────────────── */
.js-plotly-plot {
  border: 1px solid var(--hairline);
  border-radius: var(--r-md);
  overflow: hidden;
}

/* ─── Remove default Streamlit red/blue accent colours from form elements ── */
.stSelectbox [data-baseweb="select"] > div:first-child:focus-within {
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 2px rgba(24,29,38,0.15) !important;
}

/* ─── Streamlit default header override ────────────────────────────────────*/
[data-testid="stHeader"] {
  background: var(--canvas) !important;
  border-bottom: 1px solid var(--hairline) !important;
}

/* ─── Spinner text ──────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
  font-size: 14px !important;
  color: var(--muted) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
_SESSION_KEYS = [
    "timeseries_df", "categories", "data_summary",
    "series_df", "test_results", "future_forecast",
    "eval_report", "engine", "selected_cat", "freq", "horizon",
]
for key in _SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None


def _reset_forecast_state() -> None:
    for key in ["series_df", "test_results", "future_forecast", "eval_report", "engine"]:
        st.session_state[key] = None


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    with st.sidebar:
        # Brand mark
        st.markdown(
            '<p class="at-eyebrow" style="margin-top:0">Forecasting System</p>'
            '<p style="font-size:17px;font-weight:600;color:#181d26;margin:0 0 2px">Indomarco · DC</p>'
            '<p style="font-size:12px;color:#41454d;margin:0">Prophet + LightGBM Hybrid</p>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:1.25rem 0 1rem'>", unsafe_allow_html=True)

        # 1. Upload
        st.markdown('<p class="at-eyebrow">1 — Data Transaksi</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Unggah log kasir (.csv)",
            type=["csv"],
            label_visibility="collapsed",
            help="Kolom wajib: TANGGAL (DD-MM-YYYY), RTYPE, DIV, CAT_COD, QTY",
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # 2. Preprocessing config
        st.markdown('<p class="at-eyebrow">2 — Konfigurasi</p>', unsafe_allow_html=True)
        category_col = st.radio(
            "Pengelompokan",
            options=["DIV (Divisi – 55 kelompok)", "CAT_COD (Sub-Kategori – 289 kode)"],
            index=0,
            label_visibility="collapsed",
        )
        cat_col_key = "DIV" if "DIV" in category_col else "CAT_COD"

        freq_label = st.selectbox(
            "Interval Waktu",
            options=list(FREQ_OPTIONS.keys()),
            index=0,
        )
        freq = FREQ_OPTIONS[freq_label]

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # 3. Category picker
        st.markdown('<p class="at-eyebrow">3 — Kategori Produk</p>', unsafe_allow_html=True)
        categories = st.session_state.get("categories") or []
        if categories:
            selected_cat = st.selectbox(
                "Kode Divisi / Kategori",
                options=categories,
                label_visibility="collapsed",
                help=f"Hanya kategori dengan ≥ {MIN_DATA_POINTS} observasi",
            )
        else:
            st.markdown(
                '<p style="font-size:13px;color:#41454d;padding:10px 0">Unggah data CSV terlebih dahulu.</p>',
                unsafe_allow_html=True,
            )
            selected_cat = None

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # 4. Model params
        st.markdown('<p class="at-eyebrow">4 — Parameter Model</p>', unsafe_allow_html=True)
        seasonality_mode = st.selectbox(
            "Dekomposisi Musiman",
            options=SEASONALITY_MODES,
            index=0,
            format_func=lambda x: "Aditif" if x == "additive" else "Multiplikatif",
            help="Aditif: amplitudo konstan. Multiplikatif: amplitudo tumbuh seiring penjualan.",
        )
        horizon = st.slider(
            "Horizon Proyeksi (periode)",
            min_value=MIN_HORIZON,
            max_value=min(MAX_HORIZON, 52),
            value=DEFAULT_HORIZON,
            step=1,
        )

        st.markdown("<hr style='margin:1.25rem 0 1rem'>", unsafe_allow_html=True)

        # Run button
        run_forecast = st.button(
            "Jalankan Peramalan",
            use_container_width=True,
            type="primary",
            disabled=(selected_cat is None),
        )

        # Footer caption
        st.markdown(
            '<p style="font-size:11px;color:#9297a0;margin-top:1.25rem;line-height:1.6">'
            'In-Memory · Data tidak tersimpan<br>'
            '<strong style="color:#41454d">Alfiyan Nazar</strong> · 220511053<br>'
            'Teknik Informatika – UMC 2026'
            '</p>',
            unsafe_allow_html=True,
        )

    return {
        "uploaded_file":    uploaded_file,
        "freq":             freq,
        "freq_label":       freq_label,
        "category_col":     cat_col_key,
        "selected_cat":     selected_cat,
        "seasonality_mode": seasonality_mode,
        "horizon":          horizon,
        "run_forecast":     run_forecast,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def handle_data_loading(params: dict) -> None:
    uploaded_file = params["uploaded_file"]
    if uploaded_file is None:
        return

    prev_summary = st.session_state.get("data_summary")
    is_new_file  = (
        prev_summary is None or
        prev_summary.get("filename") != uploaded_file.name
    )
    if not is_new_file:
        return

    with st.spinner(f"Memproses {uploaded_file.name} …"):
        try:
            preprocessor = DataPreprocessor(freq=params["freq"])
            preprocessor.fit(
                source=uploaded_file,
                category_col=params["category_col"],
                freq=params["freq"],
            )
            summary = preprocessor.get_data_summary()
            summary["filename"] = uploaded_file.name

            st.session_state["timeseries_df"] = preprocessor.timeseries_df
            st.session_state["categories"]    = preprocessor.categories
            st.session_state["data_summary"]  = summary
            _reset_forecast_state()

            st.success(
                f"Dataset dimuat — "
                f"**{summary['total_baris_bersih']:,}** transaksi valid · "
                f"**{summary['total_kategori']}** kategori"
            )
            logger.info("Dataset loaded: %s", uploaded_file.name)
        except ValueError as e:
            st.error(f"Skema berkas tidak sesuai: {e}")
        except Exception as e:
            st.error(f"Gagal memproses berkas: {e}")
            logger.exception("Gagal memproses berkas.")


# ---------------------------------------------------------------------------
# Forecast Execution
# ---------------------------------------------------------------------------
def handle_forecast_execution(params: dict) -> None:
    if not params["run_forecast"]:
        return

    ts_df        = st.session_state.get("timeseries_df")
    selected_cat = params["selected_cat"]
    if ts_df is None or selected_cat is None:
        st.warning("Muat data terlebih dahulu.")
        return

    filter_engine = DataFilterEngine(min_points=MIN_DATA_POINTS)
    evaluator     = ModelEvaluator()

    try:
        series_df, summary, is_valid, msg = filter_engine.prepare_for_forecast(
            ts_df, selected_cat=selected_cat, category_col="category",
        )
    except ValueError as e:
        st.error(str(e))
        return

    if not is_valid:
        st.warning(msg)

    with st.spinner(f"Melatih model untuk divisi {selected_cat} …"):
        try:
            engine = HybridForecastingEngine(
                seasonality_mode=params["seasonality_mode"],
                test_size=0.20,
            )
            engine.fit(series_df, freq=params["freq"])
            test_results    = engine.predict_test()
            future_forecast = engine.forecast_future(series_df, horizon=params["horizon"])
            eval_report     = evaluator.build_evaluation_report(test_results)

            st.session_state.update({
                "series_df":      series_df,
                "test_results":   test_results,
                "future_forecast": future_forecast,
                "eval_report":    eval_report,
                "engine":         engine,
                "selected_cat":   selected_cat,
                "freq":           params["freq"],
                "horizon":        params["horizon"],
            })

            mape = eval_report["metrics_hybrid"]["MAPE_%"]
            label, emoji = evaluator.interpret_mape(mape)
            st.success(
                f"{emoji} Selesai · Divisi **{selected_cat}** · "
                f"MAPE Hibrida: **{mape:.2f}%** — {label}"
            )
            logger.info("Forecast selesai: kategori=%s, MAPE=%.2f%%", selected_cat, mape)
        except Exception as e:
            st.error(f"Gagal menjalankan peramalan: {e}")
            logger.exception("Forecast gagal.")


# ---------------------------------------------------------------------------
# WELCOME STATE — hero band + step guide
# ---------------------------------------------------------------------------
def _render_welcome() -> None:
    st.markdown(
        """
        <div class="at-cream" style="margin-top:0">
          <h4>Selamat datang di Forecasting System</h4>
          <p>Unggah log transaksi kasir POS Indomarco untuk memulai peramalan penjualan per kategori produk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("Unggah berkas CSV", "Log kasir POS — MTRAN.csv atau file serupa"),
        ("Pilih interval waktu", "Mingguan (W) atau Bulanan (M)"),
        ("Pilih kategori produk", "Berdasarkan kode Divisi atau Sub-Kategori"),
        ("Tentukan mode musiman", "Aditif (amplitudo konstan) atau Multiplikatif"),
        ("Atur horizon proyeksi", "Jumlah periode ke depan yang ingin diramalkan"),
        ("Jalankan peramalan", "Klik tombol di sidebar — hasil muncul di bawah"),
    ]
    html = ""
    for i, (title, desc) in enumerate(steps, 1):
        html += (
            f'<div class="at-step">'
            f'<div class="at-step-num">{i}</div>'
            f'<div class="at-step-body"><strong>{title}</strong><span>{desc}</span></div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)

    # CSV format callout
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="at-dark">
          <h3>Format Berkas CSV</h3>
          <p>Kolom wajib: <strong>TANGGAL</strong> (DD-MM-YYYY) &nbsp;·&nbsp;
          <strong>RTYPE</strong> ('J' = Jual) &nbsp;·&nbsp;
          <strong>DIV</strong> (kode 2 digit) &nbsp;·&nbsp;
          <strong>CAT_COD</strong> (kode 6 digit) &nbsp;·&nbsp;
          <strong>QTY</strong> (kuantitas).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------
def render_main_dashboard(params: dict) -> None:
    viz          = DashboardVisualizer()
    evaluator    = ModelEvaluator()
    series_df    = st.session_state.get("series_df")
    test_results = st.session_state.get("test_results")
    future_df    = st.session_state.get("future_forecast")
    eval_report  = st.session_state.get("eval_report")
    engine       = st.session_state.get("engine")
    selected_cat = st.session_state.get("selected_cat")
    data_summary = st.session_state.get("data_summary")

    # ── Hero band ─────────────────────────────────────────────────────────
    st.markdown(
        '<p class="at-eyebrow" style="margin-top:0">PT. Indomarco Prismatama · DC Cirebon</p>'
        '<h1 style="margin-bottom:4px">Sales Forecasting Dashboard</h1>'
        '<p style="font-size:14px;color:#41454d;margin:0">Model Hibrida Prophet + LightGBM &nbsp;·&nbsp; In-Memory Analytics</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_overview, tab_forecast, tab_eval, tab_export = st.tabs([
        "Overview",
        "Kurva Peramalan",
        "Evaluasi Akurasi",
        "Ekspor Data",
    ])

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────
    with tab_overview:
        if data_summary is None:
            _render_welcome()
            return

        # Data summary as signature coral card
        fname = data_summary.get("filename", "-")
        n_tx  = data_summary.get("total_baris_bersih", 0)
        n_cat = data_summary.get("total_kategori", 0)
        qty   = data_summary.get("total_qty", 0)
        d0    = data_summary.get("tanggal_awal")
        d1    = data_summary.get("tanggal_akhir")
        daterange = ""
        if d0 and d1:
            daterange = f"{d0.strftime('%d %b %Y')} – {d1.strftime('%d %b %Y')}"

        st.markdown(
            f"""
            <div class="at-coral">
              <h3>{fname}</h3>
              <p>{daterange} &nbsp;·&nbsp; {n_tx:,} transaksi valid &nbsp;·&nbsp; {n_cat} kategori aktif</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stat grid (demo-grid-card style)
        st.markdown(
            f"""
            <div class="at-stat-grid">
              <div class="at-stat-card peach">
                <div class="label">Total Transaksi</div>
                <div class="value">{n_tx:,}</div>
                <div class="delta">baris valid (RTYPE=J, QTY&gt;0)</div>
              </div>
              <div class="at-stat-card mint">
                <div class="label">Kategori Aktif</div>
                <div class="value">{n_cat}</div>
                <div class="delta">≥ {MIN_DATA_POINTS} titik observasi</div>
              </div>
              <div class="at-stat-card cream">
                <div class="label">Total Kuantitas</div>
                <div class="value">{qty:,}</div>
                <div class="delta">unit barang terjual</div>
              </div>
              <div class="at-stat-card yellow">
                <div class="label">Periode Data</div>
                <div class="value">{daterange or "—"}</div>
                <div class="delta">rentang historis tersedia</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preview top-5 time series
        ts_df = st.session_state.get("timeseries_df")
        if ts_df is not None:
            st.markdown("<h2 style='margin-top:2rem'>Deret Waktu — 5 Kategori Teratas</h2>", unsafe_allow_html=True)
            st.markdown(
                '<p class="at-caption-row">Volume penjualan tertinggi. Maks 100 baris ditampilkan.</p>',
                unsafe_allow_html=True,
            )
            top5 = ts_df.groupby("category")["y"].sum().nlargest(5).index.tolist()
            preview = ts_df[ts_df["category"].isin(top5)].head(100).copy()
            preview.columns = [c.upper() for c in preview.columns]
            st.dataframe(preview, use_container_width=True, hide_index=True)

    # ── TAB 2: KURVA PERAMALAN ─────────────────────────────────────────────
    with tab_forecast:
        if series_df is None or test_results is None or future_df is None:
            st.markdown(
                '<div class="at-cream"><h4>Belum ada hasil peramalan</h4>'
                '<p>Pilih kategori di sidebar lalu klik <strong>Jalankan Peramalan</strong>.</p></div>',
                unsafe_allow_html=True,
            )
            return

        # KPI metrics (4 per row)
        if eval_report:
            mape_val  = eval_report["metrics_hybrid"]["MAPE_%"]
            rmse_val  = eval_report["metrics_hybrid"]["RMSE"]
            mae_val   = eval_report["metrics_hybrid"]["MAE"]
            mape_lbl  = eval_report["mape_interpretation"]
            mape_emj  = eval_report["mape_emoji"]
            trend_pct = viz.calculate_trend_pct(series_df)

            # Signature coral header row
            st.markdown(
                f'<div class="at-eyebrow">Divisi / Kategori</div>'
                f'<h2 style="margin-top:0">{selected_cat}</h2>',
                unsafe_allow_html=True,
            )

            # KPI Cards via Streamlit metric (styled via CSS above)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "PENJUALAN HISTORIS",
                    f"{int(series_df['y'].sum()):,}",
                    delta=f"{trend_pct:+.1f}% tren",
                    help="Kuantitas kumulatif seluruh periode historis",
                )
            with c2:
                st.metric(
                    "MAPE HIBRIDA",
                    f"{mape_val:.2f}%",
                    delta=f"{mape_emj} {mape_lbl}",
                    delta_color="off",
                    help="Mean Absolute Percentage Error · < 10% = Sangat Baik",
                )
            with c3:
                st.metric(
                    "RMSE",
                    f"{rmse_val:,.2f}",
                    help="Root Mean Squared Error (unit barang)",
                )
            with c4:
                st.metric(
                    "MAE",
                    f"{mae_val:,.2f}",
                    help="Mean Absolute Error rata-rata per periode",
                )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Main forecast chart
        fig_main = viz.plot_forecast(
            historical_df=series_df,
            test_results=test_results,
            future_forecast=future_df,
            title="Penjualan Aktual vs Peramalan Hibrida",
            category_label=selected_cat,
            freq_label=params.get("freq_label", "W-MON"),
            show_prophet=True,
        )
        # Apply clean Airtable-style chart theme
        fig_main.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#333840"),
            title=dict(font=dict(size=15, weight=500, color="#181d26")),
            xaxis=dict(
                gridcolor="#dddddd", gridwidth=1,
                linecolor="#dddddd", tickfont=dict(size=11),
            ),
            yaxis=dict(
                gridcolor="#dddddd", gridwidth=1,
                linecolor="#dddddd", tickfont=dict(size=11),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#dddddd",
                borderwidth=1,
                font=dict(size=12),
            ),
            margin=dict(l=40, r=16, t=50, b=40),
        )
        st.plotly_chart(fig_main, use_container_width=True)

    # ── TAB 3: EVALUASI AKURASI ────────────────────────────────────────────
    with tab_eval:
        if eval_report is None:
            st.markdown(
                '<div class="at-cream"><h4>Belum ada laporan evaluasi</h4>'
                '<p>Jalankan peramalan terlebih dahulu.</p></div>',
                unsafe_allow_html=True,
            )
            return

        mape_val = eval_report["metrics_hybrid"]["MAPE_%"]
        lbl, emj = evaluator.interpret_mape(mape_val)

        # MAPE interpretasi sebagai dark/forest signature card
        card_class = "at-forest" if mape_val < 10 else ("at-dark" if mape_val < 20 else "at-coral")
        st.markdown(
            f'<div class="{card_class}">'
            f'<h3>Kualitas Model: {lbl}</h3>'
            f'<p>MAPE = {mape_val:.2f}% &nbsp;·&nbsp; '
            f'Kriteria: &lt;10% Sangat Baik · 10–20% Baik · 20–50% Layak · &gt;50% Kurang</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<h2>Perbandingan Model</h2>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:13px;color:#41454d;margin-bottom:1rem">'
            'Prophet Tunggal dibandingkan dengan Model Hibrida Prophet + LightGBM pada data uji (20%).'
            '</p>',
            unsafe_allow_html=True,
        )
        viz.render_comparison_table(eval_report["comparison_table"])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2>Analisis Galat Residual</h2>", unsafe_allow_html=True)

        fig_res = viz.plot_residual_analysis(
            test_results=test_results,
            category_label=selected_cat,
        )
        fig_res.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#333840"),
            xaxis=dict(gridcolor="#dddddd", linecolor="#dddddd"),
            yaxis=dict(gridcolor="#dddddd", linecolor="#dddddd"),
            margin=dict(l=40, r=16, t=50, b=40),
        )
        st.plotly_chart(fig_res, use_container_width=True)

        # Feature importance
        if engine is not None:
            fi_df = engine.get_feature_importance()
            if fi_df is not None and not fi_df.empty:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h2>Feature Importance LightGBM</h2>", unsafe_allow_html=True)
                st.markdown(
                    '<p style="font-size:13px;color:#41454d;margin-bottom:1rem">'
                    'Kontribusi setiap fitur residual terhadap prediksi koreksi LightGBM (gain-based).'
                    '</p>',
                    unsafe_allow_html=True,
                )
                fig_fi = viz.plot_feature_importance(fi_df)
                fig_fi.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#333840"),
                    margin=dict(l=160, r=16, t=50, b=40),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    # ── TAB 4: EKSPOR ─────────────────────────────────────────────────────
    with tab_export:
        if future_df is None:
            st.markdown(
                '<div class="at-cream"><h4>Belum ada data untuk diekspor</h4>'
                '<p>Jalankan peramalan terlebih dahulu.</p></div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            f'<div class="at-eyebrow">Proyeksi Masa Depan</div>'
            f'<h2 style="margin-top:0">Divisi {selected_cat} — {st.session_state.get("horizon", DEFAULT_HORIZON)} Periode</h2>',
            unsafe_allow_html=True,
        )

        viz.render_forecast_table(future_df, freq_label=st.session_state.get("freq", "W-MON"))

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # CTA band for download
        csv_bytes = viz.export_csv_bytes(future_df, category_label=selected_cat)
        st.markdown(
            '<div class="at-cta-band">'
            '<div><h3>Unduh Laporan Proyeksi</h3>'
            '<p>Berkas CSV ter-encode UTF-8. Dapat dibuka di Excel atau Google Sheets.</p></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Unduh CSV",
            data=csv_bytes,
            file_name=f"proyeksi_{selected_cat}.csv",
            mime="text/csv; charset=utf-8",
            type="primary",
        )

        if test_results is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2>Detail Hasil Uji (20% Data Terakhir)</h2>", unsafe_allow_html=True)
            st.dataframe(test_results.round(2), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    # Inject Airtable Design System CSS first
    st.markdown(AIRTABLE_CSS, unsafe_allow_html=True)

    params = render_sidebar()
    handle_data_loading(params)
    handle_forecast_execution(params)
    render_main_dashboard(params)


if __name__ == "__main__":
    main()
