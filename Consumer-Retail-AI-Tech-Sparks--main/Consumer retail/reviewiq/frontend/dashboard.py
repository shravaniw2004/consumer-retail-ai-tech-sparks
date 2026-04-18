"""
ReviewIQ v5 — CSV Upload-Driven Dashboard
Upload your CSV → Instant AI Analysis → Actionable Insights

Run: python -m streamlit run frontend/dashboard.py
"""

import os
import sys
import re
import time
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

# Import backend modules
from emotion_classifier import (
    classify_emotion_batch, get_emotion_distribution,
    get_emotion_timeline, EMOTION_TAXONOMY
)
from bot_detector import execute_bot_detection, generate_network_html
from crisis_simulator import inject_crisis, get_crisis_options, CRISIS_TYPES
from preprocessor import ReviewPreprocessor
from batch_analyzer import analyze_batches
from anomaly_detector import calculate_anomaly_scores

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ReviewIQ v5",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Premium CSS — Global Design System
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global Reset & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background-color: #0A0E1A !important;
    color: #F1F5F9 !important;
}

.stApp {
    background-color: #0A0E1A !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3B82F6; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0F172A 0%, #1A2235 50%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6, #14B8A6);
}
.app-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #F1F5F9 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 13px;
    color: #475569;
    margin-top: 6px;
    letter-spacing: 0.5px;
}

/* ── Navigation Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #1E293B !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #1A2235 !important;
    color: #F1F5F9 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.4) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.1) !important;
    transform: translateY(-1px);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.3) !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed #1E293B !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6 !important;
    background: rgba(59,130,246,0.03) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Selectbox / Dropdowns ── */
[data-baseweb="select"] {
    background: #111827 !important;
}
[data-baseweb="select"] > div {
    background: #111827 !important;
    border-color: #1E293B !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
}
[data-baseweb="select"] > div:hover {
    border-color: #3B82F6 !important;
}
[data-baseweb="popover"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
}
[data-baseweb="menu"] {
    background: #111827 !important;
}
[data-baseweb="option"] {
    background: #111827 !important;
    color: #94A3B8 !important;
}
[data-baseweb="option"]:hover {
    background: #1A2235 !important;
    color: #F1F5F9 !important;
}

/* ── Sliders ── */
[data-baseweb="slider"] [role="slider"] {
    background: #3B82F6 !important;
    border: 2px solid #F1F5F9 !important;
    box-shadow: 0 0 8px rgba(59,130,246,0.5) !important;
}

/* ── Text inputs and text areas ── */
[data-baseweb="textarea"], [data-baseweb="input"] {
    background: #111827 !important;
    border-color: #1E293B !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── Dataframe / Tables ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #1E293B !important;
}
[data-testid="stDataFrame"] iframe {
    border-radius: 10px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    background: #111827 !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #1A2235 !important;
    color: #F1F5F9 !important;
}

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-size: 14px !important;
}
.stSuccess {
    background: rgba(34,197,94,0.08) !important;
    border-left-color: #22C55E !important;
    color: #86EFAC !important;
}
.stError {
    background: rgba(239,68,68,0.08) !important;
    border-left-color: #EF4444 !important;
    color: #FCA5A5 !important;
}
.stWarning {
    background: rgba(234,179,8,0.08) !important;
    border-left-color: #EAB308 !important;
    color: #FDE047 !important;
}
.stInfo {
    background: rgba(59,130,246,0.08) !important;
    border-left-color: #3B82F6 !important;
    color: #93C5FD !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: #1E293B !important;
    border-radius: 4px !important;
}

/* ── Toggle / Checkbox ── */
[role="checkbox"] {
    accent-color: #3B82F6 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #3B82F6 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}

/* ── Dividers ── */
hr {
    border-color: #1E293B !important;
    margin: 24px 0 !important;
}

/* ── Custom Card Classes ── */
.reviewiq-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
}
.reviewiq-card:hover {
    border-color: #3B82F6;
    box-shadow: 0 4px 20px rgba(59,130,246,0.08);
}
.reviewiq-card-critical {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 4px solid #EF4444;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.reviewiq-card-warning {
    background: rgba(249,115,22,0.06);
    border: 1px solid rgba(249,115,22,0.3);
    border-left: 4px solid #F97316;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.reviewiq-card-success {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.3);
    border-left: 4px solid #22C55E;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.reviewiq-card-info {
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.3);
    border-left: 4px solid #3B82F6;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1E293B;
}
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #F1F5F9;
    margin: 0;
}
.section-badge {
    background: rgba(59,130,246,0.15);
    color: #93C5FD;
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Stat Badges ── */
.badge-critical {
    background: rgba(239,68,68,0.15);
    color: #FCA5A5;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-block;
}
.badge-warning {
    background: rgba(249,115,22,0.15);
    color: #FDBA74;
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
}
.badge-success {
    background: rgba(34,197,94,0.15);
    color: #86EFAC;
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
}
.badge-info {
    background: rgba(59,130,246,0.15);
    color: #93C5FD;
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
}

/* ── Top Gradient Bar ── */
.top-accent-bar {
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #14B8A6 100%);
    border-radius: 0 0 2px 2px;
    margin-bottom: 24px;
}

/* ── Download button override ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #14B8A6, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    color: #94A3B8 !important;
    font-size: 14px !important;
}
[data-testid="stRadio"] [role="radio"][aria-checked="true"] + div {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Helper: Plotly Chart Theme
# ──────────────────────────────────────────────────────────────

def apply_chart_theme(fig):
    """Apply the ReviewIQ dark chart theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(family='Inter, sans-serif', color='#94A3B8', size=12),
        title_font=dict(size=16, color='#F1F5F9', family='Inter, sans-serif'),
        xaxis=dict(gridcolor='#1E293B', linecolor='#1E293B', tickcolor='#475569',
                   tickfont=dict(color='#94A3B8')),
        yaxis=dict(gridcolor='#1E293B', linecolor='#1E293B', tickcolor='#475569',
                   tickfont=dict(color='#94A3B8')),
        legend=dict(bgcolor='rgba(17,24,39,0.9)', bordercolor='#1E293B',
                    borderwidth=1, font=dict(color='#94A3B8')),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor='#1A2235', font_size=13, font_family='Inter',
                        bordercolor='#3B82F6'),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Helper: Normalize uploaded CSV into standard format
# ──────────────────────────────────────────────────────────────

def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts any CSV and normalizes column names.
    Tries to detect the review text column, rating column, etc.
    """
    df = df.copy()

    # Track which source columns have already been claimed
    claimed_sources = set()
    col_map = {}

    def _detect(keywords, target_name):
        """Find the best candidate column for a target, avoiding already-claimed columns."""
        candidates = [c for c in df.columns if c not in claimed_sources and
                      any(k in c.lower() for k in keywords)]
        # If the target name already exists as a column, don't remap
        if target_name in df.columns and target_name not in col_map.values():
            claimed_sources.add(target_name)
            return
        if candidates:
            col_map[candidates[0]] = target_name
            claimed_sources.add(candidates[0])

    # -- Detect columns in priority order --
    _detect(['review', 'text', 'comment', 'feedback', 'body', 'content', 'description', 'message'], 'review_text')

    # Fallback: if no text column found, pick the one with longest average string length
    if 'review_text' not in col_map.values() and 'review_text' not in df.columns:
        str_cols = [c for c in df.select_dtypes(include='object').columns if c not in claimed_sources]
        if len(str_cols) > 0:
            avg_lens = {c: df[c].astype(str).apply(len).mean() for c in str_cols}
            best = max(avg_lens, key=avg_lens.get)
            col_map[best] = 'review_text'
            claimed_sources.add(best)

    _detect(['rating', 'star', 'score', 'stars'], 'rating')
    _detect(['product', 'item', 'brand', 'category'], 'product_name')
    _detect(['date', 'time', 'timestamp', 'created', 'posted'], 'timestamp')
    _detect(['user', 'author', 'reviewer', 'username'], 'username')

    # Apply mapping
    df = df.rename(columns=col_map)

    # Drop any duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure required columns exist
    if 'review_text' not in df.columns:
        st.error("❌ Could not detect a review text column. Please ensure your CSV has a column with review text.")
        st.stop()

    if 'product_name' not in df.columns:
        df['product_name'] = 'Product'
    if 'rating' not in df.columns:
        df['rating'] = np.nan
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now().isoformat()
    if 'username' not in df.columns:
        df['username'] = [f"user_{i}" for i in range(len(df))]

    # Add ID
    df['review_id'] = [f"R{i:04d}" for i in range(len(df))]

    # Drop rows where review text is empty/NaN
    df = df.dropna(subset=['review_text'])
    # Ensure review_text is a Series (not a DataFrame from duplicate cols)
    review_col = df['review_text']
    if isinstance(review_col, pd.DataFrame):
        review_col = review_col.iloc[:, 0]
    df['review_text'] = review_col.astype(str).str.strip()
    df = df[df['review_text'] != '']
    df = df.reset_index(drop=True)

    return df


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Run the preprocessor on review text and add columns."""
    preprocessor = ReviewPreprocessor()
    results = []
    for _, row in df.iterrows():
        res = preprocessor.process(str(row['review_text']))
        results.append(res)

    df['clean_text'] = [r.get('clean_text', r.get('processed_text')) for r in results]
    df['processed_text'] = [r.get('processed_text', '') for r in results]
    df['original_text'] = [r.get('original_text', '') for r in results]
    df['language'] = [r.get('language', 'en') for r in results]
    df['detected_language'] = [r.get('detected_language', 'EN') for r in results]
    df['trust_score'] = [r.get('trust_score', 0.0) for r in results]
    df['review_quality_score'] = [r.get('review_quality_score', 0) for r in results]
    df['is_suspicious'] = [r.get('is_suspicious', False) for r in results]
    df['low_quality'] = [r.get('low_quality', False) for r in results]
    df['emoji_tags'] = [r.get('emoji_tags', []) for r in results]
    df['caps_intensity'] = [r.get('caps_intensity', False) for r in results]
    df['preprocessing_flags'] = [r.get('preprocessing_flags', []) for r in results]
    return df


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run sentiment classification using local keyword-based method (no API calls)."""
    def _local_sentiment(text, rating=None):
        if rating is not None:
            try:
                r = float(rating)
                if r >= 4: return 'positive'
                elif r <= 2: return 'negative'
                else: return 'neutral'
            except: pass
        t = str(text).lower()
        neg = ['worst','terrible','awful','horrible','useless','waste','pathetic','broken','scam','fraud','bakwas','bekar','ghatiya','poor','bad','disappointed','angry','disgusting','refund','hate']
        pos = ['amazing','excellent','great','love','perfect','fantastic','outstanding','wonderful','best','superb','recommended','happy','good','nice','mast','zabardast']
        nc = sum(1 for w in neg if w in t)
        pc = sum(1 for w in pos if w in t)
        if nc > pc: return 'negative'
        elif pc > nc: return 'positive'
        return 'neutral'
    df['sentiment'] = df.apply(lambda row: _local_sentiment(row.get('clean_text', row.get('review_text', '')), row.get('rating')), axis=1)
    return df


# ──────────────────────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────────────────────

if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'emotion_results' not in st.session_state:
    st.session_state.emotion_results = None
if 'bot_result' not in st.session_state:
    st.session_state.bot_result = None
if 'crisis_active' not in st.session_state:
    st.session_state.crisis_active = False
if 'crisis_result' not in st.session_state:
    st.session_state.crisis_result = None
if 'crisis_reviews_df' not in st.session_state:
    st.session_state.crisis_reviews_df = None
if 'col_mapping_shown' not in st.session_state:
    st.session_state.col_mapping_shown = False
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'batches' not in st.session_state:
    st.session_state.batches = None
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = None
if 'deduped_df' not in st.session_state:
    st.session_state.deduped_df = None
if 'pipeline_complete' not in st.session_state:
    st.session_state.pipeline_complete = False


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="top-accent-bar"></div>
<div class="app-header">
    <h1 class="app-title">🔍 ReviewIQ — Review Intelligence Platform</h1>
    <p class="app-subtitle">
        v5 &nbsp;|&nbsp; Upload CSV → Instant AI Analysis → Actionable Insights
        &nbsp;|&nbsp; Hack Malenadu '26
    </p>
</div>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────
# GLOBAL PIPELINE SUMMARY BANNER
# ──────────────────────────────────────────────────────────────

if st.session_state.get('pipeline_complete', False):
    with st.expander("✅ Pipeline Analysis Summary", expanded=False):
        c_df = st.session_state.processed_df
        c_bot = st.session_state.bot_result
        c_anom = st.session_state.get('anomalies') or {}
        c_batch = st.session_state.get('batches') or {}
        
        langs = c_df['language'].value_counts() if 'language' in c_df.columns else pd.Series()
        lang_str = " ".join([f"{k.upper()}({v})" for k, v in langs.items()]) if not langs.empty else "N/A"
        
        bot_count = c_bot.confirmed_bots_removed if c_bot else 0
        camp_count = len([c for c in c_bot.clusters if c.campaign_confidence > 50]) if c_bot else 0
        anom_count = len(c_anom.get('anomalies', [])) if isinstance(c_anom, dict) and 'error' not in c_anom else 0
        batch_count = len(c_batch.get('batches', [])) if isinstance(c_batch, dict) else 0
        prods_count = c_df['product_name'].nunique() if 'product_name' in c_df.columns else 1
        
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown(f"**📊 Dataset:** {len(c_df)} clean reviews | {prods_count} products")
            st.markdown(f"**🤖 Bots:** {bot_count} removed | {camp_count} active campaigns")
            st.markdown(f"**🌐 Languages:** {lang_str}")
        with c2:
            st.markdown(f"**📦 Batches:** {batch_count} temporal chunks created")
            st.markdown(f"**🚨 Anomalies:** {anom_count} Z-Score anomalies detected")
            
# ──────────────────────────────────────────────────────────────
# TABS (each feature in exactly one tab — no duplication)
# ──────────────────────────────────────────────────────────────

tab_upload, tab_preprocess, tab_overview, tab_trends, tab_bot, tab_action, tab_batch, tab_anomaly, tab_crisis = st.tabs([
    "📤 Upload",
    "🧹 Preprocessing Report",
    "📊 Overview",
    "📈 Trends",
    "🕸️ Bot Detection",
    "📋 Action Brief",
    "📦 Batch Analysis",
    "🎯 Z-Score Anomalies",
    "☢️ Crisis Sim",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: Upload & Preprocess
# ═══════════════════════════════════════════════════════════════

with tab_upload:
    st.markdown("""
    <div class="section-header">
        <span style="font-size:24px">📤</span>
        <h2 class="section-title">Upload Your Review CSV</h2>
        <span class="section-badge">STEP 1</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="reviewiq-card-info">
        <p style="margin:0; color:#94A3B8; font-size:14px; line-height:1.6;">
            Upload any CSV file containing customer reviews. The system will automatically
            detect the review text, rating, product, and other columns.
        </p>
        <p style="margin:8px 0 0; color:#94A3B8; font-size:13px;">
            <strong style="color:#93C5FD;">Expected columns (auto-detected):</strong><br>
            Review text: <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#86EFAC;font-size:12px;">review_text</code>
            <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#86EFAC;font-size:12px;">comment</code>
            <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#86EFAC;font-size:12px;">feedback</code><br>
            Optional: <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#FDBA74;font-size:12px;">rating</code>
            <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#FDBA74;font-size:12px;">product_name</code>
            <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#FDBA74;font-size:12px;">date</code>
            <code style="background:#1A2235;padding:2px 6px;border-radius:4px;
            color:#FDBA74;font-size:12px;">username</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload CSV or JSON file",
        type=["csv", "json"],
        accept_multiple_files=False,
        key="csv_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".json"):
                raw_df = pd.read_json(uploaded_file)
            else:
                try:
                    raw_df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

        st.success(f"✅ File uploaded: {uploaded_file.name} — {len(raw_df)} rows detected")
        
        # Preview table
        st.markdown("""
        <div class="section-header" style="margin-top:20px;">
            <span style="font-size:18px">👁️</span>
            <h2 class="section-title" style="font-size:16px;">Preview</h2>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(raw_df.head(5), hide_index=True, use_container_width=True, height=200)
        
        # Column auto-detection with styled badges
        st.markdown("""
        <div class="section-header" style="margin-top:20px;">
            <span style="font-size:18px">🔎</span>
            <h2 class="section-title" style="font-size:16px;">Auto-Detected Columns</h2>
        </div>
        """, unsafe_allow_html=True)
        
        def detect_column(df, keywords):
            for col in df.columns:
                if any(kw in col.lower() for kw in keywords):
                    return col
            return None
        
        review_col = detect_column(raw_df, ['review', 'text', 'comment', 'feedback', 'body', 'content'])
        rating_col = detect_column(raw_df, ['rating', 'stars', 'score', 'rate'])
        product_col = detect_column(raw_df, ['product', 'product_name', 'item', 'name', 'title'])
        date_col = detect_column(raw_df, ['date', 'timestamp', 'created', 'time', 'posted'])
        username_col = detect_column(raw_df, ['user', 'username', 'reviewer', 'author', 'name'])
        
        col_html = ""
        for col_label, col_val in [("Review text", review_col), ("Rating", rating_col), ("Product", product_col), ("Date", date_col), ("Username", username_col)]:
            if col_val:
                col_html += f"""<span style="background:rgba(34,197,94,0.15); color:#86EFAC; border:1px solid rgba(34,197,94,0.3); border-radius:6px; padding:4px 12px; font-size:12px; font-weight:600; margin-right:8px; display:inline-block; margin-bottom:6px;">✅ {col_label} → "{col_val}"</span>"""
            else:
                col_html += f"""<span style="background:rgba(249,115,22,0.15); color:#FDBA74; border:1px solid rgba(249,115,22,0.3); border-radius:6px; padding:4px 12px; font-size:12px; font-weight:600; margin-right:8px; display:inline-block; margin-bottom:6px;">⚠️ {col_label} → not found</span>"""
        
        st.markdown(col_html, unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🚀 Execute Unified Pipeline", type="primary", key="analyze_btn", use_container_width=True):
            st.session_state.uploaded_df = raw_df
            st.session_state.pipeline_complete = False
            
            progress_container = st.container()
            with progress_container:
                st.markdown("""
                <div class="section-header" style="margin-top:16px;">
                    <span style="font-size:20px">⚙️</span>
                    <h2 class="section-title">Executing Intelligence Pipeline</h2>
                    <span class="section-badge">PROCESSING</span>
                </div>
                """, unsafe_allow_html=True)
                
                # STAGE 1
                prog_msg1 = st.empty()
                prog_msg1.markdown("Stage 1: Preprocessing... ⏳")
                try:
                    norm_df = normalize_csv(raw_df)
                    clean_df = run_preprocessing(norm_df)
                    clean_df = clean_df[clean_df['low_quality'] == False].copy()
                    st.session_state.clean_df = clean_df
                    prog_msg1.markdown(f"Stage 1: Preprocessing... ✅ Done ({len(clean_df)} reviews cleaned)")
                except Exception as e:
                    prog_msg1.markdown(f"Stage 1: Preprocessing... ⚠️ Issue ({e})")
                    clean_df = normalize_csv(raw_df)
                    st.session_state.clean_df = clean_df

                # STAGE 2
                prog_msg2 = st.empty()
                prog_msg2.markdown("Stage 2: Bot Detection... ⏳")
                try:
                    bot_res = execute_bot_detection(clean_df.to_dict('records'))
                    indivs = [ind.username for ind in bot_res.individuals if ind.status == 'CONFIRMED_BOT']
                    deduped_df = clean_df[~clean_df['username'].isin(indivs)].copy()
                    if len(deduped_df) == 0:
                        st.error("No valid reviews remain after bot removal")
                        st.stop()
                    st.session_state.bot_result = bot_res
                    st.session_state.deduped_df = deduped_df
                    prog_msg2.markdown(f"Stage 2: Bot Detection... ✅ Done ({bot_res.confirmed_bots_removed} bots removed)")
                except Exception as e:
                    prog_msg2.markdown(f"Stage 2: Bot Detection... ⚠️ Issue ({e})")
                    deduped_df = clean_df.copy()
                    st.session_state.deduped_df = deduped_df
                
                # STAGE 3
                prog_msg3 = st.empty()
                prog_msg3.markdown("Stage 3: Batch Analysis... ⏳")
                if 'timestamp' not in deduped_df.columns and 'date' not in deduped_df.columns:
                    st.session_state.batches = []
                    prog_msg3.markdown(f"Stage 3: Batch Analysis... ⚠️ Skipped (No date column found)")
                else:
                    try:
                        b_res = analyze_batches(deduped_df, mode='time', size=14)
                        st.session_state.batches = b_res
                        prog_msg3.markdown(f"Stage 3: Batch Analysis... ✅ Done ({len(b_res['batches'])} batches created)")
                    except Exception as e:
                        st.session_state.batches = []
                        prog_msg3.markdown(f"Stage 3: Batch Analysis... ⚠️ Failed ({str(e)})")
                
                # STAGE 4
                prog_msg4 = st.empty()
                prog_msg4.markdown("Stage 4: Z-Score Analysis... ⏳")
                try:
                    anom_res = calculate_anomaly_scores(deduped_df, 'All Products')
                    if "error" in anom_res:
                        st.session_state.anomalies = {"error": anom_res["error"]}
                        prog_msg4.markdown(f"Stage 4: Z-Score Analysis... ⚠️ Skipped ({anom_res['error']})")
                    else:
                        st.session_state.anomalies = anom_res
                        prog_msg4.markdown(f"Stage 4: Z-Score Analysis... ✅ Done ({len(anom_res['anomalies'])} anomalies detected)")
                except Exception as e:
                    st.session_state.anomalies = {"error": str(e)}
                    prog_msg4.markdown(f"Stage 4: Z-Score Analysis... ⚠️ Failed ({str(e)})")

                # STAGE 5
                prog_msg5 = st.empty()
                prog_msg5.markdown("Stage 5: Sentiment Analysis... ⏳")
                
                processed = run_sentiment(deduped_df)
                review_dicts = processed.to_dict('records')
                emotion_results = classify_emotion_batch(review_dicts)
                
                processed['emotion'] = [e.emotion for e in emotion_results]
                processed['emotion_label'] = [e.emotion_label for e in emotion_results]
                processed['urgency'] = [e.urgency for e in emotion_results]
                processed['emotion_confidence'] = [e.confidence for e in emotion_results]
                
                st.session_state.processed_df = processed
                st.session_state.emotion_results = emotion_results
                st.session_state.crisis_active = False
                st.session_state.crisis_result = None
                
                prog_msg5.markdown("Stage 5: Sentiment Analysis... ✅ Done")

            st.session_state.pipeline_complete = True
            
            st.divider()
            st.success(f"✅ Pipeline Complete")
            _anom = st.session_state.anomalies
            _anom_ct = len(_anom.get('anomalies', [])) if isinstance(_anom, dict) and 'error' not in _anom else 0
            _batch = st.session_state.batches
            _batch_ct = len(_batch.get('batches', [])) if isinstance(_batch, dict) else 0
            st.info(f"Input: {len(raw_df)} reviews → Clean: {len(clean_df)} → Bots removed: {len(clean_df) - len(deduped_df)} | Anomalies: {_anom_ct} | Batches: {_batch_ct}")

    # MANUAL PASTE section
    st.divider()
    with st.expander("Or paste reviews manually"):
        st.markdown("Paste one review per line or paragraph:")
        pasted_reviews = st.text_area("Reviews", height=150, key="paste_reviews")
        
        if st.button("Analyze Pasted Reviews", key="analyze_pasted"):
            if pasted_reviews.strip():
                reviews_list = [r.strip() for r in pasted_reviews.split('\n') if r.strip()]
                manual_df = pd.DataFrame({'review_text': reviews_list})
                
                with st.spinner("🔄 Processing..."):
                    norm_df = normalize_csv(manual_df)
                    processed = run_preprocessing(norm_df)
                    processed = run_sentiment(processed)
                    review_dicts = processed.to_dict('records')
                    emotion_results = classify_emotion_batch(review_dicts)
                    
                    processed['emotion'] = [e.emotion for e in emotion_results]
                    processed['emotion_label'] = [e.emotion_label for e in emotion_results]
                    processed['urgency'] = [e.urgency for e in emotion_results]
                    processed['emotion_confidence'] = [e.confidence for e in emotion_results]
                    
                    st.session_state.processed_df = processed
                    st.session_state.emotion_results = emotion_results
                    st.session_state.uploaded_df = manual_df
                    st.session_state.bot_result = None
                    st.session_state.crisis_active = False
                    st.session_state.crisis_result = None
                    st.session_state.crisis_reviews_df = None
                
                st.success(f"✅ Processed {len(processed)} reviews!")
                st.info("Navigate to the **Overview** tab to see analysis results.")
            else:
                st.warning("Please paste some reviews first.")


# ─────────────────────────── Guard ───────────────────────────
# All tabs below require processed data
def get_data():
    """Get the working dataframe (processed + any crisis injections)."""
    df = st.session_state.processed_df
    if df is None:
        return None
    # Merge crisis reviews if active
    if st.session_state.crisis_reviews_df is not None:
        crisis_df = st.session_state.crisis_reviews_df
        # Align columns
        for col in df.columns:
            if col not in crisis_df.columns:
                crisis_df[col] = None
        combined = pd.concat([df, crisis_df], ignore_index=True)
        return combined
    return df

def require_data():
    """Show warning if no data and stop."""
    if st.session_state.processed_df is None:
        st.info("📤 Please upload and process a CSV file in the **Upload** tab first.")
        st.stop()


# ═══════════════════════════════════════════════════════════════
# TAB PREPROCESS: Preprocessing Report
# ═══════════════════════════════════════════════════════════════
with tab_preprocess:
    require_data()
    df = get_data()
    st.markdown("""
    <div class="section-header">
        <span style="font-size:24px">🧹</span>
        <h2 class="section-title">Data Preprocessing Report</h2>
        <span class="section-badge">QUALITY</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    total_reviews = len(df)
    low_qual_count = df['low_quality'].sum() if 'low_quality' in df.columns else 0
    avg_qual = df['review_quality_score'].mean() if 'review_quality_score' in df.columns else 0
    hinglish_count = (df['detected_language'] == 'HINGLISH').sum() if 'detected_language' in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Total Processed", total_reviews)
    col2.metric("⭐ Avg Quality Score", f"{avg_qual:.1f}/100")
    col3.metric("⚠️ Low Quality Flags", low_qual_count)
    col4.metric("🇮🇳 Hinglish Detected", hinglish_count)

    st.divider()

    # Layout: Lang Pie Chart | Quality Distrib
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🌐</span>
            <h2 class="section-title" style="font-size:16px;">Language Breakdown</h2>
        </div>
        """, unsafe_allow_html=True)
        if 'detected_language' in df.columns:
            lang_counts = df['detected_language'].value_counts().reset_index()
            lang_counts.columns = ['Language', 'Count']
            fig_lang = px.pie(lang_counts, values='Count', names='Language', hole=0.4,
                              color_discrete_sequence=['#3B82F6', '#8B5CF6', '#14B8A6', '#F97316', '#EAB308', '#22C55E'])
            fig_lang.update_layout(height=300)
            apply_chart_theme(fig_lang)
            st.plotly_chart(fig_lang, use_container_width=True)

    with c2:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">📊</span>
            <h2 class="section-title" style="font-size:16px;">Quality Score Distribution</h2>
        </div>
        """, unsafe_allow_html=True)
        if 'review_quality_score' in df.columns:
            fig_qual = px.histogram(df, x='review_quality_score', nbins=20,
                                    color_discrete_sequence=['#22C55E'])
            fig_qual.update_layout(height=300, xaxis_title="Quality Score", yaxis_title="Review Count")
            apply_chart_theme(fig_qual)
            st.plotly_chart(fig_qual, use_container_width=True)

    st.divider()

    # Before / After samples
    st.markdown("""
    <div class="section-header">
        <span style="font-size:18px">🔍</span>
        <h2 class="section-title" style="font-size:16px;">Processing Samples (Before & After)</h2>
    </div>
    """, unsafe_allow_html=True)
    sample_size = min(3, len(df))
    samples = df.sample(sample_size, random_state=42)
    
    for _, s_row in samples.iterrows():
        with st.container():
            st.markdown(f"**Original:** `{s_row.get('original_text', '')}`")
            st.markdown(f"**Processed:** `{s_row.get('processed_text', '')}`")
            tags = s_row.get('emoji_tags', [])
            lang = s_row.get('detected_language', 'EN')
            score = s_row.get('review_quality_score', 0)
            st.caption(f"Language: {lang} | Emoji Tags: {tags} | Score: {score}")
            st.markdown("---")

    # Low quality list
    if low_qual_count > 0:
        st.markdown(f"""
        <div class="section-header">
            <span style="font-size:18px">⚠️</span>
            <h2 class="section-title" style="font-size:16px;">Low Quality Reviews ({low_qual_count})</h2>
        </div>
        """, unsafe_allow_html=True)
        low_q_df = df[df['low_quality'] == True]
        st.dataframe(low_q_df[['original_text', 'review_quality_score', 'preprocessing_flags']], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2: Overview — Health metrics, sentiment breakdown, ratings, emotions
# ═══════════════════════════════════════════════════════════════

with tab_overview:
    require_data()
    df = get_data()
    st.markdown("""
    <div class="section-header">
        <span style="font-size:24px">📊</span>
        <h2 class="section-title">Health Overview</h2>
    </div>
    """, unsafe_allow_html=True)

    # -- Metrics row --
    total = len(df)
    has_rating = df['rating'].notna().sum() if 'rating' in df.columns else 0
    avg_rating = df['rating'].mean() if has_rating > 0 else 0
    if 'sentiment' not in df.columns:
        df['sentiment'] = 'neutral'
    negative_pct = (df['sentiment'] == 'negative').sum() / max(total, 1) * 100
    positive_pct = (df['sentiment'] == 'positive').sum() / max(total, 1) * 100
    suspicious = df['is_suspicious'].sum() if 'is_suspicious' in df.columns else 0
    products = df['product_name'].nunique()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("📝 Total Reviews", total)
    m2.metric("⭐ Avg Rating", f"{avg_rating:.1f}" if has_rating else "N/A")
    m3.metric("✅ Positive", f"{positive_pct:.0f}%")
    m4.metric("❌ Negative", f"{negative_pct:.0f}%")
    m5.metric("🚩 Suspicious", int(suspicious))
    m6.metric("📦 Products", products)

    # Rating drop alert — styled HTML
    if has_rating > 0 and avg_rating < 3.0:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3);
        border-left:4px solid #EF4444; border-radius:10px; padding:16px 20px;
        margin:16px 0; display:flex; align-items:center; gap:12px;">
            <span style="font-size:20px;">🚨</span>
            <span style="color:#FCA5A5; font-size:14px; font-weight:500;">
                ALERT: Average product rating is critically low ({avg_rating:.1f}★).
                Immediate action required to address customer dissatisfaction!
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif has_rating > 0 and avg_rating < 3.5:
        st.markdown(f"""
        <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.3);
        border-left:4px solid #EAB308; border-radius:10px; padding:16px 20px;
        margin:16px 0; display:flex; align-items:center; gap:12px;">
            <span style="font-size:20px;">⚠️</span>
            <span style="color:#FDE047; font-size:14px; font-weight:500;">
                Warning: Average product rating is declining ({avg_rating:.1f}★). Monitor closely.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif has_rating > 0 and negative_pct > 40:
        st.markdown(f"""
        <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.3);
        border-left:4px solid #EAB308; border-radius:10px; padding:16px 20px;
        margin:16px 0; display:flex; align-items:center; gap:12px;">
            <span style="font-size:20px;">⚠️</span>
            <span style="color:#FDE047; font-size:14px; font-weight:500;">
                Warning: High negative review rate ({negative_pct:.0f}%). Customer sentiment is trending down.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Per-product rating drop alerts
    if products > 1 and 'rating' in df.columns and df['rating'].notna().sum() > 0:
        product_ratings = df.groupby('product_name')['rating'].mean().sort_values()
        for prod, rating in product_ratings.items():
            if rating < 3.0:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2);
                border-left:3px solid #EF4444; border-radius:8px; padding:10px 16px; margin:8px 0;">
                    <span style="color:#FCA5A5; font-size:13px; font-weight:500;">
                        🚨 <strong>{prod}</strong> — Rating critically low at {rating:.1f}★!
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif rating < 3.5:
                st.markdown(f"""
                <div style="background:rgba(234,179,8,0.06); border:1px solid rgba(234,179,8,0.2);
                border-left:3px solid #EAB308; border-radius:8px; padding:10px 16px; margin:8px 0;">
                    <span style="color:#FDE047; font-size:13px; font-weight:500;">
                        ⚠️ <strong>{prod}</strong> — Rating declining at {rating:.1f}★
                    </span>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    col_sent, col_rating = st.columns(2)

    # -- Sentiment Distribution --
    with col_sent:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">💬</span>
            <h2 class="section-title" style="font-size:16px;">Sentiment Distribution</h2>
        </div>
        """, unsafe_allow_html=True)
        sent_counts = df['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        color_map = {'positive': '#22C55E', 'negative': '#EF4444', 'neutral': '#475569', 'mixed': '#EAB308', 'ambiguous': '#EAB308'}
        fig = px.pie(sent_counts, values='Count', names='Sentiment',
                     color='Sentiment', color_discrete_map=color_map,
                     hole=0.4)
        fig.update_layout(height=350)
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # -- Rating Distribution --
    with col_rating:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">⭐</span>
            <h2 class="section-title" style="font-size:16px;">Rating Distribution</h2>
        </div>
        """, unsafe_allow_html=True)
        if has_rating > 0:
            rating_counts = df['rating'].dropna().astype(int).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_color_map = {1: '#EF4444', 2: '#F97316', 3: '#EAB308', 4: '#84CC16', 5: '#22C55E'}
            bar_colors = [rating_color_map.get(r, '#94A3B8') for r in rating_counts['Rating']]
            fig2 = go.Figure(data=[go.Bar(
                x=rating_counts['Rating'], y=rating_counts['Count'],
                marker_color=bar_colors,
                text=rating_counts['Count'], textposition='auto',
            )])
            fig2.update_layout(height=350, showlegend=False,
                             xaxis_title="Rating", yaxis_title="Count")
            apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No rating data in uploaded CSV")

    # -- Per-product breakdown --
    if products > 1:
        st.divider()
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">📦</span>
            <h2 class="section-title" style="font-size:16px;">Per-Product Sentiment</h2>
        </div>
        """, unsafe_allow_html=True)
        product_sent = df.groupby(['product_name', 'sentiment']).size().reset_index(name='Count')
        fig3 = px.bar(product_sent, x='product_name', y='Count', color='sentiment',
                     color_discrete_map=color_map, barmode='group')
        fig3.update_layout(height=350, xaxis_title="Product", yaxis_title="Reviews")
        apply_chart_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    # SECTION 2 — Emotion Intensity Analysis
    st.divider()
    st.markdown("""
    <div class="section-header" style="margin-top:32px;">
        <span style="font-size:24px">🎭</span>
        <h2 class="section-title">Emotion Intensity Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    # Re-run emotion classification if crisis data was added
    review_dicts = df.to_dict('records')
    emotion_results = classify_emotion_batch(review_dicts)
    dist = get_emotion_distribution(emotion_results)

    col_dist, col_urgency = st.columns([2, 1])

    with col_dist:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">📊</span>
            <h2 class="section-title" style="font-size:16px;">Emotion Distribution</h2>
        </div>
        """, unsafe_allow_html=True)
        emotions = list(dist.keys())
        counts = list(dist.values())
        # Override emotion colors per design spec
        emotion_color_map = {
            'delighted': '#F59E0B', 'satisfied': '#22C55E', 'neutral': '#475569',
            'disappointed': '#EAB308', 'frustrated': '#F97316', 'angry': '#EF4444',
            'furious': '#7F1D1D'
        }
        colors = [emotion_color_map.get(e, EMOTION_TAXONOMY[e]["color"]) for e in emotions]
        labels = [EMOTION_TAXONOMY[e]["label"] for e in emotions]

        fig = go.Figure(data=[go.Bar(
            x=labels, y=counts,
            marker_color=colors,
            text=counts, textposition='auto',
        )])
        fig.update_layout(height=380, xaxis_title="Emotion", yaxis_title="Count", showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_urgency:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🚦</span>
            <h2 class="section-title" style="font-size:16px;">Urgency Summary</h2>
        </div>
        """, unsafe_allow_html=True)
        urgency_counts = {}
        for emo_res in emotion_results:
            urgency_counts[emo_res.urgency] = urgency_counts.get(emo_res.urgency, 0) + 1

        # Styled urgency cards
        urgency_styles = {
            'escalate': ('#EF4444', 'rgba(239,68,68,0.1)', 'rgba(239,68,68,0.3)', '#FCA5A5'),
            'critical': ('#EF4444', 'rgba(239,68,68,0.1)', 'rgba(239,68,68,0.3)', '#FCA5A5'),
            'high': ('#F97316', 'rgba(249,115,22,0.1)', 'rgba(249,115,22,0.3)', '#FDBA74'),
            'medium': ('#3B82F6', 'rgba(59,130,246,0.1)', 'rgba(59,130,246,0.3)', '#93C5FD'),
            'low': ('#94A3B8', 'rgba(148,163,184,0.1)', 'rgba(148,163,184,0.3)', '#CBD5E1'),
            'none': ('#22C55E', 'rgba(34,197,94,0.1)', 'rgba(34,197,94,0.3)', '#86EFAC'),
        }
        
        urgency_html = '<div style="display:flex; flex-direction:column; gap:8px;">'
        for level in ['escalate', 'critical', 'high', 'medium', 'low', 'none']:
            count = urgency_counts.get(level, 0)
            if count > 0:
                accent, bg, border, text_color = urgency_styles.get(level, ('#94A3B8', 'rgba(148,163,184,0.1)', 'rgba(148,163,184,0.3)', '#CBD5E1'))
                urgency_html += f"""
                <div style="background:{bg}; border:1px solid {border};
                border-radius:10px; padding:12px 16px;">
                    <div style="color:{text_color}; font-size:11px; font-weight:600;
                    text-transform:uppercase; letter-spacing:1px;">{level.upper()}</div>
                    <div style="color:#F1F5F9; font-size:24px; font-weight:700;
                    margin-top:2px;">{count}</div>
                    <div style="color:#475569; font-size:11px;">reviews</div>
                </div>"""
        urgency_html += '</div>'
        st.markdown(urgency_html, unsafe_allow_html=True)

        # Key insight
        escalate_critical = urgency_counts.get('escalate', 0) + urgency_counts.get('critical', 0)
        if escalate_critical > 0:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3);
            border-radius:8px; padding:12px; margin-top:12px; text-align:center;">
                <span style="color:#FCA5A5; font-size:13px; font-weight:600;">
                    ⚠️ {escalate_critical} reviews need immediate attention!
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.3);
            border-radius:8px; padding:12px; margin-top:12px; text-align:center;">
                <span style="color:#86EFAC; font-size:13px; font-weight:600;">
                    ✅ No critical urgency issues detected
                </span>
            </div>
            """, unsafe_allow_html=True)

    # -- Emotion Timeline --
    st.divider()
    st.markdown("""
    <div class="section-header">
        <span style="font-size:18px">📈</span>
        <h2 class="section-title" style="font-size:16px;">Emotion Timeline</h2>
    </div>
    """, unsafe_allow_html=True)
    timeline = get_emotion_timeline(emotion_results, review_dicts, window_size=max(len(review_dicts)//6, 10))
    if timeline:
        timeline_df = pd.DataFrame(timeline)
        emotion_cols = [e for e in EMOTION_TAXONOMY.keys() if e in timeline_df.columns]
        if emotion_cols:
            fig2 = go.Figure()
            for emotion in emotion_cols:
                e_color = emotion_color_map.get(emotion, EMOTION_TAXONOMY[emotion]["color"])
                fig2.add_trace(go.Bar(
                    name=EMOTION_TAXONOMY[emotion]["label"],
                    x=timeline_df["window"],
                    y=timeline_df[emotion],
                    marker_color=e_color,
                ))
            fig2.update_layout(
                barmode='stack', height=350,
                xaxis_title="Window", yaxis_title="Reviews",
            )
            apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: Trends — Complaint rate over time, top negative features
# ═══════════════════════════════════════════════════════════════

with tab_trends:
    require_data()
    df = get_data()
    st.markdown("""
    <div class="section-header">
        <span style="font-size:24px">📈</span>
        <h2 class="section-title">Trend Analysis</h2>
        <span class="section-badge">SLIDING WINDOW</span>
    </div>
    """, unsafe_allow_html=True)

    # LEFT COLUMN — Controls (30%)
    col_controls, col_charts = st.columns([3, 7])

    with col_controls:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">⚙️</span>
            <h2 class="section-title" style="font-size:16px;">Controls</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature selection
        if 'sentiment' not in df.columns:
            df['sentiment'] = 'neutral'
        neg_df = df[df['sentiment'] == 'negative']
        
        # Extract features from reviews
        feature_keywords = ['packaging', 'battery', 'delivery', 'quality', 'price', 'service', 'cap', 'leak']
        detected_features = set()
        for _, row in neg_df.iterrows():
            text = str(row.get('clean_text', row.get('review_text', ''))).lower()
            for kw in feature_keywords:
                if kw in text:
                    detected_features.add(kw)
        
        feature_list = list(detected_features) if detected_features else ['Overall']
        selected_feature = st.selectbox("Select Feature", feature_list, key="trend_feature")
        
        # Window size with auto-adjustment
        default_window = 30
        if len(df) < default_window * 2:
            adjusted_window = max(10, len(df) // 3)
            st.info(f"ℹ️ Window size auto-adjusted to {adjusted_window} to show meaningful trends")
            default_window = adjusted_window
        
        window_size = st.slider("Window Size", 10, 100, default_window, 10, key="trend_window")
        
        # Alert threshold
        alert_threshold = st.selectbox("Alert Threshold %", [15, 20, 25, 30], index=0, key="trend_threshold")
        
        # Toggles
        show_baseline = st.checkbox("Show Baseline", value=True, key="trend_baseline")
        show_anomaly = st.checkbox("Show Anomaly Markers", value=True, key="trend_anomaly")
        
        # Current window rate metric — styled card
        if len(df) >= window_size:
            last_window = df.iloc[-window_size:]
            current_rate = (last_window['sentiment'] == 'negative').sum() / len(last_window) * 100
            rate_color = "#22C55E" if current_rate < alert_threshold else "#EF4444"
            st.markdown(f"""
            <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;padding:20px;text-align:center;'>
            <p style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin:0 0 4px 0;'>Current Window Rate</p>
            <p style='font-size:2.5rem;font-weight:bold;color:{rate_color};margin:0;'>{current_rate:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    with col_charts:
        # Compute windows
        windows = []
        for i in range(0, len(df), window_size):
            window_df = df.iloc[i:i + window_size]
            neg_count = (window_df['sentiment'] == 'negative').sum()
            pos_count = (window_df['sentiment'] == 'positive').sum()
            complaint_rate = neg_count / len(window_df) if len(window_df) > 0 else 0
            windows.append({
                'Window': f"W{i // window_size + 1}",
                'Reviews': len(window_df),
                'Negative': neg_count,
                'Positive': pos_count,
                'Rate%': round(complaint_rate * 100, 1),
            })
        
        if windows:
            windows_df = pd.DataFrame(windows)
            
            # CHART 1 — Complaint Rate Over Time
            st.markdown(f"""
            <div class="section-header">
                <span style="font-size:18px">📉</span>
                <h2 class="section-title" style="font-size:16px;">Complaint Rate Trend — {selected_feature}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Color markers based on rate value
            colors = ['#22C55E' if rate < 15 else '#EAB308' if rate <= 25 else '#EF4444' 
                     for rate in windows_df['Rate%']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=windows_df['Window'],
                y=windows_df['Rate%'],
                mode='lines+markers',
                marker=dict(color=colors, size=8),
                line=dict(color='#94A3B8', width=3),
                fill='tozeroy',
                fillcolor='rgba(239,68,68,0.05)',
                hovertemplate="Window: %{x}<br>Rate: %{y:.1f}%<br>Reviews: %{customdata[0]}<extra></extra>",
                customdata=windows_df['Reviews']
            ))
            
            # Alert threshold line
            fig.add_hline(y=alert_threshold, line_dash="dash", line_color="#EF4444", line_width=1.5,
                        annotation_text=f"Alert Threshold ({alert_threshold}%)",
                        annotation_font_color="#EF4444")
            
            # Baseline line
            if show_baseline and len(windows_df) > 0:
                baseline = windows_df['Rate%'].mean()
                fig.add_hline(y=baseline, line_dash="dot", line_color="#475569", line_width=1,
                            annotation_text=f"Baseline ({baseline:.1f}%)",
                            annotation_font_color="#475569")
            
            # Anomaly markers
            if show_anomaly and len(windows_df) > 0:
                for idx, row in windows_df.iterrows():
                    if row['Rate%'] > alert_threshold:
                        fig.add_trace(go.Scatter(
                            x=[row['Window']], y=[row['Rate%']],
                            mode='markers',
                            marker=dict(symbol='star', size=14, color='#EF4444',
                                       line=dict(color='white', width=1)),
                            showlegend=False,
                            hovertemplate=f"⚠️ Anomaly: {row['Rate%']:.1f}%<extra></extra>"
                        ))
            
            fig.update_layout(height=400, yaxis_title="Complaint Rate (%)",
                            xaxis_title="Window", yaxis_range=[0, 100])
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # CHART 2 — Window Breakdown Bar Chart
            st.markdown("""
            <div class="section-header">
                <span style="font-size:18px">📊</span>
                <h2 class="section-title" style="font-size:16px;">Review Volume per Window</h2>
            </div>
            """, unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name='Negative',
                x=windows_df['Window'],
                y=windows_df['Negative'],
                marker_color='#EF4444'
            ))
            fig2.add_trace(go.Bar(
                name='Positive',
                x=windows_df['Window'],
                y=windows_df['Positive'],
                marker_color='#22C55E'
            ))
            fig2.update_layout(barmode='stack', height=300,
                            xaxis_title="Window", yaxis_title="Review Count")
            apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            
            # ALERT BOX — styled HTML
            st.divider()
            max_rate = windows_df['Rate%'].max()
            max_window = windows_df.loc[windows_df['Rate%'].idxmax(), 'Window']
            baseline = windows_df['Rate%'].mean()
            
            if max_rate > alert_threshold:
                ratio = max_rate / baseline if baseline > 0 else 0
                st.markdown(f"""
                <div class="reviewiq-card-critical">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="font-size:20px;">🚨</span>
                        <span style="color:#FCA5A5; font-weight:700; font-size:15px;">TREND ALERT</span>
                        <span class="badge-critical">Z-SCORE ANOMALY</span>
                    </div>
                    <p style="color:#94A3B8; margin:0; font-size:14px; line-height:1.6;">
                        {selected_feature} complaint rate reached <strong style="color:#FCA5A5;">{max_rate}%</strong>
                        in {max_window} — <strong style="color:#FCA5A5;">{ratio:.1f}×</strong> above baseline.
                        Possible systemic issue detected.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="reviewiq-card-success">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:20px;">✅</span>
                        <span style="color:#86EFAC; font-weight:600; font-size:14px;">
                            No significant trend anomalies detected for {selected_feature}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # TREND SUMMARY TABLE
            st.markdown("""
            <div class="section-header" style="margin-top:20px;">
                <span style="font-size:18px">📋</span>
                <h2 class="section-title" style="font-size:16px;">Trend Summary Table</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Add Status column
            def get_status(rate):
                if rate > alert_threshold:
                    return "🔴 ALERT"
                elif rate >= alert_threshold - 5:
                    return "🟡 WATCH"
                else:
                    return "🟢 OK"
            
            windows_df['Status'] = windows_df['Rate%'].apply(get_status)
            
            # Style the dataframe
            st.dataframe(
                windows_df[['Window', 'Reviews', 'Negative', 'Positive', 'Rate%', 'Status']],
                hide_index=True,
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════
# TAB BOT: Bot Detection
# ═══════════════════════════════════════════════════════════════

with tab_bot:
    if st.session_state.get('bot_result') is None:
        st.info("Run the Unified Pipeline in the Upload tab to populate this section.")
    else:
        df = st.session_state.get('deduped_df')
        st.markdown("""
        <div class="section-header">
            <span style="font-size:24px">🕸️</span>
            <h2 class="section-title">Coordinated Bot Campaign Detection</h2>
            <span class="section-badge">NETWORK ANALYSIS</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p style="color:#475569; font-size:13px; margin-top:-12px; margin-bottom:20px;">Advanced deduplication, semantic signaling, and network-based clustering.</p>', unsafe_allow_html=True)
        res = st.session_state.bot_result

        # SECTION 1 — SUMMARY STATS ROW
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 Total Scanned", res.total_reviews_scanned)
        m2.metric("🤖 Confirmed Bots Removed", res.confirmed_bots_removed)
        m3.metric("🚩 Suspicious Flagged", res.suspicious_flagged)
        m4.metric("🚨 Campaigns Detected", res.campaigns_detected)

        st.divider()

        # SECTION 2 — INDIVIDUAL SUSPICION TABLE
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🔍</span>
            <h2 class="section-title" style="font-size:16px;">Isolated Review Suspicion Board</h2>
        </div>
        """, unsafe_allow_html=True)
        filter_val = st.radio("Filter By Status", ["Show All", "CONFIRMED_BOT", "HIGH_SUSPICION", "LOW_SUSPICION", "TRUSTED"], horizontal=True)
        
        indiv_data = []
        for ind in res.individuals:
            if filter_val == "Show All" or ind.status == filter_val:
                indiv_data.append({
                    "Reviewer": ind.username,
                    "Snippet": ind.text_snippet,
                    "Bot Score": ind.score,
                    "Signals": ", ".join(ind.signals_triggered),
                    "Status": ind.status
                })
                
        if indiv_data:
            st.dataframe(pd.DataFrame(indiv_data), use_container_width=True, hide_index=True)
        else:
            st.info("No reviews match this filter.")

        st.divider()

        # SECTION 4 — CLEAN DATASET STATS
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🧹</span>
            <h2 class="section-title" style="font-size:16px;">Data Cleansing Summary</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="reviewiq-card-info">
            <p style="color:#94A3B8; font-size:14px; line-height:1.8; margin:0;">
                <strong style="color:#F1F5F9;">After executing multi-vector deduplication and bot removal:</strong><br>
                • Original Dataset: <strong style="color:#93C5FD;">{res.total_reviews_scanned}</strong> reviews<br>
                • Duplicates Filtered: <strong style="color:#93C5FD;">{res.exact_duplicates_removed}</strong> exact, <strong style="color:#93C5FD;">{res.near_duplicates_clustered}</strong> near-duplicates<br>
                • Bots Filtered: <strong style="color:#93C5FD;">{res.confirmed_bots_removed}</strong> explicit bots<br>
                <br>
                ➡ <strong style="color:#F1F5F9;">Working Clean Dataset Size: {res.clean_dataset_count} reviews.</strong><br>
                <em style="color:#475569;">All secondary analysis properties utilize strictly the cleansed counts.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5: Action Brief — Smart recommendations for teams
# ═══════════════════════════════════════════════════════════════

with tab_action:
    require_data()
    df = get_data()
    st.markdown("""
    <div class="section-header">
        <span style="font-size:24px">📋</span>
        <h2 class="section-title">Smart Action Brief</h2>
        <span class="section-badge">TEAM-READY</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Decision Mode Toggle
    decision_mode = st.toggle("⚡ Decision Mode — Plain English Only", value=False, key="decision_mode")
    
    if not decision_mode:
        # SECTION A — Issue Priority Queue
        st.divider()
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🔴</span>
            <h2 class="section-title" style="font-size:16px;">Issue Priority Queue</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Compute complaint rate per feature
        if 'sentiment' not in df.columns:
            df['sentiment'] = 'neutral'
        neg_df = df[df['sentiment'] == 'negative']
        
        # Extract features from negative reviews (simple keyword extraction)
        feature_keywords = ['packaging', 'battery', 'delivery', 'quality', 'price', 'service', 'cap', 'leak', 'defect', 'broken', 'damaged']
        feature_counts = {}
        for _, row in neg_df.iterrows():
            text = str(row.get('clean_text', row.get('review_text', ''))).lower()
            for kw in feature_keywords:
                if kw in text:
                    feature_counts[kw] = feature_counts.get(kw, 0) + 1
        
        # Calculate complaint rates
        total_reviews = len(df)
        feature_rates = {}
        for feature, count in feature_counts.items():
            rate = (count / total_reviews) * 100
            feature_rates[feature] = rate
        
        # Sort by rate and get top 3
        top_features = sorted(feature_rates.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if not top_features:
            st.info("No significant complaint patterns detected.")
        else:
            for i, (feature, rate) in enumerate(top_features, 1):
                # Determine styling
                if rate > 25:
                    status = "SYSTEMIC"
                    rate_color = "#FCA5A5"
                    card_class = "critical"
                    priority_label = "CRITICAL"
                elif rate > 15:
                    status = "EMERGING"
                    rate_color = "#FDBA74"
                    card_class = "warning"
                    priority_label = "HIGH"
                else:
                    status = "MONITOR"
                    rate_color = "#93C5FD"
                    card_class = "info"
                    priority_label = "MEDIUM"
                
                escalation_score = int(min(rate * 2.5, 100))
                
                # Get sample review evidence
                sample_reviews = neg_df[neg_df['clean_text'].str.contains(feature, case=False, na=False)]['review_text'].head(1)
                evidence = sample_reviews.iloc[0][:80] + "..." if len(sample_reviews) > 0 else "No sample available"
                
                badge_class = "badge-critical" if escalation_score > 75 else "badge-warning" if escalation_score > 50 else "badge-info"
                
                st.markdown(f"""
                <div class="reviewiq-card-{card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div style="font-size:11px; font-weight:700; letter-spacing:1px;
                            color:#475569; text-transform:uppercase; margin-bottom:4px;">
                                #{i} PRIORITY</div>
                            <div style="font-size:18px; font-weight:700; color:#F1F5F9;">
                                {feature.upper()}</div>
                            <div style="font-size:13px; color:#94A3B8; margin-top:4px;">
                                Complaint Rate: <strong style="color:{rate_color};">{rate:.1f}%</strong>
                                &nbsp;|&nbsp; Lifecycle: <strong style="color:#F1F5F9;">{status}</strong>
                                &nbsp;|&nbsp; Score: <strong style="color:#F1F5F9;">{escalation_score}/100</strong>
                            </div>
                        </div>
                        <span class="{badge_class}">
                            {priority_label}
                        </span>
                    </div>
                    <div style="margin-top:12px; padding:10px 14px; background:rgba(0,0,0,0.2);
                    border-radius:8px; font-size:13px; color:#94A3B8; font-style:italic;">
                        "{evidence}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # SECTION B — Team Action Cards
        st.divider()
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🎯</span>
            <h2 class="section-title" style="font-size:16px;">Team Action Cards</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Card 1 — Quality Team
        with col1:
            if top_features:
                top_feature = top_features[0][0]
                if top_feature in ['packaging', 'cap', 'leak']:
                    quality_action = "Audit cap/seal on production line immediately"
                elif top_feature in ['battery', 'defect', 'broken']:
                    quality_action = "Check firmware version and battery supplier batch"
                elif top_feature in ['delivery', 'service']:
                    quality_action = "Review fulfillment SLA with logistics partner"
                else:
                    quality_action = f"Investigate top complaint feature: {top_feature}"
            else:
                quality_action = "Review quality assurance processes"
            
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #EF4444;
            border-top: 3px solid #EF4444; border-radius:12px; padding:20px; height:100%;">
                <div style="font-size:15px; font-weight:700; color:#F1F5F9;
                margin-bottom:12px;">🔧 Quality Team</div>
                <ul style="color:#94A3B8; font-size:13px; line-height:2;
                padding-left:16px; margin:0;">
                    <li>{quality_action}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Card 2 — Marketing Team
        with col2:
            if top_features:
                top_feature = top_features[0][0]
                if 'packaging' in top_feature:
                    marketing_action = "Pause all packaging-focused ad campaigns"
                else:
                    marketing_action = "Avoid promoting features currently under complaint"
            else:
                marketing_action = "Review current marketing messaging"
            
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #F97316;
            border-top: 3px solid #F97316; border-radius:12px; padding:20px; height:100%;">
                <div style="font-size:15px; font-weight:700; color:#F1F5F9;
                margin-bottom:12px;">📣 Marketing Team</div>
                <ul style="color:#94A3B8; font-size:13px; line-height:2;
                padding-left:16px; margin:0;">
                    <li>{marketing_action}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Card 3 — Do NOT Do
        with col3:
            do_not_do_items = [
                "Do not close complaints without root cause confirmation",
                "Do not run discount campaigns while defect is unresolved"
            ]
            if st.session_state.bot_result and len(st.session_state.bot_result.clusters) > 0:
                do_not_do_items.append("Do not respond to flagged bot reviews publicly")
            
            action_items_html = "".join([f"<li>{item}</li>" for item in do_not_do_items])
            
            st.markdown(f"""
            <div style="background:rgba(127,29,29,0.08); border:1px solid #7F1D1D;
            border-top: 3px solid #7F1D1D; border-radius:12px; padding:20px; height:100%;">
                <div style="font-size:15px; font-weight:700; color:#F1F5F9;
                margin-bottom:12px;">⛔ Do NOT Do</div>
                <ul style="color:#94A3B8; font-size:13px; line-height:2;
                padding-left:16px; margin:0;">
                    {action_items_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # SECTION C — Downloadable Text Brief
        st.divider()
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">📥</span>
            <h2 class="section-title" style="font-size:16px;">Download Action Brief</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate text brief content
        brief_lines = []
        brief_lines.append("=" * 60)
        brief_lines.append("REVIEWIQ ACTION BRIEF")
        brief_lines.append("=" * 60)
        brief_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        brief_lines.append(f"Total Reviews: {len(df)}")
        brief_lines.append(f"Product: {df['product_name'].iloc[0] if 'product_name' in df.columns else 'N/A'}")
        brief_lines.append("")
        brief_lines.append("TOP 3 ISSUES:")
        if top_features:
            for i, (feature, rate) in enumerate(top_features, 1):
                brief_lines.append(f"{i}. {feature.upper()} — Complaint Rate: {rate:.1f}%")
        else:
            brief_lines.append("No significant issues detected")
        brief_lines.append("")
        brief_lines.append("TEAM ACTIONS:")
        brief_lines.append(f"Quality Team: {quality_action}")
        brief_lines.append(f"Marketing Team: {marketing_action}")
        brief_lines.append("")
        brief_lines.append("DO NOT DO:")
        for item in do_not_do_items:
            brief_lines.append(f"- {item}")
        brief_lines.append("")
        brief_lines.append("=" * 60)
        
        brief_text = "\n".join(brief_lines)
        
        st.download_button(
            label="📥 Download Action Brief",
            data=brief_text.encode('utf-8'),
            file_name=f"action_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            type="primary"
        )
    else:
        # Decision Mode — Simple 3-card summary
        st.divider()
        col_stop, col_watch, col_amplify = st.columns(3)
        
        with col_stop:
            st.markdown("""
            <div style='background:rgba(239,68,68,0.1);border:2px solid #EF4444;border-radius:16px;padding:30px;text-align:center;'>
            <h2 style='color:#EF4444;font-size:2.5rem;margin:0;'>🛑 STOP</h2>
            <p style='font-size:1.1rem;color:#94A3B8;margin:12px 0 0 0;'>Pause campaigns<br>Investigate quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_watch:
            st.markdown("""
            <div style='background:rgba(249,115,22,0.1);border:2px solid #F97316;border-radius:16px;padding:30px;text-align:center;'>
            <h2 style='color:#F97316;font-size:2.5rem;margin:0;'>👁️ WATCH</h2>
            <p style='font-size:1.1rem;color:#94A3B8;margin:12px 0 0 0;'>Monitor trends<br>Review feedback</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_amplify:
            st.markdown("""
            <div style='background:rgba(34,197,94,0.1);border:2px solid #22C55E;border-radius:16px;padding:30px;text-align:center;'>
            <h2 style='color:#22C55E;font-size:2.5rem;margin:0;'>📢 AMPLIFY</h2>
            <p style='font-size:1.1rem;color:#94A3B8;margin:12px 0 0 0;'>Promote positives<br>Feature strengths</p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 6: Crisis Simulator — Inject synthetic crises
# ═══════════════════════════════════════════════════════════════

with tab_crisis:
    require_data()
    df_base = st.session_state.processed_df
    
    # HEADER — styled crisis header
    st.markdown("""
    <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2);
    border-left:4px solid #EF4444; border-radius:12px; padding:20px 24px; margin-bottom:24px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:28px;">☢️</span>
            <div>
                <h2 style="margin:0; font-size:22px; font-weight:700; color:#F1F5F9;">
                    Crisis Simulator</h2>
                <p style="margin:4px 0 0; font-size:13px; color:#94A3B8;">
                    Inject synthetic crisis scenarios to test real-time detection capability
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.3);
    border-left:3px solid #EAB308; border-radius:8px; padding:10px 16px; margin-bottom:16px;">
        <span style="color:#FDE047; font-size:13px;">⚠️ <strong>Demo Mode</strong> — Injecting synthetic data to simulate real-world crisis events</span>
    </div>
    """, unsafe_allow_html=True)
    
    # SECTION 1 — Product Selector + Crisis Type Selector
    col_product, col_crisis = st.columns(2)
    
    with col_product:
        crisis_product = st.selectbox("Target Product",
                                      df_base['product_name'].unique().tolist(),
                                      key="crisis_product")
    
    with col_crisis:
        crisis_type = st.radio("Crisis Scenario",
                               ["☢️ Packaging Crisis — 50 reviews: cap leakage complaints spike",
                                "💀 Quality Disaster — 30 reviews: furious-intensity defect reports",
                                "🤖 Bot Attack — 25 coordinated 1-star suspicious accounts"],
                               key="crisis_type")
    
    # Determine crisis key from selection
    if "Packaging" in crisis_type:
        crisis_key = "packaging_crisis"
    elif "Quality" in crisis_type:
        crisis_key = "quality_disaster"
    else:
        crisis_key = "bot_attack"
    
    # SECTION 2 — Crisis Preview Panel
    st.divider()
    with st.expander("👁️ Preview Crisis Batch (first 5 reviews)"):
        # Generate preview reviews without actually injecting
        review_dicts = df_base.to_dict('records')
        result = inject_crisis(review_dicts, crisis_key, product_name=crisis_product)
        preview_df = pd.DataFrame(result.injected_reviews[:5])
        available_cols = [c for c in ['rating', 'review_text', 'timestamp', 'username'] if c in preview_df.columns]
        st.dataframe(preview_df[available_cols], hide_index=True, use_container_width=True)
        st.caption(f"These reviews will be appended to your current {len(df_base)} reviews")
    
    # SECTION 3 — Inject Button + Crisis type cards
    st.divider()
    
    # Styled crisis selector cards
    if "Packaging" in crisis_type:
        st.markdown("""
        <div style="background:rgba(249,115,22,0.06); border:1px solid rgba(249,115,22,0.25);
        border-radius:10px; padding:16px; margin-bottom:8px; text-align:center;">
            <div style="font-size:13px; color:#94A3B8; margin-bottom:4px;">
                ☢️ <strong style="color:#FDBA74;">Packaging Crisis</strong> — 50 reviews<br>
                <span style="font-size:12px;">Cap leakage complaints spike in latest batch</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif "Quality" in crisis_type:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.25);
        border-radius:10px; padding:16px; margin-bottom:8px; text-align:center;">
            <div style="font-size:13px; color:#94A3B8; margin-bottom:4px;">
                💀 <strong style="color:#FCA5A5;">Quality Disaster</strong> — 30 reviews<br>
                <span style="font-size:12px;">Furious-intensity defect reports flooding in</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(139,92,246,0.06); border:1px solid rgba(139,92,246,0.25);
        border-radius:10px; padding:16px; margin-bottom:8px; text-align:center;">
            <div style="font-size:13px; color:#94A3B8; margin-bottom:4px;">
                🤖 <strong style="color:#C4B5FD;">Bot Attack</strong> — 25 reviews<br>
                <span style="font-size:12px;">Coordinated 1-star suspicious accounts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button(f"☢️ INJECT {crisis_type.split('—')[0].strip()} INTO DATASET", key="inject_crisis", type="primary", width='stretch'):
        with st.spinner("Injecting crisis..."):
            review_dicts = df_base.to_dict('records')
            result = inject_crisis(review_dicts, crisis_key, product_name=crisis_product)
            
            # Convert to DataFrame
            crisis_df = pd.DataFrame(result.injected_reviews)
            # Run preprocessing on crisis reviews
            crisis_df = run_sentiment(crisis_df)
            crisis_emotion = classify_emotion_batch(crisis_df.to_dict('records'))
            crisis_df['emotion'] = [e.emotion for e in crisis_emotion]
            crisis_df['emotion_label'] = [e.emotion_label for e in crisis_emotion]
            crisis_df['urgency'] = [e.urgency for e in crisis_emotion]
            
            st.session_state.crisis_reviews_df = crisis_df
            st.session_state.crisis_active = True
            st.session_state.crisis_result = result
            st.session_state.bot_result = None
            
            # Store old metrics for delta calculation
            old_total = len(df_base)
            old_rate = (df_base['sentiment'] == 'negative').sum() / len(df_base) * 100 if 'sentiment' in df_base.columns else 0
            old_trust = df_base['trust_score'].mean() if 'trust_score' in df_base.columns else 80
        
        st.rerun()
    
    # SECTION 3 (continued) — Real-time Metrics after injection
    if st.session_state.crisis_active and st.session_state.crisis_result:
        cr = st.session_state.crisis_result
        df_combined = get_data()
        
        # Calculate new metrics
        new_total = len(df_combined)
        new_rate = (df_combined['sentiment'] == 'negative').sum() / max(len(df_combined), 1) * 100 if 'sentiment' in df_combined.columns else 0
        new_trust = df_combined['trust_score'].mean() if 'trust_score' in df_combined.columns else 80
        
        # Calculate old metrics
        old_total = len(df_base)
        old_rate = (df_base['sentiment'] == 'negative').sum() / max(len(df_base), 1) * 100 if 'sentiment' in df_base.columns else 0
        old_trust = df_base['trust_score'].mean() if 'trust_score' in df_base.columns else 80
        
        # Show 4 animated metrics with delta
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📝 Total Reviews", new_total, delta=cr.reviews_injected)
        m2.metric("❌ Complaint Rate", f"{new_rate:.0f}%", delta=f"{new_rate - old_rate:.0f}%", delta_color="inverse")
        m3.metric("🛡️ Avg Trust Score", f"{new_trust:.0f}", delta=f"{new_trust - old_trust:.0f}")
        m4.metric("🚨 Alert Level", "CRITICAL", delta="was NORMAL")
        
        # Big red banner — styled
        crisis_name = cr.crisis_label if hasattr(cr, 'crisis_label') else crisis_type.split('—')[0].strip()
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.1); border:2px solid rgba(239,68,68,0.4);
        border-radius:12px; padding:24px; margin:20px 0; text-align:center;">
            <div style="font-size:32px; margin-bottom:8px;">🚨</div>
            <div style="font-size:20px; font-weight:700; color:#FCA5A5; margin-bottom:8px;">
                CRISIS DETECTED</div>
            <div style="font-size:14px; color:#94A3B8; line-height:1.6;">
                {crisis_name} injected.<br>
                Complaint rate jumped from
                <strong style="color:#F1F5F9;">{old_rate:.0f}%</strong> to
                <strong style="color:#EF4444;">{new_rate:.0f}%</strong>.
                Immediate action recommended.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # SECTION 4 — Post-Injection Analysis
        st.divider()
        st.markdown("""
        <div class="section-header">
            <span style="font-size:18px">🔬</span>
            <h2 class="section-title" style="font-size:16px;">Post-Injection Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Mini Chart 1: Trend line showing spike
        col_mini1, col_mini2, col_mini3 = st.columns(3)
        
        with col_mini1:
            st.markdown("""
            <div class="section-header">
                <h2 class="section-title" style="font-size:14px;">Trend Spike</h2>
            </div>
            """, unsafe_allow_html=True)
            # Simple before/after comparison
            fig_mini1 = go.Figure()
            fig_mini1.add_trace(go.Scatter(
                x=['Before', 'After'],
                y=[old_rate, new_rate],
                mode='lines+markers',
                marker=dict(size=10, color=['#22C55E', '#EF4444']),
                line=dict(color='#EF4444', width=3)
            ))
            fig_mini1.update_layout(height=200, yaxis_title="Complaint Rate (%)")
            apply_chart_theme(fig_mini1)
            st.plotly_chart(fig_mini1, use_container_width=True)
        
        with col_mini2:
            st.markdown("""
            <div class="section-header">
                <h2 class="section-title" style="font-size:14px;">Emotion Distribution</h2>
            </div>
            """, unsafe_allow_html=True)
            if 'emotion_label' in df_combined.columns:
                emotion_counts = df_combined['emotion_label'].value_counts()
                fig_mini2 = px.bar(x=emotion_counts.index, y=emotion_counts.values, color=emotion_counts.values,
                                color_continuous_scale=['#22C55E', '#EAB308', '#EF4444'])
                fig_mini2.update_layout(height=200)
                apply_chart_theme(fig_mini2)
                st.plotly_chart(fig_mini2, use_container_width=True)
        
        with col_mini3:
            st.markdown("""
            <div class="section-header">
                <h2 class="section-title" style="font-size:14px;">Top Complaint Features</h2>
            </div>
            """, unsafe_allow_html=True)
            neg_df = df_combined[df_combined['sentiment'] == 'negative']
            if len(neg_df) > 0:
                feature_keywords = ['packaging', 'battery', 'delivery', 'quality', 'defect']
                feature_counts = {}
                for _, row in neg_df.iterrows():
                    text = str(row.get('clean_text', row.get('review_text', ''))).lower()
                    for kw in feature_keywords:
                        if kw in text:
                            feature_counts[kw] = feature_counts.get(kw, 0) + 1
                if feature_counts:
                    feat_df = pd.DataFrame(list(feature_counts.items()), columns=['Feature', 'Count'])
                    fig_mini3 = px.bar(feat_df, x='Count', y='Feature', orientation='h',
                                    color='Count', color_continuous_scale=['#F97316', '#EF4444'])
                    fig_mini3.update_layout(height=200)
                    apply_chart_theme(fig_mini3)
                    st.plotly_chart(fig_mini3, use_container_width=True)
        
        # SECTION 5 — Crisis Resolution Guide
        st.divider()
        with st.expander("🛡️ Recommended Crisis Response Protocol"):
            if "Packaging" in crisis_type:
                st.markdown("""
                **Packaging Crisis Protocol:**
                1. Immediately halt shipments from affected batch
                2. Notify QC team within 2 hours
                3. Prepare replacement/refund SOP for support team
                4. Pause all packaging-related marketing campaigns
                5. Post public acknowledgment within 24 hours
                """)
            elif "Quality" in crisis_type:
                st.markdown("""
                **Quality Disaster Protocol:**
                1. Escalate to product engineering within 1 hour
                2. Pull affected SKU from marketplace listing temporarily
                3. Activate escalated refund protocol
                4. Conduct root cause analysis within 48 hours
                """)
            else:
                st.markdown("""
                **Bot Attack Protocol:**
                1. Report suspicious accounts to marketplace platform
                2. Do NOT respond to flagged bot reviews publicly
                3. Flag campaign evidence for legal team
                4. Monitor for new account registrations in next 72 hours
                """)
        
        # SECTION 6 — Reset Button
        st.divider()
        if st.button("🔄 Reset to Original Dataset", key="reset_crisis", type="secondary"):
            st.session_state.crisis_active = False
            st.session_state.crisis_result = None
            st.session_state.crisis_reviews_df = None
            st.session_state.bot_result = None
            st.success("✅ Dataset reset to original state")
            st.rerun()
    else:
        st.info("Select a crisis type above and click to inject it into the current dataset.")


# ═══════════════════════════════════════════════════════════════
# TAB BATCH: Batch Analysis
# ═══════════════════════════════════════════════════════════════

with tab_batch:
    batches_data = st.session_state.get('batches')
    if batches_data is None or (isinstance(batches_data, dict) and len(batches_data.get('batches', [])) == 0) or (isinstance(batches_data, list) and len(batches_data) == 0):
        st.info("Run the Unified Pipeline in the Upload tab to populate this section.")
    else:
        df = st.session_state.get('deduped_df')
        st.markdown("""
        <div class="section-header">
            <span style="font-size:24px">📦</span>
            <h2 class="section-title">Batch Analysis & Degradation Detection</h2>
            <span class="section-badge">TEMPORAL</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p style="color:#475569; font-size:13px; margin-top:-12px; margin-bottom:20px;">Track post-purchase degradation by splitting reviews into consecutive batches.</p>', unsafe_allow_html=True)
        res = batches_data
        batches = res.get("batches", []) if isinstance(res, dict) else []
        if isinstance(res, dict) and res.get("degradation_detected"):
            st.markdown(f"""
            <div class="reviewiq-card-critical">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:20px;">🚨</span>
                    <div>
                        <span style="color:#FCA5A5; font-weight:700; font-size:15px;">Post-Purchase Degradation Detected</span><br>
                        <span style="color:#94A3B8; font-size:13px;">{res.get('degradation_reason', '')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chart
        if batches:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            x_vals = [b['batch_name'] for b in batches]
            r_vals = [b['avg_rating'] for b in batches]
            c_vals = [b['complaint_rate'] for b in batches]

            fig.add_trace(go.Scatter(x=x_vals, y=r_vals, name="Avg Rating", marker_color="#3B82F6", mode="lines+markers", line=dict(width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=x_vals, y=c_vals, name="Complaint Rate %", marker_color="#EF4444", mode="lines+markers", line=dict(width=3)), secondary_y=True)

            if isinstance(res, dict) and res.get("degradation_detected"):
                fig.add_vline(x=res["degradation_batch_num"] - 1, line_dash="dash", line_color="#EF4444", annotation_text="⚠️ Degradation Detected", annotation_font_color="#EF4444")

            fig.update_layout(height=400)
            apply_chart_theme(fig)
            fig.update_yaxes(title_text="Avg Rating", secondary_y=False, range=[0, 5])
            fig.update_yaxes(title_text="Complaint Rate (%)", secondary_y=True, range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        # Table
        if batches:
            st.markdown("""
            <div class="section-header" style="margin-top:20px;">
                <span style="font-size:18px">📋</span>
                <h2 class="section-title" style="font-size:16px;">Batch Comparison Table</h2>
            </div>
            """, unsafe_allow_html=True)
            batch_df = pd.DataFrame(batches)
            display_cols = [c for c in ['batch_name', 'start_date', 'end_date', 'review_count', 'avg_rating', 'complaint_rate', 'top_complaint', 'status'] if c in batch_df.columns]
            st.dataframe(batch_df[display_cols], use_container_width=True, hide_index=True)

            # Drilldown
            st.markdown("""
            <div class="section-header" style="margin-top:20px;">
                <span style="font-size:18px">🔎</span>
                <h2 class="section-title" style="font-size:16px;">Batch Detail Drilldown</h2>
            </div>
            """, unsafe_allow_html=True)
            for b in batches:
                with st.expander(f"{b['batch_name']} ({b['review_count']} reviews) - Rating: {b['avg_rating']}★"):
                    try:
                        if df is not None and 'reviews_indices' in b:
                            sub_df = df.loc[b['reviews_indices']]
                            cols_to_show = [c for c in ['review_text', 'sentiment', 'clean_text', 'timestamp'] if c in sub_df.columns]
                            st.dataframe(sub_df[cols_to_show], use_container_width=True, hide_index=True)
                        else:
                            st.write("Details not available.")
                    except Exception as e:
                        st.write("Details not available.", e)

# ═══════════════════════════════════════════════════════════════
# TAB ANOMALY: Z-Score Anomaly Detection
# ═══════════════════════════════════════════════════════════════

with tab_anomaly:
    anomaly_data = st.session_state.get('anomalies')
    if anomaly_data is None:
        st.info("Run the Unified Pipeline in the Upload tab to populate this section.")
    else:
        df = st.session_state.get('deduped_df')
        res = anomaly_data
        
        st.markdown("""
        <div class="section-header">
            <span style="font-size:24px">🎯</span>
            <h2 class="section-title">Z-Score Anomaly Detection</h2>
            <span class="section-badge">STATISTICAL</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p style="color:#475569; font-size:13px; margin-top:-12px; margin-bottom:20px;">Statistically isolate invisible deviations before they affect overall historic averages.</p>', unsafe_allow_html=True)
        
        if isinstance(res, dict) and "error" in res:
            st.warning(res["error"])
        elif isinstance(res, dict):
            anomalies = res.get("anomalies", [])
            wk_df = pd.DataFrame(res.get("weekly_data", []))
            heatmap_df = res.get("heatmap", pd.DataFrame())
            
            # PANEL 4: 4.0 THRESHOLD PROGRESS
            st.markdown("""
            <div class="section-header">
                <span style="font-size:18px">📊</span>
                <h2 class="section-title" style="font-size:16px;">Baseline Health Tracker</h2>
            </div>
            """, unsafe_allow_html=True)
            p_c1, p_c2 = st.columns([1,3])
            p_c1.metric("Current Avg Rating", f"{res.get('curr_rating', 0):.2f}★", delta=f"{res.get('rating_drop_z', 0):.1f}z", delta_color="inverse")
            if res.get('curr_rating', 5) < 4.0 and res.get('peak_rating', 0) >= 4.0:
                st.markdown(f"""
                <div class="reviewiq-card-critical">
                    <span style="color:#FCA5A5; font-size:14px;">
                        ⚠️ <strong>Crossed below 4.0★ threshold.</strong> Drop detected: {res['peak_rating']:.1f}★ → {res['curr_rating']:.1f}★
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="reviewiq-card-success">
                    <span style="color:#86EFAC; font-size:14px;">
                        ✅ Product remains stable vs historical peak ({res.get('peak_rating', 0):.1f}★)
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # PANEL 3: ALERTS
            if anomalies:
                st.divider()
                st.markdown("""
                <div class="section-header">
                    <span style="font-size:18px">🚨</span>
                    <h2 class="section-title" style="font-size:16px;">Active Warnings</h2>
                </div>
                """, unsafe_allow_html=True)
                # Sort active anomalies by absolute severity (Z-score deviation)
                anomalies = sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
                for a in anomalies:
                    if a['type'] == "SILENT COMPLAINT GROWTH":
                        card_class = "reviewiq-card-warning"
                        badge_class = "badge-warning"
                    else:
                        card_class = "reviewiq-card-critical"
                        badge_class = "badge-critical"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="color:#F1F5F9; font-weight:700; font-size:15px;">{a['type']} — {a['feature']}</span>
                            <span class="{badge_class}">Z: {a['z_score']:.2f}</span>
                        </div>
                        <p style="color:#94A3B8; margin:4px 0; font-size:14px; font-weight:500;">{a['msg']}</p>
                        <p style="color:#94A3B8; margin:4px 0; font-size:13px;"><strong style="color:#F1F5F9;">Week:</strong> {a['week']}</p>
                        <p style="color:#94A3B8; margin:4px 0; font-size:13px;"><strong style="color:#F1F5F9;">Meaning:</strong> {a['meaning']}</p>
                        <p style="color:#94A3B8; margin:4px 0; font-size:13px;"><strong style="color:#F1F5F9;">Action:</strong> {a['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # PANEL 1: PLOTLY CHART
            st.divider()
            st.markdown("""
            <div class="section-header">
                <span style="font-size:18px">📈</span>
                <h2 class="section-title" style="font-size:16px;">Z-Score Historic Timeline</h2>
            </div>
            """, unsafe_allow_html=True)
            if len(wk_df) > 0 and 'week_label' in wk_df.columns:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                x_axis = wk_df['week_label']
                
                # Highlight Danger Zones
                fig.update_layout(
                    shapes=[
                        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=2.0, y1=3.0, fillcolor="rgba(234,179,8,0.1)", layer="below", line_width=0),
                        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=3.0, y1=10.0, fillcolor="rgba(239,68,68,0.1)", layer="below", line_width=0),
                        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-3.0, y1=-2.0, fillcolor="rgba(234,179,8,0.1)", layer="below", line_width=0),
                        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-10.0, y1=-3.0, fillcolor="rgba(239,68,68,0.1)", layer="below", line_width=0),
                    ]
                )

                fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.2)")
                fig.add_hline(y=2.0, line_dash="dash", line_color="#EAB308")
                fig.add_hline(y=3.0, line_dash="dash", line_color="#EF4444")
                fig.add_hline(y=-2.0, line_dash="dash", line_color="#EAB308")
                fig.add_hline(y=-3.0, line_dash="dash", line_color="#EF4444")
                
                if 'z_avg_rating' in wk_df.columns:
                    fig.add_trace(go.Scatter(x=x_axis, y=wk_df['z_avg_rating'], name="Rating ΔZ", marker_color="#3B82F6", mode="lines+markers", line=dict(width=3)), secondary_y=False)
                if 'z_complaint_rate' in wk_df.columns:
                    fig.add_trace(go.Scatter(x=x_axis, y=wk_df['z_complaint_rate'], name="Complaint Rate ΔZ", marker_color="#EF4444", mode="lines+markers", line=dict(width=3)), secondary_y=False)

                fig.update_layout(yaxis_range=[-5, 5], height=400)
                apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            # PANEL 2: FEATURE HEATMAP
            st.divider()
            st.markdown("""
            <div class="section-header">
                <span style="font-size:18px">🗺️</span>
                <h2 class="section-title" style="font-size:16px;">Feature Deterioration Heatmap</h2>
            </div>
            """, unsafe_allow_html=True)
            if isinstance(heatmap_df, pd.DataFrame) and len(heatmap_df) > 0 and len(heatmap_df.columns) > 0:
                fig_h = px.imshow(
                    heatmap_df.values, 
                    x=heatmap_df.columns, 
                    y=heatmap_df.index,
                    color_continuous_scale=["#111827", "#EAB308", "#EF4444"],
                    zmin=0.0, zmax=3.5,
                    aspect="auto"
                )
                fig_h.update_layout(height=350)
                apply_chart_theme(fig_h)
                st.plotly_chart(fig_h, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0;">
        <span style="font-size:24px; font-weight:700; background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚙️ ReviewIQ</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if st.session_state.processed_df is not None:
        df_sidebar = get_data()
        st.metric("📊 Total Reviews", len(df_sidebar))
        st.metric("📦 Products", df_sidebar['product_name'].nunique())

        if st.session_state.crisis_active:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
            border-radius:8px; padding:10px; margin:8px 0; text-align:center;">
                <span style="color:#FCA5A5; font-size:13px; font-weight:600;">
                    ☢️ Crisis Active: +{st.session_state.crisis_result.reviews_injected} reviews
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Quick filter — show suspicious reviews
        st.markdown("""
        <div style="font-size:14px; font-weight:600; color:#F1F5F9; margin-bottom:8px;">🚩 Suspicious Reviews</div>
        """, unsafe_allow_html=True)
        if 'is_suspicious' in df_sidebar.columns:
            susp = df_sidebar[df_sidebar['is_suspicious'] == True]
            st.caption(f"{len(susp)} flagged")
            text_col = 'clean_text' if 'clean_text' in susp.columns else 'review_text'
            for _, row in susp.head(5).iterrows():
                text = str(row.get(text_col, ''))[:60]
                trust = row.get('trust_score', 0)
                st.markdown(f"""
                <div style='background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);border-radius:8px;padding:8px;margin:4px 0;font-size:12px;color:#94A3B8;'>
                {text}...<br><strong style="color:#FDBA74;">Trust: {trust:.2f}</strong>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="reviewiq-card-info" style="text-align:center;">
            <span style="color:#93C5FD; font-size:13px;">Upload a CSV to get started</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Clear All Data", width='stretch', key="clear_all"):
        for key in ['uploaded_df', 'processed_df', 'emotion_results', 'bot_result',
                    'crisis_active', 'crisis_result', 'crisis_reviews_df',
                    'clean_df', 'deduped_df', 'anomalies', 'batches']:
            st.session_state[key] = None
        st.session_state.crisis_active = False
        st.session_state.pipeline_complete = False
        st.rerun()

    st.divider()
    st.markdown('<p style="color:#475569; font-size:11px; text-align:center;">ReviewIQ v5 | Hack Malenadu \'26 | Team Tech Sparks</p>', unsafe_allow_html=True)
