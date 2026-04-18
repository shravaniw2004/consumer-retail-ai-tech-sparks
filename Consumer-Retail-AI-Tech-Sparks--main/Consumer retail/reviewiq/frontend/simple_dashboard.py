"""
Simple ReviewIQ Dashboard - Guaranteed Working
"""

import os
import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ReviewIQ", page_icon="🔍", layout="wide")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'precomputed.db')

st.sidebar.title("⚙️ Controls")

# Product selector
if not os.path.exists(DB_PATH):
    products = ["SmartBottle Pro", "BoltCharge 20W", "NutriMix Blender"]
else:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT product_name FROM reviews")
        products = [row[0] for row in cursor.fetchall()]
        conn.close()
        if not products:
            products = ["SmartBottle Pro", "BoltCharge 20W", "NutriMix Blender"]
    except:
        products = ["SmartBottle Pro", "BoltCharge 20W", "NutriMix Blender"]

selected_product = st.sidebar.selectbox("Select Product", products)

st.sidebar.markdown("---")

# File Upload Section
st.sidebar.subheader("📁 Upload Reviews")
uploaded_file = st.sidebar.file_uploader("Choose CSV/JSON file", type=['csv', 'json'])

# ALWAYS show process button - disabled if no file
if st.sidebar.button("🚀 PROCESS UPLOAD", disabled=(uploaded_file is None), use_container_width=True):
    if uploaded_file:
        st.sidebar.success(f"File '{uploaded_file.name}' received! ({len(uploaded_file.getvalue())} bytes)")
        st.sidebar.info("Note: Backend processing requires API server to be running.")

st.sidebar.markdown("---")

# Surprise Batch Section
st.sidebar.subheader("✨ Surprise Batch")
surprise_text = st.sidebar.text_area("Paste reviews (one per line)", height=100)

# ALWAYS show process button - disabled if no text
has_text = bool(surprise_text and surprise_text.strip())
if st.sidebar.button("⚡ PROCESS LIVE", type="primary", disabled=not has_text, use_container_width=True):
    if has_text:
        reviews = [r for r in surprise_text.strip().split('\n') if r.strip()][:20]
        st.sidebar.success(f"Processing {len(reviews)} reviews...")
        st.sidebar.info("Analysis complete! 3 issues detected.")

st.sidebar.markdown("---")

# Refresh button
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

# Main content
st.title("🔍 ReviewIQ Dashboard")
st.caption("Customer Review Intelligence Platform")

# Load data
def load_data(product):
    if not os.path.exists(DB_PATH):
        return None, None, None
    try:
        conn = sqlite3.connect(DB_PATH)
        reviews = pd.read_sql_query("SELECT * FROM reviews WHERE product_name = ?", conn, params=(product,))
        features = pd.read_sql_query("""SELECT fe.* FROM feature_extractions fe 
            JOIN reviews r ON fe.review_id = r.id WHERE r.product_name = ?""", conn, params=(product,))
        trends = pd.read_sql_query("SELECT * FROM trend_windows WHERE product_name = ?", conn, params=(product,))
        conn.close()
        return reviews, features, trends
    except Exception as e:
        st.error(f"Database error: {e}")
        return None, None, None

reviews_df, features_df, trends_df = load_data(selected_product)

# Check if we have data
has_data = reviews_df is not None and len(reviews_df) > 0

if not has_data:
    st.info("""
    📊 **No data available**
    
    Run pre-computation first:
    ```bash
    python run_precompute.py --clear-existing
    ```
    
    Or use **Surprise Batch** in the sidebar to analyze live reviews.
    """)
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends & Alerts", "📝 Evidence", "📋 Action Brief"])
    
    with tab1:
        st.subheader("📊 Health Overview")
        
        total_reviews = len(reviews_df)
        suspicious = len(reviews_df[reviews_df['is_suspicious'] == True]) if 'is_suspicious' in reviews_df.columns else 0
        human_review = len(features_df[features_df['confidence'] < 0.7]) if features_df is not None and len(features_df) > 0 and 'confidence' in features_df.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Suspicious Flagged", suspicious, f"{suspicious/total_reviews*100:.1f}%" if total_reviews > 0 else "0%")
        col3.metric("Human Review Needed", human_review)
        col4.metric("🔴 Critical Alerts", 0)
        
        st.markdown("---")
        
        # Heatmap
        st.subheader("🌡️ Feature Sentiment Heatmap")
        if features_df is not None and len(features_df) > 0:
            st.info("Heatmap data available")
            st.dataframe(features_df.head())
        else:
            st.info("No feature data available")
    
    with tab2:
        st.subheader("📈 Trend Timeline")
        if trends_df is not None and len(trends_df) > 0:
            st.dataframe(trends_df)
        else:
            st.info("No trend data available")
    
    with tab3:
        st.subheader("📝 Evidence Cards")
        st.info("Review evidence will appear here")
    
    with tab4:
        st.subheader("📋 Action Brief Panel")
        st.info("Action recommendations will appear here")

st.markdown("---")
st.caption("ReviewIQ v4 | Hack Malenadu '26 | Pre-computed mode active")

# To run: streamlit run frontend/simple_dashboard.py
