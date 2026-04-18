"""
ReviewIQ Frontend Entry Point
Redirects to main dashboard.py
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Redirect to dashboard
st.set_page_config(
    page_title="ReviewIQ",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ReviewIQ")
st.markdown("---")

st.info("""
👋 Welcome to ReviewIQ! 

The main dashboard is in **dashboard.py**. Run it with:

```bash
streamlit run frontend/dashboard.py
```

Or click the link below:
""")

st.markdown("[👉 Open Dashboard](./dashboard)")

st.markdown("---")

st.markdown("""
**Quick Links:**
- [Dashboard](./dashboard) - Main analytics interface
- [Health Check](http://localhost:8000/health) - API status
- [API Docs](http://localhost:8000/docs) - Swagger documentation
""")
