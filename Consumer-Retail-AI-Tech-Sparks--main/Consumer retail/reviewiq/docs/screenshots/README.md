# ReviewIQ Screenshots

This directory contains screenshots for documentation and README.

## Required Screenshots

### 1. Decision Mode
**File:** `decision-mode.png`

**How to Capture:**
1. Start the application: `streamlit run frontend/dashboard.py`
2. Select "SmartBottle Pro" from product dropdown
3. Toggle ON "⚡ Decision Mode" at top
4. Screenshot the 3 cards view

**Should Show:**
- 🛑 STOP card (red gradient) for Packaging
- 👁️ WATCH card (orange gradient) for Durability  
- 📢 AMPLIFY card (green gradient) for Battery Life
- Team badges and action bullets
- "Download Full Brief PDF" button at bottom

### 2. Trend Timeline
**File:** `trend-timeline.png`

**How to Capture:**
1. Toggle OFF Decision Mode (Analytics view)
2. Click "Analytics" tab
3. Scroll to "Trend Timeline" panel
4. Select "packaging" from feature dropdown
5. Screenshot the dual-axis chart

**Should Show:**
- Red line: Complaint rate increasing over windows
- Blue dashed line: Z-score crossing 2.5 threshold
- Orange dotted line: Alert threshold at Z=2.5
- Trend direction indicator: "🔴 SPIKING"

### 3. Analytics Dashboard (Full)
**File:** `analytics-dashboard.png`

**How to Capture:**
1. Ensure full Analytics view is visible
2. Screenshot entire page showing all 7 panels

**Should Show:**
- Header with product selector
- Health Overview metrics (4 cards)
- Feature Sentiment Heatmap
- Trend Timeline with dropdown
- Escalation Board with progress bars
- Evidence Cards with Hinglish
- Anomaly Alerts section

### 4. Surprise Batch (Optional)
**File:** `surprise-batch.png`

**How to Capture:**
1. Open sidebar "✨ Surprise Batch"
2. Paste 10-20 sample reviews
3. Click "⚡ Analyze Now"
4. Screenshot results panel showing top issues

**Should Show:**
- Input textarea with pasted reviews
- Results with "Complete in XXs"
- Top issues list with complaint counts
- Sentiment distribution

## Screenshot Specifications

- **Format:** PNG (recommended) or JPG
- **Width:** 1920px minimum (full HD)
- **Quality:** High (no compression artifacts)
- **Annotation:** None needed (UI is self-explanatory)

## Note

These screenshots are referenced in the main README.md file.
Update them whenever the UI design changes significantly.
