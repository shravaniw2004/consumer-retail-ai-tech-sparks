# ReviewIQ Architecture

## Design Philosophy

ReviewIQ was designed specifically for **demo performance** without sacrificing analytical depth. The key insight: reviewers don't want to wait 20 minutes to see intelligence on 300 reviews. Our solution: **pre-compute everything**.

## Why Pre-Compute?

### The Problem
- Claude API takes ~3-4 seconds per review
- 300 reviews × 4 seconds = **20 minutes** of watching a spinner
- Demo judges lose interest after 30 seconds

### The Solution
- Run batch extraction **once** before demo (~5 minutes)
- Store all results in SQLite database
- Dashboard loads **instantly** from pre-computed data
- Surprise batch processes **live** in <45 seconds (20 reviews max)

### Trade-offs
| Approach | Demo Performance | Flexibility | Use Case |
|----------|-----------------|-------------|----------|
| Real-time all | ❌ 20min wait | ✅ Always fresh | Production |
| **Pre-compute** | ✅ Instant | ✅ Demo scenarios | **Hackathons** |
| Hybrid | ✅ <45s live | ✅ Best of both | **Our choice** |

## Batch Processing Strategy

### Sentiment Extraction (The Bottleneck)

```
Problem: 300 reviews, Claude takes 3-4s per review
Solution: Batch 5 reviews per API call, process in parallel
```

**Configuration:**
- `batch_size = 5` (optimal for Claude context window)
- `max_workers = 3` (rate limit friendly)
- `use_cache = True` (shelve DB for re-runs)

**Math:**
- 300 reviews ÷ 5 per batch = 60 batches
- 60 batches ÷ 3 workers = 20 parallel rounds
- 20 rounds × 4 seconds = **~80 seconds** (not 20 minutes!)

```python
# sentiment_engine.py
results = sentiment_engine.process_all_reviews_parallel(
    reviews, 
    max_workers=3,  # 3 parallel API calls
    use_cache=True   # Skip if already processed
)
```

### Surprise Batch (Live Demo)

For hackathon demos, we need **live processing** that completes while judges watch:

```
Target: <45 seconds for 20 reviews
Strategy: Maximize parallelism
```

**Pipeline:**
1. **Parse** (0.1s) - Split text into reviews
2. **Preprocess** (2-3s) - 4 parallel threads
3. **Extract** (10-15s) - batch_size=5, max_workers=3
   - 20 reviews = 4 batches
   - 4 batches ÷ 3 workers = 2 rounds (3+1)
   - 2 rounds × 5s = 10s
4. **Aggregate** (0.5s) - Build results
5. **Save** (2s) - Async DB write (non-blocking)

**Total: ~15-20 seconds** (well under 45s limit)

```python
# Parallel preprocessing
with ThreadPoolExecutor(max_workers=4) as executor:
    processed = list(executor.map(process_single, reviews))

# Parallel extraction
results = sentiment_engine.process_all_reviews_parallel(
    processed, max_workers=3, use_cache=False
)

# Async DB save (doesn't block response)
background_tasks.add_task(_save_to_db_async, enriched)
```

## Database Schema

### Pre-Compute Strategy

All tables are populated by `run_precompute.py`:

```sql
-- reviews: Preprocessed review data
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    original_text TEXT,
    clean_text TEXT,
    language TEXT,
    trust_score FLOAT,
    is_suspicious BOOLEAN
);

-- feature_extractions: Claude outputs (cached)
CREATE TABLE feature_extractions (
    id INTEGER PRIMARY KEY,
    review_id INTEGER,
    feature_name TEXT,
    sentiment TEXT,
    intensity FLOAT,
    confidence FLOAT,
    evidence TEXT
);

-- trend_windows: Sliding window aggregations
CREATE TABLE trend_windows (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    feature_name TEXT,
    window_label TEXT,
    complaint_rate FLOAT,
    praise_rate FLOAT,
    z_score FLOAT
);

-- escalations: Priority-ranked issues
CREATE TABLE escalations (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    feature_name TEXT,
    severity_score FLOAT,
    priority_rank INTEGER
);

-- action_briefs: Generated PDFs
CREATE TABLE action_briefs (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    generated_text TEXT,
    pdf_path TEXT
);
```

### Query Patterns

All dashboard queries are simple SELECTs from pre-computed tables:

```sql
-- Analytics: Get trends (instant, no calculation)
SELECT * FROM trend_windows 
WHERE product_name = 'SmartBottle Pro';

-- Decision Mode: Get escalations (instant)
SELECT * FROM escalations 
WHERE product_name = 'SmartBottle Pro' 
ORDER BY priority_rank DESC 
LIMIT 3;
```

## Architecture Layers

### 1. Ingestion Layer
```
CSV/JSON → Parse → Preprocess → Deduplicate → Store (raw)
                      ↓
              Hinglish expansion
              Trust score calculation
```

### 2. Extraction Layer
```
Raw reviews → Claude API (batch_size=5) → Feature extractions
                ↓
            Sarcasm detection (2-pass)
            Ambiguity flagging
```

### 3. Analysis Layer
```
Extractions → Trend detection (sliding windows)
            → Lifecycle tracking (6-stage FSM)
            → Escalation scoring (6-factor formula)
            → Vocabulary detection (emerging phrases)
```

### 4. Output Layer
```
Analysis → Action briefs (PDF generation)
         → Decision Mode (3-card aggregation)
         → Analytics dashboard (7 panels)
```

## Performance Optimizations

### 1. Caching
- **API Cache**: `shelve` DB stores Claude responses by content hash
- **Avoids re-calling API** during demo prep or re-runs
- Key: MD5 hash of batch content

### 2. Parallelism
| Stage | Workers | Library |
|-------|---------|---------|
| Preprocessing | 4 | `ThreadPoolExecutor` |
| API Calls | 3 | `concurrent.futures` |
| DB Writes | Async | `BackgroundTasks` |

### 3. Pre-aggregation
- Trend windows computed once, stored in DB
- Escalation scores calculated once, ranked
- No math at query time

### 4. Lazy Loading
- Surprise batch skips deduplication (not needed for 20 reviews)
- Only extracts features for pasted reviews
- DB save is async (user gets result immediately)

## Demo Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MINUTE 1: Upload CSV (background processing)                │
│  • Preprocessing (4 threads)                               │
│  • Deduplication (TF-IDF)                                  │
│  • Store in DB                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  MINUTE 2: Pre-compute (run_precompute.py)                 │
│  • Batch extraction (60 batches, 3 parallel)             │
│  • Trend detection (sliding windows)                       │
│  • Escalation scoring (6-factor formula)                   │
│  • Action brief generation (PDF)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  MINUTE 3: Dashboard (instant from DB)                     │
│  • Health metrics (cached)                                   │
│  • Sentiment heatmap (pre-computed)                        │
│  • Trend timeline (Z-scores ready)                         │
│  • Escalation board (priority-ranked)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  MINUTE 4: Decision Mode (3-card view)                     │
│  • Query: top 3 escalations from DB                        │
│  • Transform: STOP/WATCH/AMPLIFY cards                       │
│  • Render: Gradient cards with team badges                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  MINUTE 5: Surprise Batch (live <45s)                      │
│  • Parse pasted text                                       │
│  • Preprocess (4 threads, 2-3s)                          │
│  • Extract (5 per batch, 3 workers, 10-15s)                │
│  • Aggregate & return (async DB save)                      │
└─────────────────────────────────────────────────────────────┘
```

## Scalability Notes

This architecture is optimized for:
- **Dataset size**: 100-1000 reviews (hackathon demos)
- **Time constraint**: <5 minute demo
- **Cost**: Minimize Claude API calls via caching

For production:
- Add Redis for distributed caching
- Use Celery for background job queues
- Implement incremental updates (not full re-compute)
- Add vector DB for semantic search

## Key Files

| File | Purpose |
|------|---------|
| `run_precompute.py` | Pre-computes all analysis (~5min) |
| `sentiment_engine.py` | Batch Claude extraction |
| `preprocessor.py` | Parallel text preprocessing |
| `main.py` | Surprise batch endpoint (<45s) |
| `dashboard.py` | Reads from pre-computed DB |

## Running the Demo

```bash
# 1. Setup (once)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # Add ANTHROPIC_API_KEY

# 2. Generate data (once)
python -c "from data.data_generation import generate_demo_dataset; r = generate_demo_dataset(); from data.data_generation import save_to_csv; save_to_csv(r)"

# 3. Pre-compute (once, ~5min)
python run_precompute.py --clear-existing

# 4. Start services (2 terminals)
uvicorn backend.main:app --reload
streamlit run frontend/dashboard.py
```

The demo is now ready for instant analytics and <45s surprise batch processing!
