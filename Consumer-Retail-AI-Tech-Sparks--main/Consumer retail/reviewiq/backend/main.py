"""
ReviewIQ Backend - FastAPI Application
Main API with all endpoints for ReviewIQ platform.
"""

import os
import json
import uuid
import csv
import io
from typing import List, Dict, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

import sys
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import ReviewIQ modules
from preprocessor import ReviewPreprocessor, preprocess_review
from deduplication import deduplicate_reviews, get_dedup_stats
from sentiment_engine import SentimentEngine, extract_sentiments
from ambiguity_detector import flag_ambiguous_reviews
from trend_detector import detect_all_trends, get_trend_summary, save_trend_windows
from lifecycle_tracker import IssueLifecycle, LifecycleManager, track_issue_lifecycle
from escalation_scorer import EscalationScorer, classify_priority, priority_to_action_brief
from vocabulary_detector import VocabularyTracker, detect_emerging_vocabulary
from action_brief import ActionBriefGenerator, get_brief_summary
from data.database import get_db, init_db, Review, FeatureExtraction, TrendWindow, Escalation, ActionBrief
from data.database import SessionLocal, engine
from sentiment_engine import FEATURE_TAXONOMY

# Initialize FastAPI app
app = FastAPI(
    title="ReviewIQ API",
    description="Customer Review Intelligence Platform",
    version="1.0.0"
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (use Redis in production)
job_store: Dict[str, Dict] = {}
job_store_lock = threading.Lock()

# Initialize components
preprocessor = ReviewPreprocessor()
try:
    sentiment_engine = SentimentEngine()
except ValueError:
    # API key not set yet; engine will fail gracefully when called
    sentiment_engine = None  # type: ignore
escalation_scorer = EscalationScorer()
brief_generator = ActionBriefGenerator()


# Pydantic models
class ReviewIngest(BaseModel):
    review_id: str
    product_name: str
    category: str
    review_text: str
    timestamp: Optional[str] = None


class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    stage: str
    message: str
    result: Optional[Dict] = None


class SurpriseBatchRequest(BaseModel):
    reviews_text: str  # Pasted text with reviews
    product_name: str
    category: str


class SurpriseBatchProgressUpdate(BaseModel):
    """Progress update for surprise batch processing."""
    stage: str
    progress: int  # 0-100
    message: str


class SurpriseBatchResult(BaseModel):
    """Result of surprise batch processing."""
    reviews_processed: int
    time_taken_seconds: float
    top_issues: List[Dict]
    sentiment_summary: Dict
    action_recommendations: List[str]
    progress_updates: Optional[List[SurpriseBatchProgressUpdate]] = None  # For tracking


class DecisionModeResponse(BaseModel):
    product_name: str
    top_actions: List[Dict]
    severity_indicator: str


def _update_job_status(job_id: str, status: str, progress: int, stage: str, message: str, result: Optional[Dict] = None):
    """Update job status in thread-safe manner."""
    with job_store_lock:
        job_store[job_id] = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "stage": stage,
            "message": message,
            "result": result,
            "updated_at": datetime.now().isoformat()
        }


def _parse_csv(file_content: bytes) -> List[Dict]:
    """Parse CSV file content to list of dicts."""
    content = file_content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def _parse_json(file_content: bytes) -> List[Dict]:
    """Parse JSON file content to list of dicts."""
    content = file_content.decode('utf-8')
    data = json.loads(content)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'reviews' in data:
        return data['reviews']
    return [data]


@app.get("/")
async def root():
    return {
        "message": "ReviewIQ API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/ingest/reviews")
async def ingest_reviews(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Ingest reviews from CSV or JSON file.
    Runs preprocessing and deduplication, returns job_id for tracking.
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    _update_job_status(job_id, "pending", 0, "initialization", "Starting ingestion...")
    
    # Read file content
    content = await file.read()
    
    # Parse based on file type
    try:
        if file.filename.endswith('.csv'):
            raw_reviews = _parse_csv(content)
        elif file.filename.endswith('.json'):
            raw_reviews = _parse_json(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    
    # Start background processing
    background_tasks.add_task(_process_ingestion, job_id, raw_reviews, db)
    
    return {"job_id": job_id, "message": f"Ingestion started with {len(raw_reviews)} reviews", "status": "pending"}


def _process_ingestion(job_id: str, raw_reviews: List[Dict], db: Session):
    """Background task for ingestion processing."""
    try:
        # Stage 1: Preprocessing
        _update_job_status(job_id, "processing", 10, "preprocessing", f"Preprocessing {len(raw_reviews)} reviews...")
        
        processed = []
        for i, review in enumerate(raw_reviews):
            text = review.get('review_text', review.get('text', ''))
            result = preprocessor.process(text)
            processed.append({
                **review,
                **result,
                'original_text': text
            })
            
            if i % 50 == 0:
                progress = 10 + int((i / len(raw_reviews)) * 30)
                _update_job_status(job_id, "processing", progress, "preprocessing", f"Processed {i}/{len(raw_reviews)} reviews...")
        
        # Stage 2: Deduplication
        _update_job_status(job_id, "processing", 40, "deduplication", "Running deduplication...")
        deduplicated = deduplicate_reviews(processed)
        dedup_stats = get_dedup_stats(deduplicated)
        
        # Stage 3: Save to database
        _update_job_status(job_id, "processing", 60, "database", "Saving to database...")
        
        for review in deduplicated:
            db_review = Review(
                product_name=review.get('product_name', 'Unknown'),
                original_text=review.get('original_text', ''),
                clean_text=review.get('clean_text', ''),
                language=review.get('language', 'en'),
                trust_score=review.get('trust_score', 1.0),
                is_suspicious=review.get('is_suspicious', False),
                created_at=datetime.now()
            )
            db.add(db_review)
            db.flush()
            review['db_id'] = db_review.id
        
        db.commit()
        
        _update_job_status(
            job_id, "completed", 100, "completed", 
            "Ingestion completed successfully",
            {
                "total_reviews": len(raw_reviews),
                "processed_reviews": len(processed),
                "unique_reviews": dedup_stats.get('unique_reviews', len(processed)),
                "duplicates_found": dedup_stats.get('duplicates_found', 0),
                "suspicious_reviews": dedup_stats.get('suspicious_reviews', 0)
            }
        )
        
    except Exception as e:
        _update_job_status(job_id, "failed", 0, "error", f"Failed: {str(e)}")
        db.rollback()
    finally:
        db.close()


@app.get("/processing/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get the status of a processing job."""
    with job_store_lock:
        if job_id not in job_store:
            raise HTTPException(status_code=404, detail="Job not found")
        return ProcessingStatus(**job_store[job_id])


@app.post("/processing/run-extraction")
async def run_extraction(
    background_tasks: BackgroundTasks,
    product_name: Optional[str] = None,
    use_cache: bool = True
):
    """
    Trigger batch AI extraction (5 reviews per API call).
    Processes reviews that haven't been extracted yet.
    """
    job_id = str(uuid.uuid4())
    _update_job_status(job_id, "pending", 0, "initialization", "Starting extraction...")
    
    background_tasks.add_task(_process_extraction, job_id, product_name, use_cache)
    
    return {"job_id": job_id, "message": "Extraction started", "status": "pending"}


def _process_extraction(job_id: str, product_name: Optional[str], use_cache: bool):
    """Background task for batch extraction."""
    db = SessionLocal()
    
    try:
        # Get unprocessed reviews
        query = db.query(Review).filter(Review.id.notin_(
            db.query(FeatureExtraction.review_id).distinct()
        ))
        
        if product_name:
            query = query.filter(Review.product_name == product_name)
        
        reviews_to_process = query.all()
        
        if not reviews_to_process:
            _update_job_status(job_id, "completed", 100, "completed", "No new reviews to extract")
            return
        
        # Convert to dicts for processing
        review_dicts = [
            {
                'review_id': f"R{r.id}",
                'product_name': r.product_name,
                'clean_text': r.clean_text,
                'category': _detect_category(r.product_name),
                'db_id': r.id
            }
            for r in reviews_to_process
        ]
        
        _update_job_status(job_id, "processing", 10, "extraction", f"Extracting from {len(review_dicts)} reviews...")
        
        # Run extraction
        results = sentiment_engine.process_all_reviews_parallel(
            review_dicts, max_workers=3, use_cache=use_cache
        )
        
        _update_job_status(job_id, "processing", 70, "saving", "Saving extractions to database...")
        
        # Save to database
        for review_dict, result in zip(review_dicts, results):
            for feature in result.features:
                extraction = FeatureExtraction(
                    review_id=review_dict['db_id'],
                    feature_name=feature.get('feature', 'unknown'),
                    sentiment=feature.get('sentiment', 'neutral'),
                    intensity=feature.get('intensity', 0.5),
                    confidence=feature.get('confidence', 0.5),
                    ambiguity_flag=feature.get('flags', {}).get('ambiguous', False),
                    sarcasm_flag=feature.get('flags', {}).get('sarcasm', False),
                    evidence=feature.get('evidence', '')
                )
                db.add(extraction)
        
        db.commit()
        
        human_review_count = sum(1 for r in results if r.human_review_needed)
        
        _update_job_status(
            job_id, "completed", 100, "completed",
            "Extraction completed",
            {
                "processed": len(results),
                "human_review_needed": human_review_count
            }
        )
        
    except Exception as e:
        _update_job_status(job_id, "failed", 0, "error", f"Failed: {str(e)}")
        db.rollback()
    finally:
        db.close()


def _detect_category(product_name: str) -> str:
    """Detect category based on product name (simplified)."""
    name_lower = product_name.lower()
    if 'bottle' in name_lower or 'care' in name_lower:
        return "Personal Care"
    elif 'charge' in name_lower or 'electronics' in name_lower:
        return "Electronics"
    elif 'mix' in name_lower or 'blend' in name_lower:
        return "Food"
    return "General"


@app.get("/analytics/trends/{product_name}")
async def get_trends(
    product_name: str,
    window_size: int = Query(50, ge=10, le=100),
    step: int = Query(25, ge=5, le=50)
):
    """
    Get trend analysis data for a product.
    Returns complaint/praise rates and Z-scores per feature.
    """
    db = SessionLocal()
    
    try:
        # Get reviews for product with extractions
        reviews = db.query(Review).filter(
            Review.product_name == product_name
        ).all()
        
        if not reviews:
            raise HTTPException(status_code=404, detail="No reviews found for product")
        
        # Convert to dicts with extractions
        review_dicts = []
        for r in reviews:
            extractions = db.query(FeatureExtraction).filter(
                FeatureExtraction.review_id == r.id
            ).all()
            
            review_dicts.append({
                'review_id': f"R{r.id}",
                'product_name': r.product_name,
                'clean_text': r.clean_text,
                'timestamp': r.created_at.isoformat() if r.created_at else '',
                'extracted_features': [
                    {
                        'feature': e.feature_name,
                        'sentiment': e.sentiment,
                        'intensity': e.intensity,
                        'confidence': e.confidence
                    }
                    for e in extractions
                ]
            })
        
        # Detect category and get taxonomy
        category = _detect_category(product_name)
        taxonomy = FEATURE_TAXONOMY.get(category, [])
        
        # Run trend detection
        trends = detect_all_trends(
            product_name=product_name,
            reviews=review_dicts,
            feature_taxonomy=taxonomy,
            window_size=window_size,
            step=step
        )
        
        summary = get_trend_summary(trends)
        
        return {
            "product_name": product_name,
            "category": category,
            "total_reviews": len(review_dicts),
            "window_size": window_size,
            "step": step,
            "summary": summary,
            "trends": [
                {
                    "feature_name": t.feature_name,
                    "trend_direction": t.trend_direction,
                    "max_severity": t.max_severity,
                    "is_emerging_issue": t.is_emerging_issue,
                    "windows": t.windows[:5]  # First 5 windows
                }
                for t in trends
            ]
        }
        
    finally:
        db.close()


@app.get("/analytics/escalations")
async def get_escalations(
    min_priority: int = Query(40, ge=0, le=100),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get prioritized list of all escalated issues.
    Sorted by priority score (highest first).
    """
    db = SessionLocal()
    
    try:
        # Get all products with reviews
        products = db.query(Review.product_name).distinct().all()
        
        all_escalations = []
        
        for (product_name,) in products:
            # Get trends for this product
            reviews = db.query(Review).filter(
                Review.product_name == product_name
            ).all()
            
            review_dicts = []
            for r in reviews:
                extractions = db.query(FeatureExtraction).filter(
                    FeatureExtraction.review_id == r.id
                ).all()
                
                review_dicts.append({
                    'product_name': r.product_name,
                    'extracted_features': [
                        {
                            'feature': e.feature_name,
                            'sentiment': e.sentiment,
                            'intensity': e.intensity,
                            'confidence': e.confidence
                        }
                        for e in extractions
                    ]
                })
            
            # Score each feature
            category = _detect_category(product_name)
            
            # Group by feature and calculate rates
            feature_stats = {}
            for review in review_dicts:
                for feat in review.get('extracted_features', []):
                    name = feat['feature']
                    if name not in feature_stats:
                        feature_stats[name] = {
                            'mentions': 0,
                            'negative': 0,
                            'confidence_scores': []
                        }
                    
                    feature_stats[name]['mentions'] += 1
                    if feat['sentiment'] == 'negative':
                        feature_stats[name]['negative'] += 1
                    feature_stats[name]['confidence_scores'].append(feat['confidence'])
            
            # Create escalations
            for feature_name, stats in feature_stats.items():
                complaint_rate = (stats['negative'] / stats['mentions'] * 100) if stats['mentions'] > 0 else 0
                
                # Simple priority calculation
                score_data = {
                    'sentiment_intensity': 0.8 if complaint_rate > 20 else 0.5,
                    'sentiment_type': 'negative',
                    'current_complaint_rate': complaint_rate,
                    'negative_mentions': stats['negative'],
                    'total_reviews': len(review_dicts),
                    'confidence_scores': stats['confidence_scores'],
                    'lifecycle_stage': 'detected',
                    'windows_ago': 0
                }
                
                priority_score = escalation_scorer.score_issue(
                    feature_name=feature_name,
                    product_name=product_name,
                    feature_data=score_data
                )
                
                if priority_score.priority_score >= min_priority:
                    all_escalations.append({
                        "product_name": product_name,
                        "feature_name": feature_name,
                        "priority_score": priority_score.priority_score,
                        "priority_level": priority_score.priority_level,
                        "priority_color": priority_score.priority_color,
                        "complaint_rate": round(complaint_rate, 2),
                        "mentions": stats['mentions'],
                        "recommendation": priority_score.recommendation
                    })
        
        # Sort by priority score
        all_escalations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            "total_escalations": len(all_escalations),
            "escalations": all_escalations[:limit]
        }
        
    finally:
        db.close()


@app.get("/analytics/decision-mode")
async def get_decision_mode(
    product_name: Optional[str] = None
):
    """
    Returns top 3 actions in simplified format for Decision Mode UI.
    Quick-glance view for executives.
    """
    db = SessionLocal()
    
    try:
        # If no product specified, get all and pick worst
        if not product_name:
            products = db.query(Review.product_name).distinct().all()
            if not products:
                return DecisionModeResponse(
                    product_name="None",
                    top_actions=[],
                    severity_indicator="green"
                )
            product_name = products[0][0]
        
        # Get escalations for this product
        response = await get_escalations(min_priority=60, limit=10)
        product_escalations = [
            e for e in response['escalations']
            if e['product_name'] == product_name
        ]
        
        if not product_escalations:
            return DecisionModeResponse(
                product_name=product_name,
                top_actions=[
                    {
                        "action": "Monitor reviews for emerging patterns",
                        "urgency": "low",
                        "team": "Quality"
                    }
                ],
                severity_indicator="green"
            )
        
        # Map priority to severity indicator
        severity_map = {
            "critical": "red",
            "high": "orange",
            "medium": "yellow",
            "low": "green"
        }
        
        top_actions = [
            {
                "action": e['recommendation'],
                "urgency": e['priority_level'],
                "team": _get_primary_team(e['feature_name']),
                "feature": e['feature_name'],
                "priority_score": e['priority_score']
            }
            for e in product_escalations[:3]
        ]
        
        return DecisionModeResponse(
            product_name=product_name,
            top_actions=top_actions,
            severity_indicator=severity_map.get(
                product_escalations[0]['priority_level'], 'green'
            )
        )
        
    finally:
        db.close()


def _get_primary_team(feature_name: str) -> str:
    """Determine primary team for a feature issue."""
    feature_lower = feature_name.lower()
    
    if any(word in feature_lower for word in ['packaging', 'delivery', 'shipping']):
        return "Operations"
    elif any(word in feature_lower for word in ['battery', 'durability', 'performance', 'quality']):
        return "Quality"
    elif any(word in feature_lower for word in ['price', 'value', 'cost', 'marketing']):
        return "Marketing"
    else:
        return "Support"


@app.post("/action-brief/generate")
async def generate_action_brief(
    product_name: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Generate action brief and return PDF download link.
    """
    # Get top issues for product
    escalations = await get_escalations(min_priority=40, limit=10)
    product_issues = [
        e for e in escalations['escalations']
        if e['product_name'] == product_name
    ]
    
    if not product_issues:
        raise HTTPException(status_code=404, detail="No issues found for product")
    
    # Generate brief
    brief, pdf_path = brief_generator.generate_and_export(
        product_name=product_name,
        top_issues=product_issues,
        save_pdf=True
    )
    
    if not brief:
        raise HTTPException(status_code=500, detail="Failed to generate brief")
    
    return {
        "product_name": product_name,
        "pdf_url": f"/download/brief/{os.path.basename(pdf_path)}" if pdf_path else None,
        "summary": get_brief_summary(brief),
        "generated_at": brief.generated_at
    }


@app.get("/download/brief/{filename}")
async def download_brief(filename: str):
    """Download generated PDF brief."""
    filepath = os.path.join("sample_output", filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=filename
    )


import asyncio


class SurpriseBatchProgress:
    """Track progress for surprise batch processing."""
    def __init__(self):
        self.stage = "initializing"
        self.progress = 0
        self.message = "Starting..."
    
    def update(self, stage: str, progress: int, message: str):
        self.stage = stage
        self.progress = progress
        self.message = message


def _parse_reviews(text: str, product_name: str, category: str) -> List[Dict]:
    """Parse reviews from pasted text (handles newlines and paragraphs)."""
    # Split by newlines and filter empty lines
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    # Also handle paragraphs (double newlines)
    paragraphs = []
    current_para = []
    for line in lines:
        if line:
            current_para.append(line)
        else:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # Use paragraphs if we have them, otherwise use lines
    review_texts = paragraphs if len(paragraphs) >= len(lines) / 2 else lines
    
    reviews = []
    for i, text in enumerate(review_texts[:20], 1):  # Limit to 20
        if len(text) > 10:  # Minimum review length
            reviews.append({
                'review_id': f'SURPRISE_{i}',
                'product_name': product_name,
                'category': category,
                'review_text': text
            })
    
    return reviews


def _preprocess_parallel(reviews: List[Dict], progress: SurpriseBatchProgress) -> List[Dict]:
    """Preprocess reviews in parallel threads."""
    progress.update("preprocessing", 10, f"Preprocessing {len(reviews)} reviews...")
    
    def process_single(review):
        result = preprocessor.process(review['review_text'])
        return {**review, **result}
    
    # Use ThreadPoolExecutor for parallel preprocessing
    with ThreadPoolExecutor(max_workers=4) as executor:
        processed = list(executor.map(process_single, reviews))
    
    progress.update("preprocessing", 25, f"Preprocessed {len(processed)} reviews")
    return processed


def _extract_batch_optimized(reviews: List[Dict], progress: SurpriseBatchProgress) -> List[Dict]:
    """Extract sentiments with optimized batch processing (batch_size=5, max_workers=3)."""
    progress.update("extraction", 30, "Starting sentiment extraction...")
    
    # Add metadata
    for review in reviews:
        review['clean_text'] = review.get('clean_text', review.get('review_text', ''))
        review['category'] = _detect_category(review.get('product_name', ''))
    
    # Calculate expected batches
    num_batches = (len(reviews) + 4) // 5
    progress.update("extraction", 35, f"Processing {len(reviews)} reviews in {num_batches} batches...")
    
    # Run with batch_size=5, max_workers=3
    # This means: 20 reviews = 4 batches, processed in 2 parallel rounds (3+1)
    # Each API call ≈ 3-4 seconds, so total ≈ 8-10 seconds
    results = sentiment_engine.process_all_reviews_parallel(
        reviews, max_workers=3, use_cache=False
    )
    
    progress.update("extraction", 75, f"Extracted {len(results)} reviews")
    
    # Combine results
    enriched = []
    for review, result in zip(reviews, results):
        enriched.append({
            **review,
            "extracted_features": result.features,
            "overall_sentiment": result.overall_sentiment,
            "trust_indicators": result.trust_indicators,
            "human_review_needed": result.human_review_needed
        })
    
    progress.update("aggregation", 80, "Aggregating results...")
    return enriched


def _aggregate_results(reviews: List[Dict]) -> Tuple[Dict, List[Dict], List[str]]:
    """Quick aggregation of results."""
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0}
    feature_complaints = {}
    
    for review in reviews:
        sentiment_counts[review.get('overall_sentiment', 'neutral')] += 1
        
        for feat in review.get('extracted_features', []):
            if feat.get('sentiment') == 'negative':
                name = feat.get('feature', 'unknown')
                if name not in feature_complaints:
                    feature_complaints[name] = {'count': 0, 'intensity': 0}
                feature_complaints[name]['count'] += 1
                feature_complaints[name]['intensity'] += feat.get('intensity', 0.5)
    
    # Top issues
    top_issues = [
        {
            'feature_name': name,
            'complaint_count': data['count'],
            'avg_intensity': data['intensity'] / data['count'] if data['count'] > 0 else 0
        }
        for name, data in sorted(
            feature_complaints.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]
    ]
    
    # Generate recommendations
    recommendations = []
    if top_issues:
        top = top_issues[0]
        recommendations.append(f"Address '{top['feature_name']}' complaints ({top['complaint_count']} mentions)")
        
        if sentiment_counts['negative'] > sentiment_counts['positive']:
            recommendations.append("Overall negative sentiment - consider product review")
        
        if any(i['complaint_count'] >= 3 for i in top_issues):
            recommendations.append("Multiple features showing issues - escalate to quality team")
    
    return sentiment_counts, top_issues, recommendations


def _save_to_db_async(reviews: List[Dict]):
    """Save results to database asynchronously."""
    try:
        db = SessionLocal()
        
        for review in reviews:
            # Save review
            db_review = Review(
                product_name=review.get('product_name', 'Unknown'),
                original_text=review.get('original_text', review.get('review_text', '')),
                clean_text=review.get('clean_text', ''),
                language=review.get('language', 'en'),
                trust_score=review.get('trust_score', 1.0),
                is_suspicious=review.get('is_suspicious', False),
                created_at=datetime.now()
            )
            db.add(db_review)
            db.flush()
            
            # Save feature extractions
            for feature in review.get('extracted_features', []):
                extraction = FeatureExtraction(
                    review_id=db_review.id,
                    feature_name=feature.get('feature', 'unknown'),
                    sentiment=feature.get('sentiment', 'neutral'),
                    intensity=feature.get('intensity', 0.5),
                    confidence=feature.get('confidence', 0.5),
                    ambiguity_flag=feature.get('flags', {}).get('ambiguous', False),
                    sarcasm_flag=feature.get('flags', {}).get('sarcasm', False),
                    evidence=feature.get('evidence', '')
                )
                db.add(extraction)
        
        db.commit()
        db.close()
        
    except Exception as e:
        print(f"Async DB save error: {e}")


@app.post("/surprise-batch", response_model=SurpriseBatchResult)
async def surprise_batch(
    request: SurpriseBatchRequest,
    background_tasks: BackgroundTasks
):
    """
    Process pasted text with live reviews (max 20, <45 seconds guaranteed).
    
    Optimizations:
    - Parallel preprocessing (4 threads)
    - Batch extraction: batch_size=5, max_workers=3 (4 batches in 2 rounds ≈ 10s)
    - Async DB storage (non-blocking)
    """
    import time
    start_time = time.time()
    progress = SurpriseBatchProgress()
    
    try:
        # Stage 1: Parse (instant)
        progress.update("parsing", 5, "Parsing reviews...")
        raw_reviews = _parse_reviews(
            request.reviews_text,
            request.product_name,
            request.category
        )
        
        if not raw_reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found in text")
        
        # Limit to 20
        raw_reviews = raw_reviews[:20]
        progress.update("parsing", 10, f"Parsed {len(raw_reviews)} reviews")
        
        # Stage 2: Parallel Preprocessing (~2-3 seconds for 20 reviews)
        processed = await asyncio.to_thread(_preprocess_parallel, raw_reviews, progress)
        
        # Stage 3: Batch Extraction (~10-15 seconds for 20 reviews with batch_size=5, workers=3)
        enriched = await asyncio.to_thread(_extract_batch_optimized, processed, progress)
        
        # Stage 4: Aggregation (instant)
        progress.update("aggregation", 85, "Finalizing results...")
        sentiment_counts, top_issues, recommendations = _aggregate_results(enriched)
        
        # Stage 5: Async DB save (non-blocking)
        progress.update("saving", 95, "Saving to database...")
        background_tasks.add_task(_save_to_db_async, enriched)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        progress.update("complete", 100, f"Complete in {elapsed:.1f}s")
        
        return SurpriseBatchResult(
            reviews_processed=len(raw_reviews),
            time_taken_seconds=round(elapsed, 2),
            top_issues=top_issues,
            sentiment_summary=sentiment_counts,
            action_recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed after {elapsed:.1f}s: {str(e)}"
        )


@app.post("/surprise-batch-test")
async def surprise_batch_test():
    """Test endpoint to verify <45 second timing."""
    import time
    
    # Create 20 sample reviews
    sample_reviews = [
        "Cap khula hua tha packing bilkul bakwas leaked completely",
        "Great product works well but cap came loose during delivery",
        "Battery backup mast hai zabardast performance love it",
        "Motor se burning smell aa rahi hai after 2 uses very bad",
        "Cable fraying after one week poor quality disappointed",
        "Excellent build quality keeps water cold for hours",
        "Charging speed amazing best charger for my phone",
        "Smoothies banane mein bahut easy love the blender",
        "Packaging tuta hua tha seal broken on delivery",
        "Cap tight nahi tha liquid leak ho gaya everywhere",
        "Wire tutne lagi hai bilkul bekar quality hai",
        "Motor garam ho gaya burning smell jaise kuch jal raha ho",
        "Perfect product exactly what I needed daily use",
        "Value for money product highly recommend to friends",
        "Fast charging speed unexpected in this price range",
        "Der se aaya aur tuta hua bhi tha very disappointed",
        "Cap khula hua tha yaar kya bakwas product hai",
        "Burning odor from motor terrible quality control",
        "Cable coating peel ho gai after just 3 days use",
        "Absolutely love this best purchase ever made"
    ]
    
    test_request = SurpriseBatchRequest(
        reviews_text='\n'.join(sample_reviews),
        product_name="SmartBottle Pro",
        category="Personal Care"
    )
    
    # Run the optimized pipeline
    start = time.time()
    
    result = await surprise_batch(test_request, BackgroundTasks())
    
    elapsed = time.time() - start
    
    return {
        "test": "surprise_batch_timing",
        "reviews_count": 20,
        "time_taken_seconds": elapsed,
        "target_seconds": 45,
        "passed": elapsed < 45,
        "pipeline_result": result
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    print("ReviewIQ API started - Database initialized")


def test_surprise_batch_timing():
    """
    Test function to verify surprise batch completes in <45 seconds.
    Run with: python -c "from main import test_surprise_batch_timing; test_surprise_batch_timing()"
    """
    import time
    import requests
    
    print("=" * 70)
    print("🧪 Surprise Batch Timing Test")
    print("=" * 70)
    print(f"Target: <45 seconds for 20 reviews")
    print()
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API not running. Start with: uvicorn main:app --reload")
            return False
    except:
        print("❌ API not running. Start with: uvicorn main:app --reload")
        return False
    
    print("✅ API is running")
    print("🚀 Running test with 20 sample reviews...")
    print()
    
    # Call test endpoint
    start = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/surprise-batch-test",
            timeout=60
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            
            print("📊 Results:")
            print(f"   API-reported time: {data.get('time_taken_seconds', 0):.2f}s")
            print(f"   Total round-trip: {elapsed:.2f}s")
            print(f"   Target: <45s")
            print()
            
            api_time = data.get('time_taken_seconds', 999)
            passed = api_time < 45
            
            if passed:
                print(f"✅ PASS - Processed in {api_time:.2f}s (well under 45s limit)")
            else:
                print(f"❌ FAIL - Took {api_time:.2f}s (exceeds 45s limit)")
            
            print()
            print("📈 Pipeline breakdown:")
            result = data.get('pipeline_result', {})
            print(f"   Reviews processed: {result.get('reviews_processed', 0)}")
            print(f"   Top issues found: {len(result.get('top_issues', []))}")
            
            if result.get('top_issues'):
                print("   Issues detected:")
                for issue in result['top_issues'][:3]:
                    print(f"      - {issue.get('feature_name')}: {issue.get('complaint_count')} complaints")
            
            print()
            print("=" * 70)
            return passed
            
        else:
            print(f"❌ Test failed with status {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if timing test requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test-timing":
        success = test_surprise_batch_timing()
        sys.exit(0 if success else 1)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

