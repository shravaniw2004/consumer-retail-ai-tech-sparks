"""
ReviewIQ Precompute Script
Processes entire dataset before demo to avoid 20-minute waits.
Completes in <5 minutes for 300 reviews using parallel processing.

Usage:
    python run_precompute.py --input-dir data/ --output-db data/precomputed.db --clear-existing
"""

import os
import sys
import csv
import json
import time
import argparse
import glob
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from preprocessor import ReviewPreprocessor
from deduplication import deduplicate_reviews, get_dedup_stats
from sentiment_engine import SentimentEngine, extract_sentiments
from ambiguity_detector import flag_ambiguous_reviews
from trend_detector import detect_all_trends, get_trend_summary
from escalation_scorer import EscalationScorer
from action_brief import ActionBriefGenerator
from sentiment_engine import FEATURE_TAXONOMY
from data.database import init_db, Review, FeatureExtraction, TrendWindow, Escalation, ActionBrief

# Constants
BATCH_SIZE = 5
MAX_WORKERS = 5
BASELINE_RATE = 5.0


class PrecomputePipeline:
    """End-to-end precompute pipeline for ReviewIQ demo."""
    
    def __init__(self, output_db: str, clear_existing: bool = False):
        self.output_db = output_db
        self.clear_existing = clear_existing
        
        # Initialize components
        self.preprocessor = ReviewPreprocessor()
        self.sentiment_engine = SentimentEngine()
        self.escalation_scorer = EscalationScorer()
        self.brief_generator = ActionBriefGenerator()
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_reviews': 0,
            'processed_reviews': 0,
            'extracted_features': 0,
            'trends_computed': 0,
            'escalations_scored': 0,
            'briefs_generated': 0
        }
        
        # Setup database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for precomputed results."""
        if self.clear_existing and os.path.exists(self.output_db):
            os.remove(self.output_db)
            print(f"🗑️  Cleared existing database: {self.output_db}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_db) or '.', exist_ok=True)
        
        # Initialize with SQLAlchemy models
        os.environ['DATABASE_URL'] = f"sqlite:///{self.output_db}"
        init_db()
        print(f"✅ Database initialized: {self.output_db}")
    
    def load_csv_files(self, input_dir: str) -> List[Dict]:
        """Load all CSV files from input directory."""
        csv_files = glob.glob(os.path.join(input_dir, "*_reviews.csv"))
        all_reviews = []
        
        print(f"\n📁 Loading CSV files from: {input_dir}")
        print(f"   Found {len(csv_files)} files")
        
        for csv_file in csv_files:
            product_name = Path(csv_file).stem.replace('_reviews', '').replace('_', ' ').title()
            
            # Detect category from filename
            category = self._detect_category(product_name)
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    review = {
                        'review_id': row.get('review_id', f'{product_name}_{i}'),
                        'product_name': row.get('product_name', product_name),
                        'category': row.get('category', category),
                        'review_text': row.get('review_text', row.get('clean_text', '')),
                        'original_text': row.get('original_text', row.get('review_text', '')),
                        'timestamp': row.get('timestamp', datetime.now().isoformat()),
                        'sentiment': row.get('sentiment', 'neutral'),
                        'is_seeded': row.get('is_seeded', 'False').lower() == 'true'
                    }
                    all_reviews.append(review)
        
        self.stats['total_reviews'] = len(all_reviews)
        print(f"   Loaded {len(all_reviews)} reviews")
        
        return all_reviews
    
    def _detect_category(self, product_name: str) -> str:
        """Detect category from product name."""
        name_lower = product_name.lower()
        if 'bottle' in name_lower or 'care' in name_lower:
            return "Personal Care"
        elif 'charge' in name_lower or 'bolt' in name_lower:
            return "Electronics"
        elif 'mix' in name_lower or 'blend' in name_lower or 'nutri' in name_lower:
            return "Food"
        return "General"
    
    def preprocess_and_deduplicate(self, reviews: List[Dict]) -> List[Dict]:
        """Run preprocessing and deduplication."""
        print("\n🔧 Stage 1: Preprocessing & Deduplication")
        start = time.time()
        
        # Preprocess all reviews
        processed = []
        for i, review in enumerate(reviews):
            text = review.get('review_text', '')
            result = self.preprocessor.process(text)
            processed.append({**review, **result})
            
            if (i + 1) % 50 == 0:
                print(f"   Preprocessed {i + 1}/{len(reviews)} reviews...")
        
        # Deduplicate
        print(f"   Running deduplication on {len(processed)} reviews...")
        deduplicated = deduplicate_reviews(processed)
        
        dedup_stats = get_dedup_stats(deduplicated)
        elapsed = time.time() - start
        
        print(f"   ✅ Complete in {elapsed:.1f}s")
        print(f"      - Unique: {dedup_stats.get('unique_reviews', len(deduplicated))}")
        print(f"      - Duplicates: {dedup_stats.get('duplicates_found', 0)}")
        print(f"      - Suspicious: {dedup_stats.get('suspicious_reviews', 0)}")
        
        return deduplicated
    
    def extract_sentiments_parallel(self, reviews: List[Dict]) -> List[Dict]:
        """Run batch sentiment extraction with parallel processing and caching."""
        print("\n🤖 Stage 2: Sentiment Extraction (5 reviews/call)")
        start = time.time()
        
        # Add metadata for extraction
        for review in reviews:
            review['clean_text'] = review.get('clean_text', review.get('review_text', ''))
            review['category'] = self._detect_category(review.get('product_name', ''))
        
        # Calculate batches
        total_batches = (len(reviews) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   Processing {len(reviews)} reviews in {total_batches} batches (max {MAX_WORKERS} parallel)")
        
        # Process with parallel executor
        results = self.sentiment_engine.process_all_reviews_parallel(
            reviews, max_workers=MAX_WORKERS, use_cache=True
        )
        
        # Combine results
        enriched = []
        for review, result in zip(reviews, results):
            enriched_review = {
                **review,
                "extracted_features": result.features,
                "overall_sentiment": result.overall_sentiment,
                "trust_indicators": result.trust_indicators,
                "human_review_needed": result.human_review_needed,
                "extraction_error": result.error_message,
            }
            enriched.append(enriched_review)
            
            # Count features
            self.stats['extracted_features'] += len(result.features)
        
        elapsed = time.time() - start
        print(f"   ✅ Complete in {elapsed:.1f}s")
        print(f"      - Processed: {len(results)} reviews")
        print(f"      - Features extracted: {self.stats['extracted_features']}")
        print(f"      - Human review needed: {sum(1 for r in results if r.human_review_needed)}")
        
        return enriched
    
    def detect_trends(self, reviews: List[Dict]) -> Dict[str, List]:
        """Run trend detection for each product."""
        print("\n📈 Stage 3: Trend Detection (Sliding Windows + Z-Score)")
        start = time.time()
        
        # Group by product
        products = {}
        for review in reviews:
            product = review.get('product_name', 'Unknown')
            if product not in products:
                products[product] = []
            products[product].append(review)
        
        all_trends = {}
        
        for product_name, product_reviews in products.items():
            print(f"   Analyzing {product_name} ({len(product_reviews)} reviews)...")
            
            category = self._detect_category(product_name)
            taxonomy = FEATURE_TAXONOMY.get(category, [])
            
            trends = detect_all_trends(
                product_name=product_name,
                reviews=product_reviews,
                feature_taxonomy=taxonomy,
                window_size=50,
                step=25
            )
            
            all_trends[product_name] = trends
            self.stats['trends_computed'] += len(trends)
        
        elapsed = time.time() - start
        print(f"   ✅ Complete in {elapsed:.1f}s")
        print(f"      - Products analyzed: {len(products)}")
        print(f"      - Trends computed: {self.stats['trends_computed']}")
        
        return all_trends
    
    def score_escalations(self, reviews: List[Dict], trends: Dict) -> List[Dict]:
        """Run escalation scoring for all issues."""
        print("\n🎯 Stage 4: Escalation Scoring")
        start = time.time()
        
        escalations = []
        
        for product_name, product_trends in trends.items():
            for trend in product_trends:
                # Calculate complaint rate from windows
                windows = trend.windows
                if not windows:
                    continue
                
                latest_window = windows[-1]
                complaint_rate = latest_window.get('complaint_rate', 0)
                
                # Build feature data
                feature_data = {
                    'sentiment_intensity': 0.8 if complaint_rate > 20 else 0.5,
                    'sentiment_type': 'negative',
                    'current_complaint_rate': complaint_rate,
                    'negative_mentions': int(complaint_rate / 100 * 50),  # Estimate
                    'total_reviews': 50,
                    'confidence_scores': [0.85],
                    'lifecycle_stage': trend.trend_direction,
                    'windows_ago': 0,
                    'baseline_rate': BASELINE_RATE
                }
                
                result = self.escalation_scorer.score_issue(
                    feature_name=trend.feature_name,
                    product_name=product_name,
                    feature_data=feature_data
                )
                
                escalations.append({
                    'product_name': product_name,
                    'feature_name': trend.feature_name,
                    'priority_score': result.priority_score,
                    'priority_level': result.priority_level,
                    'priority_color': result.priority_color,
                    'trend_direction': trend.trend_direction,
                    'max_severity': trend.max_severity,
                    'is_emerging': trend.is_emerging_issue,
                    'recommendation': result.recommendation
                })
                
                self.stats['escalations_scored'] += 1
        
        # Sort by priority
        escalations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        elapsed = time.time() - start
        print(f"   ✅ Complete in {elapsed:.1f}s")
        print(f"      - Escalations scored: {len(escalations)}")
        print(f"      - Critical (>80): {sum(1 for e in escalations if e['priority_score'] >= 80)}")
        print(f"      - High (>60): {sum(1 for e in escalations if 60 <= e['priority_score'] < 80)}")
        
        return escalations
    
    def generate_action_briefs(self, escalations: List[Dict]) -> List[Dict]:
        """Generate action briefs for top 3 issues per product."""
        print("\n📋 Stage 5: Generating Action Briefs")
        start = time.time()
        
        # Group escalations by product
        product_escalations = {}
        for e in escalations:
            product = e['product_name']
            if product not in product_escalations:
                product_escalations[product] = []
            product_escalations[product].append(e)
        
        briefs = []
        
        for product_name, issues in product_escalations.items():
            # Get top 3 issues
            top_issues = issues[:3]
            
            if not top_issues:
                continue
            
            print(f"   Generating brief for {product_name}...")
            
            brief, pdf_path = self.brief_generator.generate_and_export(
                product_name=product_name,
                top_issues=top_issues,
                save_pdf=True
            )
            
            if brief:
                briefs.append({
                    'product_name': product_name,
                    'severity_score': brief.severity_score,
                    'priority_level': brief.priority_level,
                    'pdf_path': pdf_path,
                    'executive_summary': brief.executive_summary,
                    'top_issues': [i['feature_name'] for i in top_issues]
                })
                self.stats['briefs_generated'] += 1
        
        elapsed = time.time() - start
        print(f"   ✅ Complete in {elapsed:.1f}s")
        print(f"      - Briefs generated: {len(briefs)}")
        
        return briefs
    
    def save_to_database(self, reviews: List[Dict], trends: Dict, escalations: List[Dict], briefs: List[Dict]):
        """Save all precomputed data to SQLite database."""
        print("\n💾 Stage 6: Saving to Database")
        start = time.time()
        
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from data.database import Base
        
        engine = create_engine(f"sqlite:///{self.output_db}")
        Session = sessionmaker(bind=engine)
        db = Session()
        
        try:
            # Clear existing data
            if self.clear_existing:
                for table in [FeatureExtraction, TrendWindow, Escalation, ActionBrief, Review]:
                    db.query(table).delete()
                db.commit()
            
            # Save reviews
            review_map = {}  # old_id -> new_id
            for review in reviews:
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
                review_map[review.get('review_id', '')] = db_review.id
            
            # Save feature extractions
            for review in reviews:
                review_id = review_map.get(review.get('review_id', ''))
                if not review_id:
                    continue
                
                for feature in review.get('extracted_features', []):
                    extraction = FeatureExtraction(
                        review_id=review_id,
                        feature_name=feature.get('feature', 'unknown'),
                        sentiment=feature.get('sentiment', 'neutral'),
                        intensity=feature.get('intensity', 0.5),
                        confidence=feature.get('confidence', 0.5),
                        ambiguity_flag=feature.get('flags', {}).get('ambiguous', False),
                        sarcasm_flag=feature.get('flags', {}).get('sarcasm', False),
                        evidence=feature.get('evidence', '')
                    )
                    db.add(extraction)
            
            # Save trend windows
            for product_name, trends_list in trends.items():
                for trend in trends_list:
                    for window in trend.windows:
                        tw = TrendWindow(
                            product_name=product_name,
                            feature_name=trend.feature_name,
                            window_label=window.get('window_id', 'W1'),
                            complaint_rate=window.get('complaint_rate', 0),
                            praise_rate=window.get('praise_rate', 0),
                            z_score=window.get('z_score'),
                            timestamp=datetime.now()
                        )
                        db.add(tw)
            
            # Save escalations
            for esc in escalations:
                escalation = Escalation(
                    product_name=esc['product_name'],
                    feature_name=esc['feature_name'],
                    severity_score=esc['priority_score'],
                    lifecycle_stage=esc['trend_direction'],
                    trend_direction=esc['trend_direction'],
                    priority_rank=int(esc['priority_score'])
                )
                db.add(escalation)
            
            # Save action briefs
            for brief in briefs:
                action_brief = ActionBrief(
                    product_name=brief['product_name'],
                    generated_text=brief.get('executive_summary', ''),
                    pdf_path=brief.get('pdf_path', ''),
                    created_at=datetime.now()
                )
                db.add(action_brief)
            
            db.commit()
            
            elapsed = time.time() - start
            print(f"   ✅ Complete in {elapsed:.1f}s")
            print(f"      - Reviews saved: {len(reviews)}")
            print(f"      - Extractions saved: {sum(len(r.get('extracted_features', [])) for r in reviews)}")
            print(f"      - Trends saved: {sum(len(t.windows) for tl in trends.values() for t in tl)}")
            print(f"      - Escalations saved: {len(escalations)}")
            print(f"      - Briefs saved: {len(briefs)}")
            
        except Exception as e:
            print(f"   ❌ Error saving to database: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def run(self, input_dir: str):
        """Run the complete precompute pipeline."""
        self.stats['start_time'] = time.time()
        
        print("=" * 70)
        print("🔥 ReviewIQ Precompute Pipeline")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input: {input_dir}")
        print(f"Output: {self.output_db}")
        print("=" * 70)
        
        # Stage 0: Load data
        print("\n📥 Loading Data...")
        reviews = self.load_csv_files(input_dir)
        
        if not reviews:
            print("❌ No reviews found. Exiting.")
            return False
        
        # Stage 1: Preprocess and deduplicate
        reviews = self.preprocess_and_deduplicate(reviews)
        
        # Stage 2: Sentiment extraction
        reviews = self.extract_sentiments_parallel(reviews)
        
        # Stage 3: Trend detection
        trends = self.detect_trends(reviews)
        
        # Stage 4: Escalation scoring
        escalations = self.score_escalations(reviews, trends)
        
        # Stage 5: Generate action briefs
        briefs = self.generate_action_briefs(escalations)
        
        # Stage 6: Save to database
        self.save_to_database(reviews, trends, escalations, briefs)
        
        # Summary
        total_time = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 70)
        print("✅ PRECOMPUTE COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Reviews processed: {self.stats['total_reviews']}")
        print(f"Features extracted: {self.stats['extracted_features']}")
        print(f"Trends computed: {self.stats['trends_computed']}")
        print(f"Escalations scored: {self.stats['escalations_scored']}")
        print(f"Briefs generated: {self.stats['briefs_generated']}")
        print("=" * 70)
        print(f"\n🎯 Demo-ready! Database: {self.output_db}")
        print("   Copy this file to replace the main database before demo.")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ReviewIQ dataset for fast demo"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory containing review CSV files"
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default="data/precomputed.db",
        help="Path to output SQLite database"
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing database before processing"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = PrecomputePipeline(
        output_db=args.output_db,
        clear_existing=args.clear_existing
    )
    
    success = pipeline.run(args.input_dir)
    
    if success:
        print("\n🚀 Ready for demo!")
        sys.exit(0)
    else:
        print("\n❌ Precompute failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
