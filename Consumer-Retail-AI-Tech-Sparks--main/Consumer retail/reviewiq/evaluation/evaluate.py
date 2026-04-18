"""
ReviewIQ Evaluation Script
Verifies seeded trends are detected and calculates metrics.
"""

import os
import sys
import csv
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

from preprocessor import ReviewPreprocessor
from deduplication import deduplicate_reviews
from sentiment_engine import SentimentEngine, extract_sentiments
from trend_detector import detect_all_trends
from escalation_scorer import EscalationScorer
from vocabulary_detector import VocabularyTracker, detect_emerging_vocabulary
from sentiment_engine import FEATURE_TAXONOMY


class EvaluationMetrics:
    """Tracks evaluation metrics."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def add_result(self, test_name: str, passed: bool, actual: any, target: any, notes: str = ""):
        """Add a test result."""
        self.results.append({
            'test_name': test_name,
            'passed': passed,
            'actual': actual,
            'target': target,
            'notes': notes
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def get_summary(self) -> Dict:
        """Get summary of results."""
        total = self.passed + self.failed
        return {
            'total': total,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': (self.passed / total * 100) if total > 0 else 0
        }


def load_labelled_reviews(filepath: str) -> List[Dict]:
    """Load manually labelled reviews for precision/recall calculation."""
    reviews = []
    
    if not os.path.exists(filepath):
        print(f"Warning: Labelled reviews file not found: {filepath}")
        # Generate sample labelled data for demo
        return generate_sample_labelled_reviews()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append({
                'review_id': row.get('review_id', ''),
                'product_name': row.get('product_name', ''),
                'review_text': row.get('review_text', ''),
                # Manual labels (ground truth)
                'manual_features': json.loads(row.get('manual_features', '[]')),
                'manual_sentiment': row.get('manual_sentiment', 'neutral'),
                'is_cap_leakage': row.get('is_cap_leakage', 'false').lower() == 'true',
                'is_burning_smell': row.get('is_burning_smell', 'false').lower() == 'true',
                'is_cable_fraying': row.get('is_cable_fraying', 'false').lower() == 'true'
            })
    
    return reviews


def generate_sample_labelled_reviews() -> List[Dict]:
    """Generate sample labelled reviews for demo."""
    return [
        {
            'review_id': 'TEST001',
            'product_name': 'SmartBottle Pro',
            'review_text': 'Cap khula hua tha packing bilkul bakwas leaked completely',
            'manual_features': [{'feature': 'packaging', 'sentiment': 'negative'}],
            'manual_sentiment': 'negative',
            'is_cap_leakage': True,
            'is_burning_smell': False,
            'is_cable_fraying': False
        },
        {
            'review_id': 'TEST002',
            'product_name': 'SmartBottle Pro',
            'review_text': 'Great bottle but cap came loose during delivery',
            'manual_features': [{'feature': 'packaging', 'sentiment': 'negative'}],
            'manual_sentiment': 'mixed',
            'is_cap_leakage': True,
            'is_burning_smell': False,
            'is_cable_fraying': False
        },
        {
            'review_id': 'TEST003',
            'product_name': 'NutriMix Blender',
            'review_text': 'Motor se burning smell aa rahi hai after 2 uses',
            'manual_features': [{'feature': 'motor', 'sentiment': 'negative'}],
            'manual_sentiment': 'negative',
            'is_cap_leakage': False,
            'is_burning_smell': True,
            'is_cable_fraying': False
        },
        {
            'review_id': 'TEST004',
            'product_name': 'BoltCharge 20W',
            'review_text': 'Cable fraying after one week very poor quality',
            'manual_features': [{'feature': 'durability', 'sentiment': 'negative'}],
            'manual_sentiment': 'negative',
            'is_cap_leakage': False,
            'is_burning_smell': False,
            'is_cable_fraying': True
        }
    ]


def calculate_precision_recall(
    labelled_reviews: List[Dict],
    extracted_reviews: List[Dict]
) -> Dict:
    """Calculate precision and recall for feature extraction."""
    
    # Build lookup by review_id
    extracted_map = {r['review_id']: r for r in extracted_reviews}
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for labelled in labelled_reviews:
        review_id = labelled['review_id']
        manual_features = {f['feature'].lower() for f in labelled.get('manual_features', [])}
        
        extracted = extracted_map.get(review_id, {})
        extracted_features = {
            f.get('feature', '').lower() 
            for f in extracted.get('extracted_features', [])
        }
        
        # Calculate matches
        matches = manual_features & extracted_features
        missed = manual_features - extracted_features
        extra = extracted_features - manual_features
        
        true_positives += len(matches)
        false_negatives += len(missed)
        false_positives += len(extra)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1_score': round(f1 * 100, 1),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def verify_seeded_trends(reviews: List[Dict]) -> Tuple[EvaluationMetrics, Dict]:
    """Verify the 3 seeded trends are correctly detected."""
    metrics = EvaluationMetrics()
    
    print("\n🔍 Verifying Seeded Trends...")
    
    # Run trend detection for each product
    products = {}
    for review in reviews:
        product = review.get('product_name', 'Unknown')
        if product not in products:
            products[product] = []
        products[product].append(review)
    
    # Track detected trends
    detected_trends = {}
    
    for product_name, product_reviews in products.items():
        category = 'Personal Care' if 'bottle' in product_name.lower() else \
                   'Electronics' if 'charge' in product_name.lower() else \
                   'Food' if 'mix' in product_name.lower() else 'General'
        
        taxonomy = FEATURE_TAXONOMY.get(category, [])
        
        trends = detect_all_trends(
            product_name=product_name,
            reviews=product_reviews,
            feature_taxonomy=taxonomy,
            window_size=50,
            step=25
        )
        
        detected_trends[product_name] = trends
    
    # Test 1: SmartBottle Pro - packaging complaints
    print("\n   Test 1: SmartBottle Pro - Packaging Complaints")
    sb_trends = detected_trends.get('SmartBottle Pro', [])
    packaging_trend = None
    
    for t in sb_trends:
        if t.feature_name.lower() == 'packaging':
            packaging_trend = t
            break
    
    if packaging_trend and packaging_trend.windows:
        max_z = max(
            (w.get('z_score') or 0) for w in packaging_trend.windows
        )
        
        metrics.add_result(
            'SmartBottle Packaging Z-Score',
            max_z > 2.0,
            f"{max_z:.2f}",
            "> 2.0",
            f"Max Z-score across windows"
        )
        
        metrics.add_result(
            'SmartBottle Trend Direction',
            packaging_trend.trend_direction in ['growing', 'systemic', 'spiking'],
            packaging_trend.trend_direction,
            "growing/systemic/spiking",
            f"Final trend state"
        )
    else:
        metrics.add_result(
            'SmartBottle Packaging Z-Score',
            False,
            "Not detected",
            "> 2.0",
            "Packaging trend not found"
        )
        metrics.add_result(
            'SmartBottle Trend Direction',
            False,
            "Not detected",
            "growing/systemic",
            "Packaging trend not found"
        )
    
    # Test 2: NutriMix Blender - burning smell
    print("   Test 2: NutriMix Blender - Motor Burning Smell")
    nm_trends = detected_trends.get('NutriMix Blender', [])
    motor_trend = None
    
    for t in nm_trends:
        if any(word in t.feature_name.lower() for word in ['motor', 'smell', 'burning']):
            motor_trend = t
            break
    
    if motor_trend:
        metrics.add_result(
            'NutriMix Burning Smell Detected',
            True,
            motor_trend.feature_name,
            "motor/smell feature",
            f"Detected as: {motor_trend.feature_name}"
        )
        
        # Check if it's in the last window (as per seeding)
        last_complaint_rate = motor_trend.windows[-1].get('complaint_rate', 0) if motor_trend.windows else 0
        metrics.add_result(
            'NutriMix Last Window Complaint Rate',
            last_complaint_rate > 10,
            f"{last_complaint_rate:.1f}%",
            "> 10%",
            "Complaints in last window"
        )
    else:
        # Check vocabulary detection as fallback
        metrics.add_result(
            'NutriMix Burning Smell Detected',
            False,
            "Not detected",
            "motor/smell feature",
            "Feature extraction missed burning smell"
        )
    
    # Test 3: BoltCharge - cable fraying
    print("   Test 3: BoltCharge 20W - Cable Fraying")
    bc_trends = detected_trends.get('BoltCharge 20W', [])
    cable_trend = None
    
    for t in bc_trends:
        if any(word in t.feature_name.lower() for word in ['cable', 'wire', 'durability']):
            cable_trend = t
            break
    
    if cable_trend:
        metrics.add_result(
            'BoltCharge Cable Fraying Detected',
            True,
            cable_trend.feature_name,
            "cable/durability feature",
            f"Detected as: {cable_trend.feature_name}"
        )
    else:
        metrics.add_result(
            'BoltCharge Cable Fraying Detected',
            False,
            "Not detected",
            "cable/durability feature",
            "Feature extraction missed cable fraying"
        )
    
    return metrics, detected_trends


def verify_emerging_vocabulary(reviews: List[Dict]) -> EvaluationMetrics:
    """Verify cap leakage emerging vocabulary is detected."""
    metrics = EvaluationMetrics()
    
    print("\n📚 Verifying Emerging Vocabulary...")
    
    # Get SmartBottle reviews
    sb_reviews = [r for r in reviews if r.get('product_name') == 'SmartBottle Pro']
    
    if len(sb_reviews) < 100:
        print("   Warning: Not enough SmartBottle reviews for vocabulary detection")
        return metrics
    
    # Split into windows
    baseline = sb_reviews[:50]
    current = sb_reviews[50:100]
    
    # Detect emerging vocabulary
    emerging = detect_emerging_vocabulary(
        current, baseline, min_appearances=3, baseline_threshold=1
    )
    
    # Check for cap leakage terms
    cap_terms = ['cap', 'leak', 'khula', 'packing', 'seal']
    found_cap_terms = []
    
    for item in emerging:
        phrase = item['phrase'].lower()
        if any(term in phrase for term in cap_terms):
            found_cap_terms.append(item)
    
    metrics.add_result(
        'Cap Leakage Vocabulary Detected',
        len(found_cap_terms) >= 2,
        f"{len(found_cap_terms)} terms",
        "≥ 2 terms",
        f"Found: {[t['phrase'] for t in found_cap_terms[:3]]}"
    )
    
    # Check top emerging phrase
    if emerging:
        top_phrase = emerging[0]
        metrics.add_result(
            'Top Emerging Phrase Count',
            top_phrase['current_count'] >= 3,
            f"{top_phrase['current_count']} mentions",
            "≥ 3 mentions",
            f"Phrase: '{top_phrase['phrase']}'"
        )
    
    return metrics


def generate_report(
    trend_metrics: EvaluationMetrics,
    vocab_metrics: EvaluationMetrics,
    pr_metrics: Dict,
    output_path: str
):
    """Generate evaluation report in markdown."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# ReviewIQ Evaluation Report

**Generated:** {timestamp}

---

## Executive Summary

| Metric | Result |
|--------|--------|
| **Overall Pass Rate** | {trend_metrics.get_summary()['pass_rate']:.1f}% |
| **Trend Detection Tests** | {trend_metrics.passed}/{trend_metrics.passed + trend_metrics.failed} passed |
| **Vocabulary Tests** | {vocab_metrics.passed}/{vocab_metrics.passed + vocab_metrics.failed} passed |
| **Precision** | {pr_metrics.get('precision', 0):.1f}% |
| **Recall** | {pr_metrics.get('recall', 0):.1f}% |
| **F1 Score** | {pr_metrics.get('f1_score', 0):.1f}% |

---

## Seeded Trend Detection Tests

### SmartBottle Pro - Packaging Complaints (Cap Leakage)

| Test | Status | Actual | Target | Notes |
|------|--------|--------|--------|-------|
"""
    
    # Add SmartBottle tests
    for result in trend_metrics.results:
        if 'SmartBottle' in result['test_name']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"| {result['test_name']} | {status} | {result['actual']} | {result['target']} | {result['notes']} |\n"
    
    report += """
### NutriMix Blender - Motor Burning Smell

| Test | Status | Actual | Target | Notes |
|------|--------|--------|--------|-------|
"""
    
    for result in trend_metrics.results:
        if 'NutriMix' in result['test_name']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"| {result['test_name']} | {status} | {result['actual']} | {result['target']} | {result['notes']} |\n"
    
    report += """
### BoltCharge 20W - Cable Fraying

| Test | Status | Actual | Target | Notes |
|------|--------|--------|--------|-------|
"""
    
    for result in trend_metrics.results:
        if 'BoltCharge' in result['test_name']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"| {result['test_name']} | {status} | {result['actual']} | {result['target']} | {result['notes']} |\n"
    
    report += f"""

---

## Emerging Vocabulary Detection

| Test | Status | Actual | Target | Notes |
|------|--------|--------|--------|-------|
"""
    
    for result in vocab_metrics.results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        report += f"| {result['test_name']} | {status} | {result['actual']} | {result['target']} | {result['notes']} |\n"
    
    report += f"""

---

## Feature Extraction Performance

### Precision & Recall (vs Manual Labels)

| Metric | Value |
|--------|-------|
| Precision | {pr_metrics.get('precision', 0):.1f}% |
| Recall | {pr_metrics.get('recall', 0):.1f}% |
| F1 Score | {pr_metrics.get('f1_score', 0):.1f}% |

### Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | {pr_metrics.get('true_positives', 0)} (TP) | {pr_metrics.get('false_negatives', 0)} (FN) |
| **Actual Negative** | {pr_metrics.get('false_positives', 0)} (FP) | - |

---

## Conclusion

"""
    
    overall_pass = trend_metrics.get_summary()['pass_rate'] >= 70
    
    if overall_pass:
        report += """✅ **Evaluation PASSED**

The ReviewIQ system successfully detected the seeded trends and meets the required performance thresholds.
"""
    else:
        report += """❌ **Evaluation FAILED**

The system did not meet all required thresholds. Review the failed tests above.
"""
    
    report += f"""
---

## Appendix: Test Methodology

1. **Seeded Trends**: 300 synthetic reviews with 3 intentional complaint trends
   - SmartBottle Pro: Cap leakage (3→7→19 across windows)
   - NutriMix Blender: Motor burning smell (last 50 reviews)
   - BoltCharge 20W: Cable fraying (middle window)

2. **Detection Criteria**:
   - Z-score > 2.0 indicates statistical significance
   - Trend direction must show growth pattern
   - Emerging vocabulary must include cap-related terms

3. **Precision/Recall**: Calculated against 50 manually labelled reviews
"""
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📝 Report written to: {output_path}")
    
    return report


def main():
    """Run evaluation suite."""
    print("=" * 70)
    print("🔍 ReviewIQ Evaluation Suite")
    print("=" * 70)
    
    # Load data
    print("\n📁 Loading test data...")
    
    # Try to load generated data first
    data_files = [
        os.path.join('..', 'data', 'all_reviews.csv'),
        os.path.join('data', 'all_reviews.csv'),
        'all_reviews.csv'
    ]
    
    reviews = []
    for data_file in data_files:
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    reviews.append({
                        'review_id': row.get('review_id', ''),
                        'product_name': row.get('product_name', ''),
                        'category': row.get('category', ''),
                        'review_text': row.get('review_text', row.get('clean_text', '')),
                        'is_seeded': row.get('is_seeded', 'false').lower() == 'true'
                    })
            print(f"   Loaded {len(reviews)} reviews from {data_file}")
            break
    
    if not reviews:
        print("   Generating synthetic data...")
        from data_generation import generate_demo_dataset
        reviews_raw = generate_demo_dataset(seed=42)
        reviews = [
            {
                'review_id': r.get('review_id', f'R{i}'),
                'product_name': r.get('product_name', 'Unknown'),
                'category': r.get('category', ''),
                'review_text': r.get('review_text', ''),
                'is_seeded': r.get('is_seeded', False)
            }
            for i, r in enumerate(reviews_raw)
        ]
        print(f"   Generated {len(reviews)} reviews")
    
    # Process reviews through pipeline
    print("\n🔧 Processing reviews through pipeline...")
    preprocessor = ReviewPreprocessor()
    sentiment_engine = SentimentEngine()
    
    # Preprocess
    processed = []
    for review in reviews:
        result = preprocessor.process(review['review_text'])
        processed.append({**review, **result})
    
    # Extract sentiments (with caching)
    print("   Running sentiment extraction...")
    results = sentiment_engine.process_all_reviews_parallel(processed, max_workers=3, use_cache=True)
    
    enriched = []
    for review, result in zip(processed, results):
        enriched.append({
            **review,
            "extracted_features": result.features,
            "overall_sentiment": result.overall_sentiment
        })
    
    print(f"   Processed {len(enriched)} reviews")
    
    # Run evaluations
    print("\n" + "=" * 70)
    print("Running Evaluation Tests...")
    print("=" * 70)
    
    # 1. Seeded trends
    trend_metrics, detected_trends = verify_seeded_trends(enriched)
    
    # 2. Emerging vocabulary
    vocab_metrics = verify_emerging_vocabulary(enriched)
    
    # 3. Precision/Recall
    print("\n📊 Calculating Precision/Recall...")
    labelled_reviews = load_labelled_reviews('test_reviews_labelled.csv')
    pr_metrics = calculate_precision_recall(labelled_reviews, enriched)
    print(f"   Precision: {pr_metrics['precision']:.1f}%")
    print(f"   Recall: {pr_metrics['recall']:.1f}%")
    print(f"   F1 Score: {pr_metrics['f1_score']:.1f}%")
    
    # Generate report
    print("\n" + "=" * 70)
    print("Generating Report...")
    print("=" * 70)
    
    report_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.md')
    report = generate_report(trend_metrics, vocab_metrics, pr_metrics, report_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Trend Detection: {trend_metrics.passed}/{trend_metrics.passed + trend_metrics.failed} tests passed")
    print(f"Vocabulary Detection: {vocab_metrics.passed}/{vocab_metrics.passed + vocab_metrics.failed} tests passed")
    print(f"Precision: {pr_metrics['precision']:.1f}%")
    print(f"Recall: {pr_metrics['recall']:.1f}%")
    print(f"F1 Score: {pr_metrics['f1_score']:.1f}%")
    print("=" * 70)
    
    overall_pass = trend_metrics.get_summary()['pass_rate'] >= 70 and pr_metrics['f1_score'] >= 60
    
    if overall_pass:
        print("\n✅ EVALUATION PASSED")
    else:
        print("\n❌ EVALUATION FAILED")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
