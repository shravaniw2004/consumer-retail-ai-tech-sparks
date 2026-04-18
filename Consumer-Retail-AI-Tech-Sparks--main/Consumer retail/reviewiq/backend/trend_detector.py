"""
ReviewIQ Trend Detector - Sliding Window Analysis and Statistical Anomaly Detection
Identifies emerging issues through time-series analysis of review sentiments.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session


# Statistical thresholds
Z_SCORE_THRESHOLD = 2.5  # Alert if |Z| > 2.5
EMERGING_THRESHOLD_MULTIPLIER = 2.0  # 2x baseline increase
EMERGING_THRESHOLD_POINTS = 10.0  # >10 percentage points increase


@dataclass
class TrendWindow:
    """Represents a single analysis window."""
    window_id: str
    start_idx: int
    end_idx: int
    reviews: List[Dict]
    baseline: bool = False  # True for first window used as baseline


@dataclass
class FeatureTrend:
    """Trend analysis for a single feature across windows."""
    feature_name: str
    product_name: str
    windows: List[Dict[str, Any]] = field(default_factory=list)
    z_scores: List[Optional[float]] = field(default_factory=list)
    trend_direction: str = "stable"
    max_severity: float = 0.0
    is_emerging_issue: bool = False


def create_windows(
    reviews: List[Dict],
    window_size: int = 50,
    step: int = 25
) -> List[TrendWindow]:
    """
    Create overlapping sliding windows for trend analysis.
    
    Args:
        reviews: List of reviews sorted by timestamp
        window_size: Number of reviews per window
        step: Step size between windows (overlap = window_size - step)
    
    Returns:
        List of TrendWindow objects with labels W1, W2, etc.
    """
    if not reviews:
        return []
    
    # Sort reviews by timestamp if available
    sorted_reviews = sorted(
        reviews,
        key=lambda r: r.get('timestamp', r.get('created_at', ''))
    )
    
    windows = []
    window_num = 1
    
    for start_idx in range(0, len(sorted_reviews), step):
        end_idx = min(start_idx + window_size, len(sorted_reviews))
        
        if end_idx - start_idx < window_size // 2:  # Skip partial windows at end
            break
        
        window_reviews = sorted_reviews[start_idx:end_idx]
        
        windows.append(TrendWindow(
            window_id=f"W{window_num}",
            start_idx=start_idx,
            end_idx=end_idx,
            reviews=window_reviews,
            baseline=(window_num == 1)  # First window is baseline
        ))
        
        window_num += 1
        
        # Stop if we've covered all reviews
        if end_idx >= len(sorted_reviews):
            break
    
    return windows


def compute_feature_rates(
    windows: List[TrendWindow],
    feature_taxonomy: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute complaint and praise rates for each feature in each window.
    
    Args:
        windows: List of TrendWindow objects
        feature_taxonomy: List of valid feature names to track
    
    Returns:
        Nested dict: product -> feature -> window_id -> {complaint_rate, praise_rate, total}
    """
    # Group reviews by product
    product_windows = defaultdict(list)
    for window in windows:
        product_groups = defaultdict(list)
        for review in window.reviews:
            product = review.get('product_name', 'Unknown')
            product_groups[product].append(review)
        
        for product, reviews in product_groups.items():
            product_windows[product].append({
                'window_id': window.window_id,
                'baseline': window.baseline,
                'reviews': reviews
            })
    
    # Compute rates per product, per feature, per window
    rates = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for product, product_window_list in product_windows.items():
        for window_data in product_window_list:
            window_id = window_data['window_id']
            reviews = window_data['reviews']
            
            # Count total reviews mentioning each feature
            feature_counts = defaultdict(lambda: {
                'total': 0,
                'negative': 0,
                'positive': 0,
                'neutral': 0,
                'mixed': 0
            })
            
            for review in reviews:
                extracted_features = review.get('extracted_features', [])
                
                for feature_data in extracted_features:
                    feature_name = feature_data.get('feature', '').lower()
                    sentiment = feature_data.get('sentiment', 'neutral').lower()
                    
                    # Only track features in taxonomy
                    if feature_name in [f.lower() for f in feature_taxonomy]:
                        feature_counts[feature_name]['total'] += 1
                        if sentiment == 'negative':
                            feature_counts[feature_name]['negative'] += 1
                        elif sentiment == 'positive':
                            feature_counts[feature_name]['positive'] += 1
                        elif sentiment == 'neutral':
                            feature_counts[feature_name]['neutral'] += 1
                        elif sentiment == 'mixed':
                            feature_counts[feature_name]['mixed'] += 1
            
            # Calculate rates
            for feature_name, counts in feature_counts.items():
                total = counts['total']
                if total > 0:
                    rates[product][feature_name][window_id] = {
                        'complaint_rate': (counts['negative'] / total) * 100,
                        'praise_rate': (counts['positive'] / total) * 100,
                        'neutral_rate': (counts['neutral'] / total) * 100,
                        'mixed_rate': (counts['mixed'] / total) * 100,
                        'total_mentions': total,
                        'raw_counts': counts
                    }
    
    return dict(rates)


def z_score_anomaly_detection(
    window_rates: Dict[str, Dict[str, float]],
    baseline_window: str
) -> Dict[str, Optional[float]]:
    """
    Calculate Z-scores comparing each window to baseline.
    
    Z = (current_complaint_rate - baseline_complaint_rate) / baseline_std_dev
    
    Args:
        window_rates: Dict of window_id -> rate data
        baseline_window: ID of baseline window (usually "W1")
    
    Returns:
        Dict of window_id -> Z-score or None if cannot calculate
    """
    if baseline_window not in window_rates:
        return {w: None for w in window_rates.keys()}
    
    baseline_rate = window_rates[baseline_window].get('complaint_rate', 0)
    
    # Calculate standard deviation across all windows
    complaint_rates = [
        data.get('complaint_rate', 0)
        for data in window_rates.values()
    ]
    
    if len(complaint_rates) < 2:
        return {w: None for w in window_rates.keys()}
    
    mean_rate = sum(complaint_rates) / len(complaint_rates)
    variance = sum((r - mean_rate) ** 2 for r in complaint_rates) / len(complaint_rates)
    std_dev = math.sqrt(variance) if variance > 0 else 0.01  # Avoid div by zero
    
    # Calculate Z-scores
    z_scores = {}
    for window_id, data in window_rates.items():
        current_rate = data.get('complaint_rate', 0)
        
        if std_dev > 0:
            z_score = (current_rate - baseline_rate) / std_dev
        else:
            z_score = 0.0 if current_rate == baseline_rate else None
        
        z_scores[window_id] = z_score
    
    return z_scores


def detect_emerging_issues(
    window_rates: Dict[str, Dict[str, float]],
    baseline_window: str
) -> Dict[str, bool]:
    """
    Detect features where complaint rate has significantly increased.
    
    Criteria:
    - Complaint rate increased >2x from baseline
    - AND increase >10 percentage points
    
    Args:
        window_rates: Dict of window_id -> rate data
        baseline_window: ID of baseline window
    
    Returns:
        Dict of window_id -> True if emerging issue detected
    """
    if baseline_window not in window_rates:
        return {w: False for w in window_rates.keys()}
    
    baseline_rate = window_rates[baseline_window].get('complaint_rate', 0)
    
    emerging_flags = {}
    for window_id, data in window_rates.items():
        current_rate = data.get('complaint_rate', 0)
        
        # Calculate increases
        if baseline_rate > 0:
            multiplier_increase = current_rate / baseline_rate
        else:
            multiplier_increase = float('inf') if current_rate > 0 else 1.0
        
        point_increase = current_rate - baseline_rate
        
        # Check both criteria
        is_emerging = (
            multiplier_increase >= EMERGING_THRESHOLD_MULTIPLIER and
            point_increase >= EMERGING_THRESHOLD_POINTS
        )
        
        emerging_flags[window_id] = is_emerging
    
    return emerging_flags


def classify_trend_direction(
    complaint_rates: List[float],
    z_scores: List[Optional[float]]
) -> str:
    """
    Classify the trend direction based on complaint rates and Z-scores.
    
    Returns:
        "spiking" - rapid increase in complaints
        "stable" - consistent rates within normal variation
        "declining" - decreasing complaint rate
    """
    if len(complaint_rates) < 2:
        return "stable"
    
    # Calculate slope (simple linear trend)
    n = len(complaint_rates)
    x_vals = list(range(n))
    
    x_mean = sum(x_vals) / n
    y_mean = sum(complaint_rates) / n
    
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, complaint_rates))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    # Check for significant Z-scores
    significant_z = any(
        z is not None and abs(z) > Z_SCORE_THRESHOLD
        for z in z_scores
    )
    
    # Classify based on slope and Z-scores
    if slope > 1.0 or significant_z:
        return "spiking"
    elif slope < -1.0:
        return "declining"
    else:
        return "stable"


def detect_all_trends(
    product_name: str,
    reviews: List[Dict],
    feature_taxonomy: List[str],
    window_size: int = 50,
    step: int = 25
) -> List[FeatureTrend]:
    """
    Detect all trends for a product across all features.
    
    Args:
        product_name: Name of product to analyze
        reviews: List of all reviews (will filter by product)
        feature_taxonomy: List of feature names to track
        window_size: Reviews per window
        step: Step size between windows
    
    Returns:
        List of FeatureTrend objects with window data, Z-scores, and severity
    """
    # Filter reviews by product
    product_reviews = [
        r for r in reviews
        if r.get('product_name') == product_name
    ]
    
    if len(product_reviews) < window_size:
        return []  # Not enough reviews for trend analysis
    
    # Create windows
    windows = create_windows(product_reviews, window_size, step)
    if len(windows) < 2:
        return []  # Need at least 2 windows for comparison
    
    # Compute feature rates
    all_rates = compute_feature_rates(windows, feature_taxonomy)
    product_rates = all_rates.get(product_name, {})
    
    trends = []
    baseline_window = "W1"
    
    for feature_name, window_data in product_rates.items():
        # Calculate Z-scores
        z_scores_dict = z_score_anomaly_detection(window_data, baseline_window)
        
        # Detect emerging issues
        emerging_flags = detect_emerging_issues(window_data, baseline_window)
        
        # Build window data list
        window_list = []
        z_score_list = []
        complaint_rates = []
        
        for window_id in sorted(window_data.keys()):
            data = window_data[window_id]
            z_score = z_scores_dict.get(window_id)
            is_emerging = emerging_flags.get(window_id, False)
            
            window_list.append({
                'window_id': window_id,
                'complaint_rate': data.get('complaint_rate', 0),
                'praise_rate': data.get('praise_rate', 0),
                'total_mentions': data.get('total_mentions', 0),
                'z_score': z_score,
                'is_anomaly': z_score is not None and abs(z_score) > Z_SCORE_THRESHOLD,
                'is_emerging': is_emerging,
            })
            
            z_score_list.append(z_score)
            complaint_rates.append(data.get('complaint_rate', 0))
        
        # Classify trend direction
        trend_direction = classify_trend_direction(complaint_rates, z_score_list)
        
        # Calculate max severity (highest absolute Z-score)
        valid_z_scores = [z for z in z_score_list if z is not None]
        max_severity = max([abs(z) for z in valid_z_scores], default=0.0)
        
        # Check if feature is an emerging issue overall
        is_emerging_issue = any(w.get('is_emerging', False) for w in window_list)
        
        trends.append(FeatureTrend(
            feature_name=feature_name,
            product_name=product_name,
            windows=window_list,
            z_scores=z_score_list,
            trend_direction=trend_direction,
            max_severity=max_severity,
            is_emerging_issue=is_emerging_issue
        ))
    
    # Sort by severity (highest Z-score first)
    trends.sort(key=lambda t: t.max_severity, reverse=True)
    
    return trends


def save_trend_windows(
    db_session: Session,
    trends: List[FeatureTrend],
    timestamp: Optional[datetime] = None
) -> List[int]:
    """
    Save trend window data to database.
    
    Args:
        db_session: SQLAlchemy session
        trends: List of FeatureTrend objects
        timestamp: Optional timestamp (defaults to now)
    
    Returns:
        List of created record IDs
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    # Import here to avoid circular imports
    from data.database import TrendWindow as TrendWindowModel
    
    record_ids = []
    
    for trend in trends:
        for window_data in trend.windows:
            record = TrendWindowModel(
                product_name=trend.product_name,
                feature_name=trend.feature_name,
                window_label=window_data['window_id'],
                complaint_rate=window_data['complaint_rate'],
                praise_rate=window_data['praise_rate'],
                z_score=window_data.get('z_score'),
                timestamp=timestamp
            )
            db_session.add(record)
            db_session.flush()  # Get ID without committing
            record_ids.append(record.id)
    
    db_session.commit()
    return record_ids


def get_trend_summary(trends: List[FeatureTrend]) -> Dict[str, Any]:
    """Generate human-readable summary of detected trends."""
    if not trends:
        return {
            'status': 'insufficient_data',
            'message': 'Not enough reviews for trend analysis'
        }
    
    emerging = [t for t in trends if t.is_emerging_issue]
    spiking = [t for t in trends if t.trend_direction == 'spiking']
    anomalies = [
        t for t in trends
        if any(z is not None and abs(z) > Z_SCORE_THRESHOLD for z in t.z_scores)
    ]
    
    return {
        'total_features_tracked': len(trends),
        'emerging_issues': len(emerging),
        'spiking_trends': len(spiking),
        'statistical_anomalies': len(anomalies),
        'top_concern': trends[0].feature_name if trends else None,
        'top_severity': trends[0].max_severity if trends else 0,
        'emerging_details': [
            {
                'feature': t.feature_name,
                'trend': t.trend_direction,
                'severity': t.max_severity,
                'worst_window': max(t.windows, key=lambda w: w.get('complaint_rate', 0))['window_id']
            }
            for t in emerging[:5]  # Top 5 emerging issues
        ]
    }


# Convenience function
def analyze_product_trends(
    product_name: str,
    reviews: List[Dict],
    category: str,
    db_session: Optional[Session] = None
) -> Dict[str, Any]:
    """
    One-shot trend analysis for a product.
    
    Args:
        product_name: Product to analyze
        reviews: All reviews (will filter by product)
        category: Product category for feature taxonomy
        db_session: Optional session to save results
    
    Returns:
        Summary dict with trends and recommendations
    """
    # Get feature taxonomy for category
    from sentiment_engine import FEATURE_TAXONOMY
    features = FEATURE_TAXONOMY.get(category, [])
    
    if not features:
        return {'error': f'Unknown category: {category}'}
    
    # Detect trends
    trends = detect_all_trends(product_name, reviews, features)
    
    # Save to database if session provided
    if db_session and trends:
        save_trend_windows(db_session, trends)
    
    # Generate summary
    summary = get_trend_summary(trends)
    summary['product_name'] = product_name
    summary['category'] = category
    summary['trends'] = trends
    
    return summary


if __name__ == "__main__":
    # Test with synthetic data
    from data.data_generation import generate_demo_dataset
    
    print("Trend Detector Test\n")
    print("=" * 80)
    
    # Generate test data
    reviews = generate_demo_dataset(seed=42)
    
    # Test SmartBottle Pro trends (should show packaging complaint increase)
    print("\nAnalyzing SmartBottle Pro trends...")
    summary = analyze_product_trends(
        product_name="SmartBottle Pro",
        reviews=reviews,
        category="Personal Care"
    )
    
    print(f"\nTrend Summary:")
    if summary.get('status') == 'insufficient_data':
        print(f"  Status: {summary['status']} - {summary['message']}")
    else:
        print(f"  Total features tracked: {summary.get('total_features_tracked', 0)}")
        print(f"  Emerging issues: {summary.get('emerging_issues', 0)}")
        print(f"  Spiking trends: {summary.get('spiking_trends', 0)}")
        print(f"  Top concern: {summary.get('top_concern')} (severity: {summary.get('top_severity', 0):.2f})")
        
        if summary.get('emerging_details'):
            print("\n  Emerging Issue Details:")
            for detail in summary['emerging_details']:
                print(f"    - {detail['feature']}: {detail['trend']} (severity: {detail['severity']:.2f})")
        
        # Show window details for top trend
        if summary.get('trends'):
            top_trend = summary['trends'][0]
            print(f"\n  Window details for '{top_trend.feature_name}':")
            for window in top_trend.windows[:4]:  # First 4 windows
                z_str = f"Z={window['z_score']:.2f}" if window['z_score'] else "Z=N/A"
                print(f"    {window['window_id']}: {window['complaint_rate']:.1f}% complaints, {z_str}")
