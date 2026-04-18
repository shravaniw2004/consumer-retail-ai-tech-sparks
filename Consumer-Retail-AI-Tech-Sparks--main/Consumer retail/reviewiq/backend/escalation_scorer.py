"""
ReviewIQ Escalation Scorer - 6-Factor Weighted Priority Calculation
Prioritizes customer issues based on severity, velocity, frequency, confidence, reach, and recency.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


# Weight constants from v4 spec
WEIGHTS = {
    'severity': 0.30,
    'trend_velocity': 0.25,
    'frequency': 0.20,
    'confidence': 0.15,
    'reach': 0.07,
    'recency': 0.03
}

# Priority classification thresholds
PRIORITY_THRESHOLDS = {
    'critical': 80,  # Red
    'high': 60,      # Orange
    'medium': 40,    # Yellow
    'low': 0         # Green (anything below medium)
}


class PriorityLevel(Enum):
    """Priority classification levels."""
    CRITICAL = "critical"  # >80, Red
    HIGH = "high"          # 60-80, Orange
    MEDIUM = "medium"      # 40-60, Yellow
    LOW = "low"            # <40, Green


def calculate_severity(
    sentiment_intensity: float,
    sentiment_type: str = "negative"
) -> float:
    """
    Calculate severity score (0-100).
    
    Args:
        sentiment_intensity: 0.0 to 1.0 intensity from sentiment extraction
        sentiment_type: "negative", "mixed", "neutral", "positive"
    
    Returns:
        Severity score 0-100
        - High intensity negative = 100
        - Medium intensity = 60
        - Low intensity = 30
        - Positive/M/Neutral scaled down
    """
    if sentiment_type == "positive":
        return 0.0
    
    if sentiment_type in ["neutral", "mixed"]:
        # Mixed sentiment has some severity but reduced
        base_severity = sentiment_intensity * 50
        return base_severity
    
    # Negative sentiment
    if sentiment_intensity >= 0.8:
        return 100.0  # High intensity
    elif sentiment_intensity >= 0.5:
        return 60.0   # Medium intensity
    else:
        return 30.0   # Low intensity


def calculate_trend_velocity(
    current_rate: float,
    previous_rate: Optional[float] = None,
    baseline_rate: float = 5.0
) -> float:
    """
    Calculate trend velocity score (0-100).
    
    Formula: (current_rate - previous_rate) × 100, capped at 100
    If no previous rate, compare to baseline.
    
    Args:
        current_rate: Current window complaint rate (percentage)
        previous_rate: Previous window complaint rate (percentage)
        baseline_rate: Baseline complaint rate for comparison
    
    Returns:
        Velocity score 0-100 (absolute change magnitude, capped)
    """
    if previous_rate is not None:
        change = abs(current_rate - previous_rate)
    else:
        # Compare to baseline if no previous
        change = max(0, current_rate - baseline_rate)
    
    # Scale: 1% change = 10 points, but cap at 100
    velocity = min(100.0, change * 10)
    
    return velocity


def calculate_frequency(
    negative_mentions: int,
    total_reviews: int
) -> float:
    """
    Calculate frequency score (0-100).
    
    Percentage of reviews mentioning this feature negatively.
    
    Args:
        negative_mentions: Number of negative mentions
        total_reviews: Total number of reviews in window
    
    Returns:
        Frequency percentage 0-100
    """
    if total_reviews == 0:
        return 0.0
    
    frequency = (negative_mentions / total_reviews) * 100
    return min(100.0, frequency)


def calculate_confidence(
    confidence_scores: List[float]
) -> float:
    """
    Calculate average confidence score (0-100).
    
    Args:
        confidence_scores: List of confidence scores from Mistral API (0.0-1.0)
    
    Returns:
        Average confidence scaled to 0-100
    """
    if not confidence_scores:
        return 50.0  # Default mid confidence
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    return avg_confidence * 100


def calculate_reach(
    complaint_rate: float,
    lifecycle_stage: str = "detected"
) -> float:
    """
    Calculate reach score (0-100).
    
    - 100 if systemic (>20% of reviews complaining)
    - 50 if growing but not yet systemic
    - 20 if isolated (<5% of reviews)
    
    Args:
        complaint_rate: Percentage of reviews with complaints
        lifecycle_stage: Current lifecycle stage
    
    Returns:
        Reach score 0-100
    """
    if complaint_rate > 20:
        return 100.0  # Systemic
    elif complaint_rate > 5:
        # Growing - scale between 50 and 100
        return 50.0 + ((complaint_rate - 5) / 15) * 50
    else:
        # Isolated
        return max(20.0, complaint_rate * 4)  # Scale 0-5% to 0-20


def calculate_recency(
    windows_ago: int
) -> float:
    """
    Calculate recency score (0-100).
    
    - 100 if in last window (0 windows ago)
    - 50 if 1 window ago
    - 0 if 2+ windows ago
    
    Args:
        windows_ago: Number of windows since issue was active (0 = current)
    
    Returns:
        Recency score 0-100
    """
    if windows_ago == 0:
        return 100.0
    elif windows_ago == 1:
        return 50.0
    else:
        return 0.0


def calculate_escalation_score(feature_data: Dict[str, Any]) -> float:
    """
    Calculate escalation priority score using 6-factor weighted formula.
    
    Formula:
    Priority = (0.30 × Severity) + (0.25 × Trend_Velocity) + (0.20 × Frequency)
             + (0.15 × Confidence) + (0.07 × Reach) + (0.03 × Recency)
    
    Args:
        feature_data: Dictionary containing:
            - sentiment_intensity: float (0-1)
            - sentiment_type: str ("negative", "mixed", etc.)
            - current_complaint_rate: float (percentage)
            - previous_complaint_rate: float (percentage, optional)
            - negative_mentions: int
            - total_reviews: int
            - confidence_scores: List[float] (0-1 each)
            - lifecycle_stage: str
            - windows_ago: int (0 = current, 1 = 1 window ago, etc.)
            - baseline_rate: float (default 5.0)
    
    Returns:
        Priority score 0-100
    """
    # Extract data with defaults
    sentiment_intensity = feature_data.get('sentiment_intensity', 0.5)
    sentiment_type = feature_data.get('sentiment_type', 'negative')
    current_rate = feature_data.get('current_complaint_rate', 0.0)
    previous_rate = feature_data.get('previous_complaint_rate')
    negative_mentions = feature_data.get('negative_mentions', 0)
    total_reviews = feature_data.get('total_reviews', 1)
    confidence_scores = feature_data.get('confidence_scores', [0.7])
    lifecycle_stage = feature_data.get('lifecycle_stage', 'detected')
    windows_ago = feature_data.get('windows_ago', 0)
    baseline_rate = feature_data.get('baseline_rate', 5.0)
    
    # Calculate each factor
    severity = calculate_severity(sentiment_intensity, sentiment_type)
    velocity = calculate_trend_velocity(current_rate, previous_rate, baseline_rate)
    frequency = calculate_frequency(negative_mentions, total_reviews)
    confidence = calculate_confidence(confidence_scores)
    reach = calculate_reach(current_rate, lifecycle_stage)
    recency = calculate_recency(windows_ago)
    
    # Apply weighted formula
    priority = (
        WEIGHTS['severity'] * severity +
        WEIGHTS['trend_velocity'] * velocity +
        WEIGHTS['frequency'] * frequency +
        WEIGHTS['confidence'] * confidence +
        WEIGHTS['reach'] * reach +
        WEIGHTS['recency'] * recency
    )
    
    return round(priority, 2)


def classify_priority(score: float) -> Tuple[PriorityLevel, str]:
    """
    Classify priority score into level and color.
    
    Args:
        score: Priority score 0-100
    
    Returns:
        Tuple of (PriorityLevel, color)
        - >80: Critical, Red
        - 60-80: High, Orange
        - 40-60: Medium, Yellow
        - <40: Low, Green
    """
    if score >= PRIORITY_THRESHOLDS['critical']:
        return PriorityLevel.CRITICAL, "red"
    elif score >= PRIORITY_THRESHOLDS['high']:
        return PriorityLevel.HIGH, "orange"
    elif score >= PRIORITY_THRESHOLDS['medium']:
        return PriorityLevel.MEDIUM, "yellow"
    else:
        return PriorityLevel.LOW, "green"


@dataclass
class EscalationResult:
    """Complete escalation analysis result."""
    feature_name: str
    product_name: str
    priority_score: float
    priority_level: str
    priority_color: str
    factors: Dict[str, float]
    recommendation: str


class EscalationScorer:
    """Calculate and track escalation priorities for multiple issues."""
    
    def __init__(self, baseline_rate: float = 5.0):
        self.baseline_rate = baseline_rate
        self.escalations: List[EscalationResult] = []
    
    def score_issue(
        self,
        feature_name: str,
        product_name: str,
        feature_data: Dict[str, Any]
    ) -> EscalationResult:
        """
        Calculate escalation score for a single issue.
        
        Args:
            feature_name: Feature being analyzed
            product_name: Product name
            feature_data: Feature data dictionary
        
        Returns:
            EscalationResult with priority score and classification
        """
        # Add baseline to data
        feature_data = {**feature_data, 'baseline_rate': self.baseline_rate}
        
        # Calculate score
        score = calculate_escalation_score(feature_data)
        
        # Classify
        level, color = classify_priority(score)
        
        # Calculate factor breakdown
        factors = {
            'severity': calculate_severity(
                feature_data.get('sentiment_intensity', 0.5),
                feature_data.get('sentiment_type', 'negative')
            ),
            'trend_velocity': calculate_trend_velocity(
                feature_data.get('current_complaint_rate', 0.0),
                feature_data.get('previous_complaint_rate'),
                self.baseline_rate
            ),
            'frequency': calculate_frequency(
                feature_data.get('negative_mentions', 0),
                feature_data.get('total_reviews', 1)
            ),
            'confidence': calculate_confidence(
                feature_data.get('confidence_scores', [0.7])
            ),
            'reach': calculate_reach(
                feature_data.get('current_complaint_rate', 0.0),
                feature_data.get('lifecycle_stage', 'detected')
            ),
            'recency': calculate_recency(
                feature_data.get('windows_ago', 0)
            )
        }
        
        # Generate recommendation
        recommendation = self._generate_recommendation(level, factors)
        
        result = EscalationResult(
            feature_name=feature_name,
            product_name=product_name,
            priority_score=score,
            priority_level=level.value,
            priority_color=color,
            factors=factors,
            recommendation=recommendation
        )
        
        self.escalations.append(result)
        return result
    
    def _generate_recommendation(
        self,
        level: PriorityLevel,
        factors: Dict[str, float]
    ) -> str:
        """Generate action recommendation based on priority and factors."""
        if level == PriorityLevel.CRITICAL:
            if factors['severity'] >= 80:
                return "Immediate action required: High-severity systemic issue"
            elif factors['trend_velocity'] >= 70:
                return "Immediate action required: Rapidly escalating issue"
            else:
                return "Immediate action required: Widespread customer impact"
        
        elif level == PriorityLevel.HIGH:
            if factors['reach'] >= 70:
                return "Urgent attention: Issue affecting significant customer base"
            elif factors['trend_velocity'] >= 50:
                return "Urgent attention: Issue showing strong growth trend"
            else:
                return "Prioritize: Issue with substantial customer impact"
        
        elif level == PriorityLevel.MEDIUM:
            if factors['trend_velocity'] >= 40:
                return "Monitor closely: Issue showing growth potential"
            else:
                return "Monitor: Issue with moderate customer impact"
        
        else:  # LOW
            return "Track: Low priority issue, continue monitoring"
    
    def score_multiple(
        self,
        issues_data: List[Dict[str, Any]]
    ) -> List[EscalationResult]:
        """
        Score multiple issues and return sorted by priority.
        
        Args:
            issues_data: List of dicts with keys:
                - feature_name
                - product_name
                - feature_data
        
        Returns:
            List of EscalationResult sorted by priority (highest first)
        """
        results = []
        
        for issue in issues_data:
            result = self.score_issue(
                feature_name=issue['feature_name'],
                product_name=issue['product_name'],
                feature_data=issue['feature_data']
            )
            results.append(result)
        
        # Sort by priority score (descending)
        results.sort(key=lambda x: x.priority_score, reverse=True)
        return results
    
    def get_critical_issues(self) -> List[EscalationResult]:
        """Get all critical priority issues."""
        return [e for e in self.escalations if e.priority_level == 'critical']
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all escalations."""
        if not self.escalations:
            return {'total_issues': 0}
        
        level_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for e in self.escalations:
            level_counts[e.priority_level] += 1
        
        avg_score = sum(e.priority_score for e in self.escalations) / len(self.escalations)
        
        return {
            'total_issues': len(self.escalations),
            'priority_distribution': level_counts,
            'average_priority_score': round(avg_score, 2),
            'highest_priority': max(e.priority_score for e in self.escalations),
            'critical_issues': level_counts['critical'],
            'top_concern': self.escalations[0].feature_name if self.escalations else None
        }


def priority_to_action_brief(
    escalation: EscalationResult,
    lifecycle_stage: str = "detected"
) -> Dict[str, Any]:
    """
    Convert escalation result to action brief format.
    
    Args:
        escalation: EscalationResult object
        lifecycle_stage: Current lifecycle stage
    
    Returns:
        Action brief dictionary
    """
    urgency_map = {
        'critical': 'immediate',
        'high': 'urgent',
        'medium': 'planned',
        'low': 'track'
    }
    
    return {
        'product_name': escalation.product_name,
        'feature_name': escalation.feature_name,
        'priority_score': escalation.priority_score,
        'priority_level': escalation.priority_level,
        'lifecycle_stage': lifecycle_stage,
        'urgency': urgency_map.get(escalation.priority_level, 'track'),
        'recommendation': escalation.recommendation,
        'factor_breakdown': escalation.factors,
        'requires_attention': escalation.priority_level in ['critical', 'high']
    }


if __name__ == "__main__":
    # Test escalation scoring
    print("Escalation Scorer Test\n")
    print("=" * 80)
    
    # Test case 1: Critical systemic packaging issue
    print("\nTest 1: SmartBottle Pro - Packaging (Systemic, High Severity)")
    critical_data = {
        'sentiment_intensity': 0.9,
        'sentiment_type': 'negative',
        'current_complaint_rate': 28.0,
        'previous_complaint_rate': 22.0,
        'negative_mentions': 28,
        'total_reviews': 100,
        'confidence_scores': [0.95, 0.92, 0.88],
        'lifecycle_stage': 'systemic',
        'windows_ago': 0,
        'baseline_rate': 5.0
    }
    
    score = calculate_escalation_score(critical_data)
    level, color = classify_priority(score)
    
    print(f"  Score: {score}")
    print(f"  Priority: {level.value} ({color})")
    print(f"  Factor breakdown:")
    print(f"    Severity: {calculate_severity(0.9, 'negative'):.1f} × {WEIGHTS['severity']} = {calculate_severity(0.9, 'negative') * WEIGHTS['severity']:.1f}")
    print(f"    Velocity: {calculate_trend_velocity(28.0, 22.0, 5.0):.1f} × {WEIGHTS['trend_velocity']} = {calculate_trend_velocity(28.0, 22.0, 5.0) * WEIGHTS['trend_velocity']:.1f}")
    print(f"    Frequency: {calculate_frequency(28, 100):.1f} × {WEIGHTS['frequency']} = {calculate_frequency(28, 100) * WEIGHTS['frequency']:.1f}")
    print(f"    Confidence: {calculate_confidence([0.95, 0.92, 0.88]):.1f} × {WEIGHTS['confidence']} = {calculate_confidence([0.95, 0.92, 0.88]) * WEIGHTS['confidence']:.1f}")
    print(f"    Reach: {calculate_reach(28.0, 'systemic'):.1f} × {WEIGHTS['reach']} = {calculate_reach(28.0, 'systemic') * WEIGHTS['reach']:.1f}")
    print(f"    Recency: {calculate_recency(0):.1f} × {WEIGHTS['recency']} = {calculate_recency(0) * WEIGHTS['recency']:.1f}")
    
    # Test case 2: Growing but not yet critical
    print("\nTest 2: BoltCharge - Cable (Growing, Medium Severity)")
    growing_data = {
        'sentiment_intensity': 0.6,
        'sentiment_type': 'negative',
        'current_complaint_rate': 15.0,
        'previous_complaint_rate': 8.0,
        'negative_mentions': 15,
        'total_reviews': 100,
        'confidence_scores': [0.75, 0.80],
        'lifecycle_stage': 'growing',
        'windows_ago': 0,
        'baseline_rate': 5.0
    }
    
    score2 = calculate_escalation_score(growing_data)
    level2, color2 = classify_priority(score2)
    print(f"  Score: {score2} → {level2.value} ({color2})")
    
    # Test case 3: Low priority isolated issue
    print("\nTest 3: Low Priority Issue")
    low_data = {
        'sentiment_intensity': 0.3,
        'sentiment_type': 'negative',
        'current_complaint_rate': 3.0,
        'negative_mentions': 3,
        'total_reviews': 100,
        'confidence_scores': [0.60],
        'lifecycle_stage': 'detected',
        'windows_ago': 1,
        'baseline_rate': 5.0
    }
    
    score3 = calculate_escalation_score(low_data)
    level3, color3 = classify_priority(score3)
    print(f"  Score: {score3} → {level3.value} ({color3})")
    
    # Test batch scoring
    print("\n" + "=" * 80)
    print("\nBatch Scoring Test:")
    
    scorer = EscalationScorer()
    
    issues = [
        {
            'feature_name': 'packaging',
            'product_name': 'SmartBottle Pro',
            'feature_data': critical_data
        },
        {
            'feature_name': 'cable',
            'product_name': 'BoltCharge 20W',
            'feature_data': growing_data
        },
        {
            'feature_name': 'motor',
            'product_name': 'NutriMix Blender',
            'feature_data': low_data
        }
    ]
    
    results = scorer.score_multiple(issues)
    
    print(f"\nRanked Priorities:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.feature_name} ({result.product_name}): {result.priority_score} - {result.priority_level}")
        print(f"     Recommendation: {result.recommendation}")
    
    print(f"\nSummary: {scorer.get_summary()}")
