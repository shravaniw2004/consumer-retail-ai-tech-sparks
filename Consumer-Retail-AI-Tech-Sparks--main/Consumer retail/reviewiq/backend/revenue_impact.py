"""
ReviewIQ v5 — Module 8: Predictive Revenue Impact Engine
Transforms ReviewIQ from monitoring tool into decision support tool.
Predicts rating drops and revenue at risk based on complaint trends.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────
# Configuration Constants (Industry Benchmarks)
# ──────────────────────────────────────────────────────────────

# Sensitivity: 0.08 star drop per 1% complaint rate increase
SENSITIVITY_COEFFICIENT = 0.08

# Conversion drop: 12% revenue loss per 1-star drop (Amazon internal data benchmark)
CONVERSION_DROP_PER_STAR = 0.12

# Default monthly revenue for demo
DEFAULT_MONTHLY_REVENUE = 500000  # ₹5,00,000

# Minimum rating floor
MIN_RATING = 1.0
MAX_RATING = 5.0


@dataclass
class RevenueScenario:
    """A single scenario prediction."""
    name: str
    action: str
    predicted_rating: float
    rating_change: float
    revenue_impact: float          # Negative = loss
    revenue_impact_pct: float      # Percentage change
    color: str                     # Dashboard color
    emoji: str


@dataclass
class RevenueImpactResult:
    """Complete revenue impact analysis."""
    product_name: str
    current_rating: float
    current_complaint_rate: float
    baseline_complaint_rate: float
    complaint_delta: float
    monthly_revenue: float
    
    # Three scenarios
    do_nothing: RevenueScenario
    partial_fix: RevenueScenario
    full_fix: RevenueScenario
    
    # Summary
    max_revenue_at_risk: float
    revenue_recoverable: float
    assumptions: List[str]


def predict_rating(
    current_rating: float,
    complaint_rate_delta: float,
    sensitivity: float = SENSITIVITY_COEFFICIENT,
) -> float:
    """
    Predict future rating based on complaint rate change.
    
    Formula: Predicted_Rating = Current_Rating - (delta × sensitivity)
    """
    predicted = current_rating - (complaint_rate_delta * sensitivity)
    return max(MIN_RATING, min(MAX_RATING, round(predicted, 1)))


def calculate_revenue_impact(
    monthly_revenue: float,
    star_drop: float,
    conversion_drop_per_star: float = CONVERSION_DROP_PER_STAR,
) -> float:
    """
    Calculate revenue at risk from a rating drop.
    
    Formula: Revenue_At_Risk = Monthly_Revenue × conversion_drop × star_drop
    """
    if star_drop <= 0:
        # Rating improved — positive revenue impact
        return abs(star_drop) * conversion_drop_per_star * monthly_revenue * 0.5  # Partial recovery
    
    return -(star_drop * conversion_drop_per_star * monthly_revenue)


def compute_revenue_impact(
    product_name: str,
    current_rating: float,
    current_complaint_rate: float,
    baseline_complaint_rate: float,
    monthly_revenue: float = DEFAULT_MONTHLY_REVENUE,
    top_issues: Optional[List[Dict]] = None,
) -> RevenueImpactResult:
    """
    Main function: compute revenue impact with 3 scenarios.
    
    Args:
        product_name: Product being analyzed
        current_rating: Current product rating (e.g., 4.3)
        current_complaint_rate: Latest complaint rate as percentage (e.g., 38%)
        baseline_complaint_rate: Original baseline rate (e.g., 8%)
        monthly_revenue: Monthly revenue in ₹
        top_issues: Optional list of top issues with fix potential
    """
    complaint_delta = current_complaint_rate - baseline_complaint_rate
    
    # ─── Scenario 1: Do Nothing ───
    do_nothing_rating = predict_rating(current_rating, complaint_delta)
    do_nothing_star_drop = current_rating - do_nothing_rating
    do_nothing_revenue = calculate_revenue_impact(monthly_revenue, do_nothing_star_drop)
    
    do_nothing = RevenueScenario(
        name="Do Nothing",
        action="No change — complaints continue at current rate",
        predicted_rating=do_nothing_rating,
        rating_change=-do_nothing_star_drop,
        revenue_impact=do_nothing_revenue,
        revenue_impact_pct=(do_nothing_revenue / monthly_revenue) * 100 if monthly_revenue > 0 else 0,
        color="#F44336",
        emoji="🔴",
    )
    
    # ─── Scenario 2: Partial Fix (address top issue only) ───
    # Assume fixing top issue reduces complaint rate by 60%
    partial_delta = complaint_delta * 0.4  # 60% reduction
    partial_rating = predict_rating(current_rating, partial_delta)
    partial_star_drop = current_rating - partial_rating
    partial_revenue = calculate_revenue_impact(monthly_revenue, partial_star_drop)
    
    partial_fix = RevenueScenario(
        name="Partial Fix",
        action="Address top issue only (e.g., packaging fix)",
        predicted_rating=partial_rating,
        rating_change=-partial_star_drop,
        revenue_impact=partial_revenue,
        revenue_impact_pct=(partial_revenue / monthly_revenue) * 100 if monthly_revenue > 0 else 0,
        color="#FF9800",
        emoji="🟡",
    )
    
    # ─── Scenario 3: Full Fix (address all critical issues) ───
    # Assume full fix reduces complaint rate by 90% and adds slight improvement
    full_delta = complaint_delta * 0.1 - 2  # Near-full reduction + slight boost
    full_rating = predict_rating(current_rating, full_delta)
    full_star_drop = current_rating - full_rating
    full_revenue = calculate_revenue_impact(monthly_revenue, full_star_drop)
    
    full_fix = RevenueScenario(
        name="Full Fix",
        action="Address all critical issues comprehensively",
        predicted_rating=full_rating,
        rating_change=-full_star_drop,
        revenue_impact=full_revenue,
        revenue_impact_pct=(full_revenue / monthly_revenue) * 100 if monthly_revenue > 0 else 0,
        color="#4CAF50",
        emoji="🟢",
    )
    
    # Summary
    max_at_risk = abs(do_nothing_revenue)
    recoverable = abs(do_nothing_revenue) - abs(partial_revenue) if partial_revenue < 0 else abs(do_nothing_revenue)
    
    assumptions = [
        f"Sensitivity coefficient: {SENSITIVITY_COEFFICIENT} star drop per 1% complaint increase",
        f"Conversion drop: {CONVERSION_DROP_PER_STAR*100}% revenue loss per 1-star drop",
        f"Monthly revenue baseline: ₹{monthly_revenue:,.0f}",
        "Partial fix assumes 60% complaint reduction for top issue",
        "Full fix assumes 90% reduction + proactive improvement",
        "Industry benchmark: Amazon seller analytics (2024)",
    ]
    
    return RevenueImpactResult(
        product_name=product_name,
        current_rating=current_rating,
        current_complaint_rate=current_complaint_rate,
        baseline_complaint_rate=baseline_complaint_rate,
        complaint_delta=complaint_delta,
        monthly_revenue=monthly_revenue,
        do_nothing=do_nothing,
        partial_fix=partial_fix,
        full_fix=full_fix,
        max_revenue_at_risk=max_at_risk,
        revenue_recoverable=recoverable,
        assumptions=assumptions,
    )


def compute_slider_prediction(
    current_rating: float,
    complaint_rate: float,
    monthly_revenue: float = DEFAULT_MONTHLY_REVENUE,
) -> Dict:
    """
    Interactive slider calculation for dashboard.
    Returns prediction for a given complaint rate value.
    """
    baseline = 8.0  # Default baseline 8%
    delta = complaint_rate - baseline
    
    predicted_rating = predict_rating(current_rating, delta)
    star_drop = current_rating - predicted_rating
    revenue_impact = calculate_revenue_impact(monthly_revenue, star_drop)
    
    return {
        "complaint_rate": complaint_rate,
        "predicted_rating": predicted_rating,
        "star_drop": round(star_drop, 2),
        "revenue_impact": round(revenue_impact),
        "revenue_impact_formatted": format_currency(revenue_impact),
        "risk_level": "CRITICAL" if star_drop > 1.5 else ("HIGH" if star_drop > 0.8 else ("MEDIUM" if star_drop > 0.3 else "LOW")),
        "risk_color": "#F44336" if star_drop > 1.5 else ("#FF9800" if star_drop > 0.8 else ("#FFEB3B" if star_drop > 0.3 else "#4CAF50")),
    }


def format_currency(amount: float) -> str:
    """Format amount in Indian currency notation."""
    prefix = "+" if amount > 0 else ""
    abs_amount = abs(amount)
    
    if abs_amount >= 10000000:  # 1 Crore
        return f"{prefix}₹{abs_amount/10000000:.1f} Cr"
    elif abs_amount >= 100000:  # 1 Lakh
        return f"{prefix}₹{abs_amount/100000:.1f}L"
    elif abs_amount >= 1000:
        return f"{prefix}₹{abs_amount/1000:.0f}K"
    else:
        return f"{prefix}₹{abs_amount:,.0f}"


def generate_scenario_table(result: RevenueImpactResult) -> List[Dict]:
    """Generate a table-friendly representation of the 3 scenarios."""
    return [
        {
            "Scenario": f"{result.do_nothing.emoji} {result.do_nothing.name}",
            "Action": result.do_nothing.action,
            "Predicted Rating": f"{result.do_nothing.predicted_rating}★",
            "Revenue Impact": format_currency(result.do_nothing.revenue_impact) + "/month",
            "color": result.do_nothing.color,
        },
        {
            "Scenario": f"{result.partial_fix.emoji} {result.partial_fix.name}",
            "Action": result.partial_fix.action,
            "Predicted Rating": f"{result.partial_fix.predicted_rating}★",
            "Revenue Impact": format_currency(result.partial_fix.revenue_impact) + "/month",
            "color": result.partial_fix.color,
        },
        {
            "Scenario": f"{result.full_fix.emoji} {result.full_fix.name}",
            "Action": result.full_fix.action,
            "Predicted Rating": f"{result.full_fix.predicted_rating}★",
            "Revenue Impact": format_currency(result.full_fix.revenue_impact) + "/month",
            "color": result.full_fix.color,
        },
    ]


if __name__ == "__main__":
    print("Revenue Impact Engine Test")
    print("=" * 70)
    
    # SmartBottle Pro example from v5 spec
    result = compute_revenue_impact(
        product_name="SmartBottle Pro",
        current_rating=4.3,
        current_complaint_rate=38.0,
        baseline_complaint_rate=8.0,
        monthly_revenue=500000,
    )
    
    print(f"\nProduct: {result.product_name}")
    print(f"Current Rating: {result.current_rating}★")
    print(f"Complaint Rate: {result.baseline_complaint_rate}% → {result.current_complaint_rate}%")
    print(f"Monthly Revenue: ₹{result.monthly_revenue:,.0f}")
    
    print(f"\n{'Scenario':<20} {'Rating':<15} {'Revenue Impact':<20}")
    print("-" * 55)
    
    for scenario in [result.do_nothing, result.partial_fix, result.full_fix]:
        print(f"{scenario.name:<17} {scenario.predicted_rating}★ {'>':>5} "
              f"{format_currency(scenario.revenue_impact)}/month")
    
    print(f"\nMax Revenue at Risk: {format_currency(-result.max_revenue_at_risk)}/month")
    print(f"Recoverable with Fix: {format_currency(result.revenue_recoverable)}/month")
    
    print(f"\nAssumptions:")
    for a in result.assumptions:
        print(f"  • {a}")
    
    # Test slider
    print(f"\n\nSlider Test (10% → 60%):")
    for rate in [10, 20, 30, 40, 50, 60]:
        pred = compute_slider_prediction(4.3, rate)
        print(f"  {rate}% → {pred['predicted_rating']}★ | {pred['revenue_impact_formatted']}/mo | {pred['risk_level']}")
