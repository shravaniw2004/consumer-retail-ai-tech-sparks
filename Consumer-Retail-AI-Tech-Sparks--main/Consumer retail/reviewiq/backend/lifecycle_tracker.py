"""
ReviewIQ Lifecycle Tracker - 6-Stage State Machine for Issue Evolution
Tracks how customer complaints evolve from detection to resolution.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LifecycleStage(Enum):
    """Six stages of issue lifecycle."""
    DETECTED = "detected"
    GROWING = "growing"
    SYSTEMIC = "systemic"
    PERSISTENT = "persistent"
    RESOLVING = "resolving"
    RESOLVED = "resolved"


# Threshold constants
BASELINE_RATE = 5.0  # 5% baseline complaint rate
SIGNIFICANT_THRESHOLD = 5.0  # +5% above baseline
CONSECUTIVE_SYSTEMIC_THRESHOLD = 2  # 2+ consecutive windows
PERSISTENT_WINDOW_THRESHOLD = 3  # 3+ windows
SYSTEMIC_MULTIPLIER = 2.0  # 2x baseline


@dataclass
class LifecycleTransition:
    """Represents a stage transition with reason."""
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    reason: str
    timestamp: Optional[str] = None


@dataclass
class IssueHistory:
    """History of an issue's complaint rates across windows."""
    feature_name: str
    product_name: str
    baseline_rate: float
    windows: List[Tuple[str, float, int]] = field(default_factory=list)  # (label, rate, mentions)
    current_stage: LifecycleStage = LifecycleStage.DETECTED
    transitions: List[LifecycleTransition] = field(default_factory=list)


def compute_lifecycle_stage(
    issue_history: List[Tuple[str, float]],
    baseline_rate: float = BASELINE_RATE,
    mention_counts: Optional[List[int]] = None
) -> Tuple[LifecycleStage, str]:
    """
    Compute the current lifecycle stage based on issue history.
    
    Args:
        issue_history: List of (window_label, complaint_rate) tuples, ordered chronologically
        baseline_rate: Baseline complaint rate (default 5%)
        mention_counts: Optional list of mention counts per window (for "detected" rule)
    
    Returns:
        Tuple of (LifecycleStage, transition_reason)
    
    Rules:
    - detected: first appearance (≥3 mentions, rate > baseline + 5%)
    - growing: latest rate > previous rate
    - systemic: rate > 2× baseline for 2+ consecutive windows
    - persistent: 3+ windows above 2× baseline with no decline
    - resolving: latest < previous but still > baseline + 5%
    - resolved: latest <= baseline + 5%
    """
    if not issue_history:
        return LifecycleStage.DETECTED, "No history available"
    
    # Ensure history is sorted by window order
    sorted_history = sorted(issue_history, key=lambda x: x[0])
    
    # Extract rates
    rates = [rate for _, rate in sorted_history]
    latest_rate = rates[-1]
    
    # Not enough data - stay at detected
    if len(rates) == 1:
        # Check detection criteria
        mentions = mention_counts[0] if mention_counts else 3
        if mentions >= 3 and latest_rate > baseline_rate + SIGNIFICANT_THRESHOLD:
            return LifecycleStage.DETECTED, f"First appearance with {latest_rate:.1f}% (≥3 mentions, >baseline+5%)"
        return LifecycleStage.DETECTED, "Insufficient data for classification"
    
    previous_rate = rates[-2] if len(rates) >= 2 else baseline_rate
    
    # Calculate systemic threshold
    systemic_threshold = baseline_rate * SYSTEMIC_MULTIPLIER
    
    # Rule: resolved - latest <= baseline + 5%
    if latest_rate <= baseline_rate + SIGNIFICANT_THRESHOLD:
        return LifecycleStage.RESOLVED, f"Rate {latest_rate:.1f}% returned to baseline levels (≤{baseline_rate + SIGNIFICANT_THRESHOLD:.1f}%)"
    
    # Rule: resolving - latest < previous but still > baseline + 5%
    if latest_rate < previous_rate:
        return LifecycleStage.RESOLVING, f"Declining from {previous_rate:.1f}% to {latest_rate:.1f}%, but still above baseline"
    
    # Check for consecutive windows above systemic threshold
    consecutive_systemic = 0
    max_consecutive = 0
    windows_above_systemic = 0
    
    for rate in rates:
        if rate > systemic_threshold:
            consecutive_systemic += 1
            windows_above_systemic += 1
            max_consecutive = max(max_consecutive, consecutive_systemic)
        else:
            consecutive_systemic = 0
    
    # Rule: persistent - 3+ windows above 2× baseline with no decline
    if windows_above_systemic >= PERSISTENT_WINDOW_THRESHOLD and latest_rate >= previous_rate:
        return LifecycleStage.PERSISTENT, f"Issue chronic: {windows_above_systemic} windows above 2× baseline with sustained elevation"
    
    # Rule: systemic - rate > 2× baseline for 2+ consecutive windows
    if max_consecutive >= CONSECUTIVE_SYSTEMIC_THRESHOLD:
        return LifecycleStage.SYSTEMIC, f"Systemic problem: {max_consecutive} consecutive windows above 2× baseline ({systemic_threshold:.1f}%)"
    
    # Rule: growing - latest rate > previous rate (and not yet systemic)
    if latest_rate > previous_rate:
        return LifecycleStage.GROWING, f"Escalating: {previous_rate:.1f}% → {latest_rate:.1f}%"
    
    # Default: detected (issue exists but not actively growing)
    return LifecycleStage.DETECTED, f"Stable at {latest_rate:.1f}% but not meeting escalation criteria"


class IssueLifecycle:
    """
    Tracks the lifecycle of a customer complaint issue through its 6 stages.
    
    Stages: detected → growing → systemic → persistent → resolving → resolved
    """
    
    def __init__(
        self,
        feature_name: str,
        product_name: str,
        baseline_rate: float = BASELINE_RATE
    ):
        self.feature_name = feature_name
        self.product_name = product_name
        self.baseline_rate = baseline_rate
        self.history: IssueHistory = IssueHistory(
            feature_name=feature_name,
            product_name=product_name,
            baseline_rate=baseline_rate,
            windows=[],
            current_stage=LifecycleStage.DETECTED,
            transitions=[]
        )
        self._window_counter = 0
    
    def update(
        self,
        window_label: str,
        complaint_rate: float,
        mention_count: int = 0
    ) -> LifecycleStage:
        """
        Update the lifecycle with new window data and recompute stage.
        
        Args:
            window_label: Window identifier (e.g., "W1", "W2")
            complaint_rate: Complaint rate for this window
            mention_count: Number of mentions in this window
        
        Returns:
            Current lifecycle stage after update
        """
        # Add to history
        self.history.windows.append((window_label, complaint_rate, mention_count))
        self._window_counter += 1
        
        # Compute new stage
        issue_history = [(label, rate) for label, rate, _ in self.history.windows]
        mention_counts = [count for _, _, count in self.history.windows]
        
        new_stage, reason = compute_lifecycle_stage(
            issue_history,
            self.baseline_rate,
            mention_counts
        )
        
        # Check for transition
        old_stage = self.history.current_stage
        if new_stage != old_stage:
            transition = LifecycleTransition(
                from_stage=old_stage,
                to_stage=new_stage,
                reason=reason,
                timestamp=window_label
            )
            self.history.transitions.append(transition)
            self.history.current_stage = new_stage
        
        return self.history.current_stage
    
    def get_stage(self) -> str:
        """Get current lifecycle stage as string."""
        return self.history.current_stage.value
    
    def get_transition_reason(self) -> Optional[str]:
        """Get the reason for the most recent transition, if any."""
        if self.history.transitions:
            return self.history.transitions[-1].reason
        return None
    
    def get_full_history(self) -> Dict[str, Any]:
        """Get complete lifecycle history."""
        return {
            'feature_name': self.feature_name,
            'product_name': self.product_name,
            'baseline_rate': self.baseline_rate,
            'current_stage': self.get_stage(),
            'windows': [
                {
                    'window': label,
                    'complaint_rate': rate,
                    'mentions': count
                }
                for label, rate, count in self.history.windows
            ],
            'transitions': [
                {
                    'from': t.from_stage.value,
                    'to': t.to_stage.value,
                    'reason': t.reason,
                    'at_window': t.timestamp
                }
                for t in self.history.transitions
            ],
            'total_windows_tracked': len(self.history.windows)
        }
    
    def is_active_issue(self) -> bool:
        """Check if issue is still active (not resolved)."""
        return self.history.current_stage != LifecycleStage.RESOLVED
    
    def requires_attention(self) -> bool:
        """Check if issue requires immediate attention (systemic or persistent)."""
        return self.history.current_stage in [
            LifecycleStage.SYSTEMIC,
            LifecycleStage.PERSISTENT
        ]
    
    def get_severity_score(self) -> float:
        """
        Calculate severity score (0-100) based on lifecycle stage and rates.
        
        Returns:
            Severity score where higher is more severe
        """
        stage_scores = {
            LifecycleStage.RESOLVED: 0,
            LifecycleStage.RESOLVING: 20,
            LifecycleStage.DETECTED: 40,
            LifecycleStage.GROWING: 60,
            LifecycleStage.SYSTEMIC: 80,
            LifecycleStage.PERSISTENT: 100
        }
        
        base_score = stage_scores.get(self.history.current_stage, 50)
        
        # Adjust based on how far above baseline
        if self.history.windows:
            latest_rate = self.history.windows[-1][1]
            baseline_diff = latest_rate - self.baseline_rate
            
            # Bonus for being significantly above baseline
            if baseline_diff > 20:
                base_score = min(100, base_score + 10)
            elif baseline_diff > 10:
                base_score = min(100, base_score + 5)
        
        return base_score


class LifecycleManager:
    """Manages multiple issue lifecycles for a product."""
    
    def __init__(self, product_name: str, baseline_rate: float = BASELINE_RATE):
        self.product_name = product_name
        self.baseline_rate = baseline_rate
        self.issues: Dict[str, IssueLifecycle] = {}
    
    def get_or_create_issue(self, feature_name: str) -> IssueLifecycle:
        """Get existing issue tracker or create new one."""
        if feature_name not in self.issues:
            self.issues[feature_name] = IssueLifecycle(
                feature_name=feature_name,
                product_name=self.product_name,
                baseline_rate=self.baseline_rate
            )
        return self.issues[feature_name]
    
    def update_all(
        self,
        window_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Update all issues with new window data.
        
        Args:
            window_data: Dict of feature_name -> {window_label, complaint_rate, mentions}
        
        Returns:
            Dict of feature_name -> current_stage
        """
        results = {}
        
        for feature_name, data in window_data.items():
            issue = self.get_or_create_issue(feature_name)
            new_stage = issue.update(
                window_label=data.get('window_label', 'W1'),
                complaint_rate=data.get('complaint_rate', 0.0),
                mention_count=data.get('mentions', 0)
            )
            results[feature_name] = new_stage.value
        
        return results
    
    def get_active_issues(self) -> List[IssueLifecycle]:
        """Get all issues that are not resolved."""
        return [issue for issue in self.issues.values() if issue.is_active_issue()]
    
    def get_critical_issues(self) -> List[IssueLifecycle]:
        """Get issues requiring immediate attention (systemic/persistent)."""
        return [issue for issue in self.issues.values() if issue.requires_attention()]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked issues."""
        stage_counts = {stage.value: 0 for stage in LifecycleStage}
        
        for issue in self.issues.values():
            stage_counts[issue.get_stage()] += 1
        
        critical = self.get_critical_issues()
        active = self.get_active_issues()
        
        return {
            'product_name': self.product_name,
            'total_issues_tracked': len(self.issues),
            'active_issues': len(active),
            'resolved_issues': len(self.issues) - len(active),
            'critical_issues': len(critical),
            'stage_distribution': stage_counts,
            'top_concerns': [
                {
                    'feature': issue.feature_name,
                    'stage': issue.get_stage(),
                    'severity': issue.get_severity_score()
                }
                for issue in sorted(
                    critical,
                    key=lambda x: x.get_severity_score(),
                    reverse=True
                )[:5]
            ]
        }


# Convenience functions
def track_issue_lifecycle(
    feature_name: str,
    product_name: str,
    window_history: List[Tuple[str, float, int]],
    baseline_rate: float = BASELINE_RATE
) -> IssueLifecycle:
    """
    One-shot lifecycle tracking for an issue with complete history.
    
    Args:
        feature_name: Feature being tracked
        product_name: Product name
        window_history: List of (window_label, complaint_rate, mention_count)
        baseline_rate: Baseline complaint rate
    
    Returns:
        IssueLifecycle object with computed stage
    """
    lifecycle = IssueLifecycle(feature_name, product_name, baseline_rate)
    
    for window_label, rate, mentions in window_history:
        lifecycle.update(window_label, rate, mentions)
    
    return lifecycle


def batch_compute_stages(
    issues_data: List[Dict[str, Any]],
    baseline_rate: float = BASELINE_RATE
) -> List[Dict[str, Any]]:
    """
    Compute lifecycle stages for multiple issues.
    
    Args:
        issues_data: List of dicts with keys: feature_name, product_name, history
        baseline_rate: Baseline complaint rate
    
    Returns:
        List of results with computed stages
    """
    results = []
    
    for data in issues_data:
        feature = data.get('feature_name', 'Unknown')
        product = data.get('product_name', 'Unknown')
        history = data.get('history', [])
        
        lifecycle = track_issue_lifecycle(feature, product, history, baseline_rate)
        
        results.append({
            'feature_name': feature,
            'product_name': product,
            'current_stage': lifecycle.get_stage(),
            'severity_score': lifecycle.get_severity_score(),
            'requires_attention': lifecycle.requires_attention(),
            'transition_reason': lifecycle.get_transition_reason(),
            'full_history': lifecycle.get_full_history()
        })
    
    return results


if __name__ == "__main__":
    # Test lifecycle transitions
    print("Lifecycle Tracker Test\n")
    print("=" * 80)
    
    # Test case: SmartBottle packaging issue lifecycle
    print("\nTest: SmartBottle Pro - Packaging Issue")
    
    # Simulate the progression from detected → growing → systemic → persistent
    packaging_history = [
        ("W1", 8.0, 3),   # detected: first appearance, >baseline+5%
        ("W2", 12.0, 5),  # growing: increasing
        ("W3", 18.0, 7),  # growing: still increasing
        ("W4", 25.0, 9),  # systemic: >2× baseline for 2+ consecutive
        ("W5", 28.0, 10), # persistent: 3+ windows above 2× baseline
        ("W6", 26.0, 9),  # persistent: still elevated
        ("W7", 20.0, 7),  # resolving: declining but still elevated
        ("W8", 15.0, 5),  # resolving: continuing decline
        ("W9", 8.0, 3),   # resolved: back to baseline
    ]
    
    lifecycle = IssueLifecycle(
        feature_name="packaging",
        product_name="SmartBottle Pro",
        baseline_rate=5.0
    )
    
    print(f"\nBaseline rate: {lifecycle.baseline_rate}%")
    print("\nWindow progression:")
    
    for window_label, rate, mentions in packaging_history:
        stage = lifecycle.update(window_label, rate, mentions)
        transition = lifecycle.get_transition_reason()
        
        print(f"  {window_label}: {rate:5.1f}% ({mentions} mentions) → {stage}")
        if transition:
            print(f"       Transition: {transition}")
    
    print(f"\nFinal stage: {lifecycle.get_stage()}")
    print(f"Severity score: {lifecycle.get_severity_score()}")
    print(f"Requires attention: {lifecycle.requires_attention()}")
    
    print("\n" + "=" * 80)
    print("\nFull history:")
    history = lifecycle.get_full_history()
    for transition in history['transitions']:
        print(f"  {transition['from']} → {transition['to']}: {transition['reason']}")
