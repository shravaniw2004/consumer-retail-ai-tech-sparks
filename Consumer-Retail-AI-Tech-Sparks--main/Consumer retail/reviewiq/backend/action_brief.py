"""
ReviewIQ Action Brief Generator - Team-Specific Recommendations
Generates executive briefs and PDF exports with actionable intelligence.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from fpdf import FPDF
from mistralai.client import Mistral


# Teams that receive recommendations
TEAMS = ["Quality", "Marketing", "Support", "Operations"]

# Timeline categories
TIMELINE = ["Immediate", "This Week", "This Month"]


# Mistral prompt template for action brief generation
ACTION_BRIEF_PROMPT = """You are a senior product strategist and crisis management expert. Analyze the following product issues and generate an actionable brief for cross-functional teams.

PRODUCT: {product_name}

TOP ISSUES:
{issues_formatted}

Generate a structured action brief with the following sections:

1. EXECUTIVE SUMMARY (2 sentences):
   - First sentence: Overall severity and primary concern
   - Second sentence: Recommended immediate priority

2. PER-TEAM RECOMMENDATIONS:
   For each team (Quality, Marketing, Support, Operations), provide 2-3 specific, actionable recommendations.

3. DO NOT DO SECTION:
   List 3-5 critical actions to AVOID while these issues are active (e.g., "Do not launch packaging-focused ads while leakage issue is active").

4. TIMELINE:
   Categorize actions into:
   - Immediate (within 24 hours)
   - This Week (within 7 days)
   - This Month (within 30 days)

Return ONLY valid JSON in this exact structure:
{{
  "executive_summary": "2-sentence summary here",
  "team_recommendations": {{
    "Quality": ["action 1", "action 2", "action 3"],
    "Marketing": ["action 1", "action 2"],
    "Support": ["action 1", "action 2", "action 3"],
    "Operations": ["action 1", "action 2"]
  }},
  "do_not_do": [
    "Specific action to avoid with reason",
    "Another action to avoid with reason"
  ],
  "timeline": {{
    "Immediate": ["action 1", "action 2"],
    "This Week": ["action 1", "action 2", "action 3"],
    "This Month": ["action 1"]
  }},
  "severity_score": 0-100,
  "priority_level": "critical|high|medium|low"
}}

Guidelines:
- Recommendations must be specific and actionable
- "Do Not Do" items should be genuinely harmful actions to avoid
- Timeline should distribute actions logically
- Consider customer impact in all recommendations
- Address root causes, not just symptoms"""


class PriorityLevel(Enum):
    """Priority levels for action briefs."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TeamRecommendation:
    """Recommendation for a specific team."""
    team: str
    actions: List[str]
    priority: str = "medium"


@dataclass
class ActionBrief:
    """Complete action brief structure."""
    product_name: str
    executive_summary: str
    severity_score: int
    priority_level: str
    team_recommendations: Dict[str, List[str]]
    do_not_do: List[str]
    timeline: Dict[str, List[str]]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pdf_path: Optional[str] = None


class ActionBriefGenerator:
    """Generates team-specific action briefs using LLM and exports to PDF."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        output_dir: str = "sample_output"
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.output_dir = output_dir
        self.client = None
        
        if self.api_key:
            self.client = Mistral(api_key=self.api_key)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def _format_issues_for_prompt(self, issues: List[Dict]) -> str:
        """Format issue list for Mistral prompt."""
        formatted = []
        
        for i, issue in enumerate(issues, 1):
            feature = issue.get('feature_name', 'Unknown')
            stage = issue.get('lifecycle_stage', 'detected')
            severity = issue.get('severity_score', 50)
            priority = issue.get('priority_level', 'medium')
            trend = issue.get('trend_direction', 'stable')
            
            formatted.append(
                f"{i}. {feature}\n"
                f"   - Stage: {stage}\n"
                f"   - Severity: {severity}/100\n"
                f"   - Priority: {priority}\n"
                f"   - Trend: {trend}\n"
            )
        
        return "\n".join(formatted)
    
    def generate_brief(
        self,
        product_name: str,
        top_issues: List[Dict]
    ) -> Optional[ActionBrief]:
        """
        Generate action brief using Mistral API.
        
        Args:
            product_name: Product being analyzed
            top_issues: List of top priority issues (from escalation scorer)
        
        Returns:
            ActionBrief object or None if generation fails
        """
        if not self.client or not top_issues:
            return None
        
        # Format prompt
        issues_formatted = self._format_issues_for_prompt(top_issues)
        prompt = ACTION_BRIEF_PROMPT.format(
            product_name=product_name,
            issues_formatted=issues_formatted
        )
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            return ActionBrief(
                product_name=product_name,
                executive_summary=data.get("executive_summary", ""),
                severity_score=data.get("severity_score", 50),
                priority_level=data.get("priority_level", "medium"),
                team_recommendations=data.get("team_recommendations", {}),
                do_not_do=data.get("do_not_do", []),
                timeline=data.get("timeline", {})
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error in brief generation: {e}")
            return None
            
        except Exception as e:
            print(f"Error generating brief: {e}")
            return None
    
    def generate_executive_summary(
        self,
        issues: List[Dict]
    ) -> str:
        """
        Generate one-line executive summary of severity.
        
        Args:
            issues: List of issues
        
        Returns:
            One-sentence summary
        """
        if not issues:
            return "No critical issues detected at this time."
        
        # Count by severity
        critical = sum(1 for i in issues if i.get('priority_level') == 'critical')
        high = sum(1 for i in issues if i.get('priority_level') == 'high')
        
        # Get top issue
        top_issue = issues[0]
        top_feature = top_issue.get('feature_name', 'Unknown')
        top_stage = top_issue.get('lifecycle_stage', 'detected')
        
        if critical > 0:
            return f"CRITICAL: {critical} systemic issue{'s' if critical > 1 else ''} detected with '{top_feature}' requiring immediate cross-functional response."
        elif high > 0:
            return f"HIGH PRIORITY: {high} growing issue{'s' if high > 1 else ''} detected, primarily '{top_feature}' at {top_stage} stage, requiring urgent attention."
        else:
            return f"MEDIUM PRIORITY: {len(issues)} emerging concern{'s' if len(issues) > 1 else ''} detected with '{top_feature}' - monitor closely."
    
    def get_team_actions(
        self,
        issue: Dict,
        team_name: str
    ) -> List[str]:
        """
        Filter actions for a specific team from an issue.
        
        Args:
            issue: Issue dictionary
            team_name: Team to filter for (Quality, Marketing, Support, Operations)
        
        Returns:
            List of actions for this team
        """
        # Map lifecycle stages to team actions
        stage_team_actions = {
            'detected': {
                'Quality': ["Monitor for pattern confirmation", "Prepare investigation protocol"],
                'Marketing': ["Pause related feature promotions", "Prepare holding statement"],
                'Support': ["Update FAQ with known workaround", "Brief support team"],
                'Operations': ["Check inventory for affected batches", "Review supplier reports"]
            },
            'growing': {
                'Quality': ["Initiate root cause analysis", "Accelerate lab testing"],
                'Marketing': ["Suspend affected campaigns", "Prepare customer communication"],
                'Support': ["Escalate handling procedures", "Create specialized ticket queue"],
                'Operations': ["Quarantine suspect inventory", "Notify logistics partners"]
            },
            'systemic': {
                'Quality': ["Immediate production halt for affected lines", "Emergency recall assessment"],
                'Marketing': ["Halt all product advertising immediately", "Prepare crisis communication"],
                'Support': ["Activate crisis response team", "Prepare for 10x ticket volume"],
                'Operations': ["Emergency supply chain audit", "Prepare recall logistics"]
            },
            'persistent': {
                'Quality': ["Complete redesign of affected component", "Implement new QC protocols"],
                'Marketing': ["Long-term reputation recovery planning", "Transparent customer updates"],
                'Support': ["Permanent process changes", "Enhanced compensation authority"],
                'Operations': ["Supplier replacement program", "Process overhaul"]
            },
            'resolving': {
                'Quality': ["Verify fix effectiveness", "Monitor for regression"],
                'Marketing': ["Prepare relaunch campaign", "Gather positive testimonials"],
                'Support': ["Close ticket backlog", "Follow up with affected customers"],
                'Operations': ["Resume normal operations", "Audit fix deployment"]
            }
        }
        
        stage = issue.get('lifecycle_stage', 'detected')
        actions = stage_team_actions.get(stage, {}).get(team_name, [])
        
        # Add priority-specific modifications
        priority = issue.get('priority_level', 'medium')
        if priority == 'critical':
            actions = [f"URGENT: {action}" for action in actions]
        
        return actions
    
    def create_pdf_export(
        self,
        brief: ActionBrief,
        filename: Optional[str] = None
    ) -> str:
        """
        Create formatted PDF export of action brief.
        
        Args:
            brief: ActionBrief object
            filename: Optional custom filename
        
        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"action_brief_{brief.product_name.replace(' ', '_')}_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 15, f"Action Brief: {brief.product_name}", ln=True, align="L")
        
        # Metadata
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Generated: {brief.generated_at}", ln=True)
        pdf.cell(0, 8, f"Severity Score: {brief.severity_score}/100 | Priority: {brief.priority_level.upper()}", ln=True)
        pdf.ln(5)
        
        # Executive Summary
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 10, " Executive Summary", ln=True, fill=True)
        pdf.ln(2)
        
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, brief.executive_summary)
        pdf.ln(5)
        
        # Team Recommendations
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 10, " Team Recommendations", ln=True, fill=True)
        pdf.ln(2)
        
        team_colors = {
            'Quality': (220, 240, 220),      # Light green
            'Marketing': (240, 220, 240),    # Light purple
            'Support': (220, 240, 255),      # Light blue
            'Operations': (255, 240, 220)    # Light orange
        }
        
        for team in TEAMS:
            if team in brief.team_recommendations:
                pdf.set_font("Arial", "B", 12)
                color = team_colors.get(team, (240, 240, 240))
                pdf.set_fill_color(*color)
                pdf.cell(0, 9, f" {team}", ln=True, fill=True)
                
                pdf.set_font("Arial", "", 10)
                for action in brief.team_recommendations[team]:
                    pdf.cell(5)  # Indent
                    pdf.cell(5, 7, chr(149), ln=0)  # Bullet
                    pdf.multi_cell(0, 7, f" {action}")
                
                pdf.ln(2)
        
        # Do Not Do Section
        if brief.do_not_do:
            pdf.add_page()
            
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(180, 0, 0)  # Red
            pdf.set_fill_color(255, 220, 220)  # Light red background
            pdf.cell(0, 10, " DO NOT DO (Critical Avoidances)", ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            
            pdf.set_font("Arial", "", 10)
            for item in brief.do_not_do:
                pdf.cell(5)
                pdf.cell(5, 7, chr(151), ln=0)  # Em dash
                pdf.multi_cell(0, 7, f" {item}")
            
            pdf.ln(5)
        
        # Timeline
        if brief.timeline:
            pdf.set_font("Arial", "B", 14)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(0, 10, " Action Timeline", ln=True, fill=True)
            pdf.ln(2)
            
            timeline_colors = {
                'Immediate': (255, 200, 200),    # Red
                'This Week': (255, 230, 200),    # Orange
                'This Month': (255, 255, 200)    # Yellow
            }
            
            for period in TIMELINE:
                if period in brief.timeline and brief.timeline[period]:
                    pdf.set_font("Arial", "B", 11)
                    color = timeline_colors.get(period, (240, 240, 240))
                    pdf.set_fill_color(*color)
                    pdf.cell(0, 8, f" {period}", ln=True, fill=True)
                    
                    pdf.set_font("Arial", "", 10)
                    for action in brief.timeline[period]:
                        pdf.cell(10)
                        pdf.cell(5, 6, chr(149), ln=0)
                        pdf.multi_cell(0, 6, f" {action}")
                    
                    pdf.ln(2)
        
        # Footer
        pdf.set_y(-20)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, f"Generated by ReviewIQ | {brief.product_name} Action Brief", align="C")
        
        # Save PDF
        pdf.output(filepath)
        brief.pdf_path = filepath
        
        return filepath
    
    def generate_and_export(
        self,
        product_name: str,
        top_issues: List[Dict],
        save_pdf: bool = True
    ) -> Tuple[Optional[ActionBrief], Optional[str]]:
        """
        Generate brief and optionally export to PDF.
        
        Args:
            product_name: Product name
            top_issues: Top priority issues
            save_pdf: Whether to create PDF export
        
        Returns:
            Tuple of (ActionBrief, PDF path) or (None, None) on failure
        """
        brief = self.generate_brief(product_name, top_issues)
        
        if brief is None:
            return None, None
        
        pdf_path = None
        if save_pdf:
            pdf_path = self.create_pdf_export(brief)
        
        return brief, pdf_path


def get_brief_summary(brief: ActionBrief) -> Dict[str, Any]:
    """
    Get quick summary of action brief for dashboard display.
    
    Args:
        brief: ActionBrief object
    
    Returns:
        Summary dictionary
    """
    immediate_count = len(brief.timeline.get('Immediate', []))
    do_not_count = len(brief.do_not_do)
    
    return {
        'product_name': brief.product_name,
        'severity_score': brief.severity_score,
        'priority_level': brief.priority_level,
        'executive_summary': brief.executive_summary,
        'immediate_actions': immediate_count,
        'avoid_count': do_not_count,
        'pdf_path': brief.pdf_path,
        'teams_addressed': list(brief.team_recommendations.keys())
    }


if __name__ == "__main__":
    # Test action brief generation
    print("Action Brief Generator Test\n")
    print("=" * 80)
    
    # Mock top issues
    test_issues = [
        {
            'feature_name': 'packaging',
            'lifecycle_stage': 'systemic',
            'severity_score': 85,
            'priority_level': 'critical',
            'trend_direction': 'spiking',
            'complaint_rate': 28.0
        },
        {
            'feature_name': 'durability',
            'lifecycle_stage': 'growing',
            'severity_score': 65,
            'priority_level': 'high',
            'trend_direction': 'stable',
            'complaint_rate': 15.0
        }
    ]
    
    generator = ActionBriefGenerator()
    
    # Test executive summary
    print("\nTest 1: Executive Summary")
    summary = generator.generate_executive_summary(test_issues)
    print(f"  {summary}")
    
    # Test team actions
    print("\nTest 2: Team Actions for Systemic Issue")
    for team in TEAMS:
        actions = generator.get_team_actions(test_issues[0], team)
        print(f"\n  {team}:")
        for action in actions:
            print(f"    - {action}")
    
    # Test full brief generation (requires API key)
    print("\n" + "=" * 80)
    print("\nTest 3: Full Brief Generation (requires API key)")
    
    if os.getenv("MISTRAL_API_KEY"):
        brief, pdf_path = generator.generate_and_export(
            product_name="SmartBottle Pro",
            top_issues=test_issues,
            save_pdf=True
        )
        
        if brief:
            print(f"\n  Brief generated successfully!")
            print(f"  Severity: {brief.severity_score}/100")
            print(f"  Priority: {brief.priority_level}")
            print(f"  PDF: {pdf_path}")
            
            print(f"\n  Executive Summary:\n    {brief.executive_summary}")
            
            print(f"\n  Do Not Do ({len(brief.do_not_do)} items):")
            for item in brief.do_not_do[:3]:
                print(f"    - {item}")
        else:
            print("  Brief generation failed")
    else:
        print("  Skipped (no MISTRAL_API_KEY)")
