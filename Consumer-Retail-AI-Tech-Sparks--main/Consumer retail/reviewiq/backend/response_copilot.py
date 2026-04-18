"""
ReviewIQ v5 — Module 9: Auto-Response Copilot
Generates professional response drafts for negative reviews.
3 tone options: Professional, Empathetic, Action-Focused.
Uses Mistral API for AI-powered responses.
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from mistralai.client import Mistral
    from dotenv import load_dotenv
    load_dotenv()
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False


# ──────────────────────────────────────────────────────────────
# Response Templates (Fallback when API unavailable)
# ──────────────────────────────────────────────────────────────

TONE_DESCRIPTIONS = {
    "professional": {
        "label": "Professional",
        "emoji": "🏢",
        "description": "Formal, SLA-focused, structured",
        "use_case": "B2B / Premium brand",
    },
    "empathetic": {
        "label": "Empathetic",
        "emoji": "💝",
        "description": "Warm, apologetic, personal",
        "use_case": "D2C / Personal Care",
    },
    "action_focused": {
        "label": "Action-Focused",
        "emoji": "⚡",
        "description": "Direct, specific fix + timeline",
        "use_case": "Fast-resolution needed",
    },
}

# Pre-built templates for offline/fallback use
FALLBACK_TEMPLATES = {
    "professional": {
        "packaging": "Dear Customer, We sincerely apologize for the packaging issue you experienced with {product}. We take quality control very seriously and have flagged this with our QA team for immediate investigation. A replacement will be shipped within 2 business days. Reference: {ref_id}.",
        "quality": "Dear Customer, Thank you for reporting this quality concern with {product}. We have escalated this to our manufacturing team for root cause analysis. Please contact support@brand.com with your order ID for a priority replacement. We value your feedback.",
        "delivery": "Dear Customer, We regret the delivery experience with {product}. Our logistics partner has been notified and we're implementing additional quality checks. A refund/replacement has been initiated. Expected resolution: 3 business days.",
        "default": "Dear Customer, We appreciate your feedback regarding {product} and sincerely apologize for the {issue} you experienced. Our team is actively addressing this concern. Please reach out to us at support@brand.com for immediate resolution. Reference: {ref_id}.",
    },
    "empathetic": {
        "packaging": "Hi there, I'm really sorry about the packaging issue with your {product} — I completely understand how frustrating that must be, especially when you're excited about a new product. We've already flagged this with our packaging team and I've personally arranged a replacement for you. You should receive it within 2 days. 🙏",
        "quality": "Hi, I hear you and I'm truly sorry about the quality issue with {product}. That's not the experience we want for our customers. I've personally escalated this and our team is looking into it right away. Let me make this right — we're sending you a replacement immediately.",
        "delivery": "Hi! I'm so sorry about the delivery problem with your {product}. I can imagine how disappointing that was. We've spoken to our delivery partner about this and I'm personally ensuring your replacement reaches you quickly. Thank you for your patience! 💛",
        "default": "Hi there, I'm genuinely sorry about what happened with your {product}. Your experience with {issue} is completely unacceptable and I want to make it right. I've already escalated this to our team. Please let me know how I can help — we truly value you as a customer. 🙏",
    },
    "action_focused": {
        "packaging": "Issue noted: {product} packaging defect. Action taken: ✅ Replacement shipped (tracking in 4 hrs) ✅ QA alert raised with packaging line ✅ Full refund if replacement unsatisfactory. Contact: support@brand.com | Ref: {ref_id}",
        "quality": "Issue noted: {product} quality concern. Immediate actions: ✅ Replacement dispatched today ✅ Quality audit initiated for this batch ✅ Your case escalated to senior QA lead. Timeline: Resolution within 48 hours. Contact: {ref_id}",
        "delivery": "Issue noted: {product} delivery problem. Steps taken: ✅ Delivery partner notified ✅ Re-delivery scheduled within 24 hours ✅ Compensation credit applied to your account. Track progress: support@brand.com | {ref_id}",
        "default": "Issue noted: {product} – {issue}. Immediate actions: ✅ Case escalated to {team} ✅ Resolution timeline: 48 hours ✅ Replacement/refund initiated. Contact: support@brand.com | Ref: {ref_id}",
    },
}


RESPONSE_PROMPT = """You are a customer experience manager for an Indian e-commerce brand.
Write 3 response drafts to this negative review in these tones:
1. Professional  2. Empathetic  3. Action-Focused

Review: {review_text}
Issue detected: {feature} — {sentiment} — {intensity}
Product: {product_name}

Rules:
- Responses must be under 80 words each
- Do not be defensive
- Always acknowledge the specific issue mentioned
- Include a concrete next step
- Natural Indian English is fine
- Use the customer's language style if appropriate

Return ONLY valid JSON in this exact format:
{{
  "professional": "response text here",
  "empathetic": "response text here",
  "action_focused": "response text here"
}}"""


@dataclass
class ResponseDraft:
    """A single response draft."""
    tone: str
    tone_label: str
    tone_emoji: str
    response_text: str
    word_count: int
    method: str  # "mistral" or "template"


@dataclass 
class ResponseResult:
    """Complete response generation result."""
    review_id: str
    review_text: str
    product_name: str
    feature: str
    sentiment: str
    drafts: List[ResponseDraft]
    generation_time: float


def generate_responses_template(
    review_text: str,
    product_name: str,
    feature: str = "quality",
    review_id: str = "REV001",
) -> List[ResponseDraft]:
    """
    Generate response drafts using pre-built templates (no API needed).
    Fast fallback when AI API is unavailable.
    """
    drafts = []
    
    # Determine issue category
    feature_lower = feature.lower()
    if any(k in feature_lower for k in ["packaging", "pack", "seal", "leak", "cap"]):
        category = "packaging"
    elif any(k in feature_lower for k in ["quality", "build", "defect", "broken"]):
        category = "quality"
    elif any(k in feature_lower for k in ["delivery", "shipping", "transit"]):
        category = "delivery"
    else:
        category = "default"
    
    # Determine team
    team_map = {
        "packaging": "packaging team",
        "quality": "quality team",
        "delivery": "logistics team",
        "default": "support team",
    }
    team = team_map.get(category, "support team")
    
    for tone_key, tone_info in TONE_DESCRIPTIONS.items():
        template = FALLBACK_TEMPLATES[tone_key].get(category, FALLBACK_TEMPLATES[tone_key]["default"])
        
        response_text = template.format(
            product=product_name,
            issue=feature,
            ref_id=f"RIQ-{review_id}",
            team=team,
            product_name=product_name,
        )
        
        drafts.append(ResponseDraft(
            tone=tone_key,
            tone_label=tone_info["label"],
            tone_emoji=tone_info["emoji"],
            response_text=response_text,
            word_count=len(response_text.split()),
            method="template",
        ))
    
    return drafts


def generate_responses_mistral(
    review_text: str,
    product_name: str,
    feature: str = "quality",
    sentiment: str = "negative",
    intensity: str = "high",
    review_id: str = "REV001",
) -> List[ResponseDraft]:
    """
    Generate response drafts using Mistral API.
    Falls back to templates if API is unavailable.
    """
    if not HAS_MISTRAL:
        return generate_responses_template(review_text, product_name, feature, review_id)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return generate_responses_template(review_text, product_name, feature, review_id)
    
    try:
        client = Mistral(api_key=api_key)
        
        prompt = RESPONSE_PROMPT.format(
            review_text=review_text[:500],  # Cap text length
            feature=feature,
            sentiment=sentiment,
            intensity=intensity,
            product_name=product_name,
        )
        
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown fences if any
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        if content.endswith("```"):
            content = content[:-3]
        
        data = json.loads(content.strip())
        
        drafts = []
        for tone_key, tone_info in TONE_DESCRIPTIONS.items():
            text = data.get(tone_key, "")
            if text:
                drafts.append(ResponseDraft(
                    tone=tone_key,
                    tone_label=tone_info["label"],
                    tone_emoji=tone_info["emoji"],
                    response_text=text,
                    word_count=len(text.split()),
                    method="mistral",
                ))
        
        if drafts:
            return drafts
        
    except Exception as e:
        print(f"Mistral response generation failed: {e}")
    
    # Fallback to templates
    return generate_responses_template(review_text, product_name, feature, review_id)


def generate_responses(
    review_text: str,
    product_name: str,
    feature: str = "quality",
    sentiment: str = "negative",
    intensity: str = "high",
    review_id: str = "REV001",
    use_ai: bool = True,
) -> List[ResponseDraft]:
    """
    Main entry point: generate response drafts.
    
    Args:
        review_text: The negative review text
        product_name: Product name
        feature: Detected issue feature
        sentiment: Detected sentiment
        intensity: Detected intensity
        review_id: Review identifier
        use_ai: Whether to try Mistral API first
    """
    if use_ai:
        return generate_responses_mistral(
            review_text, product_name, feature, sentiment, intensity, review_id
        )
    else:
        return generate_responses_template(review_text, product_name, feature, review_id)


def get_negative_review_queue(reviews: List[Dict], max_items: int = 20) -> List[Dict]:
    """
    Build a queue of negative reviews needing responses.
    Sorted by urgency (intensity + confidence).
    """
    queue = []
    
    for review in reviews:
        # Check if review has negative sentiment
        overall = review.get("overall_sentiment", "neutral")
        if overall not in ("negative", "mixed"):
            continue
        
        # Get the worst feature
        features = review.get("extracted_features", [])
        worst_feature = None
        worst_intensity = 0
        
        for feat in features:
            if feat.get("sentiment") == "negative":
                intensity_val = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(
                    str(feat.get("intensity", "medium")).lower(), 0.5
                )
                if intensity_val > worst_intensity:
                    worst_intensity = intensity_val
                    worst_feature = feat
        
        if worst_feature:
            queue.append({
                "review_id": review.get("review_id", ""),
                "review_text": review.get("clean_text", review.get("review_text", "")),
                "original_text": review.get("original_text", ""),
                "product_name": review.get("product_name", "Unknown"),
                "feature": worst_feature.get("feature", "quality"),
                "sentiment": worst_feature.get("sentiment", "negative"),
                "intensity": worst_feature.get("intensity", "medium"),
                "confidence": worst_feature.get("confidence", 0.5),
                "urgency_score": worst_intensity,
            })
    
    # Sort by urgency
    queue.sort(key=lambda x: x["urgency_score"], reverse=True)
    
    return queue[:max_items]


if __name__ == "__main__":
    print("Auto-Response Copilot Test")
    print("=" * 70)
    
    test_reviews = [
        {
            "text": "Cap khula hua tha jab mila — poora bag bheeg gaya. Terrible packaging!",
            "product": "SmartBottle Pro",
            "feature": "packaging_seal",
        },
        {
            "text": "Charging speed dropped after 2 weeks. Complete waste of money.",
            "product": "BoltCharge Cable",
            "feature": "charging_speed",
        },
        {
            "text": "Taste bilkul alag hai is baar, bekar lag raha. Not buying again.",
            "product": "NutriMix Powder",
            "feature": "taste",
        },
    ]
    
    for i, review in enumerate(test_reviews):
        print(f"\n{'─' * 60}")
        print(f"Review: {review['text'][:60]}...")
        print(f"Product: {review['product']} | Feature: {review['feature']}")
        print(f"{'─' * 60}")
        
        drafts = generate_responses(
            review_text=review["text"],
            product_name=review["product"],
            feature=review["feature"],
            review_id=f"REV{i:03d}",
            use_ai=False,  # Use templates for testing
        )
        
        for draft in drafts:
            print(f"\n  {draft.tone_label} ({draft.word_count} words):")
            print(f"  {draft.response_text[:120]}...")
