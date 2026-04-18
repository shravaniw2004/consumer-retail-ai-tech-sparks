"""
ReviewIQ v5 — Module 6: Emotion Intensity Classifier
Two-layer approach: keyword/emoji scoring (fast) + Mistral AI fallback (ambiguous cases).
Classifies reviews into 7 emotion categories with urgency levels.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from mistralai.client import Mistral

# ──────────────────────────────────────────────────────────────
# Emotion Taxonomy (7 classes)
# ──────────────────────────────────────────────────────────────

EMOTION_TAXONOMY = {
    "delighted":     {"label": "🤩 Delighted",     "color": "#FFD700", "urgency": "none",     "score_range": (85, 100)},
    "satisfied":     {"label": "😊 Satisfied",     "color": "#4CAF50", "urgency": "none",     "score_range": (70, 84)},
    "neutral":       {"label": "😐 Neutral",       "color": "#9E9E9E", "urgency": "low",      "score_range": (45, 69)},
    "disappointed":  {"label": "😟 Disappointed",  "color": "#FFEB3B", "urgency": "medium",   "score_range": (30, 44)},
    "frustrated":    {"label": "😤 Frustrated",    "color": "#FF9800", "urgency": "high",     "score_range": (15, 29)},
    "angry":         {"label": "😡 Angry",         "color": "#F44336", "urgency": "critical", "score_range": (5, 14)},
    "furious":       {"label": "🤬 Furious",       "color": "#B71C1C", "urgency": "escalate", "score_range": (0, 4)},
}

# ──────────────────────────────────────────────────────────────
# Layer 1: Keyword + Emoji Scoring (Fast, No API)
# ──────────────────────────────────────────────────────────────

FURY_SIGNALS = [
    "WORST", "NEVER AGAIN", "SCAM", "FRAUD", "GARBAGE",
    "DEMANDING REFUND", "WILL POST EVERYWHERE", "LEGAL ACTION",
    "CONSUMER COURT", "NEVER BUYING", "COMPLETE SCAM",
    "TOTAL FRAUD", "RUINED", "DESTROYED",
    # Emojis
    "🤬", "💀", "☠️", "🖕",
    # Hinglish fury
    "BILKUL BAKWAS", "SABSE GHATIYA", "PAISA BARBAD",
    "DHOKA", "LOOT", "FRAUD HAI",
]

ANGER_SIGNALS = [
    "terrible", "pathetic", "useless", "waste", "horrible",
    "disgusting", "awful", "dreadful", "abysmal", "worst ever",
    "unacceptable", "defective", "broken", "ruined",
    # Emojis
    "😡", "👎", "🤮",
    # Hinglish anger
    "bakwas", "bekar", "ghatiya", "kachra", "faltu",
    "bilkul bekar", "total waste", "paisa waste",
]

FRUSTRATION_SIGNALS = [
    "annoying", "keeps failing", "again and again", "still not fixed",
    "same issue", "repeated problem", "fed up", "tired of",
    "not working properly", "inconsistent", "unreliable",
    # Emojis
    "😤", "😩", "🤦",
    # Hinglish frustration
    "phir se", "baar baar", "thak gaya", "kab theek hoga",
    "kaam nahi karta", "problem hai",
]

DISAPPOINTMENT_SIGNALS = [
    "expected better", "not satisfied", "below par", "could be better",
    "not worth", "disappointed", "let down", "underwhelming",
    "not as described", "misleading", "overrated",
    # Emojis
    "😟", "😔", "😞",
    # Hinglish disappointment
    "umeed se kam", "theek nahi", "jaise socha tha waisa nahi",
    "pehle waala better tha",
]

SATISFACTION_SIGNALS = [
    "good", "works well", "decent", "nice", "okay",
    "satisfactory", "fine", "meets expectations",
    "worth the price", "recommended",
    # Emojis
    "👍", "🙂", "✅",
    # Hinglish satisfaction
    "acha hai", "theek hai", "chalta hai", "sahi hai",
    "value for money",
]

DELIGHT_SIGNALS = [
    "amazing", "love it", "fantastic", "incredible", "perfect",
    "outstanding", "exceeded expectations", "best ever", "brilliant",
    "phenomenal", "blown away", "absolutely love",
    # Emojis
    "😍", "🤩", "❤️", "🔥", "💯", "⭐",
    # Hinglish delight
    "zabardast", "mast", "ekdum mast", "shandaar", "jhakaas",
    "kamaal", "bahut acha", "best hai",
]

NEUTRAL_SIGNALS = [
    "okay", "average", "fine", "normal", "standard",
    "nothing special", "as expected", "mediocre",
    # Hinglish neutral
    "theek thaak", "chalta hai", "normal hai",
]


@dataclass
class EmotionResult:
    """Result of emotion classification for a single review."""
    review_id: str
    emotion: str               # Key from EMOTION_TAXONOMY
    emotion_label: str         # Human-readable label with emoji
    emotion_color: str         # Dashboard color
    urgency: str               # none/low/medium/high/critical/escalate
    confidence: float          # 0.0 - 1.0
    raw_score: int             # 0-100 (higher = more positive)
    signals_detected: List[str]
    method: str                # "keyword" or "mistral"


def _count_uppercase_ratio(text: str) -> float:
    """Ratio of uppercase characters (indicator of shouting)."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    upper = sum(1 for c in alpha_chars if c.isupper())
    return upper / len(alpha_chars)


def _check_signals(text: str, signals: List[str]) -> List[str]:
    """Check if any signals are present in text. Return matched signals."""
    matched = []
    text_lower = text.lower()
    text_original = text
    
    for signal in signals:
        # Check both case-sensitive (for ALL CAPS signals) and case-insensitive
        if signal.isupper() and len(signal) > 2:
            # ALL CAPS signal — check original text
            if signal in text_original:
                matched.append(signal)
        elif signal in text_lower:
            matched.append(signal)
        # Check emoji signals (single character)
        elif len(signal) <= 2 and signal in text_original:
            matched.append(signal)
    
    return matched


def classify_emotion_layer2(review_id: str, text: str, overall_sentiment: str = "neutral") -> EmotionResult:
    """
    Layer 2: Mistral AI classification for ambiguous cases.
    
    Use this when Layer 1 confidence is low (0.4-0.65).
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        # Fallback to Layer 1 if no API key
        return classify_emotion_layer1(review_id, text, overall_sentiment)
    
    try:
        client = Mistral(api_key=api_key)
        
        prompt = f"""Classify the emotion of this product review into one of these 7 categories:
- delighted (🤩 Delighted): extremely positive, enthusiastic
- satisfied (😊 Satisfied): positive, happy
- neutral (😐 Neutral): factual, no strong emotion
- disappointed (😟 Disappointed): negative, let down
- frustrated (😤 Frustrated): annoyed, repeated problems
- angry (😡 Angry): strong negative, wants action
- furious (🤬 Furious): extreme anger, threatening, all caps

Review text: {text}
Overall sentiment: {overall_sentiment}

Return ONLY the emotion key (e.g., "delighted"). No explanation."""
        
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        
        emotion = response.choices[0].message.content.strip().lower()
        
        # Validate emotion
        if emotion not in EMOTION_TAXONOMY:
            emotion = "neutral"  # Fallback
        
        taxonomy = EMOTION_TAXONOMY[emotion]
        raw_score = taxonomy["score_range"][0] + (taxonomy["score_range"][1] - taxonomy["score_range"][0]) // 2
        
        return EmotionResult(
            review_id=review_id,
            emotion=emotion,
            emotion_label=taxonomy["label"],
            emotion_color=taxonomy["color"],
            urgency=taxonomy["urgency"],
            confidence=0.85,  # High confidence from AI
            raw_score=raw_score,
            signals_detected=[],
            method="mistral"
        )
    except Exception as e:
        # Fallback to Layer 1 on error
        return classify_emotion_layer1(review_id, text, overall_sentiment)


def classify_emotion_layer1(review_id: str, text: str, overall_sentiment: str = "neutral") -> EmotionResult:
    """
    Layer 1: Fast keyword + emoji scoring. No API calls.
    
    Returns EmotionResult. If confidence is in the ambiguous range (0.4-0.65),
    the caller should consider using Layer 2 (Mistral) for refinement.
    """
    if not text or not text.strip():
        return EmotionResult(
            review_id=review_id, emotion="neutral", emotion_label="😐 Neutral",
            emotion_color="#9E9E9E", urgency="low", confidence=0.5,
            raw_score=50, signals_detected=[], method="keyword"
        )
    
    # Detect signals for each emotion level
    fury_matches = _check_signals(text, FURY_SIGNALS)
    anger_matches = _check_signals(text, ANGER_SIGNALS)
    frustration_matches = _check_signals(text, FRUSTRATION_SIGNALS)
    disappointment_matches = _check_signals(text, DISAPPOINTMENT_SIGNALS)
    satisfaction_matches = _check_signals(text, SATISFACTION_SIGNALS)
    delight_matches = _check_signals(text, DELIGHT_SIGNALS)
    neutral_matches = _check_signals(text, NEUTRAL_SIGNALS)
    
    # Uppercase ratio (shouting detection)
    upper_ratio = _count_uppercase_ratio(text)
    
    # Scoring: accumulate weighted scores
    # Score goes from 0 (most negative/furious) to 100 (most positive/delighted)
    scores = {
        "furious":       len(fury_matches) * 25 + (30 if upper_ratio > 0.6 else 0),
        "angry":         len(anger_matches) * 15,
        "frustrated":    len(frustration_matches) * 12,
        "disappointed":  len(disappointment_matches) * 10,
        "neutral":       len(neutral_matches) * 8,
        "satisfied":     len(satisfaction_matches) * 10,
        "delighted":     len(delight_matches) * 15,
    }
    
    # All signals detected
    all_signals = (fury_matches + anger_matches + frustration_matches + 
                   disappointment_matches + satisfaction_matches + 
                   delight_matches + neutral_matches)
    
    # Get highest scoring emotion
    if not any(scores.values()):
        # No signals detected — use overall_sentiment as fallback
        sentiment_map = {
            "positive": "satisfied",
            "negative": "disappointed",
            "mixed": "neutral",
            "neutral": "neutral",
        }
        emotion = sentiment_map.get(overall_sentiment, "neutral")
        confidence = 0.3  # Low confidence — ambiguous zone
    else:
        # Find the dominant emotion
        max_score = max(scores.values())
        emotion = max(scores, key=scores.get)
        
        # Calculate confidence based on signal clarity
        total_signal_strength = sum(scores.values())
        if total_signal_strength > 0:
            confidence = min(0.95, max_score / total_signal_strength)
        else:
            confidence = 0.3
        
        # Boost confidence for strong signals
        if len(fury_matches) >= 2 or upper_ratio > 0.6:
            confidence = max(confidence, 0.85)
        if len(delight_matches) >= 3:
            confidence = max(confidence, 0.85)
    
    # Get taxonomy info
    taxonomy = EMOTION_TAXONOMY.get(emotion, EMOTION_TAXONOMY["neutral"])
    
    # Compute raw_score (0-100, higher = more positive)
    raw_score = taxonomy["score_range"][0] + (taxonomy["score_range"][1] - taxonomy["score_range"][0]) // 2
    
    return EmotionResult(
        review_id=review_id,
        emotion=emotion,
        emotion_label=taxonomy["label"],
        emotion_color=taxonomy["color"],
        urgency=taxonomy["urgency"],
        confidence=round(confidence, 2),
        raw_score=raw_score,
        signals_detected=all_signals[:10],  # Cap at 10
        method="keyword",
    )


def classify_emotion_batch(reviews: List[Dict], sentiment_results: Optional[List[Dict]] = None) -> List[EmotionResult]:
    """
    Classify emotions for a batch of reviews using Layer 1, with Layer 2 (Mistral) fallback for ambiguous cases.
    
    Args:
        reviews: List of review dicts with 'review_id' and 'clean_text'/'review_text'
        sentiment_results: Optional sentiment extraction results for fallback
    
    Returns:
        List of EmotionResult for each review
    """
    results = []
    sentiment_map = {}
    
    if sentiment_results:
        for sr in sentiment_results:
            rid = sr.get("review_id", "")
            sentiment_map[rid] = sr.get("overall_sentiment", "neutral")
    
    for review in reviews:
        review_id = review.get("review_id", "unknown")
        text = review.get("clean_text", review.get("review_text", ""))
        overall = sentiment_map.get(review_id, review.get("overall_sentiment", "neutral"))
        
        # Layer 1: Fast keyword classification
        result = classify_emotion_layer1(review_id, text, overall)
        
        # If confidence is in ambiguous range, use Layer 2 (Mistral)
        if 0.4 <= result.confidence <= 0.65:
            result = classify_emotion_layer2(review_id, text, overall)
        
        results.append(result)
    
    return results


def get_emotion_distribution(results: List[EmotionResult]) -> Dict[str, int]:
    """Get count distribution of emotions."""
    distribution = {emotion: 0 for emotion in EMOTION_TAXONOMY}
    for result in results:
        if result.emotion in distribution:
            distribution[result.emotion] += 1
    return distribution


def get_emotion_timeline(results: List[EmotionResult], reviews: List[Dict], window_size: int = 50) -> List[Dict]:
    """
    Build emotion timeline for dashboard visualization.
    Groups reviews into time windows and counts emotion distribution per window.
    """
    # Sort reviews by timestamp
    review_map = {r.get("review_id"): r for r in reviews}
    
    sorted_results = sorted(results, key=lambda r: review_map.get(r.review_id, {}).get("timestamp", ""))
    
    timeline = []
    for i in range(0, len(sorted_results), window_size):
        window = sorted_results[i:i + window_size]
        window_label = f"W{i // window_size + 1}"
        
        distribution = {emotion: 0 for emotion in EMOTION_TAXONOMY}
        for result in window:
            if result.emotion in distribution:
                distribution[result.emotion] += 1
        
        timeline.append({
            "window": window_label,
            "total": len(window),
            **distribution,
        })
    
    return timeline


def get_urgency_summary(results: List[EmotionResult]) -> Dict[str, List[str]]:
    """Get reviews grouped by urgency level."""
    urgency_groups = {
        "escalate": [],
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
        "none": [],
    }
    
    for result in results:
        urgency_groups.get(result.urgency, urgency_groups["low"]).append(result.review_id)
    
    return urgency_groups


if __name__ == "__main__":
    # Test with sample reviews
    test_reviews = [
        {"review_id": "T1", "clean_text": "WORST PRODUCT EVER TOTAL SCAM DEMANDING REFUND 🤬💀", "overall_sentiment": "negative"},
        {"review_id": "T2", "clean_text": "terrible pathetic waste of money useless product 😡", "overall_sentiment": "negative"},
        {"review_id": "T3", "clean_text": "keeps failing again and again, annoying 😤", "overall_sentiment": "negative"},
        {"review_id": "T4", "clean_text": "expected better, not satisfied with quality", "overall_sentiment": "negative"},
        {"review_id": "T5", "clean_text": "okay product, nothing special", "overall_sentiment": "neutral"},
        {"review_id": "T6", "clean_text": "good product, works well, recommended 👍", "overall_sentiment": "positive"},
        {"review_id": "T7", "clean_text": "amazing love it fantastic incredible best ever 😍🤩🔥", "overall_sentiment": "positive"},
        {"review_id": "T8", "clean_text": "bilkul bakwas hai yaar total waste 😡", "overall_sentiment": "negative"},
        {"review_id": "T9", "clean_text": "zabardast product ekdum mast shandaar 🔥", "overall_sentiment": "positive"},
    ]
    
    results = classify_emotion_batch(test_reviews)
    
    print("Emotion Intensity Classification Test")
    print("=" * 70)
    for result in results:
        # Safe print for Windows terminal (skip emoji char)
        label_text = result.emotion_label.split(" ", 1)[-1] if " " in result.emotion_label else result.emotion_label
        print(f"\n{result.review_id}: {label_text}")
        print(f"  Urgency: {result.urgency} | Confidence: {result.confidence}")
        print(f"  Signals: {result.signals_detected[:5]}")
        print(f"  Method: {result.method}")
    
    print("\n\nDistribution:")
    dist = get_emotion_distribution(results)
    for emotion, count in dist.items():
        emoji_label = EMOTION_TAXONOMY[emotion]["label"]
        safe_label = emoji_label.split(" ", 1)[-1] if " " in emoji_label else emoji_label
        print(f"  {safe_label}: {count}")
