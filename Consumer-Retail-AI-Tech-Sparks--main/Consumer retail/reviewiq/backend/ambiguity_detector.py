"""
ReviewIQ Ambiguity Detector - Two-Pass Sarcasm Detection Strategy
Combines regex-based heuristics with LLM verification for accurate sarcasm detection.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from mistralai.client import Mistral


# Pass 1: Regex patterns for obvious sarcasm indicators
SARCASM_PATTERNS = {
    # Quotes around positive words (e.g., "great" quality, "amazing" product)
    'quoted_positive': re.compile(
        r'["\'](?:great|amazing|excellent|fantastic|wonderful|perfect|awesome|best|love|nice)["\']',
        re.IGNORECASE
    ),
    
    # Excessive punctuation (3+ consecutive ! or ? or mixed)
    'excessive_punct': re.compile(r'[!?]{3,}|[!?.]{4,}'),
    
    # Contradictory adjectives (positive intensifier + negative word)
    'contradictory_adj': re.compile(
        r'\b(?:amazingly|incredibly|absolutely|totally|completely|really|so|such)\s+(?:terrible|awful|bad|worst|horrible|useless|waste|pathetic)',
        re.IGNORECASE
    ),
    
    # Explicit sarcasm markers
    'sarcasm_markers': re.compile(
        r'\b(?:yeah right|sure|obviously|clearly|totally|definitely|as if|like i care|tell me about it|oh great|just what i needed)',
        re.IGNORECASE
    ),
    
    # All caps positive word (shouting sarcasm)
    'shouting_positive': re.compile(r'\b(?:GREAT|AMAZING|EXCELLENT|FANTASTIC|PERFECT|AWESOME|LOVE IT)\b'),
}

# Common positive words that when quoted suggest sarcasm
POSITIVE_WORDS = [
    'great', 'amazing', 'excellent', 'fantastic', 'wonderful', 'perfect',
    'awesome', 'best', 'love', 'nice', 'good', 'beautiful', 'quality'
]

# Pass 2: Mistral API prompt for sarcasm analysis
SARCASM_ANALYSIS_PROMPT = """You are an expert in detecting sarcasm and irony in customer reviews. Analyze the following review for sarcastic intent.

Review Text: {review_text}
Product: {product_name}
Category: {category}

Consider these sarcasm indicators:
1. Verbal irony (saying opposite of what is meant)
2. Exaggerated praise for clearly defective products
3. Mocking tone or phrasing
4. Quotes around positive words (e.g., "great" quality)
5. Excessive punctuation (!!!, ???)
6. Contradictory statements ("amazingly terrible")
7. Context that makes positive words clearly negative

Return ONLY this JSON structure:
{{
  "sarcasm_detected": true|false,
  "confidence": 0.0-1.0,
  "explanation": "brief reason for detection or non-detection",
  "indicators": ["list of specific phrases that indicated sarcasm"]
}}

Respond with valid JSON only. No markdown, no explanations outside JSON."""


@dataclass
class SarcasmResult:
    """Result of sarcasm detection for a single review."""
    review_id: str
    sarcasm_detected: bool
    confidence: float
    explanation: str
    indicators: List[str]
    pass_1_flags: List[str]
    requires_llm_check: bool
    human_review_needed: bool = False


class AmbiguityDetector:
    """Two-pass sarcasm detection combining regex heuristics with LLM verification."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = Mistral(api_key=self.api_key)
    
    def _pass1_regex_detection(self, review_text: str) -> Tuple[bool, List[str], float]:
        """
        Pass 1: Detect obvious sarcasm patterns using regex.
        
        Returns:
            Tuple of (is_flagged, list_of_flags, confidence_score)
        """
        if not review_text:
            return False, [], 0.0
        
        text = review_text.lower()
        flags = []
        confidence_boost = 0.0
        
        # Check quoted positive words
        if SARCASM_PATTERNS['quoted_positive'].search(review_text):
            flags.append('quoted_positive_words')
            confidence_boost += 0.3
        
        # Check excessive punctuation
        if SARCASM_PATTERNS['excessive_punct'].search(review_text):
            flags.append('excessive_punctuation')
            confidence_boost += 0.2
        
        # Check contradictory adjectives
        if SARCASM_PATTERNS['contradictory_adj'].search(review_text):
            flags.append('contradictory_intensifier')
            confidence_boost += 0.35
        
        # Check explicit sarcasm markers
        matches = SARCASM_PATTERNS['sarcasm_markers'].findall(text)
        if matches:
            flags.append(f'sarcasm_markers: {matches}')
            confidence_boost += 0.3
        
        # Check shouting positive (all caps)
        if SARCASM_PATTERNS['shouting_positive'].search(review_text):
            flags.append('shouting_positive_caps')
            confidence_boost += 0.25
        
        # Additional heuristic: positive word followed by negative context
        positive_with_complaint = re.search(
            r'\b(?:great|amazing|good)\b.*?(?:but|except|however|although|though).*?(?:bad|terrible|problem|issue)',
            text
        )
        if positive_with_complaint:
            flags.append('positive_but_negative_context')
            confidence_boost += 0.2
        
        is_flagged = len(flags) > 0
        confidence = min(0.7, confidence_boost)  # Cap Pass 1 confidence at 0.7
        
        return is_flagged, flags, confidence
    
    def _pass2_llm_verification(
        self,
        review_text: str,
        product_name: str,
        category: str
    ) -> SarcasmResult:
        """
        Pass 2: Use Mistral API to verify sarcasm in ambiguous reviews.
        
        Returns:
            SarcasmResult with LLM analysis
        """
        if not self.client:
            # No API key - return uncertain result
            return SarcasmResult(
                review_id="",
                sarcasm_detected=False,
                confidence=0.0,
                explanation="No API key available for LLM verification",
                indicators=[],
                pass_1_flags=[],
                requires_llm_check=True,
                human_review_needed=True
            )
        
        prompt = SARCASM_ANALYSIS_PROMPT.format(
            review_text=review_text,
            product_name=product_name,
            category=category
        )
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up potential markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            return SarcasmResult(
                review_id="",
                sarcasm_detected=data.get("sarcasm_detected", False),
                confidence=data.get("confidence", 0.0),
                explanation=data.get("explanation", ""),
                indicators=data.get("indicators", []),
                pass_1_flags=[],
                requires_llm_check=True,
                human_review_needed=data.get("sarcasm_detected", False)
            )
            
        except json.JSONDecodeError:
            return SarcasmResult(
                review_id="",
                sarcasm_detected=False,
                confidence=0.0,
                explanation="Failed to parse LLM response",
                indicators=[],
                pass_1_flags=[],
                requires_llm_check=True,
                human_review_needed=True
            )
            
        except Exception as e:
            return SarcasmResult(
                review_id="",
                sarcasm_detected=False,
                confidence=0.0,
                explanation=f"LLM API error: {str(e)}",
                indicators=[],
                pass_1_flags=[],
                requires_llm_check=True,
                human_review_needed=True
            )
    
    def detect_sarcasm(self, review: Dict) -> SarcasmResult:
        """
        Two-pass sarcasm detection for a single review.
        
        Pass 1: Fast regex-based detection
        Pass 2: LLM verification for flagged reviews or low confidence
        
        Args:
            review: Review dictionary with clean_text, product_name, category, sentiment_confidence
        
        Returns:
            SarcasmResult with final determination
        """
        review_id = review.get("review_id", "unknown")
        text = review.get("clean_text", "")
        product = review.get("product_name", "Unknown")
        category = review.get("category", "General")
        sentiment_confidence = review.get("sentiment_confidence", 1.0)
        
        # Pass 1: Regex detection
        pass1_flagged, pass1_flags, pass1_confidence = self._pass1_regex_detection(text)
        
        # Determine if Pass 2 is needed
        requires_llm = (
            pass1_flagged or  # Flagged by regex
            sentiment_confidence < 0.6 or  # Low sentiment confidence
            review.get("overall_sentiment") == "mixed"  # Mixed sentiment often sarcastic
        )
        
        if not requires_llm:
            # Pass 1 cleared it - no sarcasm detected
            return SarcasmResult(
                review_id=review_id,
                sarcasm_detected=False,
                confidence=0.9,
                explanation="No sarcasm indicators detected in Pass 1",
                indicators=[],
                pass_1_flags=[],
                requires_llm_check=False,
                human_review_needed=False
            )
        
        # Pass 2: LLM verification
        llm_result = self._pass2_llm_verification(text, product, category)
        llm_result.review_id = review_id
        llm_result.pass_1_flags = pass1_flags
        
        # Combine Pass 1 and Pass 2 results
        if llm_result.sarcasm_detected or pass1_confidence > 0.5:
            final_confidence = max(pass1_confidence, llm_result.confidence)
            return SarcasmResult(
                review_id=review_id,
                sarcasm_detected=True,
                confidence=final_confidence,
                explanation=llm_result.explanation or f"Pass 1 flags: {pass1_flags}",
                indicators=llm_result.indicators or pass1_flags,
                pass_1_flags=pass1_flags,
                requires_llm_check=True,
                human_review_needed=True  # Sarcasm requires human review
            )
        
        return llm_result
    
    def detect_batch(self, reviews: List[Dict]) -> List[SarcasmResult]:
        """Detect sarcasm for a batch of reviews."""
        return [self.detect_sarcasm(r) for r in reviews]


def flag_ambiguous_reviews(
    reviews: List[Dict],
    api_key: Optional[str] = None,
    db_session=None
) -> List[Dict]:
    """
    Process sentiment engine output and flag ambiguous/sarcastic reviews.
    
    This function:
    1. Runs two-pass sarcasm detection on each review
    2. Updates sentiment to "ambiguous" if sarcasm detected
    3. Sets sarcasm_flag = True for sarcastic reviews
    4. Marks human_review_needed for ambiguous cases
    5. Optionally updates database if db_session provided
    
    Args:
        reviews: List of reviews from sentiment_engine (with extracted_features, overall_sentiment)
        api_key: Mistral API key (or from env)
        db_session: Optional SQLAlchemy session for database updates
    
    Returns:
        Updated reviews with sarcasm detection fields
    """
    detector = AmbiguityDetector(api_key=api_key)
    
    results = []
    sarcasm_count = 0
    human_review_count = 0
    
    print(f"\nRunning ambiguity detection on {len(reviews)} reviews...")
    
    for review in reviews:
        # Run two-pass detection
        sarcasm_result = detector.detect_sarcasm(review)
        
        # Update review based on detection
        if sarcasm_result.sarcasm_detected:
            review["overall_sentiment"] = "ambiguous"
            review["sarcasm_flag"] = True
            review["sarcasm_confidence"] = sarcasm_result.confidence
            review["sarcasm_explanation"] = sarcasm_result.explanation
            review["sarcasm_indicators"] = sarcasm_result.indicators
            review["human_review_needed"] = True
            sarcasm_count += 1
        else:
            review["sarcasm_flag"] = False
            review["sarcasm_confidence"] = sarcasm_result.confidence
        
        # Track human review needs
        if review.get("human_review_needed") or sarcasm_result.human_review_needed:
            review["human_review_needed"] = True
            human_review_count += 1
        
        # Add ambiguity detection metadata
        review["ambiguity_detection"] = {
            "pass_1_flags": sarcasm_result.pass_1_flags,
            "requires_llm_check": sarcasm_result.requires_llm_check,
            "llm_explanation": sarcasm_result.explanation,
        }
        
        results.append(review)
    
    # Update database if session provided
    if db_session:
        _update_database(db_session, results)
    
    print(f"  Sarcasm detected: {sarcasm_count}")
    print(f"  Marked for human review: {human_review_count}")
    
    return results


def _update_database(db_session, reviews: List[Dict]):
    """Update database with ambiguity detection results."""
    try:
        # Import models here to avoid circular dependencies
        from data.database import FeatureExtraction
        
        for review in reviews:
            # Update feature extractions with sarcasm flag
            for feature in review.get("extracted_features", []):
                feature["sarcasm_flag"] = review.get("sarcasm_flag", False)
                feature["ambiguity_flag"] = review.get("overall_sentiment") == "ambiguous"
        
        db_session.commit()
        print("  Database updated successfully")
        
    except Exception as e:
        print(f"  Database update failed: {e}")
        db_session.rollback()


# Convenience functions
def check_sarcasm_single(review_text: str, api_key: Optional[str] = None) -> Dict:
    """Quick check for a single review text."""
    detector = AmbiguityDetector(api_key=api_key)
    
    review = {
        "review_id": "single_check",
        "clean_text": review_text,
        "product_name": "Unknown",
        "category": "General",
        "sentiment_confidence": 0.5,
    }
    
    result = detector.detect_sarcasm(review)
    
    return {
        "sarcasm_detected": result.sarcasm_detected,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "indicators": result.indicators,
        "pass_1_flags": result.pass_1_flags,
    }


if __name__ == "__main__":
    # Test cases
    test_reviews = [
        {
            "review_id": "TEST001",
            "clean_text": 'Oh "great" quality as always! Totally worth the money... NOT',
            "product_name": "SmartBottle Pro",
            "category": "Personal Care",
            "overall_sentiment": "positive",
            "sentiment_confidence": 0.4,
        },
        {
            "review_id": "TEST002",
            "clean_text": "Amazingly terrible product, absolutely love wasting my money!!!",
            "product_name": "BoltCharge 20W",
            "category": "Electronics",
            "overall_sentiment": "negative",
            "sentiment_confidence": 0.8,
        },
        {
            "review_id": "TEST003",
            "clean_text": "Perfect product, exactly what I needed for daily use",
            "product_name": "NutriMix Blender",
            "category": "Food",
            "overall_sentiment": "positive",
            "sentiment_confidence": 0.9,
        },
        {
            "review_id": "TEST004",
            "clean_text": "Yeah right, this is definitely the BEST purchase I've ever made...",
            "product_name": "Generic Product",
            "category": "Electronics",
            "overall_sentiment": "positive",
            "sentiment_confidence": 0.5,
        },
    ]
    
    print("Ambiguity Detector Test\n")
    print("=" * 80)
    
    results = flag_ambiguous_reviews(test_reviews)
    
    print("\nDetailed Results:")
    for r in results:
        print(f"\n{r['review_id']}: {r['clean_text'][:50]}...")
        print(f"  Sarcasm: {r.get('sarcasm_flag', False)} (confidence: {r.get('sarcasm_confidence', 0):.2f})")
        print(f"  Final sentiment: {r['overall_sentiment']}")
        print(f"  Human review needed: {r.get('human_review_needed', False)}")
        if r.get('ambiguity_detection', {}).get('pass_1_flags'):
            print(f"  Pass 1 flags: {r['ambiguity_detection']['pass_1_flags']}")
