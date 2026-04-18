"""
ReviewIQ Vocabulary Detector - Emerging Complaint Phrase Detection
Uses spaCy to identify new complaint types not in predefined taxonomy.
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter

import spacy
from spacy.tokens import Doc


# Common words to filter out from noun phrases
COMMON_STOP_WORDS = {
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'and', 'or', 'but', 'so', 'yet', 'for', 'nor',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might',
    'can', 'must', 'shall',
    'very', 'really', 'quite', 'pretty', 'too', 'so',
    'just', 'only', 'even', 'also', 'still', 'already',
    'here', 'there', 'now', 'then', 'today', 'tomorrow',
    'good', 'bad', 'nice', 'great', 'okay', 'fine',
    'thing', 'things', 'stuff', 'something', 'anything',
    'one', 'two', 'first', 'second', 'way', 'ways',
    'time', 'times', 'day', 'days', 'lot', 'lots',
    'bit', 'piece', 'part', 'parts', 'kind', 'sort'
}

# Product-specific words to filter (kept generic, should be customized per domain)
PRODUCT_WORDS = {
    'product', 'item', 'order', 'delivery', 'package', 'box',
    'amazon', 'flipkart', 'seller', 'company', 'brand'
}

# Global spaCy model (lazy loaded)
_nlp = None


def get_nlp_model() -> spacy.Language:
    """
    Lazy load spaCy model.
    
    Returns:
        Loaded spaCy English model
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    return _nlp


def _clean_phrase(phrase: str) -> str:
    """
    Clean and normalize a phrase.
    
    - Lowercase
    - Remove extra whitespace
    - Strip punctuation
    """
    # Lowercase and strip
    phrase = phrase.lower().strip()
    
    # Remove extra whitespace
    phrase = ' '.join(phrase.split())
    
    # Remove trailing punctuation
    phrase = re.sub(r'[^\w\s]', '', phrase)
    
    return phrase


def _is_valid_phrase(phrase: str, min_words: int = 2, max_words: int = 4) -> bool:
    """
    Check if phrase is valid for vocabulary detection.
    
    Criteria:
    - 2-4 words (configurable)
    - Not all stop words
    - Contains at least one content word (noun, verb, adj)
    - Not just numbers
    """
    words = phrase.split()
    
    # Check word count
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Check if all words are stop words or product words
    content_words = [
        w for w in words
        if w not in COMMON_STOP_WORDS and w not in PRODUCT_WORDS
    ]
    
    if not content_words:
        return False
    
    # Check if it's just numbers
    if all(w.isdigit() for w in words):
        return False
    
    # Check minimum length (avoid single-char words)
    if all(len(w) <= 2 for w in words):
        return False
    
    return True


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases from text using spaCy.
    
    Uses spaCy noun chunks and filters for 2-4 word phrases,
    removing common stop words and product-specific terms.
    
    Args:
        text: Input text to analyze
    
    Returns:
        List of valid noun phrases (2-4 words)
    """
    if not text or not text.strip():
        return []
    
    nlp = get_nlp_model()
    doc = nlp(text)
    
    phrases = []
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        phrase = _clean_phrase(chunk.text)
        
        if _is_valid_phrase(phrase, min_words=2, max_words=4):
            phrases.append(phrase)
    
    # Also extract bigrams and trigrams from noun tokens
    # This catches phrases not in noun_chunks
    noun_tokens = [
        token.text.lower()
        for token in doc
        if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop
    ]
    
    # Generate 2-4 word combinations from adjacent noun tokens
    for i in range(len(noun_tokens)):
        for j in range(i + 1, min(i + 4, len(noun_tokens) + 1)):
            phrase = ' '.join(noun_tokens[i:j])
            if _is_valid_phrase(phrase, min_words=2, max_words=4):
                phrases.append(phrase)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_phrases = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            unique_phrases.append(phrase)
    
    return unique_phrases


def extract_phrases_from_reviews(reviews: List[Dict]) -> Counter:
    """
    Extract all noun phrases from a list of reviews.
    
    Args:
        reviews: List of review dictionaries with 'clean_text' field
    
    Returns:
        Counter of phrase -> count
    """
    phrase_counts = Counter()
    
    for review in reviews:
        text = review.get('clean_text', '') or review.get('review_text', '')
        if text:
            phrases = extract_noun_phrases(text)
            phrase_counts.update(phrases)
    
    return phrase_counts


def detect_emerging_vocabulary(
    window_current: List[Dict],
    window_baseline: List[Dict],
    min_appearances: int = 5,
    baseline_threshold: int = 2
) -> List[Dict[str, any]]:
    """
    Detect emerging vocabulary phrases.
    
    Finds phrases that appear frequently in current window but were
    rare or absent in the baseline window. Indicates new complaint types.
    
    Criteria:
    - ≥ min_appearances (default 5) in current window
    - < baseline_threshold (default 2) in baseline window
    
    Args:
        window_current: List of reviews from current time window
        window_baseline: List of reviews from baseline window
        min_appearances: Minimum appearances in current window to be considered
        baseline_threshold: Maximum appearances in baseline (exclusive)
    
    Returns:
        List of dicts with:
        - phrase: The emerging phrase
        - current_count: Count in current window
        - baseline_count: Count in baseline window
        - emergence_score: How strongly it emerged (current - baseline)
    """
    # Extract phrases from both windows
    current_phrases = extract_phrases_from_reviews(window_current)
    baseline_phrases = extract_phrases_from_reviews(window_baseline)
    
    emerging = []
    
    # Check each phrase in current window
    for phrase, current_count in current_phrases.items():
        # Must meet minimum appearances in current window
        if current_count < min_appearances:
            continue
        
        # Check baseline count
        baseline_count = baseline_phrases.get(phrase, 0)
        
        # Must be below threshold in baseline
        if baseline_count >= baseline_threshold:
            continue
        
        # Calculate emergence score
        emergence_score = current_count - baseline_count
        
        emerging.append({
            'phrase': phrase,
            'current_count': current_count,
            'baseline_count': baseline_count,
            'emergence_score': emergence_score
        })
    
    # Sort by emergence score (descending)
    emerging.sort(key=lambda x: x['emergence_score'], reverse=True)
    
    return emerging


def find_novel_complaint_types(
    current_phrases: List[str],
    taxonomy_features: List[str]
) -> List[str]:
    """
    Identify complaint types not in predefined taxonomy.
    
    Args:
        current_phrases: List of detected phrases
        taxonomy_features: List of known feature names from taxonomy
    
    Returns:
        List of novel complaint phrases not matching taxonomy
    """
    # Normalize taxonomy for comparison
    taxonomy_normalized = {f.lower().replace('_', ' ') for f in taxonomy_features}
    
    novel = []
    for phrase in current_phrases:
        phrase_normalized = phrase.lower()
        
        # Check if phrase matches or contains any taxonomy term
        is_in_taxonomy = any(
            tax in phrase_normalized or phrase_normalized in tax
            for tax in taxonomy_normalized
        )
        
        if not is_in_taxonomy:
            novel.append(phrase)
    
    return novel


class VocabularyTracker:
    """
    Track vocabulary emergence across multiple windows.
    """
    
    def __init__(self, taxonomy_features: Optional[List[str]] = None):
        self.taxonomy = taxonomy_features or []
        self.window_history: List[Tuple[str, List[Dict]]] = []
        self.emerging_cache: Dict[str, List[Dict]] = {}
    
    def add_window(self, window_label: str, reviews: List[Dict]):
        """Add a window to tracking history."""
        self.window_history.append((window_label, reviews))
    
    def detect_emerging_for_window(
        self,
        window_label: str,
        baseline_label: Optional[str] = None,
        min_appearances: int = 5
    ) -> List[Dict]:
        """
        Detect emerging vocabulary for a specific window.
        
        Args:
            window_label: Target window to analyze
            baseline_label: Baseline window (default: first window)
            min_appearances: Minimum appearances to be considered
        
        Returns:
            List of emerging phrases
        """
        # Find windows
        window_map = {label: reviews for label, reviews in self.window_history}
        
        if window_label not in window_map:
            return []
        
        current = window_map[window_label]
        
        if baseline_label and baseline_label in window_map:
            baseline = window_map[baseline_label]
        else:
            # Use first window as baseline
            baseline = self.window_history[0][1] if self.window_history else []
        
        return detect_emerging_vocabulary(
            current, baseline, min_appearances=min_appearances
        )
    
    def get_novel_complaints(
        self,
        window_label: str,
        min_appearances: int = 3
    ) -> List[str]:
        """Get complaint types not in predefined taxonomy."""
        window_map = {label: reviews for label, reviews in self.window_history}
        
        if window_label not in window_map:
            return []
        
        reviews = window_map[window_label]
        phrase_counts = extract_phrases_from_reviews(reviews)
        
        # Filter by minimum appearances
        frequent_phrases = [
            phrase for phrase, count in phrase_counts.items()
            if count >= min_appearances
        ]
        
        return find_novel_complaint_types(frequent_phrases, self.taxonomy)
    
    def get_vocabulary_trend(
        self,
        phrase: str
    ) -> List[Tuple[str, int]]:
        """Get trend of a specific phrase across all windows."""
        trend = []
        
        for label, reviews in self.window_history:
            phrase_counts = extract_phrases_from_reviews(reviews)
            count = phrase_counts.get(phrase, 0)
            trend.append((label, count))
        
        return trend


def generate_vocabulary_alert(
    emerging_phrases: List[Dict],
    product_name: str,
    window_label: str
) -> Optional[Dict]:
    """
    Generate alert for significant vocabulary emergence.
    
    Args:
        emerging_phrases: List of emerging phrase dicts
        product_name: Product name
        window_label: Current window label
    
    Returns:
        Alert dict if significant emergence detected, None otherwise
    """
    if not emerging_phrases:
        return None
    
    # Check if any highly significant emergence
    significant = [p for p in emerging_phrases if p['emergence_score'] >= 5]
    
    if not significant:
        return None
    
    return {
        'alert_type': 'emerging_vocabulary',
        'product_name': product_name,
        'window': window_label,
        'severity': 'high' if len(significant) >= 3 else 'medium',
        'emerging_phrases': significant[:10],  # Top 10
        'recommendation': 'Review for new complaint types not in taxonomy',
        'suggested_taxonomy_additions': [p['phrase'] for p in significant[:5]]
    }


if __name__ == "__main__":
    # Test vocabulary detection
    print("Vocabulary Detector Test\n")
    print("=" * 80)
    
    # Test 1: Basic noun phrase extraction
    print("\nTest 1: Noun Phrase Extraction")
    test_texts = [
        "The battery life is terrible and there's a weird smell coming from the motor.",
        "I hear a clicking noise when I turn it on, very annoying.",
        "The packaging was damaged and there was a strange odor from the box.",
        "The heating element makes a buzzing sound, not good quality.",
    ]
    
    all_phrases = []
    for text in test_texts:
        phrases = extract_noun_phrases(text)
        all_phrases.extend(phrases)
        print(f"  Text: {text[:50]}...")
        print(f"  Phrases: {phrases}")
    
    print(f"\n  Unique phrases found: {set(all_phrases)}")
    
    # Test 2: Emerging vocabulary detection
    print("\n" + "=" * 80)
    print("\nTest 2: Emerging Vocabulary Detection")
    
    # Simulate baseline window (older reviews)
    baseline_reviews = [
        {'clean_text': 'Good product works well', 'review_id': 'B1'},
        {'clean_text': 'Nice quality happy with purchase', 'review_id': 'B2'},
        {'clean_text': 'Battery lasts long good value', 'review_id': 'B3'},
    ]
    
    # Simulate current window with new complaints
    current_reviews = [
        {'clean_text': 'Weird smell from the motor very concerning', 'review_id': 'C1'},
        {'clean_text': 'Clicking noise every time I use it annoying', 'review_id': 'C2'},
        {'clean_text': 'Weird smell again not happy with this', 'review_id': 'C3'},
        {'clean_text': 'Clicking noise getting worse over time', 'review_id': 'C4'},
        {'clean_text': 'Weird smell persists even after cleaning', 'review_id': 'C5'},
        {'clean_text': 'Clicking noise and weird smell both issues', 'review_id': 'C6'},
        {'clean_text': 'Weird smell is the main problem here', 'review_id': 'C7'},
    ]
    
    emerging = detect_emerging_vocabulary(
        current_reviews,
        baseline_reviews,
        min_appearances=2,
        baseline_threshold=1
    )
    
    print(f"\n  Emerging phrases detected:")
    for item in emerging:
        print(f"    - '{item['phrase']}': {item['current_count']} current, {item['baseline_count']} baseline (score: {item['emergence_score']})")
    
    # Test 3: Vocabulary tracker
    print("\n" + "=" * 80)
    print("\nTest 3: Vocabulary Tracker")
    
    from sentiment_engine import FEATURE_TAXONOMY
    
    tracker = VocabularyTracker(taxonomy_features=FEATURE_TAXONOMY.get('Electronics', []))
    
    tracker.add_window('W1', baseline_reviews)
    tracker.add_window('W2', current_reviews)
    
    novel = tracker.get_novel_complaints('W2', min_appearances=2)
    print(f"\n  Novel complaints (not in taxonomy): {novel}")
    
    # Test alert generation
    alert = generate_vocabulary_alert(emerging, 'SmartBottle Pro', 'W2')
    if alert:
        print(f"\n  Alert generated: {alert['severity']} severity")
        print(f"  Suggested additions to taxonomy: {alert['suggested_taxonomy_additions']}")
