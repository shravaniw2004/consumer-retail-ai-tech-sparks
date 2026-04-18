"""
ReviewIQ Text Preprocessor v2
Comprehensive preprocessing for raw review data.
Handles Hinglish, emojis, encoding issues, translation, and quality scoring.
"""

import re
import math
import unicodedata
from typing import Dict, List, Tuple
from collections import Counter

import ftfy
import emoji
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# Common Hindi words written in Latin script to detect Hinglish
HINDI_WORDS = {
    'hai', 'bhi', 'ki', 'se', 'ko', 'aur', 'nahi', 'bahut', 'accha', 'acha', 
    'sab', 'kya', 'hain', 'mein', 'ho', 'gaya', 'tha', 'thi', 'hua', 'kuch', 
    'ab', 'toh', 'tum', 'main', 'mujhe', 'mera', 'meri', 'he', 'ye', 'yeh',
    'wo', 'woh', 'bahi', 'bhai', 'yaar', 'mast', 'bakwas', 'bekar', 'kharab',
    'theek', 'karo', 'kar', 'raha', 'rahi', 'ka', 'ke', 'pas', 'paas'
}

HINGLISH_ABBREV = {
    "gr8": "great",
    "pls": "please",
    "bcz": "because",
    "nahi": "not",
    "bahut": "very",
    "accha": "good",
    "acha": "good",
    "bakwas": "nonsense/bad",
    "mast": "excellent",
    "kharab": "bad/broken",
    "theek": "okay"
}

CUSTOM_EMOJI_MAP = {
    "😡": "[ANGRY]",
    "🔥": "[EXCELLENT]",
    "💔": "[DISAPPOINTED]",
    "👍": "[POSITIVE]",
    "😍": "[LOVE]",
    "🤬": "[FURIOUS]",
    "😤": "[FRUSTRATED]",
}

PRODUCT_TERMS = {'buy', 'purchased', 'product', 'quality', 'packaging', 'delivery', 'price', 'item', 'money', 'worth'}
PERSONAL_TERMS = {'i', 'my', 'me', 'bought', 'mine', 'we', 'our'}

class ReviewPreprocessor:
    def __init__(self):
        self.stats = {
            'processed_count': 0,
            'low_quality_count': 0,
        }
        self.translator = GoogleTranslator(source='auto', target='en')

    def normalize_encoding(self, text: str) -> str:
        """Step 1: Fix broken unicode, strip zero-width, normalize quotes."""
        if not text:
            return ""
        
        # ftfy fixes mojibake
        text = ftfy.fix_text(text)
        
        # Strip null bytes
        text = text.replace('\x00', '')
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero width
        for char in ['\u200b', '\u200c', '\u200d', '\ufeff', '\u180e']:
            text = text.replace(char, '')
            
        # Normalize quotes
        text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
        return text

    def handle_emojis(self, text: str) -> Tuple[str, List[str]]:
        """Step 2: Convert emojis to meaning tags preserving in text."""
        emoji_tags = []
        
        # Find all emojis
        found_emojis = [c['emoji'] for c in emoji.emoji_list(text)]
        
        for em in found_emojis:
            # Get the tag value
            tag = CUSTOM_EMOJI_MAP.get(em)
            if not tag:
                # Fallback to demojize -> :thumbs_up: -> [THUMBS UP]
                demoj = emoji.demojize(em)
                clean_tag = demoj.replace(':', '').replace('_', ' ').upper()
                tag = f"[{clean_tag}]"
            
            text = text.replace(em, f" {tag} ")
            if tag not in emoji_tags:
                emoji_tags.append(tag)
                
        # cleanup extra spaces from replacement
        text = re.sub(r'\s+', ' ', text).strip()
        return text, emoji_tags

    def detect_multilingual(self, text: str) -> Tuple[str, str, str]:
        """Step 3: Detect language, expand hinglish, and translate."""
        # Simplify text to detect language accurately
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 'OTHER', text, text
            
        lang = 'en'
        try:
            lang = detect(text)
        except LangDetectException:
            pass

        detected_language = 'EN'
        
        # Hinglish detection
        hindi_count = sum(1 for w in words if w in HINDI_WORDS)
        if words and (hindi_count / len(words)) > 0.20:
            detected_language = 'HINGLISH'
        elif lang == 'hi':
            detected_language = 'HI'
        elif lang != 'en':
            detected_language = 'OTHER'
            if lang in ['mr', 'bn', 'ta', 'te', 'gu', 'kn', 'ml', 'pa']:
                detected_language = 'HI' # Route local languages to translation

        # Apply Hinglish abbreviations
        if detected_language == 'HINGLISH' or detected_language == 'EN':
            for abbrev, full in HINGLISH_ABBREV.items():
                text = re.sub(r'\b' + abbrev + r'\b', full, text, flags=re.IGNORECASE)

        translated_text = text
        if detected_language in ['HI', 'HINGLISH', 'OTHER']:
            try:
                # Strip emoji tags temporarily for translation or rely on translator ignoring them
                translated_text = self.translator.translate(text)
                if not translated_text:
                    translated_text = text
            except Exception as e:
                print(f"Translation Error: {e}")
                
        return detected_language, translated_text, text

    def normalize_text(self, text: str) -> Tuple[str, bool]:
        """Step 4: Caps, Punctuation, Whitespace, HTML."""
        caps_intensity = False
        
        # Check ALL CAPS words and Title Case them
        words = text.split()
        normalized_words = []
        for w in words:
            # Check if alphabetic word is entirely title case or upper case
            is_alpha_upper = sum(1 for c in w if c.isalpha() and c.isupper())
            total_alpha = sum(1 for c in w if c.isalpha())
            if total_alpha > 1 and is_alpha_upper == total_alpha:
                caps_intensity = True
                normalized_words.append(w.title())
            else:
                normalized_words.append(w)
                
        text = ' '.join(normalized_words)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        # Remove repeated characters (e.g. soooo -> so)
        # Using a pattern that replaces 3 or more occurrences with 2 occurrences, unless it's an 'o' in which maybe 2?
        # A simple approach to compress massive char repetitions
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
        
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text, caps_intensity

    def score_quality(self, text: str, original: str, emoji_tags: List[str]) -> Tuple[int, bool, List[str]]:
        """Step 5: Review Quality Scoring."""
        score = 0
        flags = []
        
        words = [w for w in text.split() if not w.startswith('[')] # don't count emoji tags
        num_words = len(words)
        
        if num_words == 0 and emoji_tags:
            score = 15
            flags.append("only_emojis")
        elif num_words < 5:
            score = 10
            flags.append("too_short")
        else:
            if num_words >= 20:
                score = 80
            else:
                score = 50 + (num_words * 1.5) # Scale up to 80
                
            text_lower = text.lower()
            word_set = set(re.findall(r'\b\w+\b', text_lower))
            
            if word_set.intersection(PRODUCT_TERMS):
                score += 10
            else:
                flags.append("no_product_terms")
                
            if word_set.intersection(PERSONAL_TERMS):
                score += 10
            else:
                flags.append("no_personal_experience")
                
        score = min(math.ceil(score), 100) if 'math' in globals() else min(int(score), 100)
        low_quality = bool(score < 20)
        
        return score, low_quality, flags

    def process(self, review_text: str) -> Dict:
        """Main processing pipeline."""
        original_text = review_text
        if not original_text or str(original_text).strip() == '':
            self.stats['low_quality_count'] += 1
            return {
                'processed_text': '',
                'original_text': original_text,
                'detected_language': 'EN',
                'emoji_tags': [],
                'caps_intensity': False,
                'review_quality_score': 0,
                'low_quality': True,
                'preprocessing_flags': ['empty_text']
            }

        # Step 1
        text = self.normalize_encoding(original_text)
        
        # Step 2
        text, emoji_tags = self.handle_emojis(text)
        
        # Step 3
        detected_language, translated_text, _ = self.detect_multilingual(text)
        text = translated_text
        
        # Step 4
        text, caps_intensity = self.normalize_text(text)
        
        # Step 5
        score, low_quality, flags = self.score_quality(text, original_text, emoji_tags)
        
        self.stats['processed_count'] += 1
        if low_quality:
            self.stats['low_quality_count'] += 1

        return {
            'processed_text': text,
            'original_text': original_text,
            'detected_language': detected_language,
            'emoji_tags': emoji_tags,
            'caps_intensity': caps_intensity,
            'review_quality_score': score,
            'low_quality': low_quality,
            'preprocessing_flags': flags,
            # For backward compatibility with some other pieces, add these generic ones:
            'clean_text': text,
            'language': detected_language.lower(),
            'trust_score': score / 100.0,
            'is_suspicious': low_quality
        }

    def process_batch(self, reviews: List[str]) -> List[Dict]:
        return [self.process(r) for r in reviews]

    def get_stats(self) -> Dict:
        return self.stats

if __name__ == "__main__":
    pre = ReviewPreprocessor()
    test = [
        "AMAZING PRODUCT MUST BUY CLICK HERE!!!!!!",
        "packing tuta hua tha  😡😡",
        "bilkul bakwas dlvry, bahut kharab",
        "sooooo goood 😍",
        "👍",
        "   "
    ]
    for r in test:
        print(f"\nOriginal: {r}")
        res = pre.process(r)
        for k,v in res.items():
            if k not in ['clean_text', 'language', 'trust_score', 'is_suspicious']:
                print(f"  {k}: {v}")
