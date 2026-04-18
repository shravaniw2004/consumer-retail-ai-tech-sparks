"""
ReviewIQ Sentiment Engine - Batch Processing v4
Processes reviews in batches of 5. Uses Mistral API.
Target: ~2 minutes for 300 reviews using parallel processing.
"""

import os
import json
import hashlib
import shelve
import time
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mistralai.client import Mistral

# Feature taxonomy by category
FEATURE_TAXONOMY = {
    "Electronics": ["battery", "durability", "charging", "performance", "packaging", "price"],
    "Personal Care": ["packaging", "texture", "scent", "effectiveness", "ingredients", "price"],
    "Food": ["taste", "freshness", "packaging", "ingredients", "value", "texture"],
}

# Batch extraction prompt for Mistral
BATCH_EXTRACTION_PROMPT = """You are ReviewIQ, an expert customer review analyst. Analyze the following {batch_size} product reviews and extract structured insights for each review.

TASK: For each review, identify product features mentioned and their associated sentiments.

FEATURE TAXONOMY by Category:
- Electronics: battery, durability, charging, performance, packaging, price
- Personal Care: packaging, texture, scent, effectiveness, ingredients, price
- Food: taste, freshness, packaging, ingredients, value, texture

For each review, extract features following this exact JSON structure:

{{
  "results": [
    {{
      "review_id": "string",
      "features": [
        {{
          "feature": "string (from taxonomy or infer)",
          "sentiment": "positive|negative|neutral|mixed",
          "intensity": 0.0-1.0,
          "confidence": 0.0-1.0,
          "evidence": "quoted text snippet supporting the sentiment",
          "flags": ["ambiguous", "sarcasm", "uncertain", "hinglish"]
        }}
      ],
      "overall_sentiment": "positive|negative|neutral|mixed",
      "trust_indicators": {{
        "specific_details": true|false,
        "balanced_perspective": true|false,
        "usage_context": true|false
      }}
    }}
  ]
}}

SENTIMENT GUIDELINES:
- positive: clear praise or satisfaction
- negative: clear complaint or dissatisfaction
- neutral: factual statement without emotion
- mixed: both positive and negative aspects mentioned

INTENSITY: 0.0 (mild mention) to 1.0 (strong emphasis/urgency)
CONFIDENCE: 0.0 (uncertain interpretation) to 1.0 (clear explicit statement)

FLAGS (include when applicable):
- "ambiguous": unclear or contradictory statements
- "sarcasm": likely sarcastic tone
- "uncertain": low confidence in interpretation
- "hinglish": mixed Hindi-English detected

REVIEWS TO ANALYZE:
{reviews_block}

Return ONLY valid JSON. No markdown, no explanations, no code blocks."""


@dataclass
class ExtractionResult:
    """Result of batch extraction for a single review."""
    review_id: str
    features: List[Dict[str, Any]]
    overall_sentiment: str
    trust_indicators: Dict[str, bool]
    human_review_needed: bool = False
    error_message: Optional[str] = None


class SentimentEngine:
    """Batch sentiment extraction using Mistral API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        cache_dir: str = ".cache",
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not provided")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self) -> str:
        """Get path to cache file."""
        return os.path.join(self.cache_dir, "sentiment_cache")
    
    def _compute_batch_hash(self, reviews: List[Dict]) -> str:
        """Compute MD5 hash of batch content for caching."""
        # Create deterministic string representation
        content = json.dumps(reviews, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def chunk_reviews(self, reviews: List[Dict], batch_size: int = 5) -> List[List[Dict]]:
        """Split reviews into batches of specified size."""
        return [
            reviews[i:i + batch_size]
            for i in range(0, len(reviews), batch_size)
        ]
    
    def _format_reviews_for_prompt(self, reviews: List[Dict]) -> str:
        """Format review batch for API prompt."""
        formatted = []
        for i, review in enumerate(reviews, 1):
            review_id = review.get('review_id', f'review_{i}')
            product = review.get('product_name', 'Unknown')
            category = review.get('category', 'General')
            text = review.get('clean_text', review.get('review_text', ''))
            
            formatted.append(
                f"REVIEW {i}:\n"
                f"  ID: {review_id}\n"
                f"  Product: {product}\n"
                f"  Category: {category}\n"
                f"  Text: {text}\n"
            )
        
        return "\n".join(formatted)
    
    def extract_batch(self, reviews: List[Dict]) -> List[ExtractionResult]:
        """
        Extract sentiments from a batch of reviews using Mistral API.
        
        Args:
            reviews: List of review dictionaries (max 5)
        
        Returns:
            List of ExtractionResult objects
        """
        if not reviews:
            return []
        
        if len(reviews) > 5:
            raise ValueError("Batch size cannot exceed 5 reviews")
        
        # Build prompt
        reviews_block = self._format_reviews_for_prompt(reviews)
        prompt = BATCH_EXTRACTION_PROMPT.format(
            batch_size=len(reviews),
            reviews_block=reviews_block
        )
        
        # API call with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                
                # Extract and parse JSON
                content = response.choices[0].message.content
                result = self._parse_extraction_response(content, reviews)
                return result
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                # Final attempt failed
                return self._create_error_results(reviews, f"API error: {str(e)}")
                

        
        return self._create_error_results(reviews, "Max retries exceeded")
    
    def _parse_extraction_response(
        self,
        content: str,
        reviews: List[Dict]
    ) -> List[ExtractionResult]:
        """Parse and validate JSON response from Mistral."""
        results = []
        
        try:
            # Clean up potential markdown/code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            if "results" not in data or not isinstance(data["results"], list):
                raise ValueError("Invalid JSON structure: missing 'results' array")
            
            # Map results by review_id
            result_map = {r.get("review_id"): r for r in data["results"]}
            
            for review in reviews:
                review_id = review.get("review_id")
                
                if review_id in result_map:
                    extracted = result_map[review_id]
                    results.append(ExtractionResult(
                        review_id=review_id,
                        features=extracted.get("features", []),
                        overall_sentiment=extracted.get("overall_sentiment", "neutral"),
                        trust_indicators=extracted.get("trust_indicators", {}),
                        human_review_needed=False,
                        error_message=None
                    ))
                else:
                    # Missing in response
                    results.append(ExtractionResult(
                        review_id=review_id,
                        features=[],
                        overall_sentiment="neutral",
                        trust_indicators={},
                        human_review_needed=True,
                        error_message="Missing in API response"
                    ))
            
            return results
            
        except json.JSONDecodeError as e:
            # JSON parsing failed - mark all for human review
            return self._create_error_results(reviews, f"JSON parse error: {str(e)}")
            
        except Exception as e:
            return self._create_error_results(reviews, f"Parse error: {str(e)}")
    
    def _create_error_results(
        self,
        reviews: List[Dict],
        error_message: str
    ) -> List[ExtractionResult]:
        """Create error results when API fails."""
        return [
            ExtractionResult(
                review_id=review.get("review_id", f"unknown_{i}"),
                features=[],
                overall_sentiment="neutral",
                trust_indicators={},
                human_review_needed=True,
                error_message=error_message
            )
            for i, review in enumerate(reviews)
        ]
    
    def cached_extract_batch(self, reviews: List[Dict]) -> List[ExtractionResult]:
        """
        Extract with caching to avoid re-calling API during demo prep.
        
        Uses MD5 hash of batch content as cache key.
        """
        if not reviews:
            return []
        
        batch_hash = self._compute_batch_hash(reviews)
        cache_path = self._get_cache_path()
        
        # Try cache first
        with shelve.open(cache_path) as cache:
            if batch_hash in cache:
                return cache[batch_hash]
        
        # Cache miss - call API
        results = self.extract_batch(reviews)
        
        # Store in cache
        with shelve.open(cache_path) as cache:
            cache[batch_hash] = results
        
        return results
    
    def process_all_reviews_parallel(
        self,
        reviews: List[Dict],
        max_workers: int = 3,
        use_cache: bool = True
    ) -> List[ExtractionResult]:
        """
        Process all reviews in parallel using ThreadPoolExecutor.
        
        Target: ~2 minutes for 300 reviews (60 batches / 3 workers = 20 rounds)
        
        Args:
            reviews: List of all reviews to process
            max_workers: Number of parallel threads (default 3)
            use_cache: Whether to use shelve caching
        
        Returns:
            List of ExtractionResult for all reviews
        """
        if not reviews:
            return []
        
        # Split into batches of 5
        batches = self.chunk_reviews(reviews, batch_size=5)
        total_batches = len(batches)
        
        print(f"Processing {len(reviews)} reviews in {total_batches} batches using {max_workers} workers...")
        
        all_results = []
        processed = 0
        
        # Choose extraction method
        extract_fn = self.cached_extract_batch if use_cache else self.extract_batch
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(extract_fn, batch): batch
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    processed += len(batch)
                    
                    if processed % 25 == 0 or processed == len(reviews):
                        print(f"  Progress: {processed}/{len(reviews)} reviews processed")
                        
                except Exception as e:
                    # Handle unexpected errors - mark batch for human review
                    print(f"  Error processing batch: {e}")
                    error_results = self._create_error_results(batch, str(e))
                    all_results.extend(error_results)
                    processed += len(batch)
        
        # Sort results by review_id to maintain order
        all_results.sort(key=lambda x: x.review_id)
        
        # Print summary
        human_review_count = sum(1 for r in all_results if r.human_review_needed)
        print(f"\nCompleted: {len(all_results)} reviews")
        print(f"  Successful: {len(all_results) - human_review_count}")
        print(f"  Needs human review: {human_review_count}")
        
        return all_results
    
    def clear_cache(self):
        """Clear the shelve cache."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            with shelve.open(cache_path) as cache:
                cache.clear()
            print("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return {"entries": 0, "path": cache_path}
        
        with shelve.open(cache_path) as cache:
            return {"entries": len(cache), "path": cache_path}


# Convenience function
def extract_sentiments(
    reviews: List[Dict],
    api_key: Optional[str] = None,
    use_cache: bool = True,
    max_workers: int = 3
) -> List[Dict]:
    """
    One-shot sentiment extraction with all processing.
    
    Args:
        reviews: List of review dictionaries
        api_key: Mistral API key (or from env)
        use_cache: Use shelve caching
        max_workers: Parallel workers
    
    Returns:
        Reviews with added extraction data
    """
    engine = SentimentEngine(api_key=api_key)
    results = engine.process_all_reviews_parallel(
        reviews, max_workers=max_workers, use_cache=use_cache
    )
    
    # Convert ExtractionResult back to dict and merge with original review
    enriched = []
    for review, result in zip(reviews, results):
        enriched_review = {
            **review,
            "extracted_features": result.features,
            "overall_sentiment": result.overall_sentiment,
            "trust_indicators": result.trust_indicators,
            "human_review_needed": result.human_review_needed,
            "extraction_error": result.error_message,
        }
        enriched.append(enriched_review)
    
    return enriched


if __name__ == "__main__":
    # Test with sample reviews
    test_reviews = [
        {
            "review_id": "SB0001",
            "product_name": "SmartBottle Pro",
            "category": "Personal Care",
            "clean_text": "Cap khula hua tha packing bilkul bakwas leaked completely",
        },
        {
            "review_id": "BC0001",
            "product_name": "BoltCharge 20W",
            "category": "Electronics",
            "clean_text": "Came with cable already fraying near the connector poor quality",
        },
        {
            "review_id": "NM0050",
            "product_name": "NutriMix Blender",
            "category": "Food",
            "clean_text": "Motor started burning after two uses terrible smell",
        },
        {
            "review_id": "SB0002",
            "product_name": "SmartBottle Pro",
            "category": "Personal Care",
            "clean_text": "Excellent build quality keeps water cold for hours love it",
        },
        {
            "review_id": "NM0051",
            "product_name": "NutriMix Blender",
            "category": "Food",
            "clean_text": "Good for making smoothies but motor gets hot quickly",
        },
    ]
    
    print("Sentiment Engine Test\n")
    print("=" * 80)
    
    # Test batch extraction
    engine = SentimentEngine()
    results = engine.extract_batch(test_reviews)
    
    for result in results:
        print(f"\nReview: {result.review_id}")
        print(f"  Overall: {result.overall_sentiment}")
        print(f"  Features: {len(result.features)}")
        for feat in result.features:
            print(f"    - {feat.get('feature')}: {feat.get('sentiment')} (confidence: {feat.get('confidence')})")
        if result.human_review_needed:
            print(f"  ⚠️ Human review needed: {result.error_message}")
