"""
ReviewIQ Deduplication Engine - TF-IDF + Cosine Similarity for Near-Duplicate Detection
"""

import re
import uuid
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Similarity threshold for duplicates
DUPLICATE_THRESHOLD = 0.85

# Minimum review length to consider for deduplication
MIN_REVIEW_LENGTH = 10


def compute_similarity_matrix(reviews: List[str]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix using TF-IDF.
    
    Args:
        reviews: List of preprocessed review texts
    
    Returns:
        NxN similarity matrix where similarity[i][j] = cosine similarity
    """
    if not reviews or len(reviews) < 2:
        return np.array([])
    
    # Filter out empty/too-short reviews for vectorization
    valid_indices = [
        i for i, text in enumerate(reviews)
        if text and len(text.strip()) >= MIN_REVIEW_LENGTH
    ]
    
    if len(valid_indices) < 2:
        return np.zeros((len(reviews), len(reviews)))
    
    valid_texts = [reviews[i] for i in valid_indices]
    
    # TF-IDF vectorization with 1-2 grams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,  # Ignore terms that appear in >95% of docs
        stop_words='english',
        lowercase=True,
        norm='l2'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        # Compute cosine similarity (optimized for sparse matrices)
        valid_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except ValueError:
        # If vectorization fails (e.g., all empty), return zeros
        valid_sim_matrix = np.zeros((len(valid_texts), len(valid_texts)))
    
    # Expand back to full matrix size with invalid indices as -1
    full_matrix = np.full((len(reviews), len(reviews)), -1.0)
    for i, orig_i in enumerate(valid_indices):
        for j, orig_j in enumerate(valid_indices):
            full_matrix[orig_i][orig_j] = valid_sim_matrix[i][j]
    
    return full_matrix


def find_duplicates(
    reviews: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = DUPLICATE_THRESHOLD
) -> Dict[int, List[int]]:
    """
    Find clusters of near-duplicate reviews based on similarity threshold.
    
    Uses connected components approach - if A~B and B~C, they form a cluster.
    
    Args:
        reviews: List of review texts
        similarity_matrix: NxN similarity matrix
        threshold: Minimum similarity to be considered duplicate (default 0.85)
    
    Returns:
        Dictionary mapping group_id -> list of review indices in that group
    """
    n = len(reviews)
    if n < 2 or similarity_matrix.size == 0:
        return {}
    
    # Build adjacency list (undirected graph)
    adjacency = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
    
    # Find connected components (clusters)
    visited = set()
    clusters = {}
    group_id = 0
    
    for start_node in range(n):
        if start_node in visited:
            continue
        
        # BFS to find all connected nodes
        component = []
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(component) > 1:  # Only save clusters with duplicates
            clusters[group_id] = component
            group_id += 1
    
    return clusters


def flag_suspicious_patterns(reviews: List[Dict]) -> Dict[int, List[str]]:
    """
    Detect suspicious patterns in reviews beyond similarity.
    
    Patterns detected:
    1. Exact duplicates (identical text after normalization)
    2. Repetitive phrases (>3x same sentence within a review)
    3. Template-like reviews (same structure, only product name changes)
    
    Args:
        reviews: List of review dictionaries with 'clean_text' field
    
    Returns:
        Dictionary mapping review index -> list of suspicious pattern flags
    """
    suspicious_flags = defaultdict(list)
    
    # Pattern 1: Exact duplicates
    text_to_indices = defaultdict(list)
    for idx, review in enumerate(reviews):
        clean_text = review.get('clean_text', '').strip().lower()
        # Normalize whitespace for exact match
        normalized = ' '.join(clean_text.split())
        text_to_indices[normalized].append(idx)
    
    for text, indices in text_to_indices.items():
        if len(indices) > 1:
            for idx in indices:
                suspicious_flags[idx].append(f'exact_duplicate:{len(indices)}_copies')
    
    # Pattern 2: Repetitive phrases within single review
    for idx, review in enumerate(reviews):
        clean_text = review.get('clean_text', '')
        sentences = re.split(r'[.!?]+', clean_text.lower())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if sentences:
            sentence_counts = Counter(sentences)
            for sentence, count in sentence_counts.items():
                if count >= 3:
                    suspicious_flags[idx].append(f'repetitive_phrase:{count}x')
                    break  # Only flag once per review
    
    # Pattern 3: Template-like reviews (same structure, different product names)
    # Detect reviews that match pattern: [Product] [rest is identical]
    template_groups = defaultdict(list)
    
    for idx, review in enumerate(reviews):
        clean_text = review.get('clean_text', '').lower()
        product_name = review.get('product_name', '').lower()
        
        # Create template by replacing product name with placeholder
        if product_name and product_name in clean_text:
            template = clean_text.replace(product_name, '[PRODUCT]')
            template_groups[template].append(idx)
    
    # Flag templates that appear multiple times
    for template, indices in template_groups.items():
        if len(indices) >= 3:  # At least 3 reviews with same template
            # Check if it's really a template (not just common phrase)
            words = template.split()
            if len(words) >= 5:  # Ensure meaningful template length
                for idx in indices:
                    suspicious_flags[idx].append(f'template_review:{len(indices)}_matches')
    
    return dict(suspicious_flags)


def assign_trust_scores(
    reviews: List[Dict],
    duplicate_clusters: Dict[int, List[int]],
    suspicious_patterns: Dict[int, List[str]],
    similarity_matrix: np.ndarray
) -> Dict[int, int]:
    """
    Assign trust scores (0-100) based on deduplication and pattern analysis.
    
    Scoring:
    - Base score: 100 (trusted)
    - Near-duplicate (>0.85 similarity): -20 per duplicate group
    - Exact duplicate: -30 per occurrence
    - Repetitive phrases: -25
    - Template review: -35
    - Suspicious patterns: -15 each additional flag
    - Isolated unique review (no duplicates): +10 (max 100)
    
    Thresholds:
    - 100: Fully trusted, unique
    - 70-99: Trusted with minor issues
    - 40-69: Questionable, needs review
    - <40: Suspicious/potential bot
    - <30: Highly suspicious, likely bot/fake
    
    Args:
        reviews: List of review dictionaries
        duplicate_clusters: Group_id -> indices mapping
        suspicious_patterns: Index -> flags mapping
        similarity_matrix: Similarity matrix for fine-grained scoring
    
    Returns:
        Dictionary mapping review index -> trust score (0-100)
    """
    n = len(reviews)
    scores = {i: 100 for i in range(n)}  # Start everyone at 100
    
    # Track which reviews are in duplicate groups
    duplicate_group_membership = {}
    for group_id, indices in duplicate_clusters.items():
        for idx in indices:
            duplicate_group_membership[idx] = group_id
    
    # Penalty: Near-duplicates
    for group_id, indices in duplicate_clusters.items():
        group_size = len(indices)
        penalty = min(20 + (group_size - 2) * 5, 50)  # -20 base, +5 per extra, max -50
        for idx in indices:
            scores[idx] -= penalty
    
    # Penalty: Exact duplicates (additional to near-duplicate)
    for idx, flags in suspicious_patterns.items():
        for flag in flags:
            if flag.startswith('exact_duplicate'):
                scores[idx] -= 30
            elif flag.startswith('repetitive_phrase'):
                scores[idx] -= 25
            elif flag.startswith('template_review'):
                scores[idx] -= 35
            else:
                scores[idx] -= 15  # Generic suspicious flag
    
    # Bonus: Isolated unique reviews (highly trusted)
    for i in range(n):
        if i not in duplicate_group_membership and i not in suspicious_patterns:
            scores[i] = min(100, scores[i] + 10)  # Bonus for uniqueness
    
    # Clamp to 0-100 range
    for idx in scores:
        scores[idx] = max(0, min(100, scores[idx]))
    
    return scores


def deduplicate_reviews(reviews: List[Dict]) -> List[Dict]:
    """
    Main deduplication pipeline.
    
    Input: List of review dicts with 'clean_text' and 'product_name' fields
    Output: Same list with added fields:
        - is_duplicate: bool
        - duplicate_group_id: str (UUID) or None
        - trust_score: int (0-100)
        - is_suspicious: bool
    
    Args:
        reviews: List of review dictionaries
    
    Returns:
        Enhanced review list with deduplication metadata
    """
    if not reviews:
        return []
    
    # Extract texts for similarity computation
    texts = [r.get('clean_text', '') for r in reviews]
    
    # Step 1: Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(texts)
    
    # Step 2: Find duplicate clusters
    duplicate_clusters = find_duplicates(texts, similarity_matrix)
    
    # Step 3: Detect suspicious patterns
    suspicious_patterns = flag_suspicious_patterns(reviews)
    
    # Step 4: Assign trust scores
    trust_scores = assign_trust_scores(
        reviews, duplicate_clusters, suspicious_patterns, similarity_matrix
    )
    
    # Create group ID mapping
    group_id_map = {}
    for group_num, indices in duplicate_clusters.items():
        group_uuid = str(uuid.uuid4())[:8]  # Short UUID for readability
        for idx in indices:
            group_id_map[idx] = group_uuid
    
    # Step 5: Enrich reviews with deduplication data
    for idx, review in enumerate(reviews):
        # Determine if duplicate
        is_dup = idx in group_id_map
        
        # Determine if suspicious
        is_susp = trust_scores[idx] < 30 or idx in suspicious_patterns
        
        # Add fields
        review['is_duplicate'] = is_dup
        review['duplicate_group_id'] = group_id_map.get(idx)
        review['trust_score'] = trust_scores[idx]
        review['is_suspicious'] = is_susp
        
        # Add deduplication flags if suspicious patterns found
        if idx in suspicious_patterns:
            review['dedup_flags'] = suspicious_patterns[idx]
        else:
            review['dedup_flags'] = []
    
    return reviews


# Convenience function for statistics
def get_dedup_stats(reviews: List[Dict]) -> Dict:
    """Get statistics about deduplication results."""
    if not reviews:
        return {}
    
    total = len(reviews)
    duplicates = sum(1 for r in reviews if r.get('is_duplicate', False))
    suspicious = sum(1 for r in reviews if r.get('is_suspicious', False))
    
    # Score distribution
    score_ranges = {
        'highly_trusted': sum(1 for r in reviews if r.get('trust_score', 0) >= 90),
        'trusted': sum(1 for r in reviews if 70 <= r.get('trust_score', 0) < 90),
        'questionable': sum(1 for r in reviews if 40 <= r.get('trust_score', 0) < 70),
        'suspicious': sum(1 for r in reviews if r.get('trust_score', 0) < 40),
    }
    
    # Group sizes
    groups = Counter(r.get('duplicate_group_id') for r in reviews if r.get('duplicate_group_id'))
    
    return {
        'total_reviews': total,
        'duplicates_found': duplicates,
        'suspicious_reviews': suspicious,
        'unique_reviews': total - duplicates,
        'duplicate_groups': len(groups),
        'largest_group_size': max(groups.values()) if groups else 0,
        'trust_distribution': score_ranges,
    }


if __name__ == "__main__":
    # Test with sample reviews
    test_reviews = [
        {'clean_text': 'Amazing product, works great for my daily needs', 'product_name': 'SmartBottle'},
        {'clean_text': 'Amazing product, works great for my daily needs', 'product_name': 'BoltCharge'},  # Exact dup
        {'clean_text': 'Amazing product works great for my daily requirements', 'product_name': 'NutriMix'},  # Near dup
        {'clean_text': 'Terrible quality broke in one day', 'product_name': 'SmartBottle'},
        {'clean_text': 'Terrible quality broke in one day', 'product_name': 'BoltCharge'},  # Template
        {'clean_text': 'Terrible quality broke in one day', 'product_name': 'NutriMix'},  # Template
        {'clean_text': 'Terrible quality broke in one day', 'product_name': 'Generic'},  # Template
        {'clean_text': 'Good good good very good product', 'product_name': 'SmartBottle'},  # Repetitive
        {'clean_text': 'Unique review with no duplicates here', 'product_name': 'SmartBottle'},
    ]
    
    print("Deduplication Test\n")
    print("=" * 80)
    
    result = deduplicate_reviews(test_reviews)
    stats = get_dedup_stats(result)
    
    print(f"\nStats: {stats}\n")
    
    for i, r in enumerate(result):
        print(f"Review {i}: {r['clean_text'][:50]}...")
        print(f"  Duplicate: {r['is_duplicate']}, Group: {r['duplicate_group_id']}")
        print(f"  Trust Score: {r['trust_score']}, Suspicious: {r['is_suspicious']}")
        if r['dedup_flags']:
            print(f"  Flags: {r['dedup_flags']}")
        print()
