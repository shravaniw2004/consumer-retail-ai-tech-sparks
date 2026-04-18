"""
ReviewIQ Synthetic Data Generation
Generates 300 realistic reviews with 3 seeded trends for demo purposes.
"""

import csv
import random
import os
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# Seeded trend templates (Hinglish + English)
SMARTBOTTLE_TEMPLATES = [
    "Cap khula hua tha",
    "cap came loose",
    "packing bilkul bakwas",
    "leaked completely",
    "seal broken on delivery",
    "bottle cap tight nahi tha",
    "packaging mein dikkat thi",
    "liquid leak ho gaya",
]

NUTRIMIX_TEMPLATES = [
    "motor se burning smell aa rahi hai",
    "blender heating up too much",
    "burning odor from motor",
    "motor garam ho gaya",
    "smell of burning plastic",
    "motor burnt out after 2 uses",
    "overheating issue in motor",
    "burning smell jaise kuch jal raha ho",
]

BOLTCHARGE_TEMPLATES = [
    "cable fraying after 1 week",
    "wire tutne lagi hai",
    "cable coating peel ho gai",
    "charging wire weak quality",
    "cable junction se break ho gaya",
    "wire mein crack aa gaya",
    "cable durability bilkul bekar",
    "frayed wire near connector",
]

# Positive and neutral templates for realistic noise
POSITIVE_TEMPLATES = {
    "SmartBottle Pro": [
        "Keeps water cold for hours, very nice",
        "Build quality achi hai, recommend karta hu",
        "Love the temperature display feature",
        "Stylish design, friends ne pucha kahan se liya",
        "Works perfectly for gym use",
        "Temperature control accurate hai",
        "Value for money product",
        "Stainless steel quality top notch",
    ],
    "NutriMix Blender": [
        "Smoothies banane mein bahut easy",
        "Powerful motor, grinds everything",
        "Best blender in this price range",
        "Cleaning is super easy",
        "Multiple speed options useful hain",
        "Blends ice cubes like butter",
        "Compact design, fits in kitchen",
        "Warranty coverage acha hai",
    ],
    "BoltCharge 20W": [
        "Fast charging speed amazing",
        "20W mein itni speed unexpected",
        "Phone 30 min mein 50% charge",
        "Adapter build solid hai",
        "Best charger for iPhone",
        "No heating issue during charging",
        "Cable length perfect hai",
        "Travel friendly compact size",
    ],
}

NEGATIVE_NOISE_TEMPLATES = {
    "SmartBottle Pro": [
        "Heavy bottle, carry karna mushkil",
        "Price thoda zyada hai",
        "Display battery jaldi drain hoti hai",
        "Lid opens slowly, patience chahiye",
    ],
    "NutriMix Blender": [
        "Noise bahut zyada hai",
        "Jar material could be better",
        "Blades dull ho gaye jaldi",
        "Price ke hisaab se thoda basic",
    ],
    "BoltCharge 20W": [
        "Adapter bulky hai",
        "Only one port available",
        "No LED indicator on charger",
        "Price compared to others thoda high",
    ],
}

# Noise parameters
TYPO_CHANCE = 0.15
EMOJI_CHANCE = 0.20
INCOMPLETE_CHANCE = 0.10
HINGLISH_CHANCE = 0.25


def add_typos(text: str) -> str:
    """Add realistic typos to text."""
    if random.random() > TYPO_CHANCE:
        return text
    
    typo_map = {
        'th': 't',
        'ing': 'ng',
        'tion': 'tio',
        'ea': 'ae',
        'ou': 'u',
        'ph': 'f',
        'ch': 'c',
    }
    
    words = text.split()
    if not words:
        return text
    
    # Apply 1-2 typos
    num_typos = random.randint(1, 2)
    for _ in range(num_typos):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 3:
            for old, new in typo_map.items():
                if old in word.lower():
                    words[idx] = word.replace(old, new, 1)
                    break
    
    return " ".join(words)


def add_emojis(text: str) -> str:
    """Add contextually relevant emojis."""
    if random.random() > EMOJI_CHANCE:
        return text
    
    positive_emojis = ["👍", "😊", "✅", "⭐", "❤️", "🙏"]
    negative_emojis = ["😞", "❌", "😡", "👎", "😤", "🤦"]
    neutral_emojis = ["🤔", "😐", "💭", "📦", "🔌"]
    
    # Determine sentiment roughly
    negative_words = ["bad", "terrible", "worst", "broken", "issue", "problem", "waste", "ghatiya", "bakwas"]
    positive_words = ["good", "great", "best", "amazing", "excellent", "perfect", "nice", "acha"]
    
    lower_text = text.lower()
    neg_count = sum(1 for w in negative_words if w in lower_text)
    pos_count = sum(1 for w in positive_words if w in lower_text)
    
    if neg_count > pos_count:
        emoji_pool = negative_emojis
    elif pos_count > neg_count:
        emoji_pool = positive_emojis
    else:
        emoji_pool = neutral_emojis
    
    # Add 1-2 emojis at end
    num_emojis = random.randint(1, 2)
    emojis = " " + " ".join(random.choices(emoji_pool, k=num_emojis))
    return text + emojis


def make_incomplete(text: str) -> str:
    """Truncate to incomplete sentence."""
    if random.random() > INCOMPLETE_CHANCE:
        return text
    
    words = text.split()
    if len(words) <= 5:
        return text
    
    # Cut off at random point
    cutoff = random.randint(3, len(words) - 2)
    truncated = words[:cutoff]
    
    # Sometimes add trailing "..."
    if random.random() > 0.5:
        truncated.append("...")
    
    return " ".join(truncated)


def add_noise(text: str) -> str:
    """Apply all noise transformations."""
    text = add_typos(text)
    text = add_emojis(text)
    text = make_incomplete(text)
    return text


def inject_seeded_trend(
    reviews: List[Dict],
    product: str,
    templates: List[str],
    windows: List[Tuple[int, int, int]],  # (start, end, count)
    is_seeded: bool = True
) -> List[Dict]:
    """
    Inject seeded complaints at specific positions.
    
    Args:
        reviews: Base review list
        product: Product name to target
        templates: Complaint templates to inject
        windows: List of (start_idx, end_idx, num_complaints)
        is_seeded: Flag for hidden column
    """
    for start, end, count in windows:
        # Find indices of target product in range
        indices = [
            i for i in range(start, end)
            if reviews[i]["product_name"] == product
        ]
        
        # Select random positions
        if len(indices) >= count:
            selected = random.sample(indices, count)
        else:
            selected = indices
        
        for idx in selected:
            template = random.choice(templates)
            noisy = add_noise(template)
            reviews[idx]["review_text"] = noisy
            reviews[idx]["is_seeded"] = is_seeded
            reviews[idx]["sentiment"] = "negative"
    
    return reviews


def generate_base_dataset(num_reviews: int = 300) -> List[Dict]:
    """
    Generate base dataset with realistic noise distribution.
    
    Product distribution:
    - SmartBottle Pro: 150 reviews (0-149)
    - BoltCharge 20W: 50 reviews (100-149) - middle overlap
    - NutriMix Blender: 100 reviews (200-299)
    
    Wait - let me fix this properly:
    - SmartBottle Pro: 150 reviews (indices 0-149)
    - BoltCharge 20W: 50 reviews (indices 100-149) - shares middle window
    - NutriMix Blender: 100 reviews (indices 200-299)
    
    Actually, let's do:
    - SmartBottle Pro: 150 reviews (0-149)
    - BoltCharge 20W: 50 reviews (100-149) - shares with SmartBottle middle window
    - NutriMix Blender: 100 reviews (150-249) - wait that's only 250
    
    Correct distribution for 300 reviews:
    - SmartBottle Pro: 150 reviews (indices 0-149)
    - BoltCharge 20W: 50 reviews (indices 100-149, overlapping middle)
    - NutriMix Blender: 100 reviews (indices 150-249)
    ... that's only 250. Need to adjust.
    
    Final distribution:
    - SmartBottle Pro: 150 reviews (0-149)
    - BoltCharge 20W: 50 reviews (100-149, overlaps with SmartBottle)
    - NutriMix Blender: 100 reviews (200-299)
    - Plus 50 filler reviews (150-199) for other products
    """
    reviews = []
    categories = {
        "SmartBottle Pro": "Personal Care",
        "BoltCharge 20W": "Electronics",
        "NutriMix Blender": "Food",
    }
    
    # Generate reviews with proper distribution
    base_date = datetime(2024, 1, 1)
    
    # SmartBottle Pro: 150 reviews (indices 0-149)
    for i in range(150):
        sentiment = random.choices(
            ["positive", "neutral", "negative"],
            weights=[0.6, 0.25, 0.15]
        )[0]
        
        if sentiment == "positive":
            template = random.choice(POSITIVE_TEMPLATES["SmartBottle Pro"])
        elif sentiment == "negative":
            template = random.choice(NEGATIVE_NOISE_TEMPLATES["SmartBottle Pro"])
        else:
            template = random.choice(POSITIVE_TEMPLATES["SmartBottle Pro"] + NEGATIVE_NOISE_TEMPLATES["SmartBottle Pro"])
        
        reviews.append({
            "review_id": f"SB{i:04d}",
            "product_name": "SmartBottle Pro",
            "category": "Personal Care",
            "review_text": add_noise(template),
            "sentiment": sentiment,
            "is_seeded": False,
            "timestamp": (base_date + timedelta(days=i)).isoformat(),
        })
    
    # BoltCharge 20W: 50 reviews (indices 100-149, but we'll add them separately)
    # Actually these are overlapping windows, so we need separate handling
    # Let me restructure: generate all 300 distinct reviews
    
    # Let me restart with proper 300 distinct reviews:
    reviews = []
    
    # 0-149: SmartBottle Pro (150 reviews)
    for i in range(150):
        sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.6, 0.25, 0.15])[0]
        if sentiment == "positive":
            template = random.choice(POSITIVE_TEMPLATES["SmartBottle Pro"])
        elif sentiment == "negative":
            template = random.choice(NEGATIVE_NOISE_TEMPLATES["SmartBottle Pro"])
        else:
            template = random.choice(POSITIVE_TEMPLATES["SmartBottle Pro"])
        
        reviews.append({
            "review_id": f"SB{i:04d}",
            "product_name": "SmartBottle Pro",
            "category": "Personal Care",
            "review_text": add_noise(template),
            "sentiment": sentiment,
            "is_seeded": False,
            "timestamp": (base_date + timedelta(days=i)).isoformat(),
        })
    
    # 150-199: BoltCharge 20W (50 reviews) - middle window for cable fraying
    for i in range(50):
        idx = 150 + i
        sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.6, 0.25, 0.15])[0]
        if sentiment == "positive":
            template = random.choice(POSITIVE_TEMPLATES["BoltCharge 20W"])
        elif sentiment == "negative":
            template = random.choice(NEGATIVE_NOISE_TEMPLATES["BoltCharge 20W"])
        else:
            template = random.choice(POSITIVE_TEMPLATES["BoltCharge 20W"])
        
        reviews.append({
            "review_id": f"BC{i:04d}",
            "product_name": "BoltCharge 20W",
            "category": "Electronics",
            "review_text": add_noise(template),
            "sentiment": sentiment,
            "is_seeded": False,
            "timestamp": (base_date + timedelta(days=idx)).isoformat(),
        })
    
    # 200-299: NutriMix Blender (100 reviews) - last 50 will have burning smell
    for i in range(100):
        idx = 200 + i
        sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.6, 0.25, 0.15])[0]
        if sentiment == "positive":
            template = random.choice(POSITIVE_TEMPLATES["NutriMix Blender"])
        elif sentiment == "negative":
            template = random.choice(NEGATIVE_NOISE_TEMPLATES["NutriMix Blender"])
        else:
            template = random.choice(POSITIVE_TEMPLATES["NutriMix Blender"])
        
        reviews.append({
            "review_id": f"NM{i:04d}",
            "product_name": "NutriMix Blender",
            "category": "Food",
            "review_text": add_noise(template),
            "sentiment": sentiment,
            "is_seeded": False,
            "timestamp": (base_date + timedelta(days=idx)).isoformat(),
        })
    
    return reviews


def apply_seeded_trends(reviews: List[Dict]) -> List[Dict]:
    """Apply all three seeded trends to the dataset."""
    
    # Trend 1: SmartBottle Pro - cap leakage (3→7→19 across 0-49, 50-99, 100-149)
    reviews = inject_seeded_trend(
        reviews,
        "SmartBottle Pro",
        SMARTBOTTLE_TEMPLATES,
        windows=[(0, 50, 3), (50, 100, 7), (100, 150, 19)],
    )
    
    # Trend 2: BoltCharge 20W - cable fraying in middle window (150-199 is entire range)
    # For 50 reviews, middle window could be 165-184 (20 reviews) with cable fraying
    reviews = inject_seeded_trend(
        reviews,
        "BoltCharge 20W",
        BOLTCHARGE_TEMPLATES,
        windows=[(155, 195, 12)],  # 12 cable fraying complaints in middle
    )
    
    # Trend 3: NutriMix Blender - motor burning smell in last 50 reviews (250-299)
    reviews = inject_seeded_trend(
        reviews,
        "NutriMix Blender",
        NUTRIMIX_TEMPLATES,
        windows=[(250, 300, 15)],  # 15 burning smell complaints in last 50
    )
    
    return reviews


def save_to_csv(reviews: List[Dict], output_dir: str = None) -> Tuple[str, str, str]:
    """
    Save reviews to 3 CSV files by product category.
    
    Returns:
        Tuple of (smartbottle_path, boltcharge_path, nutrimix_path)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by product
    products = {
        "SmartBottle Pro": [],
        "BoltCharge 20W": [],
        "NutriMix Blender": [],
    }
    
    for review in reviews:
        product = review["product_name"]
        if product in products:
            products[product].append(review)
    
    # Save each to CSV
    paths = {}
    for product, product_reviews in products.items():
        # Create safe filename
        safe_name = product.lower().replace(" ", "_")
        filepath = os.path.join(output_dir, f"{safe_name}_reviews.csv")
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if product_reviews:
                fieldnames = list(product_reviews[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(product_reviews)
        
        paths[product] = filepath
        print(f"Saved {len(product_reviews)} reviews to {filepath}")
    
    # Also save combined dataset
    combined_path = os.path.join(output_dir, "all_reviews.csv")
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(reviews[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)
    
    print(f"Saved {len(reviews)} total reviews to {combined_path}")
    
    return paths["SmartBottle Pro"], paths["BoltCharge 20W"], paths["NutriMix Blender"]


def generate_demo_dataset(seed: int = 42) -> List[Dict]:
    """
    Generate complete demo dataset with all seeded trends.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        List of 300 review dictionaries
    """
    random.seed(seed)
    
    # Step 1: Generate base dataset
    reviews = generate_base_dataset()
    
    # Step 2: Inject seeded trends
    reviews = apply_seeded_trends(reviews)
    
    # Shuffle to mix seeded reviews with regular ones
    random.shuffle(reviews)
    
    return reviews


if __name__ == "__main__":
    # Generate dataset
    reviews = generate_demo_dataset(seed=42)
    
    # Print summary
    seeded_count = sum(1 for r in reviews if r["is_seeded"])
    print(f"\nDataset Summary:")
    print(f"Total reviews: {len(reviews)}")
    print(f"Seeded reviews: {seeded_count}")
    print(f"SmartBottle Pro: {sum(1 for r in reviews if r['product_name'] == 'SmartBottle Pro')}")
    print(f"BoltCharge 20W: {sum(1 for r in reviews if r['product_name'] == 'BoltCharge 20W')}")
    print(f"NutriMix Blender: {sum(1 for r in reviews if r['product_name'] == 'NutriMix Blender')}")
    
    # Count by trend
    cap_leakage = sum(1 for r in reviews if r["is_seeded"] and "cap" in r["review_text"].lower())
    cable_fray = sum(1 for r in reviews if r["is_seeded"] and "cable" in r["review_text"].lower())
    burning = sum(1 for r in reviews if r["is_seeded"] and "burn" in r["review_text"].lower())
    
    print(f"\nSeeded Trends:")
    print(f"  Cap leakage (SmartBottle): {cap_leakage}")
    print(f"  Cable fraying (BoltCharge): {cable_fray}")
    print(f"  Burning smell (NutriMix): {burning}")
    
    # Save to CSV
    print("\nSaving to CSV files...")
    paths = save_to_csv(reviews)
    print(f"\nFiles created successfully!")
