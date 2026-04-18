"""
ReviewIQ v5 — Module 10: Crisis Simulator Mode
A "God Mode" button that lets judges trigger synthetic crises during demo.
Pre-generates 3 crisis batch types and appends them to current dataset.
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────
# Crisis Types
# ──────────────────────────────────────────────────────────────

CRISIS_TYPES = {
    "packaging_crisis": {
        "emoji": "☢️",
        "label": "Packaging Crisis",
        "description": "50 packaging complaints in 1 hour",
        "count": 50,
        "target_feature": "packaging",
        "sentiment": "negative",
        "intensity": "high",
        "color": "#F44336",
    },
    "quality_disaster": {
        "emoji": "💀",
        "label": "Quality Disaster",
        "description": "30 furious-intensity quality defect reviews",
        "count": 30,
        "target_feature": "quality",
        "sentiment": "negative",
        "intensity": "extreme",
        "color": "#B71C1C",
    },
    "bot_attack": {
        "emoji": "🤖",
        "label": "Bot Attack",
        "description": "25 coordinated suspicious 1-star reviews",
        "count": 25,
        "target_feature": "overall",
        "sentiment": "negative",
        "intensity": "high",
        "color": "#9C27B0",
    },
}


# ──────────────────────────────────────────────────────────────
# Pre-generated Crisis Templates
# ──────────────────────────────────────────────────────────────

PACKAGING_CRISIS_TEMPLATES = [
    "Cap completely broken, liquid leaked everywhere in my bag 😡",
    "THIS IS THE THIRD TIME the cap was loose! How is this still a problem??",
    "Packaging is absolutely terrible. Product arrived destroyed.",
    "Cap khula hua tha — poora bag barbad ho gaya, refund do mujhe 😡😡😡",
    "Seal was broken when I opened the box. Product is contaminated.",
    "Worst packaging I've ever seen. Everything was wet and damaged.",
    "packing bilkul bakwas hai, kab sudhrega ye company? 🤬",
    "Leak all over my office bag. Do you even test your packaging?",
    "Cap doesn't close properly. Design flaw. RECALL THIS PRODUCT.",
    "Product was literally floating in liquid inside the box. Disgusting.",
    "Third replacement, SAME cap issue. Your QC is nonexistent.",
    "Packaging leaked in transit, stained all my clothes. Demanding compensation!",
    "Box damaged, cap loose, product half empty. Total disaster.",
    "Seal completely broken. This is a safety issue. CONSUMER COURT NOTICE.",
    "Cap leakage ruined my expensive laptop bag. Send me a new bag + product.",
    "Not even worth 1 star. Packaging is a JOKE. 💀",
    "Opened the delivery — liquid everywhere. Amazon should ban this seller.",
    "Cap was not even screwed on properly. Quality control ZERO.",
    "Had to throw away immediately. Cap was off, product contaminated.",
    "WILL POST ON SOCIAL MEDIA unless packaging is fixed. This is pathetic.",
    "delivery boy bhi bol raha tha ki ye product hamesha leak hota hai 😤",
    "The box was soggy when it arrived. Obviously leaked badly.",
    "I ordered 3 bottles, ALL 3 had cap issues. Not a coincidence!",
    "Packaging design needs complete overhaul. Current cap is defective.",
    "Video proof uploaded — cap comes off with gentle shake. DEFECTIVE.",
    "Even kids' bottles have better caps. This is premium? Jokes.",
    "FFS fix your packaging already! 4th order, 4th leak! 🤬🤬🤬",
    "Returned immediately. Packaging damage. No point using this product.",
    "My bag smells terrible now because of this leaky bottle. ANGRY!",
    "packaging team ko nikalo aur naye logo ko rakho, bilkul bakwas 😡",
    "Seal tampered or never sealed properly. Questioning authenticity now.",
    "Bought as gift — arrived leaked and looking used. SO EMBARRASSING.",
    "Product quality might be good but packaging kills it. STOP SELLING.",
    "Transit damage due to poor packaging. Zero protection inside box.",
    "Cap broke on first twist. Cheap plastic. Not even worth returning.",
    "leakproof likhte ho advertisement mein, totally misleading! 🤮",
    "Packaging department clearly doesn't care. Zero quality checks.",
    "Bottle exploded in my gym bag during summer heat. Cap failure.",
    "Third complaint, still no fix. Escalating to consumer forum.",
    "PACKAGING IS THE WORST. Who approved this design? Fire them.",
    "Liquid leaking even while standing upright. Cap is fundamentally broken.",
    "Why do I have to become QC inspector for every delivery?? FIX IT!",
    "Health hazard — open seal means product could be tampered with.",
    "Packing mein zero improvement pichhle 6 mahine se. Fed up!",
    "I'm a loyal customer but this packaging issue is making me switch.",
    "New batch, same old cap problem. Nothing changes. Disappointed.",
    "Delivery partner even asked if I want to refuse it. That bad.",
    "Photos uploaded in review — see the mess. This is unacceptable.",
    "Tried tightening cap myself — still leaks. Design problem, not user error.",
    "NEVER ORDERING AGAIN until packaging is completely redesigned. Done.",
]

QUALITY_DISASTER_TEMPLATES = [
    "WORST PRODUCT I HAVE EVER PURCHASED IN MY ENTIRE LIFE 🤬🤬🤬",
    "Product BROKE after ONE use. TOTAL SCAM! DEMANDING FULL REFUND!",
    "Material quality is GARBAGE. Looks nothing like the photos. FRAUD!",
    "This product literally FELL APART in my hands. ZERO build quality.",
    "sabse ghatiya product hai ye — bilkul kachra quality, paisa barbad 🤬",
    "COMPLETE AND UTTER WASTE OF MONEY. I want to SUE this company.",
    "Defective right out of the box. Did anyone QC this?? PATHETIC.",
    "Product caused SKIN REACTION. This is a HEALTH HAZARD. LEGAL ACTION.",
    "Motor burned out after 2 uses. Smelled like burning plastic. DANGEROUS!",
    "Colors faded after first wash. False advertising. Filing complaint!",
    "Build quality of a 50 rupee Chinese copy. This cost ₹2000?? SCAM!",
    "Product MALFUNCTIONED and nearly caused a fire. EXTREMELY DANGEROUS.",
    "ye kya bech rahe ho log ko? Quality check karte ho ya seedha ship?",
    "I am SO ANGRY right now. Product is DEFECTIVE. UNUSABLE after day 1.",
    "Gave as gift — receiver laughed at quality. Most embarrassing moment.",
    "BOYCOTT THIS BRAND. Quality has gone to absolute ZERO. Share widely!",
    "Tested 5 units from same batch — ALL had the same defect. Systemic!",
    "FRAUD COMPANY selling defective products at premium prices. SCAMMERS!",
    "Customer support said 'known issue' — then WHY ARE YOU STILL SELLING?",
    "Going to every review site to warn people. This product is DANGEROUS.",
    "Quality dropped DRASTICALLY from last year. What changed?? Cost cutting?",
    "Returned 3 times, each replacement equally bad. Quality control DEAD.",
    "WILL FILE CONSUMER COURT case if refund not processed immediately!",
    "product mein dum hi nahi hai — 2 din mein khatam, total bakwas 💀",
    "Do NOT buy this. Save your money. Quality is NON-EXISTENT.",
    "Allergic reaction from this product. WHERE IS YOUR QUALITY TESTING??",
    "Bought 'premium' version — quality even WORSE than regular. How??",
    "This should be RECALLED. Serious safety concern. Not acceptable.",
    "Quality audit needed YESTERDAY. Every unit I received was defective.",
    "NEVER AGAIN. Lost trust completely. Switching to competitor TODAY.",
]

BOT_ATTACK_TEMPLATES = [
    "terrible product do not buy waste of money worst experience ever",
    "worst product ever bought completely useless total waste of money",
    "do not buy this product it is total waste of money worst quality",
    "this product is worst i ever bought completely useless waste money",
    "very bad product quality worst experience money wasted do not buy",
    "totally useless product wasted my money worst purchase ever made",
    "worst quality product money wasted completely useless do not buy it",
    "do not recommend this product worst experience total waste of money",
    "terrible terrible product worst thing i ever bought total waste",
    "completely waste of money worst product do not buy this at all",
    "product is totally useless worst experience i had money wasted",
    "money wasted worst product do not buy terrible quality everything bad",
    "worst worst worst product ever do not waste your money on this",
    "total waste of money do not buy worst product in entire market",
    "this is the worst product terrible quality waste of money do not buy",
    "do not buy product is worst complete waste customer service worst too",
    "pathetic product worst buy ever money completely wasted terrible",
    "absolute garbage product worst purchase terrible quality waste money",
    "worst product bought this year total waste avoid at all costs",
    "terrible quality worst experience avoid this product save your money",
    "do not buy worst product waste of hard earned money terrible quality",
    "this product worst purchase ever do not waste money terrible quality",
    "complete garbage worst product terrible customer service waste money",
    "worst experience avoid this product waste of money terrible quality",
    "totally terrible product worst purchase worst quality waste of money",
]


@dataclass
class CrisisResult:
    """Result of a crisis injection."""
    crisis_type: str
    crisis_label: str
    crisis_emoji: str
    reviews_injected: int
    injection_time: str
    simulated_metrics: Dict[str, float]
    injected_reviews: List[Dict]


def generate_crisis_reviews(
    crisis_type: str,
    product_name: str = "SmartBottle Pro",
    category: str = "Personal Care",
    base_timestamp: Optional[datetime] = None,
) -> List[Dict]:
    """
    Generate a batch of synthetic crisis reviews.
    
    Args:
        crisis_type: One of 'packaging_crisis', 'quality_disaster', 'bot_attack'
        product_name: Target product
        category: Product category
        base_timestamp: When the crisis "starts"
    """
    if base_timestamp is None:
        base_timestamp = datetime.now()
    
    config = CRISIS_TYPES.get(crisis_type)
    if not config:
        raise ValueError(f"Unknown crisis type: {crisis_type}")
    
    # Select templates
    if crisis_type == "packaging_crisis":
        templates = PACKAGING_CRISIS_TEMPLATES
    elif crisis_type == "quality_disaster":
        templates = QUALITY_DISASTER_TEMPLATES
    elif crisis_type == "bot_attack":
        templates = BOT_ATTACK_TEMPLATES
    else:
        templates = PACKAGING_CRISIS_TEMPLATES
    
    count = config["count"]
    reviews = []
    
    for i in range(count):
        # Pick template with slight variation
        template = templates[i % len(templates)]
        
        # Add some random noise to bot attack templates for realism
        if crisis_type == "bot_attack":
            username = f"review_bot_{1000 + i}"
            rating = 1
            # Slight text variation
            words = template.split()
            if len(words) > 5:
                # Swap 2 random adjacent words
                swap_idx = random.randint(0, len(words) - 2)
                words[swap_idx], words[swap_idx + 1] = words[swap_idx + 1], words[swap_idx]
            review_text = " ".join(words)
        else:
            username = f"crisis_reviewer_{random.randint(1000, 9999)}"
            rating = random.choice([1, 1, 1, 2])
            review_text = template
        
        # Timestamps within a tight window
        time_offset = timedelta(minutes=random.randint(0, 60))
        timestamp = base_timestamp + time_offset
        
        reviews.append({
            "review_id": f"CRISIS_{crisis_type.upper()}_{i:03d}",
            "product_name": product_name,
            "category": category,
            "review_text": review_text,
            "clean_text": review_text.lower(),
            "original_text": review_text,
            "rating": rating,
            "timestamp": timestamp.isoformat(),
            "username": username,
            "is_seeded": True,
            "is_crisis": True,
            "crisis_type": crisis_type,
            "sentiment": "negative",
            "language": "english",
            "trust_score": 20 if crisis_type == "bot_attack" else 80,
            "overall_sentiment": "negative",
        })
    
    return reviews


def inject_crisis(
    existing_reviews: List[Dict],
    crisis_type: str,
    product_name: str = "SmartBottle Pro",
    category: str = "Personal Care",
) -> CrisisResult:
    """
    Inject a crisis into the existing review dataset.
    Returns the crisis result with injected reviews and simulated metrics.
    """
    config = CRISIS_TYPES[crisis_type]
    crisis_reviews = generate_crisis_reviews(crisis_type, product_name, category)
    
    # Calculate simulated post-crisis metrics
    total_after = len(existing_reviews) + len(crisis_reviews)
    
    # Count existing negatives
    existing_negatives = sum(
        1 for r in existing_reviews 
        if r.get("sentiment") == "negative" or r.get("overall_sentiment") == "negative"
    )
    
    new_negatives = existing_negatives + len(crisis_reviews)
    new_complaint_rate = (new_negatives / total_after * 100) if total_after > 0 else 0
    
    # Simulate Z-score spike
    baseline_rate = existing_negatives / max(len(existing_reviews), 1) * 100
    if baseline_rate > 0:
        z_score = (new_complaint_rate - baseline_rate) / max(baseline_rate * 0.3, 1.0)
    else:
        z_score = 4.0
    
    simulated_metrics = {
        "complaint_rate_before": round(baseline_rate, 1),
        "complaint_rate_after": round(new_complaint_rate, 1),
        "z_score": round(z_score, 1),
        "total_reviews": total_after,
        "reviews_injected": len(crisis_reviews),
        "estimated_rating_impact": round(-(new_complaint_rate - baseline_rate) * 0.08, 2),
        "revenue_at_risk": round(500000 * 0.12 * abs((new_complaint_rate - baseline_rate) * 0.08)),
    }
    
    return CrisisResult(
        crisis_type=crisis_type,
        crisis_label=config["label"],
        crisis_emoji=config["emoji"],
        reviews_injected=len(crisis_reviews),
        injection_time=datetime.now().isoformat(),
        simulated_metrics=simulated_metrics,
        injected_reviews=crisis_reviews,
    )


def get_crisis_options() -> List[Dict]:
    """Get available crisis options for dashboard buttons."""
    return [
        {
            "key": key,
            "emoji": config["emoji"],
            "label": config["label"],
            "description": config["description"],
            "count": config["count"],
            "color": config["color"],
        }
        for key, config in CRISIS_TYPES.items()
    ]


if __name__ == "__main__":
    print("Crisis Simulator Test")
    print("=" * 70)
    
    # Generate some existing reviews
    existing = [
        {"review_id": f"EX{i}", "product_name": "SmartBottle Pro", "sentiment": "positive"}
        for i in range(100)
    ] + [
        {"review_id": f"EX{100+i}", "product_name": "SmartBottle Pro", "sentiment": "negative"}
        for i in range(15)
    ]
    
    for crisis_key in CRISIS_TYPES:
        print(f"\n--- {CRISIS_TYPES[crisis_key]['label']} (Injected) ---")
        
        result = inject_crisis(existing, crisis_key)
        
        print(f"  Reviews injected: {result.reviews_injected}")
        print(f"  Complaint rate: {result.simulated_metrics['complaint_rate_before']}% -> "
              f"{result.simulated_metrics['complaint_rate_after']}%")
        print(f"  Z-Score: {result.simulated_metrics['z_score']}")
        print(f"  Rating impact: {result.simulated_metrics['estimated_rating_impact']}★")
        print(f"  Revenue at risk: ₹{result.simulated_metrics['revenue_at_risk']:,}")
        print(f"  Sample review: {result.injected_reviews[0]['review_text'][:80]}...")
    
    # Test crisis options
    print(f"\nAvailable crisis buttons:")
    for option in get_crisis_options():
        print(f"  {option['emoji']} {option['label']}: {option['description']}")
