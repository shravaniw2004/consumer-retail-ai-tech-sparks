"""
ReviewIQ v5 — Module: Coordinated Bot Campaign Detector (Rebuilt)
Detects organized reviewer campaigns using hashing deduplications,
deep NLP suspicion mapping, and NetworkX relational graphs.
"""
import os
import re
import math
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

@dataclass
class CampaignCluster:
    campaign_type: str
    reviewers: List[str]
    review_ids: List[str]
    campaign_confidence: float
    probability_label: str
    time_window: Optional[str]
    avg_rating: float
    summary: str

@dataclass
class SuspiciousIndividual:
    review_id: str
    username: str
    text_snippet: str
    score: int
    signals_triggered: List[str]
    status: str
    is_duplicate: bool
    duplicate_type: str

@dataclass
class BotDetectionResult:
    total_reviews_scanned: int
    confirmed_bots_removed: int
    suspicious_flagged: int
    campaigns_detected: int
    clean_dataset_count: int
    
    exact_duplicates_removed: int
    near_duplicates_clustered: int
    near_duplicate_cluster_details: List[Dict]
    
    individuals: List[SuspiciousIndividual]
    clusters: List[CampaignCluster]
    network_data: Optional[Dict] = None

# Part 1: Deduplication
def run_deduplication(reviews: List[Dict]) -> Tuple[List[Dict], int, int, List[Dict]]:
    exact_duplicates = 0
    near_duplicates = 0
    near_cluster_details = []
    
    # 1. Exact Duplicates
    seen_hashes = {}
    for r in reviews:
        r['is_duplicate'] = False
        r['duplicate_type'] = ""
        text = r.get("clean_text", r.get("review_text", ""))
        clean_norm = str(text).lower().replace(" ", "").strip()
        r_hash = hashlib.md5(clean_norm.encode()).hexdigest()
        
        if r_hash in seen_hashes:
            r['is_duplicate'] = True
            r['duplicate_type'] = "EXACT_DUPLICATE"
            exact_duplicates += 1
        else:
            seen_hashes[r_hash] = 1

    # Filter for near-duplicate scoring
    clean_reviews = [r for r in reviews if not r.get("is_duplicate")]
    texts = [r.get("clean_text", r.get("review_text", "")) for r in clean_reviews]
    
    if len(texts) > 1:
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([t if t else "empty" for t in texts])
            sim_matrix = cosine_similarity(tfidf_matrix)
            
            G = nx.Graph()
            for i in range(len(texts)):
                G.add_node(i)
                
            for i in range(len(valid_texts:=texts)):
                for j in range(i+1, len(valid_texts)):
                    if sim_matrix[i][j] > 0.85:
                        G.add_edge(i, j)
                        
            components = list(nx.connected_components(G))
            
            for comp in components:
                indices = list(comp)
                if len(indices) >= 2:
                    # Keep first, flag rest
                    indices.sort()
                    rep_idx = indices[0]
                    cluster_members = [clean_reviews[rep_idx].get("username", "Unknown")]
                    for dup_idx in indices[1:]:
                        clean_reviews[dup_idx]['is_duplicate'] = True
                        clean_reviews[dup_idx]['duplicate_type'] = "NEAR_DUPLICATE"
                        near_duplicates += 1
                        cluster_members.append(clean_reviews[dup_idx].get("username", "Unknown"))
                        
                    near_cluster_details.append({
                        "representative": clean_reviews[rep_idx].get("clean_text", "")[:50] + "...",
                        "count": len(indices),
                        "members": cluster_members
                    })
        except Exception:
            pass

    return reviews, exact_duplicates, near_duplicates, near_cluster_details


# Part 2: Individual Scoring
def analyze_individual(r: Dict, sim_matrix, idx: int) -> Tuple[int, List[str]]:
    score = 0
    signals = []
    
    text = str(r.get("clean_text", r.get("review_text", ""))).lower()
    try:
        rating = float(r.get("rating", 3))
    except:
        rating = 3.0

    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # SIGNAL 1: Rating vs Sentiment Mismatch (0-20)
    neg_kw = ['bad', 'worst', 'broken', 'terrible', 'waste', 'fraud', 'fake', 'pathetic']
    pos_kw = ['amazing', 'excellent', 'great', 'love', 'perfect', 'fantastic', 'best']
    
    n_count = sum(1 for w in words if w in neg_kw)
    p_count = sum(1 for w in words if w in pos_kw)
    
    if rating >= 4.0 and n_count > 3:
        score += 20
        signals.append("RATING_SENTIMENT_MISMATCH")
    elif rating <= 2.0 and p_count > 3:
        score += 20
        signals.append("RATING_SENTIMENT_MISMATCH")

    # SIGNAL 2: Lexical Diversity (0-20)
    if word_count > 0:
        unique_words = len(set(words))
        ttr = unique_words / word_count
        if ttr < 0.4:
            score += 20
            signals.append("LOW_LEXICAL_DIVERSITY")
        elif ttr <= 0.6:
            score += 10
            signals.append("SUSPICIOUS_LEXICAL_DIVERSITY")

    # SIGNAL 3: Review Length / Generic (0-20)
    generic_kw = ['amazing', 'great', 'good', 'quality', 'recommend', 'everyone', 'value', 'money', 'best', 'purchase', 'excellent']
    gen_count = sum(1 for w in words if w in generic_kw)
    spec_kw = ['buy', 'delivery', 'battery', 'price', 'item', 'support', 'worked', 'broke', 'arrived', 'packaging']
    has_spec = any(w in words for w in spec_kw)
    
    if word_count < 5:
        score += 15
        signals.append("TOO_SHORT")
    elif gen_count >= 5 and not has_spec:
        score += 20
        signals.append("GENERIC_PRAISE")

    # SIGNAL 4: Reviewer Pattern (0-20)
    username = str(r.get('username', ''))
    if re.match(r'^user_?\d+$', username, re.IGNORECASE):
        score += 15
        signals.append("AUTO_GENERATED_USERNAME")
    verified = r.get('verified_purchase')
    if verified is False or verified == 'False':
        score += 10
        signals.append("UNVERIFIED_PURCHASE")

    # SIGNAL 5: Text Similarity
    if sim_matrix is not None and idx < len(sim_matrix):
        row = sim_matrix[idx]
        sim_scores = [val for j, val in enumerate(row) if j != idx]
        if any(v > 0.70 for v in sim_scores):
            score += 20
            signals.append("HIGH_SIMILARITY_MATCH")
        elif sum(1 for v in sim_scores if v > 0.50) >= 3:
            score += 20
            signals.append("NETWORK_SIMILARITY")

    score = min(score, 100)
    return score, signals

# Part 3: Campaigns
def detect_coordinated_campaigns(reviews: List[Dict], sim_matrix=None) -> List[CampaignCluster]:
    clusters = []
    
    # Rule 1: Timestamp Burst
    timestamped = []
    for i, r in enumerate(reviews):
        ts_str = r.get("timestamp", "")
        if ts_str:
            try:
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        ts = datetime.strptime(ts_str[:19], fmt[:min(len(fmt), 19)])
                        timestamped.append((i, ts, r.get('rating'), r))
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
                
    timestamped.sort(key=lambda x: x[1])
    window_delta = timedelta(hours=2)
    
    visited = set()
    for start_idx in range(len(timestamped)):
        if start_idx in visited: continue
        st = timestamped[start_idx][1]
        et = st + window_delta
        
        in_window = [x for i, x in enumerate(timestamped[start_idx:]) if x[1] <= et]
        
        if len(in_window) >= 3:
            ratings = [x[2] for x in in_window]
            if len(set(ratings)) == 1 and str(ratings[0]) in ("1", "5", "1.0", "5.0"):
                cluster_inds = [x[0] for x in in_window]
                for x in in_window:
                    visited.add(x[0]) # approx
                clusters.append(CampaignCluster(
                    "TIMESTAMP_BURST",
                    reviewers=[reviews[i].get('username', '') for i in cluster_inds],
                    review_ids=[reviews[i].get('review_id', '') for i in cluster_inds],
                    campaign_confidence=35 + 40, # bump up
                    probability_label="HIGH PROBABILITY COORDINATED ATTACK",
                    time_window=f"{st.strftime('%Y-%m-%d %H:%M')} — {et.strftime('%Y-%m-%d %H:%M')}",
                    avg_rating=float(ratings[0]),
                    summary=f"{len(in_window)} reviews submitted identically within 2 hours."
                ))

    # Rule 2: Seq Usernames
    usernames = [(i, r.get("username", "")) for i, r in enumerate(reviews)]
    pattern = re.compile(r'^(.+?)(\d+)$')
    prefix_map = defaultdict(list)
    for idx, un in usernames:
        match = pattern.match(un)
        if match:
            prefix_map[match.group(1)].append((int(match.group(2)), idx, un))
            
    for prefix, entries in prefix_map.items():
        if len(entries) >= 3:
            entries.sort(key=lambda x: x[0])
            seq = [entries[0]]
            for i in range(1, len(entries)):
                if entries[i][0] - seq[-1][0] <= 2:
                    seq.append(entries[i])
                else:
                    if len(seq) >= 3:
                        inds = [x[1] for x in seq]
                        clusters.append(CampaignCluster(
                            "USERNAME_PATTERN",
                            reviewers=[x[2] for x in seq],
                            review_ids=[reviews[x].get('review_id') for x in inds],
                            campaign_confidence=75,
                            probability_label="HIGH PROBABILITY COORDINATED ATTACK",
                            time_window=None,
                            avg_rating=3.0,
                            summary=f"{len(seq)} sequential usernames detected."
                        ))
                    seq = [entries[i]]
            if len(seq) >= 3:
                inds = [x[1] for x in seq]
                clusters.append(CampaignCluster(
                    "USERNAME_PATTERN",
                    reviewers=[x[2] for x in seq],
                    review_ids=[reviews[x].get('review_id') for x in inds],
                    campaign_confidence=75,
                    probability_label="HIGH PROBABILITY COORDINATED ATTACK",
                    time_window=None,
                    avg_rating=3.0,
                    summary=f"{len(seq)} sequential usernames detected."
                ))

    # Rule 3: Text Sim
    if sim_matrix is not None:
        G = nx.Graph()
        for i in range(len(sim_matrix)):
            G.add_node(i)
        for i in range(len(sim_matrix)):
            for j in range(i+1, len(sim_matrix)):
                if sim_matrix[i][j] > 0.65:
                    G.add_edge(i, j)
        comps = list(nx.connected_components(G))
        for comp in comps:
            if len(comp) >= 3:
                inds = list(comp)
                clusters.append(CampaignCluster(
                    "TEXT_SIMILARITY",
                    reviewers=[reviews[x].get('username') for x in inds],
                    review_ids=[reviews[x].get('review_id') for x in inds],
                    campaign_confidence=60, # 30base + something -> SUSPICIOUS
                    probability_label="SUSPICIOUS CLUSTER",
                    time_window=None,
                    avg_rating=3.0,
                    summary=f"{len(inds)} functionally identical reviews found."
                ))
                
    return clusters

def build_network_map(individuals, clusters):
    nodes = []
    edges = []
    seen = set()
    
    for c in clusters:
        for ru in c.reviewers:
            if ru not in seen:
                seen.add(ru)
                color = "#F44336" if c.campaign_confidence >= 70 else "#FF9800"
                nodes.append({
                    "id": ru, "label": ru[:15], "color": color, 
                    "size": 25, "title": f"{ru} - Campaign Member"
                })
        for i in range(len(c.reviewers)):
            for j in range(i+1, len(c.reviewers)):
                edges.append({
                    "from": c.reviewers[i], "to": c.reviewers[j],
                    "color": "#F44336" if c.campaign_confidence >= 70 else "#FF9800"
                })
                
    # Add scattered nodes
    for ind in individuals[:50]:
        if ind.username not in seen:
            seen.add(ind.username)
            color = "#4CAF50" # Trusted
            if ind.status == "CONFIRMED_BOT": color = "#F44336"
            elif ind.status == "HIGH_SUSPICION": color = "#FF9800"
            elif ind.status == "LOW_SUSPICION": color = "#FFEB3B"
            nodes.append({
                "id": ind.username, "label": ind.username[:15], "color": color, 
                "size": 15, "title": f"Status: {ind.status}"
            })
            
    return {"nodes": nodes, "edges": edges}

def execute_bot_detection(reviews: List[Dict]) -> BotDetectionResult:
    # 1. Dedup
    reviews, em_count, nm_count, n_clust = run_deduplication(reviews)
    
    # Prep TFIDF
    texts = [r.get("clean_text", r.get("review_text", "")) for r in reviews]
    sim_matrix = None
    if len(texts) > 1:
        v = TfidfVectorizer(max_features=2000, stop_words='english')
        try:
            m = v.fit_transform([t if t else "e" for t in texts])
            sim_matrix = cosine_similarity(m)
        except:
            pass

    indivs = []
    conf_bot_ct = 0
    susp_ct = 0

    for i, r in enumerate(reviews):
        if r.get('is_duplicate'): continue
        
        score, signals = analyze_individual(r, sim_matrix, i)
        if score <= 25:
            s_label = "TRUSTED"
        elif score <= 50:
            s_label = "LOW_SUSPICION"
            susp_ct += 1
        elif score <= 75:
            s_label = "HIGH_SUSPICION"
            susp_ct += 1
        else:
            s_label = "CONFIRMED_BOT"
            conf_bot_ct += 1
            
        indivs.append(SuspiciousIndividual(
            review_id=r.get("review_id", ""),
            username=r.get("username", "Unknown"),
            text_snippet=str(r.get("clean_text", r.get("review_text", "")))[:50] + "...",
            score=score,
            signals_triggered=signals,
            status=s_label,
            is_duplicate=False,
            duplicate_type=""
        ))
        r['bot_status'] = s_label

    clusters = detect_coordinated_campaigns(reviews, sim_matrix)
    net_data = build_network_map(indivs, clusters)
    
    total = len(reviews)
    clean = total - em_count - nm_count - conf_bot_ct

    return BotDetectionResult(
        total_reviews_scanned=total,
        confirmed_bots_removed=conf_bot_ct,
        suspicious_flagged=susp_ct,
        campaigns_detected=len(clusters),
        clean_dataset_count=clean,
        exact_duplicates_removed=em_count,
        near_duplicates_clustered=nm_count,
        near_duplicate_cluster_details=n_clust,
        individuals=indivs,
        clusters=clusters,
        network_data=net_data
    )

def generate_network_html(network_data: Dict, output_path: str = "sample_output/bot_network.html") -> str:
    try:
        from pyvis.network import Network
        net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="white")
        for node in network_data.get("nodes", []):
            net.add_node(node["id"], label=node["label"], color=node["color"], size=node["size"], title=node["title"])
        for edge in network_data.get("edges", []):
            net.add_edge(edge["from"], edge["to"], color=edge["color"])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        net.save_graph(output_path)
        return output_path
    except ImportError:
        # Fallback to simple generic stub if missing
        with open(output_path, "w") as f:
            f.write("<html><body><h3>PyVis not installed natively. Networks unavailable.</h3></body></html>")
        return output_path
