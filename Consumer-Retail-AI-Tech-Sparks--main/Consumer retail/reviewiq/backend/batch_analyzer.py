import pandas as pd
from typing import Dict, List, Any
import collections
import re

FEATURE_KEYWORDS = ['battery', 'packaging', 'delivery', 'quality', 'price', 
                   'service', 'screen', 'camera', 'speed', 'design', 'support',
                   'cap', 'leak', 'material']

def _extract_top_word(texts: List[str]) -> str:
    words = []
    for t in texts:
        text_lower = str(t).lower()
        for kw in FEATURE_KEYWORDS:
            if kw in text_lower:
                words.append(kw)
    if words:
        most_common = collections.Counter(words).most_common(1)
        if most_common:
            return most_common[0][0].title()
    return "None"

def analyze_batches(df: pd.DataFrame, mode: str, size: Any) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"error": "No data available."}
        
    df = df.copy()
    if 'timestamp' not in df.columns:
        return {"error": "No timestamp column found."}
        
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')
    
    batches = []
    
    if mode == 'time':
        # size is days like 7
        freq = f"{size}D"
        grouped = df.groupby(pd.Grouper(key='timestamp_dt', freq=freq))
        
        batch_idx = 1
        for name, group in grouped:
            if len(group) == 0:
                continue
            batches.append({
                'id': batch_idx,
                'name': f"Week {batch_idx}" if size == 7 else f"Period {batch_idx}",
                'num_days': size,
                'start_date': group['timestamp_dt'].min(),
                'end_date': group['timestamp_dt'].max(),
                'indices': group.index.tolist(),
                'data': group
            })
            batch_idx += 1
            
    elif mode == 'fixed':
        batch_idx = 1
        for i in range(0, len(df), size):
            group = df.iloc[i:i+size]
            batches.append({
                'id': batch_idx,
                'name': f"Batch {batch_idx}",
                'num_days': None,
                'start_date': group['timestamp_dt'].min(),
                'end_date': group['timestamp_dt'].max(),
                'indices': group.index.tolist(),
                'data': group
            })
            batch_idx += 1
            
    if not batches:
        return {"error": "Unable to create batches."}
        
    results = []
    prev_rating = None
    prev_complaint_rate = None
    
    for b in batches:
        group = b['data']
        count = len(group)
        avg_rating = group['rating'].astype(float).mean() if 'rating' in group.columns else 0.0
        
        if 'sentiment' not in group.columns:
            sentiment_series = group['rating'].apply(lambda r: 'positive' if r >= 4 else ('negative' if r <= 2 else 'neutral'))
        else:
            sentiment_series = group['sentiment']
            
        pos_count = (sentiment_series == 'positive').sum()
        neg_count = (sentiment_series == 'negative').sum()
        
        sentiment_score = (pos_count / count * 100) if count > 0 else 0
        complaint_rate = (neg_count / count * 100) if count > 0 else 0
        
        # Extact top issue/praise
        if 'clean_text' not in group.columns:
            group['clean_text'] = group['review_text'] if 'review_text' in group.columns else ''
            
        neg_texts = group[sentiment_series == 'negative']['clean_text'].tolist()
        pos_texts = group[sentiment_series == 'positive']['clean_text'].tolist()
        
        top_complaint = _extract_top_word(neg_texts)
        top_praise = _extract_top_word(pos_texts)
        
        # Deltas
        rating_delta = (avg_rating - prev_rating) if prev_rating is not None else 0
        complaint_delta = (complaint_rate - prev_complaint_rate) if prev_complaint_rate is not None else 0
        
        status = '🟢 Stable'
        flag = None
        if rating_delta < -0.3:
            status = '🔴 Alert'
            flag = 'RATING DROP'
        elif complaint_delta > 15:
            status = '🔴 Alert'
            flag = 'COMPLAINT SPIKE'
        elif rating_delta < -0.1 or complaint_delta > 5:
            status = '🟡 Watch'
            
        results.append({
            'batch_num': b['id'],
            'batch_name': b['name'],
            'start_date': b['start_date'].strftime('%Y-%m-%d') if pd.notnull(b['start_date']) else '',
            'end_date': b['end_date'].strftime('%Y-%m-%d') if pd.notnull(b['end_date']) else '',
            'review_count': count,
            'avg_rating': round(avg_rating, 2),
            'sentiment_score': round(sentiment_score, 1),
            'complaint_rate': round(complaint_rate, 1),
            'top_complaint': top_complaint,
            'top_praise': top_praise,
            'rating_delta': round(rating_delta, 2),
            'complaint_delta': round(complaint_delta, 1),
            'status': status,
            'flag': flag,
            'reviews_indices': b['indices'] # pass indices for drilldown
        })
        
        prev_rating = avg_rating
        prev_complaint_rate = complaint_rate
        
    # Check Post-Purchase Degradation
    has_degradation = False
    degradation_batch = None
    degradation_reason = ""
    main_issue = ""
    batch1_rating = results[0]['avg_rating'] if len(results) > 0 else 0
    batch1_2_rating = results[0]['avg_rating']
    if len(results) > 1:
        batch1_2_rating = (results[0]['avg_rating'] + results[1]['avg_rating']) / 2
        
    degraded_batch_num = -1
    recent_rating = 0
    if batch1_rating > 4.0 and len(results) >= 3:
        for r in results[2:]:
            if r['avg_rating'] < 3.5:
                has_degradation = True
                degradation_batch = r
                degraded_batch_num = r['batch_num']
                main_issue = r['top_complaint']
                break
                
    if has_degradation:
        degradation_reason = f"Product rated {batch1_rating:.1f}★ in first batch, dropped to {degradation_batch['avg_rating']:.1f}★ by {degradation_batch['batch_name']}. Likely quality/durability issue."
        # Calculate recent rating
        recent_rating = sum([r['avg_rating'] for r in results[-2:]]) / min(2, len(results))
        
    return {
        "batches": results,
        "degradation_detected": has_degradation,
        "degradation_batch_num": degraded_batch_num,
        "degradation_reason": degradation_reason,
        "early_rating": round(batch1_2_rating, 1),
        "recent_rating": round(recent_rating, 1),
        "main_issue": main_issue,
        "degradation_date_range": f"{degradation_batch['start_date']} to {degradation_batch['end_date']}" if has_degradation else ""
    }
