import pandas as pd
import numpy as np

def _safe_z(val, mean, std):
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return 0.0
    return (val - mean) / std

def get_actionable_insight(anomaly_type, feature):
    if anomaly_type == "RATING CRASH":
        return "Review recent product changes and halt immediate marketing spend.", "Rating dropped significantly below historical norms indicating potential broken update or bad batch."
    if anomaly_type == "COMPLAINT SPIKE":
        if feature:
            return f"Audit {feature} issues immediately.", f"{feature.title()} complaints are heavily spiking above normal baseline."
        return "Investigate recent negative reviews immediately.", "Complaints spiked drastically above normal baseline."
    if anomaly_type == "SILENT COMPLAINT GROWTH":
        return "Preemptively catch the structural issue before rating drops further.", "Continuous long-term rise in complaints detected across 3 weeks."
    if anomaly_type == "VOLUME SPIKE with LOW RATING":
        return "Check social media for viral complaints or coordinated actions.", "Immediate influx of negative reviews detected."
    return "Monitor closely.", "Statistical anomaly detected."

def calculate_anomaly_scores(df: pd.DataFrame, target_product: str):
    if target_product != 'All Products':
        df = df[df['product_name'] == target_product].copy()
    else:
        df = df.copy()
        
    if len(df) == 0:
        return {"error": "No data available."}
        
    if 'timestamp' not in df.columns:
        return {"error": "No timestamp column found."}
        
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp_dt'])
    
    if len(df) == 0:
        return {"error": "No valid timestamps."}
        
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
        
    df['is_neg'] = (df['sentiment'] == 'negative').astype(int)
    df['is_pos'] = (df['sentiment'] == 'positive').astype(int)
    
    # Process by weekly groups
    df = df.sort_values('timestamp_dt')
    weekly = df.groupby(pd.Grouper(key='timestamp_dt', freq='W-MON')).agg(
        weekly_avg_rating=('rating', 'mean'),
        neg_count=('is_neg', 'sum'),
        pos_count=('is_pos', 'sum'),
        weekly_review_volume=('rating', 'count')
    ).reset_index()
    
    # Filter empty weeks if they occur at edges, but keeping continuous is better for time series
    # Let's drop leading and trailing 0 volume weeks
    weekly = weekly[weekly['weekly_review_volume'] > 0].reset_index(drop=True)
    
    if len(weekly) < 4:
        return {"error": "Need more data — 4+ weeks required for anomaly detection."}
        
    weekly['weekly_complaint_rate'] = (weekly['neg_count'] / weekly['weekly_review_volume']) * 100
    weekly['weekly_sentiment_score'] = (weekly['pos_count'] / weekly['weekly_review_volume']) * 100
    weekly['week_label'] = [f"W{i+1}" for i in range(len(weekly))]
    
    # Calculate Z-Scores mapping historical expanding window
    for col in ['weekly_avg_rating', 'weekly_complaint_rate', 'weekly_review_volume', 'weekly_sentiment_score']:
        s = weekly[col]
        # expanding mean excluding current row
        prev_mean = s.shift(1).expanding(min_periods=2).mean()
        prev_std = s.shift(1).expanding(min_periods=2).std(ddof=1)
        z_col = f"z_{col.replace('weekly_', '')}"
        
        # Calculate
        weekly[z_col] = [
            _safe_z(val, prev_mean.iloc[i], prev_std.iloc[i]) 
            for i, val in enumerate(s)
        ]
        
    # Anomaly classification logic
    anomalies = []
    
    for i in range(3, len(weekly)):
        row = weekly.iloc[i]
        week_num = row['week_label']
        
        z_rating = row['z_avg_rating']
        z_comp = row['z_complaint_rate']
        z_vol = row['z_review_volume']
        rating = row['weekly_avg_rating']
        complaint_rate = row['weekly_complaint_rate']
        baseline_comp = weekly['weekly_complaint_rate'].iloc[:i].mean()
        
        type_str = None
        action, meaning = "", ""
        
        # TYPE 1
        if z_rating < -2.0 and rating < 4.0:
            type_str = "RATING CRASH"
            msg = f"Rating dropped below 4.0★ threshold — {abs(z_rating):.1f} standard deviations below historical average"
            action, meaning = get_actionable_insight(type_str, "")
            anomalies.append({"type": type_str, "week": week_num, "z_score": z_rating, "feature": "Overall", "msg": msg, "action": action, "meaning": meaning, "metric": rating, "baseline": weekly['weekly_avg_rating'].iloc[:i].mean()})
            continue # Prioritize crash
            
        # TYPE 2
        if z_comp > 2.0:
            type_str = "COMPLAINT SPIKE"
            msg = f"Complaint rate {complaint_rate:.1f}% — {z_comp:.1f} standard deviations above normal baseline of {baseline_comp:.1f}%"
            action, meaning = get_actionable_insight(type_str, "")
            anomalies.append({"type": type_str, "week": week_num, "z_score": z_comp, "feature": "Overall", "msg": msg, "action": action, "meaning": meaning, "metric": complaint_rate, "baseline": baseline_comp})
        
        # TYPE 4
        if z_vol > 2.0 and rating < 3.0:
            type_str = "VOLUME SPIKE with LOW RATING"
            msg = "Sudden influx of negative reviews — possible viral complaint or coordinated attack"
            action, meaning = get_actionable_insight(type_str, "")
            anomalies.append({"type": type_str, "week": week_num, "z_score": z_vol, "feature": "Overall", "msg": msg, "action": action, "meaning": meaning, "metric": row['weekly_review_volume'], "baseline": weekly['weekly_review_volume'].iloc[:i].mean()})

    # TYPE 3 - Silent complaint growth
    for i in range(2, len(weekly)):
        if (weekly['z_complaint_rate'].iloc[i] > 1.5 and 
            weekly['z_complaint_rate'].iloc[i-1] > 1.5 and 
            weekly['z_complaint_rate'].iloc[i-2] > 1.5):
            
            type_str = "SILENT COMPLAINT GROWTH"
            week_num = weekly['week_label'].iloc[i]
            # Avoid duplicate if already covered
            if not any(a['type'] == type_str and a['week'] == week_num for a in anomalies):
                msg = "⚠️ Slow-burn issue detected: Complaint rate has been above normal for 3 consecutive weeks. This often precedes a major rating drop."
                action, meaning = get_actionable_insight(type_str, "")
                anomalies.append({"type": type_str, "week": week_num, "z_score": weekly['z_complaint_rate'].iloc[i], "feature": "Overall", "msg": msg, "action": action, "meaning": meaning, "metric": weekly['weekly_complaint_rate'].iloc[i], "baseline": weekly['weekly_complaint_rate'].iloc[:i-2].mean()})
                
    # Step 3: Feature Level Heatmap
    features = ['battery', 'packaging', 'delivery', 'durability', 'support', 'ease_of_use', 'taste', 'price']
    
    if 'clean_text' not in df.columns:
        df['clean_text'] = df['review_text'] if 'review_text' in df.columns else ""
        
    df['clean_text_dl'] = df['clean_text'].astype(str).str.lower()
    
    feature_matrices = {}
    
    for f in features:
        df[f'has_{f}'] = df['clean_text_dl'].str.contains(f).astype(int)
        df[f'neg_{f}'] = (df[f'has_{f}'] & df['is_neg']).astype(int)
        
    weekly_f = df.groupby(pd.Grouper(key='timestamp_dt', freq='W-MON')).agg(
        **{f"{f}_neg": (f'neg_{f}', 'sum') for f in features},
        count=('rating', 'count')
    ).reset_index()
    weekly_f = weekly_f[weekly_f['count'] > 0].reset_index(drop=True)
    
    heatmap_data = []
    
    for f in features:
        s = (weekly_f[f"{f}_neg"] / weekly_f['count']) * 100
        prev_mean = s.shift(1).expanding(min_periods=2).mean()
        prev_std = s.shift(1).expanding(min_periods=2).std(ddof=1)
        z_scores = [_safe_z(val, prev_mean.iloc[i], prev_std.iloc[i]) for i, val in enumerate(s)]
        heatmap_data.append(z_scores)
        
        # Check specific feature anomalies
        for i in range(3, len(z_scores)):
            z_f = z_scores[i]
            if z_f > 3.0:
                week_num = f"W{i+1}"
                type_str = "COMPLAINT SPIKE"
                msg = f"Packaging complaints are {z_f:.1f}× above normal — systemic defect" if f == "packaging" else f"{f.title()} complaints are {z_f:.1f}× above normal — systemic defect"
                action, meaning = get_actionable_insight(type_str, f)
                
                # Check for duplicates
                exists = any(a['feature'] == f and a['week'] == week_num for a in anomalies)
                if not exists:
                    anomalies.append({"type": type_str, "week": week_num, "z_score": z_f, "feature": f.title(), "msg": msg, "action": action, "meaning": meaning, "metric": s.iloc[i], "baseline": s.iloc[:i].mean()})
    
    heatmap_df = pd.DataFrame(heatmap_data, index=[f.title() for f in features], columns=[f"W{i+1}" for i in range(len(heatmap_data[0])) if len(heatmap_data) > 0])
    
    # Base peak rating
    peak_rating = df.groupby(pd.Grouper(key='timestamp_dt', freq='W-MON'))['rating'].mean().max()
    curr_rating = weekly['weekly_avg_rating'].iloc[-1]
    rating_drop_z = weekly['z_avg_rating'].iloc[-1]
    
    return {
        "weekly_data": weekly.to_dict('records'),
        "anomalies": anomalies,
        "heatmap": heatmap_df,
        "peak_rating": peak_rating,
        "curr_rating": curr_rating,
        "rating_drop_z": rating_drop_z
    }
