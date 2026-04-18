"""
Test script for Surprise Batch timing verification.
Verifies that 20 reviews process in <45 seconds.
"""

import time
import requests
import sys

API_BASE_URL = "http://localhost:8000"


def test_surprise_batch_timing():
    """
    Test the surprise batch endpoint timing.
    
    Expected timeline for 20 reviews:
    - Parsing: <0.1s
    - Parallel preprocessing (4 threads): ~2-3s
    - Batch extraction (batch_size=5, max_workers=3): ~10-15s
      - 20 reviews = 4 batches
      - 4 batches with 3 workers = 2 parallel rounds (3+1)
      - Each API call ≈ 3-4s, total ≈ 8-10s
    - Aggregation: <0.5s
    - Async DB save: ~2s (non-blocking)
    - Total: ~15-20s (well under 45s limit)
    """
    
    print("=" * 70)
    print("🧪 ReviewIQ Surprise Batch Timing Test")
    print("=" * 70)
    print(f"API: {API_BASE_URL}")
    print(f"Target: <45 seconds for 20 reviews")
    print(f"Expected: ~15-20 seconds with optimizations")
    print()
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API not responding correctly")
            print("   Start the API with: uvicorn backend.main:app --reload")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Start the API with: uvicorn backend.main:app --reload")
        return False
    
    print("✅ API is healthy")
    print()
    
    # Run timing test
    print("🚀 Running surprise batch test with 20 sample reviews...")
    print()
    
    start = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/surprise-batch-test",
            timeout=60
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            
            print("📊 Results:")
            print(f"   API-reported processing time: {data.get('time_taken_seconds', 0):.2f}s")
            print(f"   Total round-trip time: {elapsed:.2f}s")
            print(f"   Target: <45s")
            print()
            
            api_time = data.get('time_taken_seconds', 999)
            passed = api_time < 45
            
            # Color coding
            if passed:
                if api_time < 20:
                    speed = "🚀 Excellent"
                elif api_time < 30:
                    speed = "✅ Good"
                else:
                    speed = "⚠️ Acceptable"
                
                print(f"{speed} - Processed in {api_time:.2f}s (under 45s limit)")
            else:
                print(f"❌ FAIL - Took {api_time:.2f}s (exceeds 45s limit)")
            
            print()
            print("📈 Pipeline Results:")
            result = data.get('pipeline_result', {})
            print(f"   Reviews processed: {result.get('reviews_processed', 0)}")
            print(f"   Top issues found: {len(result.get('top_issues', []))}")
            
            if result.get('top_issues'):
                print("   Top issues detected:")
                for i, issue in enumerate(result['top_issues'][:3], 1):
                    print(f"      {i}. {issue.get('feature_name')}: {issue.get('complaint_count')} complaints (intensity: {issue.get('avg_intensity', 0):.2f})")
            
            if result.get('sentiment_summary'):
                sentiment = result['sentiment_summary']
                print("   Sentiment distribution:")
                print(f"      Positive: {sentiment.get('positive', 0)}")
                print(f"      Negative: {sentiment.get('negative', 0)}")
                print(f"      Neutral: {sentiment.get('neutral', 0)}")
                print(f"      Mixed: {sentiment.get('mixed', 0)}")
            
            print()
            print("=" * 70)
            
            if passed:
                print("✅ TEST PASSED - Surprise batch meets timing requirements")
            else:
                print("❌ TEST FAILED - Processing took too long")
            
            print("=" * 70)
            return passed
            
        else:
            print(f"❌ Test failed with HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def test_with_progress_simulation():
    """
    Simulate how progress tracking would work in frontend.
    Shows expected progress stages and timing.
    """
    print("\n" + "=" * 70)
    print("📊 Progress Tracking Simulation")
    print("=" * 70)
    print()
    print("For 20 reviews, expected progress timeline:")
    print()
    
    stages = [
        ("parsing", 5, 0.1, "Parsing reviews from text"),
        ("preprocessing", 10, 2.5, "Preprocessing 20 reviews (4 threads parallel)"),
        ("extraction", 30, 0.5, "Starting sentiment extraction..."),
        ("extraction", 35, 3.0, "Processing batch 1/4 (reviews 1-5)"),
        ("extraction", 50, 3.0, "Processing batch 2/4 (reviews 6-10)"),
        ("extraction", 65, 3.0, "Processing batch 3/4 (reviews 11-15)"),
        ("extraction", 75, 2.0, "Processing batch 4/4 (reviews 16-20)"),
        ("aggregation", 85, 0.5, "Aggregating results..."),
        ("saving", 95, 0.5, "Saving to database (async)..."),
        ("complete", 100, 0.5, "Complete in 18.5s"),
    ]
    
    cumulative_time = 0
    for stage, progress, duration, message in stages:
        cumulative_time += duration
        print(f"   [{progress:3d}%] {message}")
    
    print()
    print(f"   Total estimated time: {cumulative_time:.1f}s")
    print()
    print("This timeline enables st.progress() bar updates in Streamlit:")
    print("   - Initial update at 5% (parsing complete)")
    print("   - Preprocessing updates at 10-25%")
    print("   - Batch progress at 35%, 50%, 65%, 75%")
    print("   - Final stages at 85%, 95%, 100%")


def main():
    """Main test runner."""
    # Run timing test
    success = test_surprise_batch_timing()
    
    # Show progress simulation
    test_with_progress_simulation()
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
