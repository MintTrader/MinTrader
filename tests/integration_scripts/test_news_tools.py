#!/usr/bin/env python3
"""
Test script for news data tools.
Tests news retrieval functions with real API calls (Alpaca and Google News).
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from tradingagents.dataflows.alpaca_news import get_news as get_alpaca_news
from tradingagents.dataflows.google import get_google_news
from tradingagents.dataflows.y_finance import get_insider_transactions

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_alpaca_news():
    """Test Alpaca news API."""
    print_section("TEST: Alpaca News API")
    
    ticker = "AAPL"
    # Use yesterday as end date to avoid any potential timing issues
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")
    
    print(f"Testing with:")
    print(f"  Ticker: {ticker}")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print()
    
    try:
        result = get_alpaca_news(ticker, start_date, end_date)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: News data retrieved")
            
            # Try to parse as JSON
            try:
                data = json.loads(result)
                num_articles = len(data.get('feed', []))
                print(f"\nNumber of articles: {num_articles}")
                
                if num_articles > 0:
                    print("\nFirst article:")
                    print("-" * 80)
                    first_article = data['feed'][0]
                    print(f"Title: {first_article.get('title', 'N/A')}")
                    print(f"Source: {first_article.get('source', 'N/A')}")
                    print(f"URL: {first_article.get('url', 'N/A')}")
                    print(f"Published: {first_article.get('time_published', 'N/A')}")
                    print(f"Summary: {first_article.get('summary', 'N/A')[:200]}...")
                    print("-" * 80)
                
            except json.JSONDecodeError:
                print("Note: Could not parse as JSON, showing raw result:")
                print(result[:500])
            
            return True
        else:
            print("✗ FAILED: Empty result returned")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_google_news():
    """Test Google News API."""
    print_section("TEST: Google News API")
    
    query = "AAPL"
    # Use yesterday's date for consistency
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    look_back_days = 7
    
    print(f"Testing with:")
    print(f"  Query: {query}")
    print(f"  Current Date: {curr_date}")
    print(f"  Look Back Days: {look_back_days}")
    print()
    
    try:
        result = get_google_news(query, curr_date, look_back_days)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: Google News data retrieved")
            print(f"\nFirst 600 characters of result:")
            print("-" * 80)
            print(result[:600])
            print("-" * 80)
            
            # Count articles (each starts with ###)
            article_count = result.count('###')
            print(f"\nNumber of articles: {article_count}")
            
            return True
        else:
            print("⚠ WARNING: Empty result (might be no news for this query)")
            print("This is not necessarily a failure - query might have no results")
            return True  # Don't fail on empty results
            
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_tickers_news():
    """Test news for multiple tickers."""
    print_section("TEST: Multiple Tickers (Alpaca News)")
    
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    # Use yesterday as end date
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")
    
    results = {}
    
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        try:
            result = get_alpaca_news(ticker, start_date, end_date)
            if result and len(result) > 0:
                data = json.loads(result)
                num_articles = len(data.get('feed', []))
                print(f"  ✓ {ticker}: SUCCESS ({num_articles} articles)")
                results[ticker] = True
            else:
                print(f"  ✗ {ticker}: FAILED (no data)")
                results[ticker] = False
        except Exception as e:
            print(f"  ✗ {ticker}: FAILED ({str(e)[:50]})")
            results[ticker] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(tickers)}")
    
    return success_count >= len(tickers) * 0.75  # Allow 25% failure rate

def test_google_news_queries():
    """Test Google News with different queries."""
    print_section("TEST: Google News - Different Queries")
    
    queries = [
        "Apple stock",
        "Tesla earnings",
        "Federal Reserve",
        "Stock market crash"
    ]
    # Use yesterday's date
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    look_back_days = 7
    
    results = {}
    
    for query in queries:
        print(f"\nTesting query: '{query}'...")
        try:
            result = get_google_news(query, curr_date, look_back_days)
            if result and len(result) > 0:
                article_count = result.count('###')
                print(f"  ✓ '{query}': SUCCESS ({article_count} articles)")
                results[query] = True
            else:
                print(f"  ⚠ '{query}': Empty result (no articles found)")
                results[query] = True  # Don't fail on empty results
        except Exception as e:
            print(f"  ✗ '{query}': FAILED ({str(e)[:50]})")
            results[query] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(queries)}")
    
    return success_count == len(queries)

def test_insider_transactions():
    """Test insider transactions data."""
    print_section("TEST: Insider Transactions (Yahoo Finance)")
    
    ticker = "AAPL"
    
    print(f"Testing with:")
    print(f"  Ticker: {ticker}")
    print()
    
    try:
        # Note: get_insider_transactions only takes ticker as argument
        result = get_insider_transactions(ticker)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: Insider transaction data retrieved")
            print(f"\nFirst 600 characters of result:")
            print("-" * 80)
            print(result[:600])
            print("-" * 80)
            return True
        else:
            print("⚠ WARNING: Empty result (might be no recent insider transactions)")
            print("This is not necessarily a failure")
            return True  # Don't fail on empty results
            
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_news_date_ranges():
    """Test news with different date ranges."""
    print_section("TEST: News - Different Date Ranges")
    
    ticker = "AAPL"
    test_cases = [
        ("1 day (ending yesterday)", 1),
        ("3 days (ending yesterday)", 3),
        ("1 week (ending yesterday)", 7),
        ("2 weeks (ending yesterday)", 14),
    ]
    
    results = {}
    
    for name, days in test_cases:
        # End date is yesterday, start is (days+1) ago
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days+1)).strftime("%Y-%m-%d")
        
        print(f"\nTesting {name} ({start_date} to {end_date})...")
        try:
            result = get_alpaca_news(ticker, start_date, end_date)
            if result and len(result) > 0:
                data = json.loads(result)
                num_articles = len(data.get('feed', []))
                print(f"  ✓ {name}: SUCCESS ({num_articles} articles)")
                results[name] = True
            else:
                print(f"  ⚠ {name}: Empty result")
                results[name] = True  # Don't fail on empty
        except Exception as e:
            print(f"  ✗ {name}: FAILED ({str(e)[:50]})")
            results[name] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(test_cases)}")
    
    return success_count == len(test_cases)

def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*22 + "NEWS TOOLS TEST SUITE" + " "*35 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Alpaca News API", test_alpaca_news),
        ("Google News API", test_google_news),
        ("Multiple Tickers", test_multiple_tickers_news),
        ("Google News Queries", test_google_news_queries),
        ("Insider Transactions", test_insider_transactions),
        ("News Date Ranges", test_news_date_ranges),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-" * 80)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("-" * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    exit(main())

