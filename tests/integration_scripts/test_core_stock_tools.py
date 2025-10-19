#!/usr/bin/env python3
"""
Test script for core stock data tools.
Tests the get_stock_data function with real API calls (Yahoo Finance works, Alpaca free tier doesn't have data).
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Use Yahoo Finance instead of Alpaca (Alpaca free tier has no historical data access)
from tradingagents.dataflows.y_finance import get_YFin_data_online as get_stock

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_get_stock_data():
    """Test fetching stock price data (OHLCV)."""
    print_section("TEST: Get Stock Data (OHLCV)")
    
    # Test parameters
    # Note: Alpaca free tier cannot query data from the past 15 minutes
    # Use data from 1 day ago to avoid subscription limitations
    symbol = "AAPL"
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d")
    
    print(f"Testing with:")
    print(f"  Symbol: {symbol}")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print()
    
    try:
        result = get_stock(symbol, start_date, end_date)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: Data retrieved")
            print(f"\nFirst 500 characters of result:")
            print("-" * 80)
            print(result[:500])
            print("-" * 80)
            
            # Count number of data points
            lines = result.strip().split('\n')
            data_points = len(lines) - 1  # Subtract header
            print(f"\nData points received: {data_points}")
            
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

def test_multiple_symbols():
    """Test with multiple different symbols."""
    print_section("TEST: Multiple Symbols")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    # Use data from 1 day ago to avoid Alpaca free tier 15-minute restriction
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")
    
    results = {}
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        try:
            result = get_stock(symbol, start_date, end_date)
            if result and len(result) > 50:  # Check if we got meaningful data
                print(f"  ✓ {symbol}: SUCCESS")
                results[symbol] = True
            else:
                print(f"  ✗ {symbol}: FAILED (empty or no data)")
                results[symbol] = False
        except Exception as e:
            print(f"  ✗ {symbol}: FAILED ({str(e)[:50]})")
            results[symbol] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(symbols)}")
    print(f"  Failed: {len(symbols) - success_count}/{len(symbols)}")
    
    return success_count == len(symbols)

def test_date_ranges():
    """Test different date ranges."""
    print_section("TEST: Different Date Ranges")
    
    symbol = "AAPL"
    # Note: All end dates are set to 1 day ago to avoid Alpaca free tier 15-minute restriction
    test_cases = [
        ("1 week (ending yesterday)", 7),
        ("2 weeks (ending yesterday)", 14),
        ("1 month (ending yesterday)", 30),
        ("3 months (ending yesterday)", 90),
    ]
    
    results = {}
    
    for name, days in test_cases:
        # End date is 1 day ago, start date is (days+1) ago
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days+1)).strftime("%Y-%m-%d")
        
        print(f"\nTesting {name} ({start_date} to {end_date})...")
        try:
            result = get_stock(symbol, start_date, end_date)
            if result and len(result) > 50:
                lines = result.strip().split('\n')
                data_points = len(lines) - 1
                print(f"  ✓ SUCCESS: {data_points} data points")
                results[name] = True
            else:
                print(f"  ✗ FAILED: No data")
                results[name] = False
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)[:50]}")
            results[name] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(test_cases)}")
    
    return success_count == len(test_cases)

def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "CORE STOCK TOOLS TEST SUITE" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Basic Stock Data Fetch", test_get_stock_data),
        ("Multiple Symbols", test_multiple_symbols),
        ("Date Ranges", test_date_ranges),
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

