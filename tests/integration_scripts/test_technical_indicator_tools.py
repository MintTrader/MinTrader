#!/usr/bin/env python3
"""
Test script for technical indicator tools.
Tests the get_indicators function with real Alpaca API calls.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from tradingagents.dataflows.alpaca_indicator import get_indicator

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_single_indicator(indicator_name):
    """Test a specific indicator."""
    symbol = "AAPL"
    # Use yesterday's date to avoid Alpaca free tier 15-minute restriction
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    look_back_days = 30
    
    print(f"\nTesting {indicator_name}...")
    print(f"  Symbol: {symbol}")
    print(f"  Current Date: {curr_date}")
    print(f"  Look Back Days: {look_back_days}")
    print()
    
    try:
        result = get_indicator(symbol, indicator_name, curr_date, look_back_days)
        
        if result and len(result) > 0 and not result.startswith("Error"):
            print(f"  ✓ {indicator_name}: SUCCESS")
            print(f"\nFirst 400 characters:")
            print("-" * 80)
            print(result[:400])
            print("-" * 80)
            return True
        else:
            print(f"  ✗ {indicator_name}: FAILED")
            print(f"  Result: {result[:200]}")
            return False
            
    except Exception as e:
        print(f"  ✗ {indicator_name}: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_indicators():
    """Test all supported technical indicators."""
    print_section("TEST: All Technical Indicators")
    
    indicators = [
        "close_50_sma",
        "close_200_sma",
        "close_10_ema",
        "macd",
        "macds",
        "macdh",
        "rsi",
        "boll",
        "boll_ub",
        "boll_lb",
        "atr",
        "vwma"
    ]
    
    results = {}
    
    for indicator in indicators:
        results[indicator] = test_single_indicator(indicator)
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(indicators)}")
    print(f"  Failed: {len(indicators) - success_count}/{len(indicators)}")
    
    return success_count == len(indicators)

def test_different_lookback_periods():
    """Test different lookback periods."""
    print_section("TEST: Different Lookback Periods")
    
    symbol = "AAPL"
    indicator = "rsi"
    # Use yesterday's date to avoid Alpaca free tier 15-minute restriction
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    lookback_periods = [7, 14, 30, 60, 90]
    results = {}
    
    for days in lookback_periods:
        print(f"\nTesting {days} days lookback...")
        try:
            result = get_indicator(symbol, indicator, curr_date, days)
            if result and len(result) > 0 and not result.startswith("Error"):
                # Count data points in result
                lines = [line for line in result.split('\n') if ':' in line and any(c.isdigit() for c in line)]
                print(f"  ✓ {days} days: SUCCESS ({len(lines)} data points)")
                results[days] = True
            else:
                print(f"  ✗ {days} days: FAILED")
                results[days] = False
        except Exception as e:
            print(f"  ✗ {days} days: FAILED ({str(e)[:50]})")
            results[days] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(lookback_periods)}")
    
    return success_count == len(lookback_periods)

def test_multiple_symbols():
    """Test indicators with different symbols."""
    print_section("TEST: Multiple Symbols")
    
    symbols = ["AAPL", "MSFT", "TSLA"]
    indicator = "close_50_sma"
    # Use yesterday's date to avoid Alpaca free tier 15-minute restriction
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    look_back_days = 30
    
    results = {}
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        try:
            result = get_indicator(symbol, indicator, curr_date, look_back_days)
            if result and len(result) > 0 and not result.startswith("Error"):
                print(f"  ✓ {symbol}: SUCCESS")
                results[symbol] = True
            else:
                print(f"  ✗ {symbol}: FAILED")
                results[symbol] = False
        except Exception as e:
            print(f"  ✗ {symbol}: FAILED ({str(e)[:50]})")
            results[symbol] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(symbols)}")
    
    return success_count == len(symbols)

def test_moving_averages():
    """Test all moving average indicators together."""
    print_section("TEST: Moving Average Indicators")
    
    symbol = "AAPL"
    # Use yesterday's date to avoid Alpaca free tier 15-minute restriction
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    look_back_days = 30
    
    ma_indicators = ["close_50_sma", "close_200_sma", "close_10_ema", "vwma"]
    results = {}
    
    print(f"Testing with {symbol} over {look_back_days} days...\n")
    
    for indicator in ma_indicators:
        try:
            result = get_indicator(symbol, indicator, curr_date, look_back_days)
            if result and not result.startswith("Error"):
                print(f"  ✓ {indicator}: SUCCESS")
                results[indicator] = True
            else:
                print(f"  ✗ {indicator}: FAILED")
                results[indicator] = False
        except Exception as e:
            print(f"  ✗ {indicator}: FAILED ({str(e)[:50]})")
            results[indicator] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(ma_indicators)}")
    
    return success_count == len(ma_indicators)

def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "TECHNICAL INDICATOR TOOLS TEST SUITE" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("All Indicators", test_all_indicators),
        ("Different Lookback Periods", test_different_lookback_periods),
        ("Multiple Symbols", test_multiple_symbols),
        ("Moving Averages", test_moving_averages),
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

