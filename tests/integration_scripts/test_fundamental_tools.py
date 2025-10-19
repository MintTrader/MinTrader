#!/usr/bin/env python3
"""
Test script for fundamental data tools.
Tests financial statement and fundamental data retrieval with real API calls.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from tradingagents.dataflows.y_finance import (
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.dataflows.openai import get_fundamentals_openai

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_balance_sheet():
    """Test balance sheet data retrieval."""
    print_section("TEST: Balance Sheet Data")
    
    ticker = "AAPL"
    freq = "quarterly"
    # Use yesterday's date for consistency (though fundamentals don't have 15-min restriction)
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Testing with:")
    print(f"  Ticker: {ticker}")
    print(f"  Frequency: {freq}")
    print(f"  Current Date: {curr_date}")
    print()
    
    try:
        result = get_balance_sheet(ticker, freq, curr_date)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: Balance sheet data retrieved")
            print(f"\nFirst 600 characters of result:")
            print("-" * 80)
            print(result[:600])
            print("-" * 80)
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

def test_cashflow_statement():
    """Test cash flow statement data retrieval."""
    print_section("TEST: Cash Flow Statement Data")
    
    ticker = "AAPL"
    freq = "quarterly"
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Testing with:")
    print(f"  Ticker: {ticker}")
    print(f"  Frequency: {freq}")
    print(f"  Current Date: {curr_date}")
    print()
    
    try:
        result = get_cashflow(ticker, freq, curr_date)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: Cash flow statement data retrieved")
            print(f"\nFirst 600 characters of result:")
            print("-" * 80)
            print(result[:600])
            print("-" * 80)
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

def test_income_statement():
    """Test income statement data retrieval."""
    print_section("TEST: Income Statement Data")
    
    ticker = "AAPL"
    freq = "quarterly"
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Testing with:")
    print(f"  Ticker: {ticker}")
    print(f"  Frequency: {freq}")
    print(f"  Current Date: {curr_date}")
    print()
    
    try:
        result = get_income_statement(ticker, freq, curr_date)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: Income statement data retrieved")
            print(f"\nFirst 600 characters of result:")
            print("-" * 80)
            print(result[:600])
            print("-" * 80)
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

def test_multiple_tickers():
    """Test fundamental data for multiple tickers."""
    print_section("TEST: Multiple Tickers - Balance Sheet")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    freq = "quarterly"
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    results = {}
    
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        try:
            result = get_balance_sheet(ticker, freq, curr_date)
            if result and len(result) > 100:  # Check if we got meaningful data
                print(f"  ✓ {ticker}: SUCCESS")
                results[ticker] = True
            else:
                print(f"  ✗ {ticker}: FAILED (empty or insufficient data)")
                results[ticker] = False
        except Exception as e:
            print(f"  ✗ {ticker}: FAILED ({str(e)[:50]})")
            results[ticker] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(tickers)}")
    
    return success_count == len(tickers)

def test_frequencies():
    """Test different reporting frequencies."""
    print_section("TEST: Different Frequencies - Income Statement")
    
    ticker = "AAPL"
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    frequencies = ["quarterly", "annual"]
    
    results = {}
    
    for freq in frequencies:
        print(f"\nTesting {freq} frequency...")
        try:
            result = get_income_statement(ticker, freq, curr_date)
            if result and len(result) > 100:
                print(f"  ✓ {freq}: SUCCESS")
                results[freq] = True
            else:
                print(f"  ✗ {freq}: FAILED")
                results[freq] = False
        except Exception as e:
            print(f"  ✗ {freq}: FAILED ({str(e)[:50]})")
            results[freq] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(frequencies)}")
    
    return success_count == len(frequencies)

def test_all_statements_single_ticker():
    """Test all financial statements for a single ticker."""
    print_section("TEST: All Financial Statements - Single Ticker")
    
    ticker = "MSFT"
    freq = "quarterly"
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    statements = {
        "Balance Sheet": get_balance_sheet,
        "Cash Flow": get_cashflow,
        "Income Statement": get_income_statement
    }
    
    results = {}
    
    print(f"Testing all statements for {ticker}...\n")
    
    for name, func in statements.items():
        print(f"\nTesting {name}...")
        try:
            result = func(ticker, freq, curr_date)
            if result and len(result) > 100:
                print(f"  ✓ {name}: SUCCESS")
                results[name] = True
            else:
                print(f"  ✗ {name}: FAILED")
                results[name] = False
        except Exception as e:
            print(f"  ✗ {name}: FAILED ({str(e)[:50]})")
            results[name] = False
    
    print("\n" + "-" * 80)
    print("Summary:")
    success_count = sum(1 for v in results.values() if v)
    print(f"  Successful: {success_count}/{len(statements)}")
    
    return success_count == len(statements)

def test_fundamentals_openai():
    """Test OpenAI-based fundamentals (if API key is available)."""
    print_section("TEST: OpenAI Fundamentals (Optional)")
    
    ticker = "AAPL"
    curr_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Testing with:")
    print(f"  Ticker: {ticker}")
    print(f"  Current Date: {curr_date}")
    print()
    
    print("Note: This test requires OPENAI_API_KEY to be set")
    print("      Will skip gracefully if not available\n")
    
    try:
        result = get_fundamentals_openai(ticker, curr_date)
        
        if result and len(result) > 0:
            print("✓ SUCCESS: OpenAI fundamentals data retrieved")
            print(f"\nFirst 400 characters of result:")
            print("-" * 80)
            print(result[:400])
            print("-" * 80)
            return True
        else:
            print("⚠ WARNING: Empty result (might be API key issue)")
            return True  # Don't fail if API key not configured
            
    except Exception as e:
        error_msg = str(e).lower()
        if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            print(f"⚠ SKIPPED: OpenAI API not configured ({str(e)[:100]})")
            return True  # Don't fail on missing API key
        else:
            print(f"✗ FAILED: Exception occurred")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*18 + "FUNDAMENTAL DATA TOOLS TEST SUITE" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Balance Sheet", test_balance_sheet),
        ("Cash Flow Statement", test_cashflow_statement),
        ("Income Statement", test_income_statement),
        ("Multiple Tickers", test_multiple_tickers),
        ("Different Frequencies", test_frequencies),
        ("All Statements Single Ticker", test_all_statements_single_ticker),
        ("OpenAI Fundamentals (Optional)", test_fundamentals_openai),
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

