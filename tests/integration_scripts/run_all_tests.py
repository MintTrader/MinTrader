#!/usr/bin/env python3
"""
Master script to run all integration tests.
This will execute all tool validation scripts against real APIs.
"""

import sys
import os
import subprocess
from datetime import datetime

def print_header():
    """Print the master header."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "MINTRADER INTEGRATION TEST SUITE" + " "*30 + "║")
    print("║" + " "*20 + "Testing Against Real APIs" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\nTest Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def run_test_script(script_name, description):
    """Run a single test script and return the result."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*80)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        return result.returncode == 0
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR running {script_name}: {e}")
        return False

def main():
    """Run all integration tests."""
    print_header()
    
    # Define all test scripts
    tests = [
        ("test_core_stock_tools.py", "Core Stock Data Tools (OHLCV)"),
        ("test_technical_indicator_tools.py", "Technical Indicator Tools"),
        ("test_news_tools.py", "News Data Tools (Alpaca & Google)"),
        ("test_fundamental_tools.py", "Fundamental Data Tools (Financial Statements)"),
        ("test_trading_tools.py", "Trading Tools (Paper Trading Only)"),
    ]
    
    results = {}
    
    print("\nℹ️  Note: These tests use REAL API calls:")
    print("   - Alpaca APIs (paper trading for trades)")
    print("   - Google News")
    print("   - Yahoo Finance")
    print("   - OpenAI (if configured)")
    print()
    
    # Run each test
    for script_name, description in tests:
        results[description] = run_test_script(script_name, description)
    
    # Print final summary
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "FINAL SUMMARY" + " "*40 + "║")
    print("╚" + "="*78 + "╝\n")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*80)
    print(f"Total Test Suites: {total} | Passed: {passed} | Failed: {failed}")
    print("="*80)
    print(f"\nTest Run Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == total:
        print("\n🎉 ALL TEST SUITES PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST SUITE(S) FAILED")
        return 1

if __name__ == "__main__":
    exit(main())

