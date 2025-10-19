#!/usr/bin/env python3
"""
Test script for Alpaca trading tools.
Tests paper trading functions with real Alpaca API calls.

⚠️  WARNING: This script uses PAPER TRADING mode only.
    It will NOT place real trades or use real money.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from tradingagents.dataflows.alpaca_trading import (
    get_account,
    get_positions,
    get_position,
    place_market_order,
    place_limit_order,
    get_orders,
    cancel_order,
    cancel_all_orders
)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_warning():
    """Print warning about paper trading."""
    print("\n" + "⚠"*40)
    print("   PAPER TRADING MODE - NO REAL MONEY INVOLVED")
    print("   All trades are simulated for testing purposes")
    print("⚠"*40 + "\n")

def test_get_account():
    """Test getting account information."""
    print_section("TEST: Get Account Information")
    
    try:
        account = get_account()
        
        if "error" in account:
            print(f"✗ FAILED: {account['error']}")
            return False
        
        print("✓ SUCCESS: Account information retrieved")
        print("\nAccount Details:")
        print("-" * 80)
        print(f"  Account Number: {account.get('account_number', 'N/A')}")
        print(f"  Status: {account.get('status', 'N/A')}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"  Paper Trading: {account.get('paper_trading', 'Unknown')}")
        print(f"  Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
        print(f"  Trading Blocked: {account.get('trading_blocked', 'N/A')}")
        print("-" * 80)
        
        # Verify paper trading mode
        if not account.get('paper_trading', False):
            print("\n⚠️  WARNING: Not in paper trading mode!")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_positions():
    """Test getting all positions."""
    print_section("TEST: Get All Positions")
    
    try:
        positions = get_positions()
        
        if positions and len(positions) > 0 and "error" in positions[0]:
            print(f"✗ FAILED: {positions[0]['error']}")
            return False
        
        print(f"✓ SUCCESS: Retrieved {len(positions)} position(s)")
        
        if len(positions) > 0:
            print("\nCurrent Positions:")
            print("-" * 80)
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('qty', 0)
                current_price = pos.get('current_price', 0)
                market_value = pos.get('market_value', 0)
                unrealized_pl = pos.get('unrealized_pl', 0)
                
                print(f"  {symbol}: {qty} shares @ ${current_price:.2f}")
                print(f"    Market Value: ${market_value:.2f}")
                print(f"    Unrealized P/L: ${unrealized_pl:.2f}")
                print()
            print("-" * 80)
        else:
            print("\nNo open positions found (this is normal for a new account)")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_orders():
    """Test getting orders."""
    print_section("TEST: Get Orders")
    
    statuses = ["open", "closed", "all"]
    results = {}
    
    for status in statuses:
        print(f"\nGetting {status} orders...")
        try:
            orders = get_orders(status)
            
            if orders and len(orders) > 0 and "error" in orders[0]:
                print(f"  ✗ {status}: FAILED - {orders[0]['error']}")
                results[status] = False
            else:
                print(f"  ✓ {status}: SUCCESS - Found {len(orders)} order(s)")
                results[status] = True
                
                if len(orders) > 0 and len(orders) <= 3:  # Show details for first few
                    for order in orders[:3]:
                        print(f"    - {order.get('symbol')} {order.get('side')} {order.get('qty')} @ {order.get('type')} - {order.get('status')}")
        except Exception as e:
            print(f"  ✗ {status}: FAILED - {str(e)[:50]}")
            results[status] = False
    
    success_count = sum(1 for v in results.values() if v)
    return success_count == len(statuses)

def test_place_market_order():
    """Test placing a market order (will be executed immediately in paper trading)."""
    print_section("TEST: Place Market Order")
    
    print("⚠️  This will place a PAPER TRADING order")
    print("   It will NOT use real money")
    print()
    
    symbol = "AAPL"
    qty = 1
    side = "buy"
    
    print(f"Placing order:")
    print(f"  Symbol: {symbol}")
    print(f"  Quantity: {qty}")
    print(f"  Side: {side}")
    print(f"  Type: Market")
    print()
    
    try:
        result = place_market_order(symbol, qty, side)
        
        if "error" in result:
            print(f"✗ FAILED: {result['error']}")
            return False
        
        print("✓ SUCCESS: Market order placed")
        print("\nOrder Details:")
        print("-" * 80)
        print(f"  Order ID: {result.get('id', 'N/A')}")
        print(f"  Symbol: {result.get('symbol', 'N/A')}")
        print(f"  Quantity: {result.get('qty', 'N/A')}")
        print(f"  Side: {result.get('side', 'N/A')}")
        print(f"  Status: {result.get('status', 'N/A')}")
        print(f"  Paper Trading: {result.get('paper_trading', 'N/A')}")
        print("-" * 80)
        
        # Store order ID for potential cancellation
        if 'id' in result:
            return result['id']
        return True
        
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_place_limit_order():
    """Test placing a limit order."""
    print_section("TEST: Place Limit Order")
    
    print("⚠️  This will place a PAPER TRADING limit order")
    print("   It will NOT use real money")
    print()
    
    symbol = "AAPL"
    qty = 1
    side = "buy"
    limit_price = 100.00  # Set below market to avoid execution
    
    print(f"Placing limit order:")
    print(f"  Symbol: {symbol}")
    print(f"  Quantity: {qty}")
    print(f"  Side: {side}")
    print(f"  Limit Price: ${limit_price}")
    print()
    
    try:
        result = place_limit_order(symbol, qty, side, limit_price)
        
        if "error" in result:
            print(f"✗ FAILED: {result['error']}")
            return False
        
        print("✓ SUCCESS: Limit order placed")
        print("\nOrder Details:")
        print("-" * 80)
        print(f"  Order ID: {result.get('id', 'N/A')}")
        print(f"  Symbol: {result.get('symbol', 'N/A')}")
        print(f"  Quantity: {result.get('qty', 'N/A')}")
        print(f"  Side: {result.get('side', 'N/A')}")
        print(f"  Limit Price: ${result.get('limit_price', 'N/A')}")
        print(f"  Status: {result.get('status', 'N/A')}")
        print(f"  Paper Trading: {result.get('paper_trading', 'N/A')}")
        print("-" * 80)
        
        # Return order ID for potential cancellation
        if 'id' in result:
            return result['id']
        return True
        
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cancel_all_orders():
    """Test cancelling all orders."""
    print_section("TEST: Cancel All Orders")
    
    print("Attempting to cancel all open orders...")
    print()
    
    try:
        result = cancel_all_orders()
        
        if not result.get('success', False):
            print(f"⚠ Note: {result.get('error', 'Unknown error')}")
            # Don't fail if there are no orders to cancel
            if 'no orders' in str(result.get('error', '')).lower():
                print("  (This is normal if no orders exist)")
                return True
            return False
        
        cancelled_count = result.get('cancelled_count', 0)
        print(f"✓ SUCCESS: Cancelled {cancelled_count} order(s)")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: Exception occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRADING TOOLS TEST SUITE" + " "*33 + "║")
    print("║" + " "*25 + "(PAPER TRADING ONLY)" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    print_warning()
    
    tests = [
        ("Get Account Information", test_get_account),
        ("Get Positions", test_get_positions),
        ("Get Orders", test_get_orders),
        ("Place Market Order", test_place_market_order),
        ("Place Limit Order", test_place_limit_order),
        ("Cancel All Orders", test_cancel_all_orders),
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
    
    print_warning()
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    exit(main())

