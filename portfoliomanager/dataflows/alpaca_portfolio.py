"""
Alpaca Portfolio Management Implementation

Vendor-specific implementation for portfolio management operations using Alpaca.
Returns raw position data - agent uses stock tools to analyze further.
"""

from typing import Dict, List, Any, Optional
from tradingagents.dataflows.alpaca_trading import (
    get_account,
    get_positions,
    place_market_order,
    get_open_orders,
    cancel_order,
    replace_order
)


def get_alpaca_account_info() -> Dict[str, Any]:
    """
    Get account information from Alpaca.
    
    Returns:
        Dictionary with account details
    """
    account = get_account()
    return {
        'cash': account.get('cash', 0),
        'buying_power': account.get('buying_power', 0),
        'portfolio_value': account.get('portfolio_value', 0),
        'equity': account.get('equity', 0),
        'paper_trading': account.get('paper_trading', True)
    }


def get_alpaca_positions() -> List[Dict[str, Any]]:
    """
    Get current positions from Alpaca.
    Returns raw position data - agent can use stock tools to analyze further.
    
    Returns:
        List of positions with basic information
    """
    positions = get_positions()
    
    # Return simplified position data
    result = []
    for pos in positions:
        result.append({
            'ticker': pos['symbol'],
            'qty': pos['qty'],
            'side': pos.get('side', 'long'),
            'avg_entry_price': pos['avg_entry_price'],
            'current_price': pos['current_price'],
            'market_value': pos['market_value'],
            'cost_basis': pos['cost_basis'],
            'unrealized_pl': pos['unrealized_pl'],
            'unrealized_pl_pct': pos['unrealized_plpc'] * 100,  # Convert to percentage
            'change_today': pos.get('change_today', 0)
        })
    
    return result


def get_alpaca_position_details(ticker: str) -> Dict[str, Any]:
    """
    Get detailed position information from Alpaca for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Position details or error
    """
    positions = get_alpaca_positions()
    
    for pos in positions:
        if pos['ticker'].upper() == ticker.upper():
            return pos
    
    return {'error': f'No position found for {ticker}'}


def execute_alpaca_trade(
    ticker: str,
    action: str,
    quantity: int,
    reasoning: str
) -> Dict[str, Any]:
    """
    Execute a trade through Alpaca.
    
    Args:
        ticker: Stock ticker symbol
        action: "BUY" or "SELL"
        quantity: Number of shares
        reasoning: Explanation for the trade
        
    Returns:
        Trade execution result
    """
    if action.upper() not in ['BUY', 'SELL']:
        return {'error': f'Invalid action: {action}. Must be BUY or SELL'}
    
    if quantity <= 0:
        return {'error': f'Invalid quantity: {quantity}. Must be greater than 0'}
    
    # Execute the trade
    result = place_market_order(
        symbol=ticker,
        qty=quantity,
        side=action.lower()
    )
    
    if 'error' in result:
        return {'error': result['error'], 'ticker': ticker, 'action': action}
    
    return {
        'success': True,
        'ticker': ticker,
        'action': action,
        'quantity': quantity,
        'order_id': result.get('id', ''),
        'reasoning': reasoning
    }


def get_alpaca_open_orders() -> List[Dict[str, Any]]:
    """
    Get all open orders from Alpaca.
    
    Returns:
        List of open orders with details
    """
    orders = get_open_orders()
    
    # Format orders for display
    result = []
    for order in orders:
        if 'error' in order:
            continue
            
        result.append({
            'order_id': order.get('id', ''),
            'ticker': order.get('symbol', ''),
            'side': order.get('side', '').upper(),
            'qty': order.get('qty', 0),
            'order_type': order.get('type', ''),
            'status': order.get('status', ''),
            'time_in_force': order.get('time_in_force', ''),
            'created_at': order.get('created_at', ''),
            'filled_qty': order.get('filled_qty', 0),
            'filled_avg_price': order.get('filled_avg_price'),
            'limit_price': order.get('limit_price'),
            'stop_price': order.get('stop_price')
        })
    
    return result


def get_alpaca_all_orders(status: str = "all") -> List[Dict[str, Any]]:
    """
    Get all orders from Alpaca with specified status filter.
    
    Args:
        status: Order status filter - 'open', 'closed', or 'all'
        
    Returns:
        List of orders with details
    """
    from tradingagents.dataflows.alpaca_trading import get_orders
    
    orders = get_orders(status=status)
    
    # Format orders for display
    result = []
    for order in orders:
        if 'error' in order:
            continue
            
        result.append({
            'id': order.get('id', ''),
            'symbol': order.get('symbol', ''),
            'side': order.get('side', '').upper(),
            'qty': order.get('qty', 0),
            'type': order.get('type', ''),
            'status': order.get('status', ''),
            'time_in_force': order.get('time_in_force', ''),
            'created_at': order.get('created_at', ''),
            'filled_qty': order.get('filled_qty', 0),
            'filled_avg_price': order.get('filled_avg_price'),
            'limit_price': order.get('limit_price'),
            'stop_price': order.get('stop_price')
        })
    
    return result


def cancel_alpaca_order(order_id: str) -> Dict[str, Any]:
    """
    Cancel an open order in Alpaca.
    
    Args:
        order_id: The order ID to cancel
        
    Returns:
        Cancellation result
    """
    result = cancel_order(order_id)
    return result


def modify_alpaca_order(
    order_id: str,
    qty: Optional[float] = None,
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: Optional[str] = None
) -> Dict[str, Any]:
    """
    Modify an existing order in Alpaca.
    
    Args:
        order_id: The order ID to modify
        qty: New quantity (optional)
        limit_price: New limit price (optional)
        stop_price: New stop price (optional)
        time_in_force: New time in force (optional)
        
    Returns:
        Modified order result
    """
    result = replace_order(
        order_id=order_id,
        qty=qty,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force
    )
    return result

