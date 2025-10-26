"""
Alpaca Portfolio Management Implementation

Vendor-specific implementation for portfolio management operations using Alpaca.
Returns raw position data - agent uses stock tools to analyze further.
"""

from typing import Dict, List, Any, Optional
import os
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Alpaca client
def _get_trading_client() -> TradingClient:
    """Get initialized Alpaca trading client"""
    api_key = os.getenv("ALPACA_API_KEY")
    # Support both ALPACA_API_SECRET and ALPACA_SECRET_KEY
    api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    if not api_key or not api_secret:
        raise ValueError(
            f"ALPACA_API_KEY and ALPACA_API_SECRET (or ALPACA_SECRET_KEY) must be set in environment. "
            f"Got API_KEY={'set' if api_key else 'NOT SET'}, "
            f"API_SECRET={'set' if api_secret else 'NOT SET'}"
        )
    
    return TradingClient(api_key, api_secret, paper=paper)

def _get_data_client() -> StockHistoricalDataClient:
    """Get initialized Alpaca data client"""
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in environment"
        )
    
    return StockHistoricalDataClient(api_key, api_secret)

def get_account() -> Dict[str, Any]:
    """Get account information from Alpaca"""
    client = _get_trading_client()
    account = client.get_account()
    return {
        'cash': float(account.cash),  # type: ignore
        'buying_power': float(account.buying_power),  # type: ignore
        'portfolio_value': float(account.portfolio_value),  # type: ignore
        'equity': float(account.equity),  # type: ignore
        'paper_trading': str(account.account_number).startswith('P')  # type: ignore
    }

def get_positions() -> List[Dict[str, Any]]:
    """Get all positions from Alpaca"""
    client = _get_trading_client()
    positions = client.get_all_positions()
    return [{  # type: ignore
        'symbol': pos.symbol,
        'qty': float(pos.qty),
        'side': pos.side,
        'market_value': float(pos.market_value),
        'cost_basis': float(pos.cost_basis),
        'unrealized_pl': float(pos.unrealized_pl),
        'unrealized_plpc': float(pos.unrealized_plpc),
        'current_price': float(pos.current_price),
        'avg_entry_price': float(pos.avg_entry_price),
        'change_today': float(pos.change_today) if hasattr(pos, 'change_today') and pos.change_today else 0.0
    } for pos in positions]

def get_open_orders() -> List[Dict[str, Any]]:
    """Get all open orders from Alpaca"""
    client = _get_trading_client()
    orders = client.get_orders()
    return [{  # type: ignore
        'id': order.id,
        'symbol': order.symbol,
        'qty': float(order.qty) if order.qty else None,
        'notional': float(order.notional) if order.notional else None,
        'side': order.side.value,
        'type': order.type.value,
        'status': order.status.value,
        'created_at': order.created_at.isoformat() if order.created_at else None,
    } for order in orders]

def place_market_order(symbol: str, qty: float, side: str) -> Dict[str, Any]:
    """Place a market order"""
    client = _get_trading_client()
    order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )
    order = client.submit_order(request)
    return {  # type: ignore
        'id': order.id,
        'symbol': order.symbol,
        'status': order.status.value
    }

def get_market_clock() -> Dict[str, Any]:
    """Get market clock information from Alpaca"""
    client = _get_trading_client()
    clock = client.get_clock()
    return {  # type: ignore
        'is_open': clock.is_open,
        'next_open': clock.next_open.isoformat() if clock.next_open else None,
        'next_close': clock.next_close.isoformat() if clock.next_close else None,
        'timestamp': clock.timestamp.isoformat() if clock.timestamp else None
    }




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


def get_alpaca_market_clock() -> Dict[str, Any]:
    """
    Get market clock information from Alpaca.
    
    Returns:
        Dictionary with market status and times
    """
    return get_market_clock()


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
    client = _get_trading_client()
    
    # Get orders - no status filter in API, filter manually
    raw_orders = client.get_orders()  # type: ignore
    
    # Convert to dict format
    orders = [{  # type: ignore
        'id': order.id,
        'symbol': order.symbol,
        'qty': float(order.qty) if order.qty else None,
        'side': order.side.value,
        'type': order.type.value,
        'status': order.status.value,
        'time_in_force': order.time_in_force.value if order.time_in_force else None,
        'created_at': order.created_at.isoformat() if order.created_at else None,
        'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
        'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
        'limit_price': float(order.limit_price) if order.limit_price else None,
        'stop_price': float(order.stop_price) if order.stop_price else None
    } for order in raw_orders]
    
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


def get_intraday_bars(symbol: str, timeframe: str = "5Min", limit: int = 24) -> List[Dict[str, Any]]:
    """
    Get intraday bars for a symbol from Alpaca.
    
    Args:
        symbol: Stock ticker symbol
        timeframe: Timeframe for bars (e.g., "1Min", "5Min", "15Min", "1Hour")
        limit: Maximum number of bars to fetch
        
    Returns:
        List of bar data with open, high, low, close, volume
    """
    client = _get_data_client()
    
    # Map timeframe string to TimeFrame enum
    timeframe_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day
    }
    
    tf = timeframe_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))
    
    # Calculate start time based on limit and timeframe
    # For 5-min bars, 24 bars = 2 hours
    now = datetime.now()
    if "Min" in timeframe:
        minutes_per_bar = int(timeframe.replace("Min", ""))
        hours_back = (limit * minutes_per_bar) / 60
        start = now - timedelta(hours=hours_back + 1)  # Add buffer
    elif "Hour" in timeframe:
        hours_per_bar = int(timeframe.replace("Hour", ""))
        start = now - timedelta(hours=limit * hours_per_bar + 24)  # Add buffer
    else:
        start = now - timedelta(days=limit + 5)  # Add buffer
    
    # Create request
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        limit=limit
    )
    
    # Fetch data
    try:
        bars_data = client.get_stock_bars(request)
        
        # Convert to list of dicts
        result = []
        if symbol in bars_data:
            for bar in bars_data[symbol]:
                result.append({
                    'timestamp': bar.timestamp.isoformat() if bar.timestamp else None,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
        
        return result
    except Exception as e:
        logger.error(f"Error getting intraday bars for {symbol}: {e}")
        return []


def get_daily_bars(symbol: str, limit: int = 60) -> List[Dict[str, Any]]:
    """
    Get daily bars for a symbol from Alpaca.
    
    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of bars to fetch (days)
        
    Returns:
        List of bar data with open, high, low, close, volume
    """
    client = _get_data_client()
    
    # Calculate start time
    now = datetime.now()
    start = now - timedelta(days=limit + 10)  # Add buffer for weekends/holidays
    
    # Create request
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        limit=limit
    )
    
    # Fetch data
    try:
        bars_data = client.get_stock_bars(request)
        
        # Convert to list of dicts
        result = []
        if symbol in bars_data:
            for bar in bars_data[symbol]:
                result.append({
                    'timestamp': bar.timestamp.isoformat() if bar.timestamp else None,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
        
        return result
    except Exception as e:
        logger.error(f"Error getting daily bars for {symbol}: {e}")
        return []
