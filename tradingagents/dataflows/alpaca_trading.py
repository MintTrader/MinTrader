"""
Alpaca Trading API integration for paper trading.

This module provides functions to interact with Alpaca's Trading API
for placing orders, checking positions, and managing accounts.

All trading operations use paper trading mode by default for safety.
"""

from typing import Dict, List, Optional
import json
import logging
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from .alpaca_common import get_trading_client, is_paper_mode

# Get logger for this module
logger = logging.getLogger(__name__)


def get_account() -> Dict:
    """
    Get account information including buying power, cash, and portfolio value.
    
    Returns:
        Dictionary containing account information
    """
    try:
        client = get_trading_client()
        account = client.get_account()
        
        return {
            "id": account.id,
            "account_number": account.account_number,
            "status": account.status,
            "currency": account.currency,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "transfers_blocked": account.transfers_blocked,
            "account_blocked": account.account_blocked,
            "created_at": str(account.created_at),
            "paper_trading": is_paper_mode()
        }
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        return {"error": str(e)}


def get_positions() -> List[Dict]:
    """
    Get all open positions.
    
    Returns:
        List of dictionaries containing position information
    """
    try:
        client = get_trading_client()
        positions = client.get_all_positions()
        
        result = []
        for position in positions:
            result.append({
                "symbol": position.symbol,
                "qty": float(position.qty),
                "side": position.side,
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "market_value": float(position.market_value),
                "cost_basis": float(position.cost_basis),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "unrealized_intraday_pl": float(position.unrealized_intraday_pl),
                "unrealized_intraday_plpc": float(position.unrealized_intraday_plpc),
                "change_today": float(position.change_today)
            })
        
        return result
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return [{"error": str(e)}]


def get_position(symbol: str) -> Optional[Dict]:
    """
    Get position for a specific symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary containing position information or None if no position
    """
    try:
        client = get_trading_client()
        position = client.get_open_position(symbol)
        
        return {
            "symbol": position.symbol,
            "qty": float(position.qty),
            "side": position.side,
            "avg_entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "market_value": float(position.market_value),
            "cost_basis": float(position.cost_basis),
            "unrealized_pl": float(position.unrealized_pl),
            "unrealized_plpc": float(position.unrealized_plpc),
            "unrealized_intraday_pl": float(position.unrealized_intraday_pl),
            "unrealized_intraday_plpc": float(position.unrealized_intraday_plpc),
            "change_today": float(position.change_today)
        }
    except Exception as e:
        logger.error(f"Error fetching position for {symbol}: {e}")
        return None


def place_market_order(
    symbol: str,
    qty: float,
    side: str,
    time_in_force: str = "day"
) -> Dict:
    """
    Place a market order.
    
    Args:
        symbol: Stock ticker symbol
        qty: Quantity to trade
        side: "buy" or "sell"
        time_in_force: Order duration ("day", "gtc", "ioc", "fok")
        
    Returns:
        Dictionary containing order information
    """
    try:
        if not is_paper_mode():
            logger.warning("⚠️  WARNING: Placing order in LIVE trading mode!")
        
        client = get_trading_client()
        
        # Convert side to enum
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # Convert time_in_force to enum
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
        
        # Create order request
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif
        )
        
        # Submit order
        order = client.submit_order(order_data)
        
        return {
            "id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "type": order.type.value,
            "time_in_force": order.time_in_force.value,
            "status": order.status.value,
            "created_at": str(order.created_at),
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "paper_trading": is_paper_mode()
        }
    except Exception as e:
        logger.error(f"Error placing market order for {symbol}: {e}")
        return {"error": str(e)}


def place_limit_order(
    symbol: str,
    qty: float,
    side: str,
    limit_price: float,
    time_in_force: str = "day"
) -> Dict:
    """
    Place a limit order.
    
    Args:
        symbol: Stock ticker symbol
        qty: Quantity to trade
        side: "buy" or "sell"
        limit_price: Limit price for the order
        time_in_force: Order duration ("day", "gtc", "ioc", "fok")
        
    Returns:
        Dictionary containing order information
    """
    try:
        if not is_paper_mode():
            logger.warning("⚠️  WARNING: Placing order in LIVE trading mode!")
        
        client = get_trading_client()
        
        # Convert side to enum
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # Convert time_in_force to enum
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
        
        # Create order request
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
            limit_price=limit_price
        )
        
        # Submit order
        order = client.submit_order(order_data)
        
        return {
            "id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "type": order.type.value,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "time_in_force": order.time_in_force.value,
            "status": order.status.value,
            "created_at": str(order.created_at),
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "paper_trading": is_paper_mode()
        }
    except Exception as e:
        logger.error(f"Error placing limit order for {symbol}: {e}")
        return {"error": str(e)}


def get_orders(status: str = "open") -> List[Dict]:
    """
    Get orders filtered by status.
    
    Args:
        status: Order status filter ("open", "closed", "all")
        
    Returns:
        List of dictionaries containing order information
    """
    try:
        client = get_trading_client()
        
        # Create request
        if status.lower() == "all":
            request = GetOrdersRequest()
        else:
            request = GetOrdersRequest(status=status)
        
        orders = client.get_orders(request)
        
        result = []
        for order in orders:
            result.append({
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "type": order.type.value,
                "time_in_force": order.time_in_force.value,
                "status": order.status.value,
                "created_at": str(order.created_at),
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0
            })
        
        return result
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        return [{"error": str(e)}]


def cancel_order(order_id: str) -> Dict:
    """
    Cancel an open order.
    
    Args:
        order_id: The order ID to cancel
        
    Returns:
        Dictionary with cancellation status
    """
    try:
        client = get_trading_client()
        client.cancel_order_by_id(order_id)
        
        return {
            "success": True,
            "order_id": order_id,
            "message": "Order cancelled successfully"
        }
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        return {
            "success": False,
            "order_id": order_id,
            "error": str(e)
        }


def cancel_all_orders() -> Dict:
    """
    Cancel all open orders.
    
    Returns:
        Dictionary with cancellation status
    """
    try:
        client = get_trading_client()
        cancelled = client.cancel_orders()
        
        return {
            "success": True,
            "cancelled_count": len(cancelled),
            "message": f"Cancelled {len(cancelled)} orders"
        }
    except Exception as e:
        logger.error(f"Error cancelling all orders: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_market_clock() -> Dict:
    """
    Get current market status and clock information from Alpaca.
    
    Returns:
        Dictionary with market clock information including is_open status
    """
    try:
        client = get_trading_client()
        clock = client.get_clock()
        
        return {
            "timestamp": str(clock.timestamp),
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close)
        }
    except Exception as e:
        logger.error(f"Error fetching market clock: {e}")
        return {
            "error": str(e),
            "is_open": False  # Default to closed on error
        }


def get_open_orders() -> List[Dict]:
    """
    Get all open orders.
    
    Returns:
        List of dictionaries containing open order information
    """
    return get_orders(status="open")


def replace_order(order_id: str, qty: Optional[float] = None, 
                 limit_price: Optional[float] = None, 
                 stop_price: Optional[float] = None,
                 time_in_force: Optional[str] = None) -> Dict:
    """
    Replace/modify an existing order.
    
    Args:
        order_id: The order ID to replace
        qty: New quantity (optional)
        limit_price: New limit price (optional, for limit orders)
        stop_price: New stop price (optional, for stop orders)
        time_in_force: New time in force (optional)
        
    Returns:
        Dictionary with the new order information
    """
    try:
        from alpaca.trading.requests import ReplaceOrderRequest
        
        client = get_trading_client()
        
        # Build replacement request with only provided parameters
        replace_params = {}
        if qty is not None:
            replace_params['qty'] = qty
        if limit_price is not None:
            replace_params['limit_price'] = limit_price
        if stop_price is not None:
            replace_params['stop_price'] = stop_price
        if time_in_force is not None:
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK
            }
            replace_params['time_in_force'] = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
        
        request = ReplaceOrderRequest(**replace_params)
        order = client.replace_order_by_id(order_id, request)
        
        return {
            "success": True,
            "id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "type": order.type.value,
            "time_in_force": order.time_in_force.value,
            "status": order.status.value,
            "created_at": str(order.created_at),
            "updated_at": str(order.updated_at) if hasattr(order, 'updated_at') else None,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "message": "Order replaced successfully"
        }
    except Exception as e:
        logger.error(f"Error replacing order {order_id}: {e}")
        return {
            "success": False,
            "order_id": order_id,
            "error": str(e)
        }

