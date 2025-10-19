"""
Portfolio Management Tools

Tools for portfolio agents using vendor routing pattern.
Similar to tradingagents/agents/utils/core_stock_tools.py
"""

from langchain_core.tools import tool
from typing import Annotated, Dict, List, Any, Optional
from portfoliomanager.dataflows.portfolio_interface import route_to_vendor


@tool
def get_account_info() -> Dict[str, Any]:
    """
    Get current account information including balance, buying power, and portfolio value.
    Uses the configured broker vendor (default: Alpaca).
    
    Returns:
        Dictionary with account details: cash, buying_power, portfolio_value, equity, paper_trading
    """
    return route_to_vendor("get_account_info")


@tool
def get_current_positions() -> List[Dict[str, Any]]:
    """
    Get all current portfolio positions with basic information.
    Uses the configured broker vendor (default: Alpaca).
    
    Returns basic position data: ticker, quantity, entry_price, current_price, P&L.
    
    For deeper analysis (holding periods, momentum, technical indicators), use:
    - get_stock_data() to fetch historical prices and calculate holding period
    - get_indicators() to analyze momentum and technical patterns
    
    Returns:
        List of positions with ticker, qty, avg_entry_price, current_price, 
        unrealized_pl, unrealized_pl_pct, market_value, cost_basis
    """
    return route_to_vendor("get_current_positions")


@tool
def get_position_details(
    ticker: Annotated[str, "Stock ticker symbol"]
) -> Dict[str, Any]:
    """
    Get detailed information about a specific position.
    Uses the configured broker vendor (default: Alpaca).
    
    Returns basic position data. For analysis, use stock tools:
    - get_stock_data() to see price history
    - get_indicators() to check momentum and technicals
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA)
        
    Returns:
        Position details including entry price, current price, P&L
    """
    return route_to_vendor("get_position_details", ticker)


@tool
def execute_trade(
    ticker: Annotated[str, "Stock ticker symbol"],
    action: Annotated[str, "Trading action: BUY or SELL"],
    quantity: Annotated[int, "Number of shares to trade"],
    reasoning: Annotated[str, "Explanation for why this trade should be executed"]
) -> Dict[str, Any]:
    """
    Execute a trade (BUY or SELL) through the configured broker.
    Uses the configured broker vendor (default: Alpaca).
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, NVDA)
        action: "BUY" or "SELL"
        quantity: Number of shares (must be positive integer)
        reasoning: Detailed explanation for why this trade maximizes profits
        
    Returns:
        Trade execution result with order_id or error message
    """
    return route_to_vendor("execute_trade", ticker, action, quantity, reasoning)


@tool
def get_last_iteration_summary() -> str:
    """
    Get the summary from the last portfolio management iteration.
    This helps the agent learn from previous decisions.
    
    Returns:
        Summary text describing what was done in the last iteration,
        or message indicating this is the first run
    """
    from portfoliomanager.dataflows.s3_client import S3ReportManager
    from portfoliomanager.config import PORTFOLIO_CONFIG
    
    s3_client = S3ReportManager(
        str(PORTFOLIO_CONFIG['s3_bucket_name']),  # type: ignore
        str(PORTFOLIO_CONFIG['s3_region'])  # type: ignore
    )
    
    summary = s3_client.get_last_summary()
    return summary if summary else "No previous iteration summary found (first run)"


@tool
def get_watchlist_stocks() -> List[str]:
    """
    Get the list of stocks in the watchlist.
    These are stocks configured for monitoring and potential investment.
    
    Returns:
        List of ticker symbols in the watchlist (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    """
    from portfoliomanager.config import PORTFOLIO_CONFIG
    watchlist = PORTFOLIO_CONFIG.get('watchlist', [])
    return list(watchlist) if watchlist else []  # type: ignore


@tool
def screen_new_opportunities(
    max_picks: Annotated[int, "Maximum number of stocks to return"] = 3
) -> List[str]:
    """
    Screen for new investment opportunities with positive momentum.
    Identifies stocks meeting screening criteria (volume, price, momentum).
    
    Args:
        max_picks: Maximum number of stocks to return (default: 3)
        
    Returns:
        List of ticker symbols that meet screening criteria and show upward momentum
    """
    from portfoliomanager.agents.stock_screener import StockScreener
    from portfoliomanager.config import PORTFOLIO_CONFIG
    
    # Get existing positions to avoid duplicates
    positions = get_current_positions.invoke({})
    existing_tickers = [p['ticker'] for p in positions]
    
    # Create screener and find opportunities
    screener = StockScreener(PORTFOLIO_CONFIG)
    opportunities = screener.screen_opportunities(existing_tickers=existing_tickers)
    
    return opportunities[:max_picks]


@tool
def get_trading_constraints() -> Dict[str, Any]:
    """
    Get the current trading constraints and limits.
    These constraints must be respected when making trading decisions.
    
    Returns:
        Dictionary of trading constraints including:
        - max_position_size_pct: Maximum percentage of portfolio per position
        - max_trades_per_day: Maximum trades allowed per day
        - min_cash_reserve_pct: Minimum cash to keep as reserve
        - stop_loss_pct: Automatic stop-loss threshold
        - min_holding_days: Minimum days to hold before selling (unless stop-loss)
        - strategy_objective: Overall strategy goal
        - trading_style: Trading time horizon
    """
    from portfoliomanager.config import PORTFOLIO_CONFIG
    
    return {
        'max_position_size_pct': PORTFOLIO_CONFIG.get('max_position_size_pct', 10),
        'max_trades_per_day': PORTFOLIO_CONFIG.get('max_trades_per_day', 10),
        'min_cash_reserve_pct': PORTFOLIO_CONFIG.get('min_cash_reserve_pct', 5),
        'stop_loss_pct': PORTFOLIO_CONFIG.get('stop_loss_pct', 5),
        'min_holding_days': PORTFOLIO_CONFIG.get('min_holding_days', 7),
        'strategy_objective': PORTFOLIO_CONFIG.get('strategy_objective', 'maximize_profits'),
        'trading_style': PORTFOLIO_CONFIG.get('trading_style', 'medium_term')
    }


@tool
def get_open_orders() -> List[Dict[str, Any]]:
    """
    Get all open orders from the broker.
    Uses the configured broker vendor (default: Alpaca).
    
    Returns:
        List of open orders with details including:
        - order_id: Unique order identifier
        - ticker: Stock symbol
        - side: BUY or SELL
        - qty: Quantity of shares
        - order_type: Type of order (market, limit, stop, etc.)
        - status: Order status
        - time_in_force: How long order is valid
        - limit_price: Limit price (for limit orders)
        - stop_price: Stop price (for stop orders)
    """
    return route_to_vendor("get_open_orders")


@tool
def get_all_orders(status: Annotated[str, "Order status filter: 'open', 'closed', or 'all'"] = "all") -> str:
    """
    Get all orders from the broker with the specified status, formatted as a table.
    Uses the configured broker vendor (default: Alpaca).
    
    Order statuses include:
    - open: Orders that are currently active (new, partially_filled, accepted, pending_new)
    - closed: Orders that are no longer active (filled, canceled, expired, replaced, rejected)
    - all: All orders regardless of status
    
    Args:
        status: Filter by status - 'open', 'closed', or 'all' (default: 'all')
        
    Returns:
        Formatted table string showing all orders with their details
    """
    orders = route_to_vendor("get_all_orders", status)
    
    if not orders or len(orders) == 0:
        return f"No {status} orders found."
    
    # Build a formatted table
    lines = []
    lines.append(f"\n{'='*150}")
    lines.append(f"ALL ORDERS (Status: {status.upper()})")
    lines.append(f"{'='*150}")
    lines.append(f"{'Ticker':<8} | {'Side':<4} | {'Qty':<8} | {'Type':<10} | {'Status':<15} | {'Limit $':<10} | {'Stop $':<10} | {'Filled':<8} | {'TIF':<5} | {'Order ID':<36}")
    lines.append(f"{'-'*150}")
    
    for order in orders:
        if 'error' in order:
            continue
        
        ticker = order.get('symbol', order.get('ticker', 'N/A'))[:8]
        side = order.get('side', '').upper()[:4]
        qty = f"{float(order.get('qty', 0)):.2f}"[:8]
        order_type = order.get('type', order.get('order_type', 'N/A'))[:10]
        status_val = order.get('status', 'N/A')[:15]
        
        limit_price = order.get('limit_price')
        limit_str = f"${float(limit_price):.2f}" if limit_price else "-"
        limit_str = limit_str[:10]
        
        stop_price = order.get('stop_price')
        stop_str = f"${float(stop_price):.2f}" if stop_price else "-"
        stop_str = stop_str[:10]
        
        filled_qty = order.get('filled_qty', 0)
        filled_str = f"{float(filled_qty):.2f}"[:8]
        
        tif = order.get('time_in_force', 'N/A')[:5]
        order_id = order.get('id', order.get('order_id', 'N/A'))[:36]
        
        lines.append(f"{ticker:<8} | {side:<4} | {qty:<8} | {order_type:<10} | {status_val:<15} | {limit_str:<10} | {stop_str:<10} | {filled_str:<8} | {tif:<5} | {order_id}")
    
    lines.append(f"{'='*150}")
    lines.append(f"Total Orders: {len(orders)}\n")
    
    return "\n".join(lines)


@tool
def cancel_order(
    order_id: Annotated[str, "Order ID to cancel"]
) -> Dict[str, Any]:
    """
    Cancel an open order.
    Uses the configured broker vendor (default: Alpaca).
    
    Args:
        order_id: The unique order ID to cancel
        
    Returns:
        Cancellation result with success status
    """
    return route_to_vendor("cancel_order", order_id)


@tool
def modify_order(
    order_id: Annotated[str, "Order ID to modify"],
    qty: Annotated[Optional[float], "New quantity (optional)"] = None,
    limit_price: Annotated[Optional[float], "New limit price (optional)"] = None,
    stop_price: Annotated[Optional[float], "New stop price (optional)"] = None,
    time_in_force: Annotated[Optional[str], "New time in force: 'day', 'gtc', 'ioc', 'fok' (optional)"] = None
) -> Dict[str, Any]:
    """
    Modify an existing open order.
    Uses the configured broker vendor (default: Alpaca).
    
    At least one parameter besides order_id must be provided.
    
    Args:
        order_id: The unique order ID to modify
        qty: New quantity of shares (optional)
        limit_price: New limit price for limit orders (optional)
        stop_price: New stop price for stop orders (optional)
        time_in_force: New time in force - 'day', 'gtc', 'ioc', 'fok' (optional)
        
    Returns:
        Modified order result
    """
    return route_to_vendor("modify_order", order_id, qty, limit_price, stop_price, time_in_force)

