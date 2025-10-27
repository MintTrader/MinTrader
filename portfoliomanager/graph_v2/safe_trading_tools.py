"""
Safe Trading Tools for Portfolio Manager

This module filters Alpaca MCP tools to only allow safe operations and adds
a validated bracket order tool for placing trades with mandatory exit strategies.

Approach:
1. Filter existing Alpaca MCP tools to only include read operations
2. Add a new @tool for placing bracket orders with validation
3. Block dangerous operations (shorts, naked orders without exits)
"""

from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import tool
import logging
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

logger = logging.getLogger(__name__)


# ==================== Safe Tool Names (Read Operations Only) ====================

ALLOWED_TOOL_NAMES = {
    # Account & Portfolio (Read Only)
    "get_account",
    "get_positions",
    "get_position",
    "get_open_orders",
    "get_orders",
    "get_order",
    
    # Market Data (Read Only)
    "get_stock_quote",
    "get_stock_bars",
    "get_stock_latest_bar",
    "get_stock_latest_quote",
    "get_stock_latest_trade",
    "get_stock_snapshot",
    "get_market_clock",
    
    # Historical Data (Read Only)
    "get_stock_historical_bars",
    "get_stock_historical_quotes",
    "get_stock_historical_trades",
    
}


def filter_safe_tools(all_tools: List) -> List:
    """
    Filter Alpaca MCP tools to only include safe read operations.
    
    Args:
        all_tools: List of all available Alpaca MCP tools
        
    Returns:
        Filtered list of safe tools
    """
    safe_tools = []
    
    for tool in all_tools:
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        
        if tool_name in ALLOWED_TOOL_NAMES:
            safe_tools.append(tool)
            logger.debug(f"✓ Allowed tool: {tool_name}")
        else:
            logger.debug(f"✗ Blocked tool: {tool_name}")
    
    logger.info(f"Filtered tools: {len(safe_tools)}/{len(all_tools)} allowed")
    
    return safe_tools


# ==================== Bracket Order Tool (With Validation) ====================

@tool
def place_buy_bracket_order(
    symbol: str,
    qty: float,
    stop_loss_price: float,
    take_profit_price: float,
    type: str = "market",
    limit_price: Optional[float] = None,
    time_in_force: str = "day",
    extended_hours: bool = False
) -> Dict[str, Any]:
    """
    Place a BUY bracket order with mandatory stop-loss and take-profit.
    
    A bracket order includes:
    - Entry order (market or limit) - BUY ONLY
    - Automatic stop-loss order
    - Automatic take-profit order
    
    When one exit order triggers, the other is automatically cancelled (OCO).
    
    LONG POSITIONS ONLY - This function ONLY places BUY orders. Short selling is not supported.
    
    ⚠️ CRITICAL: BRACKET ORDERS ONLY SUPPORT WHOLE SHARE QUANTITIES (qty parameter)
    - You MUST use qty (number of shares) - NOT notional (dollar amount)
    - Alpaca does not allow fractional/notional orders with bracket orders
    - Calculate shares: qty = round(dollar_amount / current_price)
    - Always use whole numbers for qty (e.g., 10, 25, 100)
    
    EXTENDED HOURS TRADING:
    - Set extended_hours=True to trade outside regular hours (4am-8pm ET)
    - Extended hours requires: type="limit", time_in_force="day", limit_price
    - Regular hours (9:30am-4pm ET): Can use market or limit orders
    
    Args:
        symbol: Stock ticker (e.g., "AAPL")
        qty: Number of WHOLE shares to buy (REQUIRED - cannot use dollar amounts)
        stop_loss_price: Price to sell if market drops (REQUIRED)
        take_profit_price: Price to sell if market rises (REQUIRED)
        type: Order type - "market" (regular hours only) or "limit"
        limit_price: Limit price if type="limit" or extended_hours=True
        time_in_force: "day" (required for extended hours), "gtc", "ioc", "fok"
        extended_hours: True to trade in extended hours (pre/after market)
        
    Returns:
        Order result from Alpaca API
        
    Examples:
        # Buy 27 shares of AAPL (calculated from ~$5000 budget at $180/share)
        # Current price: $180
        place_buy_bracket_order(
            symbol="AAPL",
            qty=27,                    # MUST be whole number
            stop_loss_price=171.00,    # -5%
            take_profit_price=198.00,  # +10%
            type="market"
        )
        
        # Buy 50 shares of MSFT with specific entry price
        place_buy_bracket_order(
            symbol="MSFT",
            qty=50,                    # MUST be whole number
            type="limit",
            limit_price=380.00,
            stop_loss_price=361.00,    # -5%
            take_profit_price=418.00   # +10%
        )
        
        # Extended hours trading (pre-market or after-hours)
        place_buy_bracket_order(
            symbol="GOOGL",
            qty=20,                    # MUST be whole number (calculated from budget)
            type="limit",              # REQUIRED for extended hours
            limit_price=145.50,
            stop_loss_price=138.25,
            take_profit_price=159.00,
            time_in_force="day",       # REQUIRED for extended hours
            extended_hours=True        # Enable extended hours
        )
    """
    
    # ==================== Validation ====================
    
    # Hardcode side to BUY only
    side = "buy"
    
    # 2. Require qty (bracket orders don't support notional)
    if qty is None or qty <= 0:
        error = "qty (number of shares) is REQUIRED and must be positive. Bracket orders do not support notional (dollar amount)."
        logger.error(error)
        return {"error": error, "status": "rejected"}
    
    # Ensure qty is a whole number (no fractional shares in bracket orders)
    if qty != int(qty):
        error = f"qty must be a whole number (got {qty}). Bracket orders do not support fractional shares."
        logger.error(error)
        return {"error": error, "status": "rejected"}
    
    # 3. Require stop-loss (already required in function signature, but validate)
    if stop_loss_price is None or stop_loss_price <= 0:
        error = "stop_loss_price is REQUIRED for all orders and must be positive. Never trade without a stop-loss!"
        logger.error(error)
        return {"error": error, "status": "rejected"}
    
    # 4. Require take-profit (already required in function signature, but validate)
    if take_profit_price is None or take_profit_price <= 0:
        error = "take_profit_price is REQUIRED for all orders and must be positive. Never trade without a profit target!"
        logger.error(error)
        return {"error": error, "status": "rejected"}
    
    # 5. Extended hours validation
    if extended_hours:
        # Extended hours requires limit orders
        if type != "limit":
            error = "Extended hours trading requires type='limit' (market orders not allowed)"
            logger.error(error)
            return {"error": error, "status": "rejected"}
        
        # Extended hours requires time_in_force="day"
        if time_in_force != "day":
            error = "Extended hours trading requires time_in_force='day'"
            logger.error(error)
            return {"error": error, "status": "rejected"}
        
        # Extended hours requires limit_price
        if limit_price is None:
            error = "Extended hours trading requires limit_price"
            logger.error(error)
            return {"error": error, "status": "rejected"}
    
    # 6. If limit order, require limit_price
    if type == "limit" and limit_price is None:
        error = "limit_price is required when type='limit'"
        logger.error(error)
        return {"error": error, "status": "rejected"}
    
    # 7. Validate prices make sense (for limit orders)
    if limit_price:
        if stop_loss_price >= limit_price:
            error = f"stop_loss_price ({stop_loss_price}) must be BELOW entry price ({limit_price})"
            logger.error(error)
            return {"error": error, "status": "rejected"}
        
        if take_profit_price <= limit_price:
            error = f"take_profit_price ({take_profit_price}) must be ABOVE entry price ({limit_price})"
            logger.error(error)
            return {"error": error, "status": "rejected"}
        
        # Calculate risk/reward ratio
        risk = limit_price - stop_loss_price
        reward = take_profit_price - limit_price
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < 1.0:
            logger.warning(
                f"⚠️  Risk/reward ratio is {risk_reward:.2f}. "
                f"Consider better ratio (risk=${risk:.2f}, reward=${reward:.2f})"
            )
    
    # ==================== Build Bracket Order ====================
    
    try:
        # Import Alpaca trading client
        from portfoliomanager.dataflows.alpaca_portfolio import _get_trading_client
        
        client = _get_trading_client()
        
        # Convert string side to OrderSide enum
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # Convert time_in_force to enum
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK
        }
        time_in_force_enum = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
        
        # Create TakeProfit and StopLoss requests
        # These are already validated as not None earlier
        assert take_profit_price is not None
        assert stop_loss_price is not None
        take_profit = TakeProfitRequest(limit_price=round(take_profit_price, 2))
        stop_loss = StopLossRequest(stop_price=round(stop_loss_price, 2))
        
        # Build order request based on order type
        # Convert qty to int to ensure whole shares
        qty_int = int(qty)
        
        order_request: Union[MarketOrderRequest, LimitOrderRequest]
        if type == "market":
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty_int,
                side=order_side,
                time_in_force=time_in_force_enum,
                order_class=OrderClass.BRACKET,
                take_profit=take_profit,
                stop_loss=stop_loss,
                extended_hours=extended_hours
            )
        else:  # limit order
            # limit_price is already validated for limit orders
            assert limit_price is not None
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty_int,
                side=order_side,
                time_in_force=time_in_force_enum,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit=take_profit,
                stop_loss=stop_loss,
                extended_hours=extended_hours
            )
        
        logger.info(
            f"📊 Placing bracket order: {symbol} "
            f"{qty_int} shares @ {limit_price or 'market'}, "
            f"Stop: {stop_loss_price:.2f}, Target: {take_profit_price:.2f}"
            f"{' (EXTENDED HOURS)' if extended_hours else ''}"
        )
        
        # Submit order to Alpaca
        order = client.submit_order(order_data=order_request)
        
        # Format response
        result = {
            "status": "success",
            "order_id": order.id,  # type: ignore
            "symbol": order.symbol,  # type: ignore
            "side": order.side,  # type: ignore
            "qty": int(order.qty) if order.qty else None,  # type: ignore
            "type": order.type,  # type: ignore
            "order_class": order.order_class,  # type: ignore
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "submitted_at": str(order.submitted_at),  # type: ignore
            "message": "Bracket order placed successfully. Alpaca will automatically manage exits."
        }
        
        # Log order IDs if available
        if hasattr(order, 'legs') and order.legs:
            result["stop_loss_order_id"] = order.legs[0].id if len(order.legs) > 0 else None  # type: ignore
            result["take_profit_order_id"] = order.legs[1].id if len(order.legs) > 1 else None  # type: ignore
            logger.info(f"  ✓ Entry order: {order.id}")  # type: ignore
            logger.info(f"  ✓ Stop-loss order: {result['stop_loss_order_id']}")
            logger.info(f"  ✓ Take-profit order: {result['take_profit_order_id']}")
        
        logger.info(f"✅ Order placed successfully: {order.id}")  # type: ignore
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to place order: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "error": error_msg
        }


# ==================== Tool Assembly ====================

def get_safe_trading_tools(alpaca_mcp_tools: List) -> List:
    """
    Get the complete set of safe trading tools for the LLM.
    
    This includes:
    1. Filtered read-only Alpaca tools (prices, portfolio, etc.)
    2. The validated buy bracket order tool (safe write operation - BUY ONLY)
    
    Args:
        alpaca_mcp_tools: All available Alpaca MCP tools
        
    Returns:
        List of safe tools to provide to the LLM
    """
    
    # 1. Filter to safe read operations
    safe_read_tools = filter_safe_tools(alpaca_mcp_tools)
    
    # 2. Add buy bracket order tool (BUY ONLY)
    safe_write_tools = [place_buy_bracket_order]
    
    # 3. Combine
    all_safe_tools = safe_read_tools + safe_write_tools
    
    logger.info(
        f"Safe trading tools ready: "
        f"{len(safe_read_tools)} read tools + "
        f"{len(safe_write_tools)} write tools = "
        f"{len(all_safe_tools)} total"
    )
    
    return all_safe_tools


# ==================== Usage Example ====================

if __name__ == "__main__":
    """
    Example showing how to use safe trading tools with LLM.
    """
    
    print("Safe Trading Tools Module")
    print("=" * 60)
    print()
    print("This module provides:")
    print("1. Filtered Alpaca MCP tools (read operations only)")
    print("2. Validated BUY bracket order tool (BUY ONLY - no short selling)")
    print()
    print("Usage in portfolio manager:")
    print("-" * 60)
    print("""
from portfoliomanager.graph_v2.mcp_adapter import get_alpaca_mcp_tools
from portfoliomanager.graph_v2.safe_trading_tools import get_safe_trading_tools

# Get all Alpaca MCP tools
all_tools = get_alpaca_mcp_tools()

# Filter to safe tools only
safe_tools = get_safe_trading_tools(all_tools)

# Bind to LLM
llm_with_safe_tools = llm.bind_tools(safe_tools)

# LLM can now:
# ✓ Read prices, portfolio, positions
# ✓ Place BUY bracket orders (with mandatory stop-loss/take-profit)
# ✗ Cannot short sell (BUY orders only)
# ✗ Cannot place orders without stop-loss/take-profit
""")

