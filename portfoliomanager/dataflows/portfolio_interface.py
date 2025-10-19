"""
Portfolio Management Interface

Routing layer for portfolio management operations to different broker vendors.
Similar to tradingagents/dataflows/interface.py pattern.
"""

from typing import Optional, Dict, Any
from .alpaca_portfolio import (
    get_alpaca_account_info,
    get_alpaca_positions,
    get_alpaca_position_details,
    execute_alpaca_trade,
    get_alpaca_open_orders,
    get_alpaca_all_orders,
    cancel_alpaca_order,
    modify_alpaca_order
)


# Tools organized by category
TOOLS_CATEGORIES = {
    "account_operations": {
        "description": "Account balance and information",
        "tools": [
            "get_account_info",
            "get_current_positions",
            "get_position_details"
        ]
    },
    "trading_operations": {
        "description": "Trade execution",
        "tools": [
            "execute_trade"
        ]
    },
    "order_operations": {
        "description": "Order management",
        "tools": [
            "get_open_orders",
            "get_all_orders",
            "cancel_order",
            "modify_order"
        ]
    },
    "portfolio_analysis": {
        "description": "Portfolio analysis and screening",
        "tools": [
            "get_watchlist_stocks",
            "screen_new_opportunities",
            "get_last_iteration_summary",
            "get_trading_constraints"
        ]
    }
}

VENDOR_LIST = [
    "alpaca",
    # Future: "interactive_brokers", "robinhood", etc.
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # Account operations
    "get_account_info": {
        "alpaca": get_alpaca_account_info,
    },
    "get_current_positions": {
        "alpaca": get_alpaca_positions,
    },
    "get_position_details": {
        "alpaca": get_alpaca_position_details,
    },
    # Trading operations
    "execute_trade": {
        "alpaca": execute_alpaca_trade,
    },
    # Order operations
    "get_open_orders": {
        "alpaca": get_alpaca_open_orders,
    },
    "get_all_orders": {
        "alpaca": get_alpaca_all_orders,
    },
    "cancel_order": {
        "alpaca": cancel_alpaca_order,
    },
    "modify_order": {
        "alpaca": modify_alpaca_order,
    },
}  # type: Dict[str, Dict[str, Any]]


def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: Optional[str] = None) -> str:
    """
    Get the configured vendor for a portfolio operation category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    
    For now, defaults to 'alpaca' for all operations.
    In the future, this could read from config to support multiple brokers.
    """
    # TODO: Read from config when supporting multiple brokers
    # from portfoliomanager.config import PORTFOLIO_CONFIG
    # return PORTFOLIO_CONFIG.get("broker_vendor", "alpaca")
    
    return "alpaca"  # Default broker


def route_to_vendor(method: str, *args, **kwargs):
    """
    Route method calls to appropriate broker vendor implementation with fallback support.
    
    Args:
        method: Name of the method to call
        *args: Positional arguments for the method
        **kwargs: Keyword arguments for the method
        
    Returns:
        Result from the vendor implementation
        
    Raises:
        ValueError: If method is not supported
        RuntimeError: If all vendor implementations fail
    """
    # Check if method requires vendor routing
    if method not in VENDOR_METHODS:
        # Method doesn't require vendor routing (e.g., config-based methods)
        raise ValueError(f"Method '{method}' does not require vendor routing")
    
    category = get_category_for_method(method)
    vendor = get_vendor(category, method)
    
    print(f"DEBUG: Routing {method} to vendor '{vendor}'")
    
    if vendor not in VENDOR_METHODS[method]:  # type: ignore
        raise ValueError(f"Vendor '{vendor}' not supported for method '{method}'")
    
    vendor_impl = VENDOR_METHODS[method][vendor]  # type: ignore
    
    try:
        print(f"DEBUG: Calling {vendor_impl.__name__} from vendor '{vendor}'...")
        result = vendor_impl(*args, **kwargs)
        print(f"SUCCESS: {vendor_impl.__name__} from vendor '{vendor}' completed successfully")
        return result
    except Exception as e:
        print(f"FAILED: {vendor_impl.__name__} from vendor '{vendor}' failed: {e}")
        raise RuntimeError(f"Vendor implementation failed for method '{method}': {e}")

