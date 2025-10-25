"""
Alpaca MCP Adapter

Wrapper around langchain-mcp to provide Alpaca trading tools.
Replaces all custom trading, account, and market data tools.
"""

import os
from typing import List
from langchain_core.tools import BaseTool
import asyncio


async def _init_alpaca_toolkit_async():
    """Internal async function to initialize Alpaca MCP toolkit"""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_core.tools import StructuredTool
    import asyncio
    
    # Validate environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper_trade = os.getenv("ALPACA_PAPER_TRADE", "true")
    
    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment"
        )
    
    # Initialize MCP client with Alpaca server
    # Use the alpaca-mcp-server command installed by the package
    client = MultiServerMCPClient({
        "alpaca": {
            "command": "alpaca-mcp-server",
            "args": ["serve"],
            "transport": "stdio",
            "env": {
                "ALPACA_API_KEY": api_key,
                "ALPACA_SECRET_KEY": secret_key,
                "ALPACA_PAPER_TRADE": paper_trade,
            }
        }
    })
    
    # Get all available tools (silently)
    tools = await client.get_tools()
    
    # Wrap async tools to support sync invocation
    sync_tools = []
    for tool in tools:
        # Create a sync wrapper for each async tool
        def make_sync_wrapper(async_tool):
            def sync_func(**kwargs):
                """Sync wrapper that runs async tool in event loop"""
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(async_tool.ainvoke(kwargs))
            
            return StructuredTool(
                name=async_tool.name,
                description=async_tool.description,
                func=sync_func,
                args_schema=async_tool.args_schema,
                coroutine=async_tool.ainvoke  # Keep async version too
            )
        
        sync_tools.append(make_sync_wrapper(tool))
    
    # Store client reference to keep session alive
    if not hasattr(_init_alpaca_toolkit_async, '_client'):
        _init_alpaca_toolkit_async._client = client
    
    return sync_tools


def get_alpaca_mcp_tools() -> List[BaseTool]:
    """
    Initialize and return Alpaca MCP tools.
    
    This replaces 30+ custom tools with standardized MCP server tools:
    - Account management: get_account_info
    - Positions: get_positions, get_open_position, close_position, close_all_positions
    - Orders: place_stock_order, place_crypto_order, cancel_order_by_id, cancel_all_orders, get_orders
    - Market data: get_stock_bars, get_stock_quote, get_stock_trades, get_stock_latest_trade
    - Market info: get_asset_info, get_all_assets, get_market_clock, get_market_calendar
    - Watchlists: create_watchlist, get_watchlists, update_watchlist
    - And many more...
    
    Returns:
        List of LangChain BaseTool objects from Alpaca MCP server
        
    Environment Variables:
        ALPACA_API_KEY: Alpaca API key
        ALPACA_SECRET_KEY: Alpaca secret key
        ALPACA_PAPER_TRADE: Set to "true" for paper trading (default)
    
    Example:
        >>> tools = get_alpaca_mcp_tools()
        >>> print([tool.name for tool in tools])
        ['get_account_info', 'get_positions', 'place_stock_order', ...]
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        raise ImportError(
            "langchain-mcp not installed.\n"
            "\n"
            "To install, run:\n"
            "  poetry install\n"
            "\n"
            "Then make sure to:\n"
            "1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file\n"
            "2. Run: poetry run portfoliomanager\n"
            "\n"
            "Note: The MCP client will automatically start the Alpaca server using Python."
        )
    
    # Run async initialization in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(_init_alpaca_toolkit_async())
    except (FileNotFoundError, ModuleNotFoundError) as e:
        error_msg = str(e)
        if 'alpaca_mcp_server' in error_msg or 'alpaca-mcp-server' in error_msg:
            raise RuntimeError(
                "alpaca-mcp-server not installed. MCP tools require this package.\n"
                "\n"
                "To install:\n"
                "  poetry install\n"
                "\n"
                "Or manually:\n"
                "  pip install alpaca-mcp-server\n"
            ) from e
        raise


def get_alpaca_tool(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """
    Get a specific tool by name from the Alpaca MCP toolkit.
    
    Args:
        tools: List of tools from get_alpaca_mcp_tools()
        tool_name: Name of the tool to retrieve
        
    Returns:
        The requested tool
        
    Raises:
        ValueError: If tool not found
        
    Example:
        >>> tools = get_alpaca_mcp_tools()
        >>> account_tool = get_alpaca_tool(tools, "get_account_info")
        >>> result = account_tool.invoke({})
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    
    available = [t.name for t in tools]
    raise ValueError(
        f"Tool '{tool_name}' not found. Available tools: {available}"
    )


def create_tool_node_for_alpaca(tools: List[BaseTool]):
    """
    Create a ToolNode for use in LangGraph with Alpaca tools.
    
    Args:
        tools: List of Alpaca MCP tools
        
    Returns:
        ToolNode configured with Alpaca tools
        
    Example:
        >>> from langgraph.prebuilt import ToolNode
        >>> tools = get_alpaca_mcp_tools()
        >>> tool_node = create_tool_node_for_alpaca(tools)
        >>> # Add to graph
        >>> graph.add_node("tools", tool_node)
    """
    from langgraph.prebuilt import ToolNode
    return ToolNode(tools)


# ==================== Tool Information ====================

def list_available_tools(tools: List[BaseTool]) -> dict:
    """
    Get a summary of all available MCP tools.
    
    Args:
        tools: Alpaca MCP tools
        
    Returns:
        Dictionary with tool names, descriptions, and categories
        
    Example:
        >>> tools = get_alpaca_mcp_tools()
        >>> info = list_available_tools(tools)
        >>> print(f"Found {len(info['tools'])} tools")
    """
    tool_info = []
    for tool in tools:
        tool_info.append({
            "name": tool.name,
            "description": tool.description,
        })
    
    return {
        "count": len(tools),
        "tools": tool_info
    }

