"""
Orchestrator Tools

Tools for the LLM orchestrator agent to use web search, request analysis, and read reports.
"""

from langchain_core.tools import tool
from typing import Annotated, Dict, Any, List
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import httpx


# OpenAI client for web search
_http_client = httpx.Client(verify=False)


@tool
def web_search_market_context(query: Annotated[str, "Search query about market conditions, stocks, or news"]) -> str:
    """
    Search the web for current market context, news, and information about stocks.
    Use this to understand current market conditions, find trending stocks, or get recent news.
    
    Args:
        query: What to search for (e.g., "trending tech stocks today", "market conditions October 2025")
        
    Returns:
        Search results with relevant information
    """
    import os
    
    backend_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    client = OpenAI(base_url=backend_url, http_client=_http_client)
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": query,
                        }
                    ],
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "medium",
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )
        
        # Extract the text from the response - handle different response types
        try:
            if hasattr(response, 'output') and response.output and len(response.output) > 1:
                output_item = response.output[1]  # type: ignore
                if hasattr(output_item, 'content') and output_item.content:  # type: ignore
                    content_item = output_item.content[0]  # type: ignore
                    if hasattr(content_item, 'text'):
                        return str(content_item.text)  # type: ignore
            return "No results found from web search"
        except (IndexError, AttributeError) as e:
            return f"Error parsing web search response: {str(e)}"
            
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def request_stock_analysis(
    ticker: Annotated[str, "Stock ticker symbol to analyze"],
    reasoning: Annotated[str, "Why you want to analyze this stock"]
) -> str:
    """
    Request a comprehensive TradingAgents analysis for a stock.
    This will run the full analysis pipeline and generate reports.
    
    ⚠️ IMPORTANT: You can only request up to 3 stock analyses per iteration.
    Use this tool wisely - only analyze stocks you're seriously considering trading.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, NVDA)
        reasoning: Explain why this stock needs analysis
        
    Returns:
        Confirmation that analysis was requested
    """
    # This is a marker tool - actual execution happens in the manager
    return f"Analysis requested for {ticker}. Reasoning: {reasoning}"


@tool
def read_analysis_report(
    ticker: Annotated[str, "Stock ticker symbol"],
    report_type: Annotated[str, "Type of report: 'investment_plan' or 'final_trade_decision' or 'market_report'"]
) -> str:
    """
    Read the analysis report for a stock that was previously analyzed.
    
    Available report types:
    - 'investment_plan': Comprehensive investment strategy and analysis
    - 'final_trade_decision': Final recommendation (BUY/SELL/HOLD) with rationale
    - 'market_report': Technical analysis and market conditions
    
    Args:
        ticker: Stock ticker symbol
        report_type: Which report to read
        
    Returns:
        The report content
    """
    # This will be intercepted by the manager to provide actual report content
    return f"Reading {report_type} for {ticker}"


@tool  
def get_analysis_status() -> Dict[str, Any]:
    """
    Check how many stock analyses you've requested and how many remain.
    You can analyze up to 3 stocks per iteration.
    
    Returns:
        Dictionary with:
        - analyses_requested: Number of analyses requested so far
        - analyses_remaining: Number of analyses still available
        - analyzed_stocks: List of stocks analyzed so far
    """
    # This will be intercepted by the manager to provide actual status
    return {
        "analyses_requested": 0,
        "analyses_remaining": 3,
        "analyzed_stocks": []
    }


@tool
def get_recently_analyzed_stocks(days: Annotated[int, "Number of days to look back"] = 14) -> Dict[str, Any]:
    """
    Get list of stocks that were recently analyzed (within the past N days).
    This helps you avoid re-analyzing stocks unnecessarily and informs your decisions.
    
    Use this to:
    - See what stocks were analyzed recently and their decisions
    - Avoid redundant analysis of stocks already reviewed
    - Understand recent portfolio activity and decisions
    
    Args:
        days: Number of days to look back (default: 14)
        
    Returns:
        Dictionary with:
        - recently_analyzed: List of dicts with ticker, date, and decision
        - days_threshold: The days parameter used
        - total_count: Number of stocks recently analyzed
    """
    # This will be intercepted by the manager to provide actual data
    return {
        "recently_analyzed": [],
        "days_threshold": days,
        "total_count": 0
    }


@tool
def place_buy_order(
    ticker: Annotated[str, "Stock ticker symbol to buy"],
    order_value: Annotated[float, "Dollar amount to invest (e.g., 1000.0)"],
    reasoning: Annotated[str, "Detailed reasoning for this BUY decision"]
) -> str:
    """
    Place a BUY order for a stock.
    
    Use this after reviewing analysis reports and deciding to buy based on:
    - TradingAgents analysis recommendations
    - Current portfolio state and diversification
    - Available cash and buying power
    - Market conditions
    
    ⚠️ IMPORTANT CHECKS:
    - Verify you have sufficient buying power
    - Check if you already have a position in this stock
    - Check if there's already a pending BUY order for this stock
    - Consider portfolio concentration limits
    
    Args:
        ticker: Stock ticker to buy (e.g., "AAPL", "TSLA")
        order_value: Dollar amount to invest (minimum $1000)
        reasoning: Why you're buying this stock
        
    Returns:
        Confirmation message with order details
    """
    # This will be intercepted by the manager to execute actual trade
    return f"BUY order placed for {ticker} (${order_value:,.2f}). Reasoning: {reasoning}"


@tool
def place_sell_order(
    ticker: Annotated[str, "Stock ticker symbol to sell"],
    quantity: Annotated[int, "Number of shares to sell, or 'all' to sell entire position"],
    reasoning: Annotated[str, "Detailed reasoning for this SELL decision"]
) -> str:
    """
    Place a SELL order for a stock you currently hold.
    
    Use this after reviewing analysis reports and deciding to sell based on:
    - TradingAgents analysis recommendations
    - Position performance and profit/loss
    - Risk management considerations
    - Portfolio rebalancing needs
    
    ⚠️ IMPORTANT CHECKS:
    - Verify you have a position in this stock
    - Check if there's already a pending SELL order for this stock
    - Consider tax implications of selling
    
    Args:
        ticker: Stock ticker to sell (e.g., "AAPL", "TSLA")
        quantity: Number of shares to sell (use "all" to sell entire position)
        reasoning: Why you're selling this stock
        
    Returns:
        Confirmation message with order details
    """
    # This will be intercepted by the manager to execute actual trade
    return f"SELL order placed for {ticker} ({quantity} shares). Reasoning: {reasoning}"


@tool
def cancel_order(
    ticker: Annotated[str, "Stock ticker symbol of order to cancel"],
    reasoning: Annotated[str, "Why you're canceling this order"]
) -> str:
    """
    Cancel a pending order for a stock.
    
    Use this when:
    - Market conditions have changed since order was placed
    - Analysis indicates the order is no longer advisable
    - You want to modify order parameters (cancel then replace)
    - Order has been pending too long and unlikely to fill
    
    ⚠️ NOTE: Only pending orders can be cancelled. Filled orders cannot be reversed.
    
    Args:
        ticker: Stock ticker of the order to cancel
        reasoning: Why you're canceling the order
        
    Returns:
        Confirmation message
    """
    # This will be intercepted by the manager to cancel actual order
    return f"Order for {ticker} cancelled. Reasoning: {reasoning}"


@tool
def review_and_decide() -> str:
    """
    Signal that you've reviewed all analysis and made your trading decisions.
    
    Use this when:
    - You've reviewed all stock analyses
    - You've made all trading decisions (buy/sell/hold)
    - You're ready to complete this iteration
    
    This marks the end of the decision-making process for this iteration.
    
    Returns:
        Confirmation message
    """
    # This signals to the manager that the LLM is done making decisions
    return "Decision-making complete. Ready to proceed to next iteration."

