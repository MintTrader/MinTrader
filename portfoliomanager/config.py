"""
Portfolio Manager Configuration

Configuration settings for the autonomous portfolio management system.
"""

import os
from tradingagents.default_config import DEFAULT_CONFIG

PORTFOLIO_CONFIG = {
    # S3 Settings
    "s3_bucket_name": os.getenv("S3_BUCKET_NAME", "mintrader-reports"),
    "s3_region": os.getenv("S3_REGION", "us-east-1"),
    
    # Trading Strategy
    "strategy_objective": "maximize_profits",
    "trading_style": "medium_term",  # weeks to months
    "min_conviction_score": 7,        # Only execute high-conviction trades
    "min_holding_days": 7,            # Discourage selling too quickly
    
    # Trading Constraints
    "max_position_size_pct": 10,      # Max 10% per position
    "max_portfolio_concentration": 30, # Max 30% in any sector
    "max_trades_per_day": 10,
    "min_cash_reserve_pct": 5,        # Keep 5% cash
    "stop_loss_pct": 5,               # Auto stop-loss at -5%
    
    # Watchlist
    "watchlist": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
    
    # Screener Settings (focus on upward momentum)
    "enable_screener": True,
    "screener_criteria": {
        "min_volume": 1000000,
        "min_price": 10,
        "max_price": 500,
        "market_cap_min": 1e9,
        "momentum_filter": "positive",  # Only screen stocks with positive momentum
    },
    "max_screener_picks": 3,
    
    # Schedule
    "schedule_times": ["09:35", "12:00", "14:30", "15:45"],
    
    # Stock Discovery Settings (news-based)
    "max_stocks_to_analyze": 3,  # Limit to 3 stocks per iteration
    "max_news_stocks": 3,         # Max stocks to discover from news
    
    # Cost Optimization Settings
    "enable_web_search": True,                 # Enable/disable web search (expensive!)
    "max_web_searches_per_iteration": 1,      # Single web search per iteration
    
    # Note: Single-pass workflow eliminates the need for:
    # - Multiple orchestrator iterations (now just one pass)
    # - Context truncation (no accumulating context)
    # - Iteration limits (deterministic workflow)
    
    # Analysis Config (passed to TradingAgentsGraph)
    # Using only 2 analysts reduces LLM calls by 33% while maintaining quality
    "analysis_analysts": ["market", "news"],  # Removed "fundamentals" for cost optimization
    "analysis_config": {
        **DEFAULT_CONFIG,
        # Use GPT-4.1 nano for all analysis to minimize costs (cheapest model)
        "deep_think_llm": "gpt-4.1-nano",      # Use nano for all analysis
        "quick_think_llm": "gpt-4.1-nano",     # Use nano for all analysis
        "backend_url": "https://api.openai.com/v1",
        # Minimize debate rounds to reduce LLM calls
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        # Use only fast news sources - avoid slow Reddit/local sources
        "data_vendors": {
            "core_stock_apis": "alpaca",
            "technical_indicators": "alpaca",
            "fundamental_data": "yfinance",
            "news_data": "alpaca,openai",  # Only use fast news sources
        },
        # Override specific tools to use fastest vendors
        "tool_vendors": {
            "get_global_news": "openai",  # Use only OpenAI for global news (fastest)
            "get_news": "alpaca,openai",   # Use Alpaca first, OpenAI as fallback
        },
    },
}

