"""
Portfolio Manager Configuration

Configuration settings for the autonomous portfolio management system.

Environment Variables for LLM Configuration:
- LLM_MODEL: Primary model to use (default: gpt-4o-mini)
- LLM_PROVIDER: Provider name (openai, ollama, anthropic) - auto-detected if not set
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

Environment Variables for LangSmith Tracing:
- LANGSMITH_TRACING: Enable tracing (true/false)
- LANGSMITH_API_KEY: Your LangSmith API key
- LANGSMITH_WORKSPACE_ID: Optional workspace ID
- LANGSMITH_PROJECT: Optional project name (default: "default")
"""

import os

PORTFOLIO_CONFIG = {
    # LLM Settings
    # Note: Agents require tool-calling support. Ollama models with tool support work great!
    # In production (Render), use OpenAI. Locally, use Ollama.
    "llm_model": os.getenv("AGENT_LLM_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini"),  # Agent LLM (ReAct, tool calling)
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),    # Default to openai for cloud deployment
    "backend_url": os.getenv("BACKEND_URL"),  # Optional: Only set if using custom endpoint
    
    # S3 Settings
    "s3_bucket_name": os.getenv("S3_BUCKET_NAME", "mintrader-reports"),
    "s3_region": os.getenv("S3_REGION", "us-east-1"),
    
    # Trading Strategy
    "strategy_objective": "maximize_profits",
    "trading_style": "medium_term",  # weeks to months, hold winners
    "min_holding_days": 7,            # Don't panic sell - give positions time to work
    
    # Trading Constraints
    "max_position_size_pct": 10,      # Max 10% per position (can go to 15% for high conviction)
    "max_portfolio_concentration": 30, # Max 30% in any sector
    "max_trades_per_day": 10,
    "min_cash_reserve_pct": 5,        # Keep 5% cash
    "stop_loss_pct": 15,              # Stop-loss at -15% (strategic threshold, not panic selling)
    
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
    # Use minimal analysts to avoid slow/expensive operations
    "analysis_analysts": ["market", "fundamentals"],  # Removed news & social to avoid OpenAI
    "analysis_config": {
        # Start fresh - don't use DEFAULT_CONFIG to avoid OpenAI dependencies
        "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
        "data_cache_dir": os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "tradingagents/dataflows/data_cache"),
        
        # Use OpenAI for cloud deployment (Ollama for local development)
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "deep_think_llm": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "quick_think_llm": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "backend_url": os.getenv("BACKEND_URL"),
        
        # Minimize debate rounds to reduce LLM calls
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        
        # Use only free/local data sources - NO OpenAI
        "data_vendors": {
            "core_stock_apis": "alpaca",
            "technical_indicators": "alpaca",
            "fundamental_data": "yfinance",
            "news_data": "alpaca",  # Only Alpaca (free)
        },
        # Override specific tools to use free vendors only
        "tool_vendors": {
            "get_global_news": "alpaca",
            "get_news": "alpaca",
        },
        
        # Alpaca configuration
        "alpaca_paper_mode": True,
        "requests_verify_ssl": False,
    },
}

