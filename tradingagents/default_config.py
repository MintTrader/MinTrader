import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "alpaca",         # Options: alpaca, yfinance, local
        "technical_indicators": "alpaca",    # Options: alpaca, yfinance, local
        "fundamental_data": "yfinance",      # Options: yfinance, openai, local
        "news_data": "alpaca,google",        # Options: alpaca, google, openai, local (comma-separated for hybrid)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpaca",  # Override category default
        # Example: "get_news": "openai",        # Override category default
    },
    # Alpaca configuration
    "alpaca_paper_mode": True,  # Always use paper trading by default for safety
    # SSL/TLS configuration
    "requests_verify_ssl": False,  # Disable SSL verification globally
}
