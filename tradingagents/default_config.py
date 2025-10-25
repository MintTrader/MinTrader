"""
TradingAgents Default Configuration

Environment Variables for LLM Configuration:
- LLM_MODEL: Primary model to use (overrides deep_think_llm and quick_think_llm)
- LLM_PROVIDER: Provider name (openai, ollama, anthropic, google) - auto-detected if not set
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

Environment Variables for LangSmith Tracing:
- LANGSMITH_TRACING: Enable tracing (true/false)
- LANGSMITH_API_KEY: Your LangSmith API key
- LANGSMITH_WORKSPACE_ID: Optional workspace ID
- LANGSMITH_PROJECT: Optional project name (default: "default")

To use local LLMs with Ollama:
1. Install: brew install ollama (macOS) or see https://ollama.ai
2. Start: ollama serve
3. Pull model: ollama pull llama3
4. Set: export LLM_MODEL="llama3"
"""

import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings (optimized for local development with Ollama)
    "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
    "deep_think_llm": os.getenv("LLM_MODEL", "gpt-oss:20b"),      # Local Ollama model
    "quick_think_llm": os.getenv("LLM_MODEL", "gpt-oss:20b"),     # Local Ollama model
    "backend_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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
