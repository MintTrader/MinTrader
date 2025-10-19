from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"  # Use a different model
config["quick_think_llm"] = "gpt-4o-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses Alpaca for market data)
config["data_vendors"] = {
    "core_stock_apis": "alpaca",           # Options: alpaca, yfinance, local
    "technical_indicators": "alpaca",      # Options: alpaca, yfinance, local
    "fundamental_data": "yfinance",        # Options: yfinance, openai, local
    "news_data": "alpaca,google",          # Options: alpaca, google, openai, local (comma-separated for hybrid)
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
