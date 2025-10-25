"""
LangGraph-based Portfolio Manager (Version 2)

Modern implementation using:
- LangGraph v1.0 for state management and workflow orchestration
- Alpaca MCP Server for all trading operations
- Built-in persistence, streaming, and autonomous decision-making
- Fully automated trading with no human approval required
"""

from .state import PortfolioState, TradeDecision, AnalysisResult
from .portfolio_graph import create_portfolio_graph, run_portfolio_iteration

__all__ = [
    "PortfolioState",
    "TradeDecision", 
    "AnalysisResult",
    "create_portfolio_graph",
    "run_portfolio_iteration"
]

