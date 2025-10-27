"""
LangGraph State Schema for Portfolio Manager

Defines the complete state schema for the portfolio management graph.
Uses TypedDict for type safety and proper state management.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class PortfolioState(TypedDict):
    """
    Complete state schema for portfolio management workflow.
    
    This state is automatically persisted at each node execution when using
    a checkpointer, enabling features like:
    - Automatic state recovery after errors
    - Time-travel debugging
    - Human-in-the-loop interrupts
    - Replay of past executions
    """
    
    # ==================== Message History ====================
    # Automatically managed by LangGraph's add_messages reducer
    messages: Annotated[list[AnyMessage], add_messages]
    
    # ==================== Portfolio Data ====================
    # Current account information
    account: dict  # {cash, portfolio_value, buying_power, ...}
    
    # Current positions
    positions: list[dict]  # [{symbol, qty, market_value, unrealized_plpc, ...}]
    
    # Open orders
    open_orders: list[dict]  # [{symbol, side, qty, status, ...}]
    
    # Market clock/status
    market_clock: dict  # {is_open, next_open, next_close, ...}
    
    # ==================== Agent Memory ====================
    # Previous iteration summary (agent's memory across runs)
    last_summary: str
    
    # ==================== Trading Decisions ====================
    # Trades pending execution (awaiting approval if HITL enabled)
    pending_trades: list[dict]  # [{ticker, action, quantity, reasoning, ...}]
    
    # Trades that have been executed
    executed_trades: list[dict]  # [{ticker, action, quantity, order_id, ...}]
    
    # ==================== Workflow Control ====================
    # Unique iteration identifier
    iteration_id: str
    
    # Current phase of workflow
    phase: Literal[
        "init",
        "assess",
        "decide",
        "execute",
        "complete",
        "market_closed",
        "error"
    ]
    
    # Error message if any
    error: str | None
    
    # Configuration for this run
    config: dict  # From PORTFOLIO_CONFIG


# ==================== Helper State Types ====================

class TradeDecision(TypedDict):
    """
    Structure for a single trade decision.
    """
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    quantity: int | str  # Can be number or "all" for sell
    reasoning: str
    order_value: float | None  # For BUY orders
    analysis_date: str


class AnalysisResult(TypedDict):
    """
    Structure for analysis results from TradingAgents.
    """
    ticker: str
    date: str
    final_trade_decision: str
    investment_plan: str
    market_report: str
    news_report: str | None
    fundamentals_report: str | None
    recommendation: Literal["BUY", "SELL", "HOLD"]

