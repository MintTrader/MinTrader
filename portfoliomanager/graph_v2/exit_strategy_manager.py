"""
Exit Strategy Prompt Guidance for Stock Portfolio

This module provides prompt guidance to teach the LLM how to use Alpaca MCP 
tools with proper exit strategies (bracket orders).

The LLM has access to all Alpaca trading tools via MCP. This teaches it 
HOW to use those tools with bracket order parameters for automatic exits.
"""

EXIT_STRATEGY_GUIDANCE = """
BRACKET ORDER REQUIREMENTS:
===========================
Every BUY must use place_bracket_order with stop-loss & take-profit.

Example:
place_bracket_order(
    symbol="AAPL",
    notional=5000,              # Dollar amount to invest
    side="buy",
    type="market",
    stop_loss_price=171.00,     # Entry × 0.95 (-5%)
    take_profit_price=198.00    # Entry × 1.10 (+10%)
)

CALCULATION:
Entry: $180.00
Stop-Loss (-5%): $180 × 0.95 = $171.00
Take-Profit (+10%): $180 × 1.10 = $198.00

RULES:
- Stop-loss: 3-8% depending on volatility
- Take-profit: 2:1 reward/risk minimum (if -5% stop → +10% target)
- Validate: stop < entry < target
- NEVER place orders without exits
"""
