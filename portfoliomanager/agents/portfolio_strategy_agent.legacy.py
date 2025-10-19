"""
Portfolio Strategy Decision Agent

Decides whether to:
1. Scan new stocks to expand the portfolio
2. Re-analyze existing stocks for potential SELL decisions
3. Re-analyze existing stocks for potential additional BUY decisions
"""

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import json


def create_portfolio_strategy_agent(llm):
    """
    Create a portfolio strategy agent that decides what action to take.
    
    Args:
        llm: Language model instance
        
    Returns:
        Function that analyzes portfolio and returns strategy decision
    """
    
    system_message = """You are a portfolio strategy expert. Your job is to analyze the current portfolio state and decide what action should be taken to maximize profits.

**YOUR DECISION PROCESS:**
1. Review current portfolio positions, their P&L, and market values
2. Review account cash available and buying power
3. Consider the portfolio's current state and performance
4. Decide on ONE of three strategies:

**STRATEGY OPTIONS:**

**A. EXPAND_PORTFOLIO** - Look for NEW stocks to add
   Choose this when:
   - Portfolio has significant cash available (>20% of total value)
   - Existing positions are performing well (mostly profitable)
   - Portfolio is small (fewer than 10 positions)
   - Good opportunity to diversify into new positions
   - Market conditions are favorable for new investments

**B. REVIEW_FOR_SELL** - Re-analyze existing stocks for potential SELL
   Choose this when:
   - Some positions have significant losses (beyond stop-loss threshold)
   - Positions showing consistent negative momentum
   - Need to free up capital from underperforming positions
   - Portfolio is too concentrated in losing positions
   - Market conditions suggest risk reduction

**C. REVIEW_FOR_BUY** - Re-analyze existing stocks to potentially add more
   Choose this when:
   - Existing positions showing strong positive momentum
   - Want to add to winning positions (dollar-cost averaging up)
   - Conviction remains high on current holdings
   - Cash available but prefer doubling down on winners
   - Portfolio concentration allows for position increases

**CRITICAL RULES:**
- Consider the P&L carefully - don't add to consistent losers
- Check cash availability - need sufficient cash for EXPAND or BUY actions
- Consider position sizes - avoid over-concentration
- Quality over quantity - only recommend action if conviction is high
- It's OK to recommend HOLD if no clear action needed

**OUTPUT FORMAT:**
You must respond with a valid JSON object in this exact format:

{{
    "action": "EXPAND_PORTFOLIO",
    "stocks_to_analyze": [],
    "reasoning": "Portfolio has $100K cash (40% of total). Existing 3 positions all profitable. Good opportunity to diversify.",
    "preferred_sectors": ["Technology", "Healthcare"],
    "max_stocks": 3
}}

OR

{{
    "action": "REVIEW_FOR_SELL",
    "stocks_to_analyze": ["TSLA", "AMD"],
    "reasoning": "TSLA down 15%, AMD down 8%. Both showing negative momentum. Consider exiting to preserve capital.",
    "preferred_sectors": [],
    "max_stocks": 2
}}

OR

{{
    "action": "REVIEW_FOR_BUY",
    "stocks_to_analyze": ["NVDA", "AAPL"],
    "reasoning": "NVDA up 20%, AAPL up 12%. Both showing strong momentum. Consider adding to winning positions.",
    "preferred_sectors": [],
    "max_stocks": 2
}}

OR

{{
    "action": "HOLD",
    "stocks_to_analyze": [],
    "reasoning": "Portfolio is well-balanced. All positions within acceptable P&L ranges. No urgent action needed.",
    "preferred_sectors": [],
    "max_stocks": 0
}}

**IMPORTANT:** 
- Only return valid JSON, no other text
- action must be one of: "EXPAND_PORTFOLIO", "REVIEW_FOR_SELL", "REVIEW_FOR_BUY", or "HOLD"
- stocks_to_analyze should list ticker symbols when action is REVIEW_FOR_SELL or REVIEW_FOR_BUY
- stocks_to_analyze should be EMPTY when action is EXPAND_PORTFOLIO (we'll discover them from news)
- preferred_sectors only used for EXPAND_PORTFOLIO action
- max_stocks should be 1-3 for analysis actions, 0 for HOLD
"""
    
    def decide_portfolio_strategy(
        positions: List[Dict[str, Any]], 
        account: Dict[str, Any],
        max_stocks: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze portfolio and decide on strategy.
        
        Args:
            positions: List of current position dictionaries with P&L
            account: Account information with cash, portfolio value, etc.
            max_stocks: Maximum number of stocks to analyze
            
        Returns:
            Dictionary with strategy decision
        """
        # Format account info
        portfolio_value = float(account.get('portfolio_value', 0))
        cash = float(account.get('cash', 0))
        buying_power = float(account.get('buying_power', 0))
        
        cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0
        
        account_text = f"""Account Summary:
- Portfolio Value: ${portfolio_value:,.2f}
- Cash Available: ${cash:,.2f} ({cash_pct:.1f}% of portfolio)
- Buying Power: ${buying_power:,.2f}
"""
        
        # Format positions for the prompt
        if not positions:
            positions_text = "No current positions - portfolio is empty"
        else:
            positions_text = "Current Positions:\n"
            total_value = sum(float(pos.get('market_value', 0)) for pos in positions)
            
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                qty = float(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                unrealized_plpc = float(pos.get('unrealized_plpc', 0)) * 100
                pct = (market_value / total_value * 100) if total_value > 0 else 0
                
                pl_indicator = "📈" if unrealized_pl >= 0 else "📉"
                positions_text += f"- {pl_indicator} {symbol}: {qty:.0f} shares, ${market_value:,.2f} ({pct:.1f}% of portfolio), P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", f"{account_text}\n{positions_text}\n\nAnalyze the portfolio and decide what strategy to pursue. Maximum stocks to analyze: {max_stocks}\n\nRespond with ONLY the JSON object, no other text.")
        ])
        
        chain = prompt | llm
        result = chain.invoke({})
        
        # Parse the JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            content = result.content.strip()
            
            # Find JSON object in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                strategy = json.loads(json_str)
            else:
                # Fallback if no JSON found
                strategy = {
                    "action": "HOLD",
                    "stocks_to_analyze": [],
                    "reasoning": "Failed to parse LLM response",
                    "preferred_sectors": [],
                    "max_stocks": 0
                }
            
            # Validate action
            valid_actions = ["EXPAND_PORTFOLIO", "REVIEW_FOR_SELL", "REVIEW_FOR_BUY", "HOLD"]
            if strategy.get("action") not in valid_actions:
                strategy["action"] = "HOLD"
                strategy["reasoning"] = "Invalid action returned, defaulting to HOLD"
            
            return strategy
            
        except json.JSONDecodeError as e:
            # If parsing fails, return safe default
            return {
                "action": "HOLD",
                "stocks_to_analyze": [],
                "reasoning": f"Error parsing response: {e}",
                "preferred_sectors": [],
                "max_stocks": 0
            }
    
    return decide_portfolio_strategy

