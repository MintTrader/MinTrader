"""
Portfolio Agent

LLM-based agent that makes strategic portfolio decisions with profit maximization focus.
Following TradingAgents pattern with prompts and tools.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Any
from .utils.portfolio_management_tools import (
    get_account_info,
    get_current_positions,
    get_position_details,
    get_last_iteration_summary,
    get_trading_constraints,
    execute_trade,
    get_watchlist_stocks,
    get_open_orders,
    get_all_orders,
    cancel_order,
    modify_order
)
# Import core stock tools from TradingAgents
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.technical_indicators_tools import get_indicators


def create_portfolio_agent(llm):
    """
    Create a portfolio management agent focused on profit maximization.
    
    Args:
        llm: Language model instance
        
    Returns:
        Agent function
    """
    
    def portfolio_agent_node(state):
        """
        Portfolio agent node that analyzes stock recommendations and makes trading decisions.
        
        Args:
            state: Agent state containing stock_recommendations and other context
            
        Returns:
            Dictionary with portfolio decisions
        """
        # Get stock recommendations from state
        stock_recommendations = state.get('stock_recommendations', {})
        
        # Format recommendations for prompt
        recommendations_text = "\n\nSTOCK ANALYSIS REPORTS:"
        for ticker, rec in stock_recommendations.items():
            recommendations_text += f"\n\n{ticker} Analysis:"
            recommendations_text += f"\nDecision: {rec.get('decision', 'N/A')}"
            recommendations_text += f"\nSummary: {rec.get('summary', 'No summary available')}"
        
        # Define tools available to the agent
        # Portfolio management tools
        # NOTE: screen_new_opportunities is NOT included here because:
        # - Stock discovery is handled by the PortfolioManager before this agent runs
        # - This agent should ONLY evaluate stocks that have already been analyzed
        # - Including it would allow the agent to bypass the analysis pipeline
        portfolio_tools = [
            get_account_info,
            get_current_positions,
            get_position_details,
            get_last_iteration_summary,
            get_trading_constraints,
            get_watchlist_stocks,
            execute_trade,
            get_open_orders,
            get_all_orders,
            cancel_order,
            modify_order
        ]
        
        # Stock analysis tools (from TradingAgents)
        stock_analysis_tools = [
            get_stock_data,
            get_indicators
        ]
        
        # Combine all tools
        tools = portfolio_tools + stock_analysis_tools
        
        system_message = f"""You are an autonomous portfolio manager focused on MAXIMIZING PROFITS through selective, medium-term trading.

**STRATEGIC OBJECTIVES:**
- Primary Goal: Maximize profits through quality positions held for weeks to months
- You are NOT a day trader - focus on quality over quantity
- Only trade when there is strong conviction based on comprehensive analysis
- Avoid buying stocks showing declining trends - only enter on upward momentum
- Consider how long positions have been held before deciding to sell
- HOLD is often the best decision - don't trade just to be active

**CRITICAL: STOCK EVALUATION SCOPE**
⚠️  You must ONLY evaluate the stocks provided in the "STOCK ANALYSIS REPORTS" section below.
⚠️  DO NOT use `screen_new_opportunities` - stocks have already been discovered and analyzed.
⚠️  Your job is to decide WHETHER to buy the analyzed stocks, not to find new stocks.
⚠️  If you decide not to buy any of the provided stocks, that's perfectly fine - explain why.

**YOUR WORKFLOW:**
1. First, call `get_trading_constraints()` to understand your operating limits
2. Call `get_account_info()` to see available cash and portfolio value
3. Call `get_current_positions()` to see all current holdings with their P&L
4. Call `get_last_iteration_summary()` to learn from previous decisions
5. Review the stock analysis reports provided below - THESE ARE THE ONLY STOCKS TO CONSIDER
6. For deeper analysis of the provided stocks, use stock tools:
   - `get_stock_data(ticker, start_date, end_date)` to fetch historical prices
   - `get_indicators(ticker, indicators, date)` to analyze momentum and technicals
   - Calculate holding periods, momentum, and trends yourself using this data
7. Make strategic decisions based on:
   - Stock analysis recommendations (ONLY for the provided stocks)
   - Position P&L and momentum (calculated using stock tools)
   - Portfolio constraints
   - Your profit-maximization mandate
8. Use `execute_trade(ticker, action, quantity, reasoning)` to execute approved trades
   - Only execute trades for stocks in the STOCK ANALYSIS REPORTS below
   - Only execute trades where you have strong conviction
   - Provide detailed reasoning for each trade
   - Consider constraints: position size, cash reserves, holding periods

**CRITICAL: BUY DECISION CRITERIA**
A BUY decision requires ALL of the following to be satisfied:
1. **TradingAgents Recommendation**: The stock analysis shows BUY with strong conviction
2. **Available Cash**: You have sufficient cash for the purchase PLUS maintaining minimum cash reserve
3. **Portfolio Diversity**: 
   - The new position won't create over-concentration (max position size constraint)
   - Consider sector exposure - avoid having too many stocks in the same sector
   - Balance between existing winners and new opportunities
4. **Risk Management**:
   - Position sizing must fit within constraints (check max position size)
   - Total portfolio risk remains acceptable after the purchase
   - You have exit strategy in mind (stop-loss levels, target gains)
5. **Technical Confirmation**: 
   - Strong upward momentum (use indicators to verify)
   - Positive fundamentals from analysis
   - Favorable entry point (not buying at peak)

**DO NOT BUY if any of these conditions fail**, even if TradingAgents recommends BUY.
Remember: It's better to miss an opportunity than to make a poor allocation decision.

**DECISION GUIDELINES:**
- BUY: Only when ALL buy criteria above are met
- SELL: Gains materialized, stop-loss triggered, or fundamental deterioration detected
- HOLD: Insufficient conviction or any buy criteria not met (this is often the right choice)

**IMPORTANT CONSTRAINTS:**
- Maximum position size: Check constraints
- Minimum cash reserve: Check constraints
- Stop-loss threshold: Check constraints
- Minimum holding period: Check constraints (unless stop-loss triggered)
- Portfolio concentration: Avoid over-concentration in single stocks or sectors

When you have completed your analysis and executed all necessary trades, provide a summary of:
1. What trades were executed and why
2. What opportunities were passed on and why (especially important for rejected BUY recommendations)
3. Current portfolio balance, diversity, and risk profile
4. Key observations from the portfolio
5. Strategy for the next iteration

{recommendations_text}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant managing a portfolio."
                " Use the provided tools to analyze the portfolio and make trading decisions."
                " Execute what you can to maximize profits."
                " When you have completed your analysis and trades, provide a comprehensive summary."
                " You have access to the following tools: {tool_names}.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        
        chain = prompt | llm.bind_tools(tools)
        
        result = chain.invoke(state["messages"])
        
        # Extract summary from result
        summary = ""
        if len(result.tool_calls) == 0:
            summary = result.content
        
        return {
            "messages": [result],
            "portfolio_decision": summary,
        }
    
    return portfolio_agent_node

