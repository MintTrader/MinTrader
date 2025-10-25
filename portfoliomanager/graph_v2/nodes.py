"""
LangGraph Nodes for Portfolio Manager

Each node is a focused, single-responsibility function that:
1. Receives the current state
2. Performs one specific operation
3. Returns state updates

Nodes are:
- Easy to test independently
- Easy to debug and trace
- Composable into different workflows
- Automatically checkpointed by LangGraph
"""

from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from langgraph.pregel import Pregel

from .state import PortfolioState
from .mcp_adapter import get_alpaca_mcp_tools
from shared.llm_factory import get_agent_llm, get_quick_llm


# ==================== Portfolio Assessment ====================

def assess_portfolio_node(state: PortfolioState) -> Dict[str, Any]:
    """
    Gather current portfolio state using Alpaca MCP tools via ReAct agent.
    
    The agent autonomously calls the necessary tools to gather portfolio information.
    This is the first node in the workflow - it establishes the baseline state.
    
    Returns:
        State updates with portfolio snapshot
    """
    print("\n📊 [ASSESS] Gathering portfolio information...")
    
    try:
        all_tools = get_alpaca_mcp_tools()
        
        # Filter to only the assessment tools we want to force
        required_tool_names = {
            'get_account_info',
            'get_positions', 
            'get_orders',
            'get_market_clock'
        }
        
        assessment_tools = [t for t in all_tools if t.name in required_tool_names]
        
        print(f"📋 Using {len(assessment_tools)}/{len(all_tools)} assessment tools: {[t.name for t in assessment_tools]}")
        
        # Create agent with ONLY the assessment tools
        # This forces the agent to use these tools (it has no other options)
        llm = get_agent_llm(state["config"])
        agent: Pregel = create_agent(llm, assessment_tools)
        
        # Ask agent to gather portfolio info
        prompt = """Gather the current portfolio state. Call ALL of these tools in PARALLEL (in a single response):
1. get_account_info - Get account information (cash, portfolio value)
2. get_positions - Get current positions
3. get_orders (with status='open') - Get open orders
4. get_market_clock - Check market clock status

After you receive all tool results, summarize the portfolio state clearly.

IMPORTANT: Call all 4 tools at once in your first response, don't wait for results before calling the next tool."""
        
        # Stream agent execution - live output as it happens
        portfolio_summary = ""
        all_messages = []
        
        for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode="values"):
            # Stream mode "values" gives us the full state at each step
            messages = chunk.get("messages", [])
            if messages:
                all_messages = messages  # Keep latest full message list
                last_msg = messages[-1]
                if hasattr(last_msg, 'type'):
                    if last_msg.type == 'ai':
                        if last_msg.content:
                            print(f"🤖 {last_msg.content}")
                            portfolio_summary = last_msg.content
                        # Show tool calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"🔧 {tc.get('name', 'unknown')}({tc.get('args', {})})")
                    elif last_msg.type == 'tool':
                        # Show full tool result (no truncation)
                        print(f"✅ {last_msg.name}:\n{last_msg.content}")
        
        # Extract structured data from tool responses
        account_info = {}
        positions = []
        
        for msg in all_messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                if msg.name == 'get_account_info':
                    # Parse account info from text response
                    import re
                    content = str(msg.content)
                    
                    # Extract key values
                    cash_match = re.search(r'Cash Balance[:\s]+\$?([\d,\.]+)', content)
                    portfolio_match = re.search(r'Portfolio Value[:\s]+\$?([\d,\.]+)', content)
                    buying_power_match = re.search(r'Buying Power[:\s]+\$?([\d,\.]+)', content)
                    
                    if cash_match:
                        account_info['cash'] = float(cash_match.group(1).replace(',', ''))
                    if portfolio_match:
                        account_info['portfolio_value'] = float(portfolio_match.group(1).replace(',', ''))
                    if buying_power_match:
                        account_info['buying_power'] = float(buying_power_match.group(1).replace(',', ''))
                
                elif msg.name == 'get_positions':
                    # Parse positions from text response
                    import re
                    content = str(msg.content)
                    
                    # Find all position blocks
                    symbol_pattern = r'Symbol:\s*(\w+)'
                    qty_pattern = r'Quantity:\s*([\d,\.]+)\s*shares'
                    value_pattern = r'Market Value:\s*\$?([\d,\.]+)'
                    entry_pattern = r'Average Entry Price:\s*\$?([\d,\.]+)'
                    current_pattern = r'Current Price:\s*\$?([\d,\.]+)'
                    unrealized_pattern = r'Unrealized.*?\$?([-\d,\.]+)\s*\(([-+\d\.]+)%\)'
                    
                    # Split by symbol to get individual positions
                    position_blocks = re.split(r'(?=Symbol:)', content)
                    
                    for block in position_blocks:
                        symbol_match = re.search(symbol_pattern, block)
                        if symbol_match:
                            symbol = symbol_match.group(1)
                            qty_match = re.search(qty_pattern, block)
                            value_match = re.search(value_pattern, block)
                            entry_match = re.search(entry_pattern, block)
                            current_match = re.search(current_pattern, block)
                            unrealized_match = re.search(unrealized_pattern, block)
                            
                            position = {
                                'symbol': symbol,
                                'qty': float(qty_match.group(1).replace(',', '')) if qty_match else 0,
                                'market_value': float(value_match.group(1).replace(',', '')) if value_match else 0,
                                'avg_entry_price': float(entry_match.group(1).replace(',', '')) if entry_match else 0,
                                'current_price': float(current_match.group(1).replace(',', '')) if current_match else 0,
                            }
                            
                            if unrealized_match:
                                position['unrealized_plpc'] = float(unrealized_match.group(2).replace(',', '')) / 100
                            
                            positions.append(position)
        
        
        return {
            "phase": "assess",
            "last_summary": portfolio_summary or "Portfolio assessed",
            "account": account_info,
            "positions": positions
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error assessing portfolio: {e}")
        return {
            "phase": "error",
            "error": str(e)
        }


# ==================== Stock Selection (with Market Research) ====================

def select_stocks_node(state: PortfolioState) -> Dict[str, Any]:
    """
    Select 0-3 stocks for deep analysis using LLM recommendations.
    
    This unified node:
    1. Takes current portfolio state (positions, cash, orders)
    2. Uses LLM to recommend stocks based on portfolio needs
    3. Recommends 0-3 specific tickers for analysis
    
    Input from state:
    - account: Current cash and portfolio value
    - positions: Current holdings and performance
    - recently_analyzed: Stocks analyzed recently (to avoid redundancy)
    
    Returns:
        State updates with selected stocks
    """
    print("\n🎯 [SELECT] Selecting stocks for analysis...")
    
    try:
        config = state.get("config", {})
        max_analyses = config.get("max_stocks_to_analyze", 3)
        
        # Check if stock selection is enabled
        if not config.get("enable_stock_selection", True):
            print("ℹ️  Stock selection disabled, no stocks selected")
            return {
                "stocks_to_analyze": []
            }
        
        llm = get_quick_llm(config)
        
        # Gather portfolio state
        account = state.get("account", {})
        positions = state.get("positions", [])
        recently_analyzed = state.get("recently_analyzed", {})
        
        # Build exclusion list (stocks analyzed in past 3 days)
        stocks_to_exclude = set()
        for item in recently_analyzed.get("recently_analyzed", []):
            if item.get("days_ago", 999) < 3:
                stocks_to_exclude.add(item["ticker"])
        
        position_tickers: list[str] = [
            str(p.get("symbol")) for p in positions if p.get("symbol")
        ]
        
        # Build comprehensive prompt for research + selection
        prompt = f"""You are a portfolio manager. Recommend 1-{max_analyses} stocks for deep analysis.

CURRENT PORTFOLIO STATE:
💰 Cash Available: ${account.get('cash', 0):,.2f}
📦 Positions: {len(positions)}
{chr(10).join([f"  - {p.get('symbol')}: {p.get('qty')} shares @ ${p.get('current_price', 0):.2f}, P&L: {(p.get('unrealized_plpc', 0) * 100):+.1f}%" for p in positions]) if positions else "  (Empty portfolio - perfect opportunity to find new positions!)"}

🚫 DO NOT RECOMMEND: {', '.join(sorted(stocks_to_exclude)) if stocks_to_exclude else 'None'}
   (Analyzed in past 3 days)

TASK:
Based on your knowledge of markets, recommend 1-{max_analyses} high-quality stocks for analysis.

Consider:
- Existing positions: {', '.join(position_tickers) if position_tickers else 'None (empty portfolio - recommend growth stocks!)'}
- Diversification across sectors
- Liquid, established companies (avoid penny stocks and microcaps)
- Mix of growth and value opportunities

RESPONSE FORMAT - CRITICAL:
Output ONLY valid JSON. No markdown code blocks, no explanations, no extra text.
Start your response with {{ and end with }}.

Required JSON structure:
{{
  "stocks": [
    {{"ticker": "TICKER", "rationale": "Specific reason for recommendation", "sector": "sector"}}
  ]
}}

**MANDATORY RULES**:
1. Output ONLY the JSON object, nothing else
2. Do NOT wrap in ```json``` or ``` tags
3. The "stocks" array MUST contain 1-{max_analyses} stocks
4. Use only well-known ticker symbols (e.g., AAPL, MSFT, GOOGL, NVDA, META, TSLA, AMD, etc.)
"""
        
        print(f"🔎 Selecting stocks for analysis...")
        
        response = llm.invoke([
            SystemMessage(content="""You are a portfolio manager. Recommend stocks for analysis.

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.
Start your response with { and end with }. Do not wrap in ```json``` tags."""),
            HumanMessage(content=prompt)
        ])
        
        # Parse response
        import json
        import re
        content = str(response.content)
        
        # Extract JSON from response - handle both raw JSON and markdown wrapped
        # First try to extract from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Otherwise extract raw JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
            else:
                json_str = None
        
        stocks: list[str] = []
        
        if json_str:
            try:
                data = json.loads(json_str)
                
                # Extract from "stocks" field
                recommended = data.get("stocks", [])
                
                # Extract tickers
                for rec in recommended:
                    ticker = rec.get("ticker", "").upper()
                    rationale = rec.get("rationale", "")
                    sector = rec.get("sector", "Unknown")
                    
                    if ticker and ticker not in stocks_to_exclude:
                        stocks.append(ticker)
                        print(f"  • {ticker} ({sector}): {rationale}")
                        
                        # Show why it was selected
                        if any(p.get('symbol') == ticker for p in positions):
                            print(f"    → Existing position review")
                
                # Limit to max_analyses
                stocks = stocks[:max_analyses]
                
                if stocks:
                    print(f"\n🎯 Recommended Stocks ({len(stocks)})")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Agent response is not valid JSON. Response:")
                print(content)
                print("\nNo stocks selected.")
        else:
            print("⚠️  Agent response is not valid JSON. Response:")
            print(content)
            print("\nNo stocks selected.")
        
        return {
            "stocks_to_analyze": stocks
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in stock selection: {e}")
        return {
            "stocks_to_analyze": []
        }


# ==================== Stock Analysis ====================

def analyze_stocks_node(state: PortfolioState) -> Dict[str, Any]:
    """
    Run TradingAgents analysis for selected stocks.
    
    This node runs the existing TradingAgentsGraph for each selected stock.
    Each analysis produces comprehensive reports with BUY/SELL/HOLD recommendations.
    
    Returns:
        State updates with analysis results
    """
    print("\n📊 [ANALYZE] Running stock analysis...")
    
    stocks = state.get("stocks_to_analyze", [])
    
    if not stocks:
        print("ℹ️  No stocks selected for analysis")
        return {
            "analysis_results": {},
            "phase": "analyze"
        }
    
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        
        config = state.get("config", {})
        analysis_config = config.get("analysis_config", {})
        
        # Initialize TradingAgents
        trading_agents = TradingAgentsGraph(
            selected_analysts=config.get("analysis_analysts", ["market", "news", "fundamentals"]),
            debug=False,
            config=analysis_config
        )
        
        results = {}
        trade_date = datetime.now()
        
        for ticker in stocks:
            print(f"  📈 Analyzing {ticker}...")
            
            final_state, processed_signal = trading_agents.propagate(ticker, trade_date)
            
            results[ticker] = {
                "final_trade_decision": final_state.get("final_trade_decision", ""),
                "investment_plan": final_state.get("investment_plan", ""),
                "market_report": final_state.get("market_report", ""),
                "recommendation": processed_signal,
                "date": str(trade_date.date())
            }
            
            print(f"  ✅ {ticker}: {processed_signal}")
        
        print(f"✅ Analyzed {len(results)} stocks")
        
        return {
            "analysis_results": results,
            "phase": "analyze"
        }
        
    except Exception as e:
        print(f"❌ Error analyzing stocks: {e}")
        return {
            "analysis_results": {},
            "phase": "analyze",
            "error": str(e)
        }


# ==================== Trading Decisions ====================

def make_decisions_node(state: PortfolioState) -> Dict[str, Any]:
    """
    LLM makes trading decisions based on analysis results.
    
    This node:
    1. Reviews all analysis reports
    2. Considers portfolio state and constraints
    3. Decides on BUY/SELL/HOLD for each stock
    4. Generates pending trades for approval (if HITL enabled)
    
    Returns:
        State updates with pending trades
    """
    print("\n💭 [DECIDE] Making trading decisions...")
    
    try:
        config = state.get("config", {})
        llm = get_quick_llm(config)
        
        account = state.get("account", {})
        positions = state.get("positions", [])
        analysis_results = state.get("analysis_results", {})
        
        if not analysis_results:
            print("ℹ️  No analysis results to decide on")
            return {
                "pending_trades": [],
                "phase": "decide"
            }
        
        # Build decision prompt
        prompt = f"""Review analysis results and decide on trades.

ACCOUNT:
Cash: ${account.get('cash', 0):,.2f}
Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
Reserve (5%): ${account.get('portfolio_value', 0) * 0.05:,.2f}
Available for purchases: ${max(0, account.get('cash', 0) - account.get('portfolio_value', 0) * 0.05):,.2f}

CURRENT POSITIONS:
{chr(10).join([f"  - {p.get('symbol')}: {p.get('qty')} shares, ${p.get('market_value', 0):,.2f} ({(p.get('unrealized_plpc', 0) * 100):+.1f}%)" for p in positions])}

ANALYSIS RESULTS:
"""
        
        for ticker, result in analysis_results.items():
            prompt += f"\n{ticker}:\n"
            prompt += f"  Recommendation: {result.get('recommendation', 'UNKNOWN')}\n"
            prompt += f"  Decision: {result.get('final_trade_decision', '')}\n"
        
        prompt += """
TASK: For each analyzed stock, autonomously decide BUY, SELL, or HOLD based on analysis.

RULES:
1. Make autonomous decisions - no human approval needed
2. Only BUY if you have sufficient cash (check available for purchases)
3. Only SELL if you have a position
4. Respect 5% cash reserve
5. Max 10% per position
6. Execute ALL recommended BUY/SELL decisions from analysis

Respond with JSON array of trades:
[
  {"ticker": "AAPL", "action": "BUY", "quantity": 0, "reasoning": "...", "order_value": 5000},
  {"ticker": "TSLA", "action": "SELL", "quantity": 10, "reasoning": "..."}
]

Or [] if no trades warranted.
"""
        
        response = llm.invoke([
            SystemMessage(content="You are a portfolio manager making trading decisions."),
            HumanMessage(content=prompt)
        ])
        
        # Parse trades
        import json
        import re
        content = str(response.content)
        
        start = content.find('[')
        end = content.rfind(']') + 1
        if start >= 0 and end > start:
            trades = json.loads(content[start:end])
            
            print(f"✅ Generated {len(trades)} trading decisions:")
            for trade in trades:
                action = trade.get("action")
                ticker = trade.get("ticker")
                qty = trade.get("quantity", 0)
                value = trade.get("order_value", 0)
                
                if action == "BUY":
                    print(f"  📈 BUY {ticker}: ${value:,.2f}")
                elif action == "SELL":
                    print(f"  📉 SELL {ticker}: {qty} shares")
                else:
                    print(f"  ⏸️  HOLD {ticker}")
            
            return {
                "pending_trades": trades,
                "phase": "decide"
            }
        else:
            print("ℹ️  No trades generated")
            return {
                "pending_trades": [],
                "phase": "decide"
            }
            
    except Exception as e:
        print(f"❌ Error making decisions: {e}")
        return {
            "pending_trades": [],
            "phase": "decide",
            "error": str(e)
        }


# ==================== Trade Execution ====================

def execute_trades_node(state: PortfolioState) -> Dict[str, Any]:
    """
    Execute approved trades using Alpaca MCP tools via ReAct agent.
    
    The agent autonomously executes trades using the appropriate tools,
    handling the order placement logic.
    
    Returns:
        State updates with executed trades
    """
    print("\n⚡ [EXECUTE] Executing trades...")
    
    pending_trades = state.get("pending_trades", [])
    
    if not pending_trades:
        print("ℹ️  No trades to execute")
        return {
            "executed_trades": [],
            "phase": "execute"
        }
    
    try:
        tools = get_alpaca_mcp_tools()
        
        # Create ReAct agent with tools
        llm = get_agent_llm(state["config"])
        agent: Pregel = create_agent(llm, tools)
        
        # Format trades for agent
        trades_str = "\n".join([
            f"- {trade['action']} {trade['ticker']}: "
            f"${trade.get('order_value', 0):,.2f} (Reason: {trade.get('reasoning', 'N/A')})"
            for trade in pending_trades
        ])
        
        # Ask agent to execute trades
        prompt = f"""Execute the following trades using the available tools:

{trades_str}

For BUY orders:
- Use place_stock_order with side="buy", type="market", notional=<dollar amount>

For SELL orders:
- Use close_position to close entire position, or place_stock_order with side="sell", qty=<shares>

Execute each trade and report the order IDs."""
        
        # Stream agent execution - live output as it happens
        for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode="values"):
            # Stream mode "values" gives us the full state at each step
            messages = chunk.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'type'):
                    if last_msg.type == 'ai':
                        if last_msg.content:
                            print(f"🤖 {last_msg.content}")
                        # Show tool calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                args = tc.get('args', {})
                                # Format trade details nicely
                                arg_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                                print(f"🔧 {tc.get('name', 'unknown')}({arg_str})")
                    elif last_msg.type == 'tool':
                        # Show full tool result (no truncation)
                        print(f"✅ {last_msg.name}:\n{last_msg.content}")
        
        # Mark all as submitted (agent handled execution)
        executed = [
            {
                **trade,
                "status": "submitted",
                "executed_at": datetime.now().isoformat()
            }
            for trade in pending_trades
        ]
        
        return {
            "executed_trades": executed,
            "phase": "execute"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error executing trades: {e}")
        return {
            "executed_trades": [],
            "phase": "execute",
            "error": str(e)
        }


# ==================== Conditional Logic ====================

def should_analyze_stocks(state: PortfolioState) -> str:
    """
    Decide whether to analyze stocks or skip to decisions.
    
    Returns:
        "analyze" if stocks selected, "decide" if no analysis needed
    """
    stocks = state.get("stocks_to_analyze", [])
    
    if stocks and len(stocks) > 0:
        return "analyze"
    else:
        return "decide"


def should_execute_trades(state: PortfolioState) -> str:
    """
    Decide whether to execute trades or complete iteration.
    
    Returns:
        "execute" if trades pending, "complete" if nothing to do
    """
    pending = state.get("pending_trades", [])
    
    if pending and len(pending) > 0:
        return "execute"
    else:
        return "complete"

