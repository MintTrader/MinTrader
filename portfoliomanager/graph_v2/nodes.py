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

import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from langgraph.pregel import Pregel

from .state import PortfolioState
from .mcp_adapter import get_alpaca_mcp_tools
from shared.llm_factory import get_agent_llm, get_quick_llm
from portfoliomanager.dataflows.s3_client import S3ReportManager

# Get logger for this module
logger = logging.getLogger(__name__)


# ==================== Portfolio Assessment ====================

def assess_portfolio_node(state: PortfolioState) -> Dict[str, Any]:
    """
    Gather current portfolio state using Alpaca MCP tools via ReAct agent.
    
    The agent autonomously calls the necessary tools to gather portfolio information.
    This is the first node in the workflow - it establishes the baseline state.
    
    Also fetches:
    - Last iteration summary from S3 (for continuity)
    - Recently analyzed stocks from S3 (to avoid redundant analysis)
    
    Returns:
        State updates with portfolio snapshot, last summary, and recently analyzed stocks
    """
    logger.info("📊 [ASSESS] Gathering portfolio information...")
    
    try:
        config = state.get("config", {})
        s3_bucket = config.get("s3_bucket_name")
        s3_region = config.get("s3_region", "us-east-1")
        
        # Fetch last iteration summary from S3
        last_summary = ""
        if s3_bucket:
            try:
                logger.info("📜 Fetching last iteration summary from S3...")
                s3_manager = S3ReportManager(s3_bucket, s3_region)
                last_summary = s3_manager.get_last_summary() or ""
                
                if last_summary:
                    logger.info("=" * 70)
                    logger.info("📜 LAST ITERATION SUMMARY")
                    logger.info("=" * 70)
                    logger.info(last_summary)
                    logger.info("=" * 70)
                else:
                    logger.info("ℹ️  No previous summary found (first run)")
            except Exception as e:
                logger.warning(f"Could not fetch last summary from S3: {e}")
        
        # Fetch recently analyzed stocks from S3
        recently_analyzed_data = {}
        try:
            if s3_bucket:
                logger.info("📦 Fetching recently analyzed stocks from S3...")
                s3_manager = S3ReportManager(s3_bucket, s3_region)
                
                # Get stocks analyzed in the last 14 days (2 weeks)
                stock_history = s3_manager.get_analyzed_stocks_history(days_threshold=14)
                
                # Format for state
                recently_analyzed_list = []
                for ticker, dates in stock_history.items():
                    if dates:
                        # Calculate days ago for most recent analysis
                        most_recent_date = dates[0]
                        try:
                            analysis_date = datetime.strptime(most_recent_date, "%Y-%m-%d")
                            days_ago = (datetime.now() - analysis_date).days
                            recently_analyzed_list.append({
                                "ticker": ticker,
                                "date": most_recent_date,
                                "days_ago": days_ago
                            })
                        except ValueError:
                            continue
                
                recently_analyzed_data = {
                    "recently_analyzed": recently_analyzed_list,
                    "total_count": len(recently_analyzed_list)
                }
                
                logger.info(f"📦 Found {len(recently_analyzed_list)} recently analyzed stocks in S3")
            else:
                logger.info("ℹ️  S3 not configured, skipping recently analyzed fetch")
                
        except Exception as e:
            logger.warning(f"Could not fetch recently analyzed stocks from S3: {e}")
            recently_analyzed_data = {}
        
        all_tools = get_alpaca_mcp_tools()
        
        # Filter to only the assessment tools we want to force
        required_tool_names = {
            'get_account_info',
            'get_positions', 
            'get_orders',
            'get_market_clock'
        }
        
        assessment_tools = [t for t in all_tools if t.name in required_tool_names]
        
        logger.info(f"📋 Using {len(assessment_tools)}/{len(all_tools)} assessment tools: {[t.name for t in assessment_tools]}")
        
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
                            logger.info(f"🤖 {last_msg.content}")
                            portfolio_summary = last_msg.content
                        # Show tool calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                logger.debug(f"🔧 {tc.get('name', 'unknown')}({tc.get('args', {})})")
                    elif last_msg.type == 'tool':
                        # Show full tool result (no truncation)
                        logger.debug(f"✅ {last_msg.name}:\n{last_msg.content}")
        
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
            "last_summary": last_summary,  # Last iteration summary from S3
            "account": account_info,
            "positions": positions,
            "recently_analyzed": recently_analyzed_data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Error assessing portfolio: {e}", exc_info=True)
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
    2. Uses LLM to recommend stocks based on portfolio needs and market knowledge
    3. Excludes stocks analyzed in the past 2 weeks (from S3 history)
    4. Recommends 0-3 specific tickers for analysis
    
    Input from state:
    - account: Current cash and portfolio value
    - positions: Current holdings and performance
    - recently_analyzed: Stocks analyzed recently (to avoid redundancy)
    
    Returns:
        State updates with selected stocks
    """
    logger.info("🎯 [SELECT] Selecting stocks for analysis...")
    
    try:
        config = state.get("config", {})
        max_analyses = config.get("max_stocks_to_analyze", 3)
        
        # Check if stock selection is enabled
        if not config.get("enable_stock_selection", True):
            logger.info("ℹ️  Stock selection disabled, no stocks selected")
            return {
                "stocks_to_analyze": []
            }
        
        llm = get_quick_llm(config)
        
        # Gather portfolio state
        account = state.get("account", {})
        positions = state.get("positions", [])
        recently_analyzed = state.get("recently_analyzed", {})
        
        # Build exclusion list (stocks analyzed in past 2 weeks)
        stocks_to_exclude = set()
        recently_analyzed_details = []
        for item in recently_analyzed.get("recently_analyzed", []):
            ticker = item.get("ticker")
            days_ago = item.get("days_ago", 999)
            date = item.get("date", "")
            if days_ago < 14:  # Past 2 weeks
                stocks_to_exclude.add(ticker)
                recently_analyzed_details.append(f"{ticker} ({days_ago} days ago on {date})")
        
        position_tickers: list[str] = [
            str(p.get("symbol")) for p in positions if p.get("symbol")
        ]
        
        # Build comprehensive prompt for stock selection
        prompt = f"""You are an expert portfolio manager. Your task is to recommend 1-{max_analyses} high-quality stocks for deep analysis.

CURRENT PORTFOLIO STATE:
💰 Cash Available: ${account.get('cash', 0):,.2f}
📈 Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
📦 Current Positions: {len(positions)}
{chr(10).join([f"  - {p.get('symbol')}: {p.get('qty')} shares @ ${p.get('current_price', 0):.2f}, P&L: {(p.get('unrealized_plpc', 0) * 100):+.1f}%" for p in positions]) if positions else "  (Empty portfolio - perfect opportunity to find new positions!)"}

🚫 STOCKS TO EXCLUDE (analyzed in past 2 weeks):
{chr(10).join([f"  - {detail}" for detail in recently_analyzed_details]) if recently_analyzed_details else "  None"}

DO NOT recommend any of these: {', '.join(sorted(stocks_to_exclude)) if stocks_to_exclude else 'None'}

CURRENT HOLDINGS: {', '.join(position_tickers) if position_tickers else 'None (empty portfolio)'}

SELECTION CRITERIA:
1. Diversification: Consider sectors not heavily represented in current positions
2. Quality: Focus on established companies with strong fundamentals
3. Liquidity: Avoid penny stocks and microcaps (min market cap $1B)
4. Current momentum: Look for positive price action and catalysts
5. Mix: Balance between growth and value opportunities

IMPORTANT: Do NOT recommend stocks we analyzed in the past 2 weeks (see exclusion list above).

Provide your final recommendations in this EXACT JSON format:
{{
  "stocks": [
    {{"ticker": "SYMBOL", "rationale": "Brief reason for recommendation", "sector": "sector name"}}
  ]
}}

Output ONLY the JSON. No markdown, no code blocks, no extra text."""
        
        logger.info("🔎 Selecting stocks for analysis...")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        response_content = response.content
        
        # Parse response
        import json
        import re
        content = str(response_content)
        
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
                        logger.info(f"  • {ticker} ({sector}): {rationale}")
                        
                        # Show why it was selected
                        if any(p.get('symbol') == ticker for p in positions):
                            logger.info(f"    → Existing position review")
                
                # Limit to max_analyses
                stocks = stocks[:max_analyses]
                
                if stocks:
                    logger.info(f"🎯 Recommended Stocks ({len(stocks)})")
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Agent response is not valid JSON. Response: {content}")
                logger.warning("No stocks selected.")
        else:
            logger.warning(f"⚠️  Agent response is not valid JSON. Response: {content}")
            logger.warning("No stocks selected.")
        
        return {
            "stocks_to_analyze": stocks
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Error in stock selection: {e}", exc_info=True)
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
    logger.info("📊 [ANALYZE] Running stock analysis...")
    
    stocks = state.get("stocks_to_analyze", [])
    
    if not stocks:
        logger.info("ℹ️  No stocks selected for analysis")
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
        
        # Get results directory from config
        results_base_dir = Path(analysis_config.get("results_dir", "./results"))
        
        for ticker in stocks:
            logger.info(f"  📈 Analyzing {ticker}...")
            
            # TradingAgentsGraph.propagate() will log its own progress
            final_state, processed_signal = trading_agents.propagate(ticker, trade_date)
            
            # Save markdown reports to date-based directory structure
            # This matches what the CLI does: results/<ticker>/YYYY-MM-DD/reports/
            analysis_date_str = str(trade_date.date())
            ticker_date_dir = results_base_dir / ticker / analysis_date_str
            reports_dir = ticker_date_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual markdown reports
            report_sections = {
                "market_report": final_state.get("market_report", ""),
                "sentiment_report": final_state.get("sentiment_report", ""),
                "news_report": final_state.get("news_report", ""),
                "fundamentals_report": final_state.get("fundamentals_report", ""),
                "investment_plan": final_state.get("investment_plan", ""),
                "trader_investment_plan": final_state.get("trader_investment_plan", ""),
                "final_trade_decision": final_state.get("final_trade_decision", ""),
            }
            
            for section_name, content in report_sections.items():
                if content:
                    report_file = reports_dir / f"{section_name}.md"
                    with open(report_file, "w", encoding="utf-8") as f:
                        f.write(content)
            
            # Generate HTML report
            try:
                from tradingagents.utils.report_generator import ReportGenerator
                if ReportGenerator.generate_for_analysis(ticker_date_dir):
                    logger.info(f"  📄 Generated HTML report for {ticker}")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not generate HTML report for {ticker}: {e}")
            
            results[ticker] = {
                "final_trade_decision": final_state.get("final_trade_decision", ""),
                "investment_plan": final_state.get("investment_plan", ""),
                "market_report": final_state.get("market_report", ""),
                "recommendation": processed_signal,
                "date": analysis_date_str
            }
            
            logger.info(f"  ✅ {ticker}: {processed_signal}")
        
        logger.info(f"✅ Analyzed {len(results)} stocks")
        
        return {
            "analysis_results": results,
            "phase": "analyze"
        }
        
    except Exception as e:
        logger.error(f"❌ Error analyzing stocks: {e}", exc_info=True)
        return {
            "analysis_results": {},
            "phase": "analyze",
            "error": str(e)
        }


# ==================== Trading Decisions ====================

def make_decisions_node(state: PortfolioState) -> Dict[str, Any]:
    """
    LLM makes trading decisions and executes operations using Alpaca MCP tools.
    
    This node:
    1. Reviews all analysis reports
    2. Considers portfolio state and constraints
    3. Uses OpenAI function calling with all available Alpaca MCP tools
    4. Can perform web search and check current market conditions
    5. Autonomously executes any operations to maximize profits and maintain balanced portfolio
    
    Returns:
        State updates with executed trades
    """
    logger.info("💭 [DECIDE & EXECUTE] Making trading decisions with full tool access...")
    
    try:
        config = state.get("config", {})
        
        # Get LLM with function calling capabilities
        llm = get_agent_llm(config)
        
        # Get all Alpaca MCP tools
        all_tools = get_alpaca_mcp_tools()
        
        # Bind tools to LLM for native OpenAI function calling
        llm_with_tools = llm.bind_tools(all_tools)
        
        account = state.get("account", {})
        positions = state.get("positions", [])
        analysis_results = state.get("analysis_results", {})
        
        # Build comprehensive tool list for the prompt
        tool_descriptions = []
        for tool in all_tools:
            tool_descriptions.append(f"  - {tool.name}: {tool.description}")
        tools_text = "\n".join(tool_descriptions)
        
        # Create a set of stocks we currently hold
        held_tickers = {p.get('symbol') for p in positions}
        
        # Build comprehensive decision prompt with tool access
        prompt = f"""You are an autonomous portfolio manager with full access to trading operations via Alpaca MCP tools.

CURRENT PORTFOLIO STATE:
======================
💰 Cash Available: ${account.get('cash', 0):,.2f}
📈 Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
🏦 Cash Reserve (5%): ${account.get('portfolio_value', 0) * 0.05:,.2f}
💵 Available for Purchases: ${max(0, account.get('cash', 0) - account.get('portfolio_value', 0) * 0.05):,.2f}

CURRENT POSITIONS:
{chr(10).join([f"  • {p.get('symbol')}: {p.get('qty')} shares @ ${p.get('current_price', 0):.2f}, Market Value: ${p.get('market_value', 0):,.2f}, P&L: {(p.get('unrealized_plpc', 0) * 100):+.1f}%" for p in positions]) if positions else "  (Empty portfolio)"}

ANALYSIS RESULTS FROM THIS ITERATION:
{"=" * 40}
"""
        
        for ticker, result in analysis_results.items():
            prompt += f"\n{ticker}:\n"
            prompt += f"  Recommendation: {result.get('recommendation', 'UNKNOWN')}\n"
            prompt += f"  Decision Rationale:\n{result.get('final_trade_decision', 'N/A')}\n"
            has_position = ticker in held_tickers
            prompt += f"  Current Position: {'YES (can SELL)' if has_position else 'NO (can only BUY)'}\n"
        
        prompt += f"""

AVAILABLE TOOLS:
================
You have access to ALL Alpaca MCP tools for trading operations:

{tools_text}

CAPABILITIES:
=============
✅ You can use ANY of the above tools to execute operations
✅ You can check current market conditions using get_stock_quote, get_stock_bars, etc.
✅ You can perform web searches (via your built-in search capabilities) to get latest market news
✅ You can place market orders (place_stock_order)
✅ You can close positions (close_position)
✅ You can check orders (get_orders)
✅ You can get real-time quotes and market data
✅ You can cancel orders if needed (cancel_order_by_id, cancel_all_orders)

YOUR MISSION:
=============
Maximize profits and maintain a well-balanced portfolio according to our strategy.

STRATEGY GUIDELINES:
====================
1. 🎯 Diversification: Maintain balanced exposure across sectors
2. 💰 Cash Reserve: Always keep 5% cash reserve (${account.get('portfolio_value', 0) * 0.05:,.2f})
3. 📊 Position Sizing: Max 10% of portfolio per position (${account.get('portfolio_value', 0) * 0.10:,.2f})
4. 📈 Follow Analysis: Strongly weight the analysis results from this iteration
5. 🔍 Market Context: Check current market conditions if needed before making decisions
6. ⚡ Execute Decisively: Don't just plan - actually execute the trades using the tools
7. 🛡️ Risk Management: Only SELL positions we currently hold (see Current Positions above)
8. 💡 Be Autonomous: Make decisions and execute them - no human approval needed

DECISION PROCESS:
=================
1. Review the analysis results for each stock
2. Consider current portfolio state and strategy guidelines
3. Check current market conditions if you need more context (using get_stock_quote or get_market_clock)
4. Decide on appropriate actions (BUY, SELL, HOLD) for each analyzed stock
5. **EXECUTE the trades** using place_stock_order for BUYs, close_position for SELLs
6. Verify execution by checking order status if needed

IMPORTANT RULES:
================
• Only BUY if you have sufficient available cash
• Only SELL positions you currently hold
• Use market orders for simplicity: place_stock_order with side="buy"/"sell", type="market"
• For BUY orders, use notional=<dollar_amount> instead of qty
• For SELL orders, use close_position to close entire position or place_stock_order with qty=<shares>
• Execute ALL recommended actions from analysis (if they meet constraints)
• You can use web search to get the latest news/market context if helpful

NOW: Review the analysis, consider market conditions, and EXECUTE the appropriate trading operations to maximize profits and maintain portfolio balance."""
        
        # Create messages list for conversation
        messages = [
            SystemMessage(content="You are an autonomous portfolio manager with full trading authority."),
            HumanMessage(content=prompt)
        ]
        
        # Track executed trades
        executed_trades = []
        
        # Iteratively call LLM with tools until it decides it's done
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            logger.info(f"🔄 Decision iteration {iteration + 1}/{max_iterations}")
            
            # Invoke LLM with tools
            response = llm_with_tools.invoke(messages)
            
            # Add AI response to messages
            messages.append(response)
            
            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"🔧 LLM requested {len(response.tool_calls)} tool call(s)")
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    tool_id = tool_call.get('id', '')
                    
                    # Log the tool call
                    args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                    logger.info(f"  🔧 Calling: {tool_name}({args_str})")
                    
                    # Find and execute the tool
                    matching_tools = [t for t in all_tools if t.name == tool_name]
                    
                    if matching_tools:
                        tool = matching_tools[0]
                        try:
                            # Execute tool
                            result = tool.invoke(tool_args)
                            logger.info(f"  ✅ {tool_name} result: {str(result)[:200]}...")
                            
                            # Track trade executions
                            if tool_name == 'place_stock_order':
                                executed_trades.append({
                                    'ticker': tool_args.get('symbol', 'UNKNOWN'),
                                    'action': tool_args.get('side', 'UNKNOWN').upper(),
                                    'quantity': tool_args.get('qty', 0),
                                    'order_value': tool_args.get('notional', 0),
                                    'order_type': tool_args.get('type', 'market'),
                                    'status': 'submitted',
                                    'executed_at': datetime.now().isoformat(),
                                    'tool_result': str(result)[:500]
                                })
                            elif tool_name == 'close_position':
                                executed_trades.append({
                                    'ticker': tool_args.get('symbol', 'UNKNOWN'),
                                    'action': 'SELL',
                                    'quantity': 'ALL',
                                    'status': 'submitted',
                                    'executed_at': datetime.now().isoformat(),
                                    'tool_result': str(result)[:500]
                                })
                            
                            # Add tool result to messages
                            from langchain_core.messages import ToolMessage
                            messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_id
                            ))
                            
                        except Exception as e:
                            error_msg = f"Error executing {tool_name}: {str(e)}"
                            logger.error(f"  ❌ {error_msg}")
                            
                            # Add error to messages
                            from langchain_core.messages import ToolMessage
                            messages.append(ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_id
                            ))
                    else:
                        logger.warning(f"  ⚠️  Tool {tool_name} not found")
                
                # Continue loop - LLM may want to call more tools
                iteration += 1
                
            else:
                # No more tool calls - LLM is done
                if response.content:
                    logger.info(f"🤖 Final response: {response.content}")
                logger.info("✅ LLM completed decision-making process")
                break
        
        if iteration >= max_iterations:
            logger.warning(f"⚠️  Reached maximum iterations ({max_iterations})")
        
        logger.info(f"✅ Executed {len(executed_trades)} trade operations")
        
        return {
            "executed_trades": executed_trades,
            "phase": "execute"
        }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Error in decision making: {e}", exc_info=True)
        return {
            "executed_trades": [],
            "phase": "execute",
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
    logger.info("⚡ [EXECUTE] Executing trades...")
    
    pending_trades = state.get("pending_trades", [])
    
    if not pending_trades:
        logger.info("ℹ️  No trades to execute")
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
                            logger.info(f"🤖 {last_msg.content}")
                        # Show tool calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                args = tc.get('args', {})
                                # Format trade details nicely
                                arg_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                                logger.info(f"🔧 {tc.get('name', 'unknown')}({arg_str})")
                    elif last_msg.type == 'tool':
                        # Show full tool result (no truncation)
                        logger.debug(f"✅ {last_msg.name}:\n{last_msg.content}")
        
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
        logger.error(f"❌ Error executing trades: {e}", exc_info=True)
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


# ==================== S3 Upload ====================

def upload_results_to_s3_node(state: PortfolioState) -> Dict[str, Any]:
    """
    Upload all analysis results, summary, and logs to S3.
    
    This node runs at the end of the workflow to persist:
    - Analysis results for each stock (in results dir structure)
    - Iteration summary
    - Execution logs (if available)
    
    Returns:
        State updates with upload status
    """
    logger.info("📦 [UPLOAD] Uploading results to S3...")
    
    try:
        config = state.get("config", {})
        s3_bucket = config.get("s3_bucket_name")
        s3_region = config.get("s3_region", "us-east-1")
        
        if not s3_bucket:
            logger.info("ℹ️  S3 not configured, skipping upload")
            return {"phase": "complete"}
        
        s3_manager = S3ReportManager(s3_bucket, s3_region)
        iteration_id = state.get("iteration_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        analysis_results = state.get("analysis_results", {})
        
        # Upload analysis results for each analyzed stock
        results_dir = Path(config.get("analysis_config", {}).get("results_dir", "./results"))
        upload_count = 0
        
        logger.info(f"📊 Looking for reports in: {results_dir}")
        
        for ticker, result in analysis_results.items():
            try:
                analysis_date = result.get("date", datetime.now().strftime("%Y-%m-%d"))
                
                # Look for reports in the standard directory structure:
                # results/<ticker>/<date>/reports/ contains .md files
                # results/<ticker>/<date>/index.html is the HTML report
                reports_dir = results_dir / ticker / analysis_date / "reports"
                
                if not reports_dir.exists():
                    logger.warning(f"  ⚠️  Reports directory not found: {reports_dir}")
                    # List what's available to help debug
                    ticker_base_dir = results_dir / ticker
                    if ticker_base_dir.exists():
                        subdirs = [d.name for d in ticker_base_dir.iterdir() if d.is_dir()]
                        logger.debug(f"  Available directories in {ticker_base_dir}: {subdirs}")
                        date_dir = ticker_base_dir / analysis_date
                        if date_dir.exists():
                            date_contents = [f.name for f in date_dir.iterdir()]
                            logger.debug(f"  Contents of {date_dir}: {date_contents}")
                    continue
                
                # Count files in reports directory
                md_files = list(reports_dir.glob('*.md'))
                html_file = reports_dir.parent / 'index.html'
                
                logger.info(f"  🔍 Found {len(md_files)} markdown reports for {ticker}")
                if html_file.exists():
                    logger.info(f"  🔍 Found index.html for {ticker}")
                
                # Upload reports (this will upload .md files + index.html if it exists)
                success = s3_manager.upload_reports(
                    ticker=ticker,
                    date=analysis_date,
                    reports_dir=reports_dir
                )
                
                if success:
                    upload_count += 1
                    files_uploaded = len(md_files) + (1 if html_file.exists() else 0)
                    logger.info(f"  ✅ Uploaded {files_uploaded} files for {ticker} to S3 (reports/{ticker}/{analysis_date}/)")
                else:
                    logger.warning(f"  ⚠️  Could not upload {ticker} analysis to S3")
                    
            except Exception as e:
                logger.error(f"  ❌ Error uploading {ticker} results: {e}", exc_info=True)
        
        logger.info(f"📦 Uploaded {upload_count}/{len(analysis_results)} stock analyses to S3")
        
        # Generate intelligent summary with LLM guidance for next run
        try:
            logger.info("📝 Generating iteration summary with agent guidance...")
            
            # Prepare data for summary
            account = state.get('account', {})
            positions = state.get('positions', [])
            executed_trades = state.get('executed_trades', [])
            last_summary = state.get('last_summary', '')
            
            # Build context for LLM
            summary_prompt = f"""You are a portfolio management system. Generate a comprehensive summary of this iteration 
and provide clear guidance/tasks for the next iteration.

ITERATION: {iteration_id}
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PREVIOUS ITERATION NOTES:
{last_summary if last_summary else "(This is the first iteration)"}

CURRENT PORTFOLIO STATE:
- Cash Available: ${account.get('cash', 0):,.2f}
- Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
- Number of Positions: {len(positions)}

POSITIONS:
{chr(10).join([f"  - {p.get('symbol')}: {p.get('qty')} shares @ ${p.get('current_price', 0):.2f}, P&L: {(p.get('unrealized_plpc', 0) * 100):+.1f}%" for p in positions]) if positions else "  (No positions)"}

THIS ITERATION:
- Stocks Analyzed: {len(analysis_results)} ({', '.join(analysis_results.keys()) if analysis_results else 'None'})
- Trades Executed: {len(executed_trades)}

EXECUTED TRADES:
{chr(10).join([f"  - {t.get('action')} {t.get('ticker')}: {t.get('reasoning', 'N/A')}" for t in executed_trades]) if executed_trades else "  (No trades executed)"}

ANALYSIS RESULTS:
"""
            
            for ticker, result in analysis_results.items():
                summary_prompt += f"\n{ticker}:\n"
                summary_prompt += f"  Recommendation: {result.get('recommendation', 'UNKNOWN')}\n"
                # Include a snippet of the decision
                decision = result.get('final_trade_decision', '')[:200]
                summary_prompt += f"  Decision: {decision}...\n"
            
            summary_prompt += """

Generate a summary with these sections:

## ITERATION SUMMARY
(Brief overview of what happened this iteration)

## PORTFOLIO STATUS
(Current health, performance, concerns)

## ACTIONS TAKEN
(What trades were executed and why)

## TASKS FOR NEXT RUN
(Specific actionable tasks and focus areas for the next iteration)
- Check on [specific stocks/positions]
- Monitor [specific metrics/events]
- Consider [specific opportunities]

## NOTES
(Any important observations or context for future iterations)

Keep it concise but actionable. Focus on continuity and learning from this iteration."""
            
            # Generate summary using LLM
            llm = get_quick_llm(config)
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary: str = str(response.content)  # Ensure it's a string
            
            # Print summary before uploading
            logger.info("=" * 70)
            logger.info("📊 ITERATION SUMMARY")
            logger.info("=" * 70)
            logger.info(summary)
            logger.info("=" * 70)
            
            s3_manager.save_summary(summary, iteration_id)
            logger.info("📦 Uploaded iteration summary to S3")
            
        except Exception as e:
            logger.error(f"Could not generate/upload summary to S3: {e}")
            # Fallback to simple summary if LLM fails
            try:
                simple_summary = f"""Portfolio Manager Iteration: {iteration_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
Cash: ${account.get('cash', 0):,.2f}
Positions: {len(positions)}

Stocks Analyzed: {', '.join(analysis_results.keys()) if analysis_results else 'None'}
Trades Executed: {len(executed_trades)}
"""
                s3_manager.save_summary(simple_summary, iteration_id)
                logger.info("📦 Uploaded fallback summary to S3")
            except Exception as fallback_error:
                logger.error(f"Even fallback summary failed: {fallback_error}")
        
        # Upload logs - get the log file path from config
        try:
            log_file_path = config.get('log_file_path')
            
            if log_file_path and Path(log_file_path).exists():
                s3_manager.upload_log(iteration_id, Path(log_file_path))
                logger.info(f"📦 Uploaded logs to S3: logs/{iteration_id}/message_tool.log")
                logger.info(f"   Source: {log_file_path}")
            else:
                logger.warning(f"⚠️  No log file available for upload!")
                logger.warning(f"   log_file_path from config: {log_file_path}")
                logger.warning(f"   File exists: {Path(log_file_path).exists() if log_file_path else 'N/A'}")
                
        except Exception as e:
            logger.error(f"❌ Could not upload logs to S3: {e}")
        
        return {
            "phase": "complete"
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading to S3: {e}", exc_info=True)
        return {
            "phase": "complete",
            "error": f"S3 upload error: {str(e)}"
        }

