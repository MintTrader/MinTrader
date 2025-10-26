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
from langchain_core.messages import SystemMessage, HumanMessage

from .state import PortfolioState
from .mcp_adapter import get_alpaca_mcp_tools
from .safe_trading_tools import get_safe_trading_tools
from shared.llm_factory import get_agent_llm
from portfoliomanager.dataflows.s3_client import S3ReportManager

# Get logger for this module
logger = logging.getLogger(__name__)


# ==================== Portfolio Assessment ====================

def assess_portfolio_node(state: PortfolioState) -> Dict[str, Any]:
    """
    [PROGRAMMATIC NODE - NO LLM]
    
    Gather current portfolio state by directly calling Alpaca APIs.
    This is pure Python code that fetches data - no LLM involved.
    
    Direct API calls:
    - Account information (cash, buying power, portfolio value)
    - Current positions (all holdings with P&L)
    - Open orders (pending trades)
    - Market clock (open/closed status)
    - Last iteration summary from S3 (for continuity)
    - Recently analyzed stocks from S3 (to avoid redundant analysis)
    
    The gathered data is provided to downstream LLM nodes in the prompt,
    eliminating the need for the LLM to waste tool calls on data fetching.
    
    Returns:
        State updates with portfolio snapshot, last summary, and recently analyzed stocks
    """
    logger.info("📊 [ASSESS] Gathering portfolio information...")
    
    try:
        # ==================== STEP 1: CHECK MARKET CLOCK FIRST ====================
        # Check if market is open BEFORE fetching any portfolio data
        # This saves time and API calls if market is closed
        
        from portfoliomanager.dataflows.alpaca_portfolio import (
            get_alpaca_account_info,
            get_alpaca_positions,
            get_alpaca_open_orders,
            get_alpaca_market_clock
        )
        
        logger.info("🕐 [STEP 1/4] Checking market status...")
        market_clock = {}
        try:
            market_clock = get_alpaca_market_clock()
            is_open = market_clock.get('is_open', False)
            
            if is_open:
                logger.info("✅ Market is OPEN - proceeding with portfolio assessment")
            else:
                next_open = market_clock.get('next_open', 'Unknown')
                logger.warning("=" * 70)
                logger.warning("🚫 MARKET IS CLOSED")
                logger.warning("=" * 70)
                logger.warning(f"Next market open: {next_open}")
                logger.warning("Skipping portfolio assessment - no trading possible")
                logger.warning("=" * 70)
                
                # Return early with market closed status
                return {
                    "phase": "market_closed",
                    "error": f"Market is closed. Next open: {next_open}",
                    "market_clock": market_clock,
                    "account": {},
                    "positions": [],
                    "open_orders": [],
                    "last_summary": ""
                }
        except Exception as e:
            logger.error(f"❌ Could not fetch market clock: {e}")
            # If we can't check market status, fail fast
            return {
                "phase": "error",
                "error": f"Could not check market status: {str(e)}",
                "market_clock": {},
                "account": {},
                "positions": [],
                "open_orders": [],
                "last_summary": ""
            }
        
        # ==================== STEP 2: FETCH LAST SUMMARY FROM S3 ====================
        
        config = state.get("config", {})
        s3_bucket = config.get("s3_bucket_name")
        s3_region = config.get("s3_region", "us-east-1")
        
        # Fetch last iteration summary from S3 (ALWAYS, even when running locally)
        last_summary = ""
        if not s3_bucket:
            logger.error("❌ S3 bucket not configured! S3 operations are REQUIRED.")
            raise ValueError("S3_BUCKET_NAME must be configured in environment variables")
        
        try:
            logger.info("📜 [STEP 2/4] Fetching last iteration summary from S3...")
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
        
        # ==================== STEP 3: FETCH PORTFOLIO DATA ====================
        
        logger.info("📊 [STEP 3/4] Fetching portfolio data from Alpaca...")
        
        # Get account information
        logger.info("  💰 Fetching account info...")
        account_info = get_alpaca_account_info()
        logger.info(f"     Cash: ${account_info.get('cash', 0):,.2f}")
        logger.info(f"     Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        logger.info(f"     Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        
        # Get current positions
        logger.info("  📈 Fetching positions...")
        positions = get_alpaca_positions()
        logger.info(f"     Found {len(positions)} positions")
        for pos in positions:
            logger.info(f"       {pos['ticker']}: {pos['qty']} shares @ ${pos['current_price']:.2f}, "
                       f"P&L: {pos['unrealized_pl_pct']:+.1f}%")
        
        # Get open orders
        logger.info("  📋 Fetching open orders...")
        open_orders = get_alpaca_open_orders()
        logger.info(f"     Found {len(open_orders)} open orders")
        for order in open_orders:
            logger.info(f"       {order['side']} {order['ticker']}: {order['qty']} shares ({order['status']})")
        
        # Format positions for state (convert to format expected by downstream nodes)
        formatted_positions = []
        for pos in positions:
            formatted_positions.append({
                'symbol': pos['ticker'],
                'qty': pos['qty'],
                'market_value': pos['market_value'],
                'avg_entry_price': pos['avg_entry_price'],
                'current_price': pos['current_price'],
                'unrealized_plpc': pos['unrealized_pl_pct'] / 100,  # Convert back to decimal
                'unrealized_pl': pos['unrealized_pl']
            })
        
        logger.info("✅ [STEP 4/4] Portfolio data fetched successfully")
        
        return {
            "phase": "assess",
            "last_summary": last_summary,
            "account": account_info,
            "positions": formatted_positions,
            "open_orders": open_orders,
            "market_clock": market_clock
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Error assessing portfolio: {e}", exc_info=True)
        return {
            "phase": "error",
            "error": str(e)
        }


# ==================== Trading Decisions ====================

def make_decisions_node(state: PortfolioState) -> Dict[str, Any]:
    """
    [LLM NODE - AUTONOMOUS TRADING]
    
    LLM makes trading decisions and executes operations using Alpaca MCP tools.
    
    This node:
    1. Reviews current portfolio state
    2. Considers portfolio constraints and strategy
    3. Uses OpenAI function calling with all available Alpaca MCP tools
    4. Can perform web search and check current market conditions
    5. Autonomously executes any operations to maximize profits and maintain balanced portfolio
    
    Returns:
        State updates with executed trades
    """
    logger.info("💭 [DECIDE & EXECUTE] Making trading decisions with full tool access...")
    
    try:
        # Check if market is closed - skip trading if so
        phase = state.get("phase", "")
        # if phase == "market_closed":
        #     logger.warning("🚫 Market is closed - skipping trading decisions")
        #     return {
        #         "phase": "market_closed",
        #         "executed_trades": []
        #     }
        
        # Check if there was an error in assessment
        if phase == "error":
            error_msg = state.get("error", "Unknown error in assessment")
            logger.error(f"❌ Cannot make decisions due to error: {error_msg}")
            return {
                "phase": "error",
                "executed_trades": []
            }
        
        config = state.get("config", {})
        
        # Import exit strategy and stock template (only in this node to save tokens)
        from .exit_strategy_manager import EXIT_STRATEGY_GUIDANCE
        from .stock_prompt_template import generate_stock_portfolio_prompt
        
        # Get LLM with function calling capabilities
        llm = get_agent_llm(config)
        
        # Get SAFE trading tools (filtered + bracket order tool)
        all_alpaca_tools = get_alpaca_mcp_tools()
        safe_tools = get_safe_trading_tools(all_alpaca_tools)
        
        # Bind safe tools to LLM for native OpenAI function calling
        llm_with_tools = llm.bind_tools(safe_tools)
        
        # Parse run count and start time from memory
        last_summary = state.get('last_summary', '')
        iteration_id = state.get("iteration_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        run_count = 1
        start_time = datetime.now()
        if last_summary and "Run #" in last_summary:
            import re
            match = re.search(r'Run #(\d+)', last_summary)
            if match:
                run_count = int(match.group(1))
            
            # Try to parse start time from iteration_id
            try:
                start_time = datetime.strptime(iteration_id, "%Y%m%d_%H%M%S")
            except:
                start_time = datetime.now()
        
        # Generate comprehensive stock portfolio prompt with live data
        # Convert state to dict for the prompt generator
        from typing import cast
        state_dict = cast(Dict[str, Any], dict(state))
        stock_context = generate_stock_portfolio_prompt(
            state=state_dict,
            iteration_count=run_count,
            start_time=start_time
        )
        
        # Build tool list for the prompt
        tool_descriptions = []
        for tool in safe_tools:
            tool_descriptions.append(f"  - {tool.name}: {tool.description}")
        tools_text = "\n".join(tool_descriptions)
        
        # Build comprehensive decision prompt with tool access
        prompt = f"""{stock_context}

AVAILABLE SAFE TOOLS:
====================
{tools_text}

{EXIT_STRATEGY_GUIDANCE}


NOW: Review the portfolio and market data, and EXECUTE appropriate bracket orders for new positions."""
        
        # Create messages list for conversation
        messages = [
            SystemMessage(content="You are an autonomous portfolio manager with full trading authority."),
            HumanMessage(content=prompt)
        ]
        
        # Track executed trades
        executed_trades = []
        
        # Make ONE LLM call - it can request multiple tool calls in parallel
        # This runs every minute, so no need for loops
        logger.info("🤖 LLM is analyzing portfolio and making trading decisions...")
        
        # First call: LLM analyzes and decides on actions
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Execute any tool calls requested by the LLM
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"🔧 LLM requested {len(response.tool_calls)} tool call(s)")
            
            from langchain_core.messages import ToolMessage
            
            # Execute all tool calls (LLM can request multiple at once)
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                tool_id = tool_call.get('id', '')
                
                # Log the tool call
                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                logger.info(f"  🔧 Calling: {tool_name}({args_str})")
                
                # Find and execute the tool
                matching_tools = [t for t in safe_tools if t.name == tool_name]
                
                if matching_tools:
                    tool = matching_tools[0]
                    try:
                        # Execute tool
                        result = tool.invoke(tool_args)
                        logger.info(f"  ✅ {tool_name} result: {str(result)[:200]}...")
                        
                        # Track trade executions (only place_bracket_order is allowed)
                        if tool_name == 'place_bracket_order':
                            executed_trades.append({
                                'ticker': tool_args.get('symbol', 'UNKNOWN'),
                                'action': tool_args.get('side', 'UNKNOWN').upper(),
                                'quantity': tool_args.get('qty', 0),
                                'order_value': tool_args.get('notional', 0),
                                'order_type': tool_args.get('type', 'market'),
                                'stop_loss_price': tool_args.get('stop_loss_price'),
                                'take_profit_price': tool_args.get('take_profit_price'),
                                'status': 'submitted',
                                'executed_at': datetime.now().isoformat(),
                                'tool_result': str(result)[:500]
                            })
                        
                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        ))
                        
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        logger.error(f"  ❌ {error_msg}")
                        
                        # Add error to messages
                        messages.append(ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_id
                        ))
                else:
                    logger.warning(f"  ⚠️  Tool {tool_name} not found")
            
            # Give LLM one final call to summarize what it did
            final_response = llm_with_tools.invoke(messages)
            if final_response.content:
                logger.info(f"🤖 Summary: {final_response.content}")
        else:
            # No tool calls - LLM decided to hold/do nothing
            if response.content:
                logger.info(f"🤖 Decision: {response.content}")
            logger.info("✅ No trades executed this minute")
        
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


# ==================== Update Summary (Agent Memory) ====================

def update_summary_node(state: PortfolioState) -> Dict[str, Any]:
    """
    [LLM NODE - MEMORY UPDATE]
    
    Generate and save iteration summary to S3.
    This summary acts as the agent's memory across runs.
    
    Tracks:
    - Portfolio state changes
    - Trades executed
    - Market context
    - Run count and timing
    - Next steps and focus areas
    
    Returns:
        State updates with completion status
    """
    logger.info("📝 [SUMMARY] Updating agent memory...")
    
    try:
        # Check if market was closed - skip summary creation entirely
        phase = state.get("phase", "")
        if phase == "market_closed":
            market_clock = state.get("market_clock", {})
            next_open = market_clock.get("next_open", "Unknown")
            
            logger.warning("🚫 Market was closed - skipping summary creation")
            logger.info("📝 No summary will be saved (market closed)")
            
            # Return early without saving anything
            return {
                "phase": "complete",
                "run_count": 0  # Don't increment run count for market closed
            }
        
        # Check if there was an error in assessment - save error summary
        if phase == "error":
            error_msg = state.get("error", "Unknown error occurred")
            iteration_id = state.get("iteration_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            logger.error("❌ Error occurred - creating error summary")
            
            summary = f"""ERROR SUMMARY
Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Iteration ID: {iteration_id}
Status: Error occurred during portfolio assessment

Error: {error_msg}

Action Taken: No trading operations performed (error in assessment)
Next Steps: Check logs and fix the issue before running again
"""
            
            # ALWAYS try to save error summary to S3
            config = state.get("config", {})
            s3_bucket = config.get("s3_bucket_name")
            s3_region = config.get("s3_region", "us-east-1")
            
            if not s3_bucket:
                logger.error("❌ S3 bucket not configured! Cannot save error summary.")
                return {
                    "phase": "complete",
                    "summary": summary,
                    "run_count": 0,
                    "error": error_msg
                }
            
            # Try to save error summary to S3
            try:
                s3_manager = S3ReportManager(s3_bucket, s3_region)
                s3_manager.save_summary(summary, iteration_id)
                logger.info("✅ Error summary saved to S3")
                logger.info(f"📁 S3 Path: s3://{s3_bucket}/portfolio_manager/summaries/")
            except Exception as e:
                logger.error(f"❌ Failed to save error summary to S3: {e}")
            
            return {
                "phase": "complete",
                "summary": summary,
                "run_count": 0,
                "error": error_msg
            }
        
        config = state.get("config", {})
        s3_bucket = config.get("s3_bucket_name")
        s3_region = config.get("s3_region", "us-east-1")
        
        # S3 operations are REQUIRED, even when running locally
        if not s3_bucket:
            logger.error("❌ S3 bucket not configured! S3 operations are REQUIRED.")
            raise ValueError("S3_BUCKET_NAME must be configured in environment variables")
        
        s3_manager = S3ReportManager(s3_bucket, s3_region)
        iteration_id = state.get("iteration_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Gather state for summary
        account = state.get('account', {})
        positions = state.get('positions', [])
        executed_trades = state.get('executed_trades', [])
        last_summary = state.get('last_summary', '')
        
        # Parse previous run count and calculate new
        run_count = 1
        if last_summary and "Run #" in last_summary:
            import re
            match = re.search(r'Run #(\d+)', last_summary)
            if match:
                run_count = int(match.group(1)) + 1
        
        # Build context for LLM summary
        summary_prompt = f"""You are a portfolio management system. Generate a comprehensive summary that will serve as MEMORY for the next run.

ITERATION: {iteration_id}
RUN NUMBER: {run_count}
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT PORTFOLIO STATE:
- Cash Available: ${account.get('cash', 0):,.2f}
- Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
- Number of Positions: {len(positions)}

POSITIONS:
{chr(10).join([f"  - {p.get('symbol')}: {p.get('qty')} shares @ ${p.get('current_price', 0):.2f}, Market Value: ${p.get('market_value', 0):,.2f}, P&L: {(p.get('unrealized_plpc', 0) * 100):+.1f}%" for p in positions]) if positions else "  (No positions)"}

THIS RUN:
- Trades Executed: {len(executed_trades)}

EXECUTED TRADES:
{chr(10).join([f"  - {t.get('action')} {t.get('ticker')}: Qty={t.get('quantity', 'N/A')}, Stop Loss=${t.get('stop_loss_price', 'N/A')}, Take Profit=${t.get('take_profit_price', 'N/A')}" for t in executed_trades]) if executed_trades else "  (No trades executed)"}

Generate a memory summary in this EXACT format:

## MEMORY SUMMARY
Run #{run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### PORTFOLIO STATUS
(Current state and performance)

### RECENT ACTIONS
(What was done this run)

### MARKET CONTEXT
(Current market conditions and observations)

### NEXT RUN FOCUS
(What to check/monitor/consider in the next run)

Keep it concise but informative. This is the agent's memory for continuity."""
        
        # Generate summary using LLM
        from shared.llm_factory import get_quick_llm
        llm = get_quick_llm(config)
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = str(response.content)
        
        # Print summary
        logger.info("=" * 70)
        logger.info("📊 AGENT MEMORY UPDATE")
        logger.info("=" * 70)
        logger.info(summary)
        logger.info("=" * 70)
        
        # Save to S3
        s3_manager.save_summary(summary, iteration_id)
        logger.info(f"✅ Updated agent memory in S3 (Run #{run_count})")
        
        return {
            "phase": "complete",
            "run_count": run_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating summary: {e}", exc_info=True)
        return {
            "phase": "complete",
            "error": str(e)
        }


