"""
Trading Executor Module

Handles LLM-driven trading decisions and order execution.
"""

from typing import Dict, List, Any, Union
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage


class TradingExecutor:
    """Handles trading decision-making and order execution"""
    
    def __init__(self, orchestrator_llm, logger, analysis_handler, s3_client):
        """
        Initialize trading executor.
        
        Args:
            orchestrator_llm: LLM with tools bound
            logger: PortfolioLogger instance
            analysis_handler: AnalysisHandler instance
            s3_client: S3ReportManager instance
        """
        self.orchestrator_llm = orchestrator_llm
        self.logger = logger
        self.analysis_handler = analysis_handler
        self.s3_client = s3_client
        self.trades_executed: List[Dict[str, Any]] = []
    
    def make_trading_decisions(
        self, account, positions, open_orders, market_open, 
        market_context, last_summary, analyzed_stocks: Dict[str, Dict]
    ):
        """
        Let LLM review analysis and make trading decisions using tools.
        
        Args:
            account: Account information
            positions: Current positions
            open_orders: Current open orders
            market_open: Whether market is open
            market_context: Web search results  
            last_summary: Previous iteration summary
            analyzed_stocks: Dictionary of analyzed stocks from this iteration
        """
        # Build prompt for LLM
        prompt = self._build_trading_prompt(
            account, positions, open_orders, market_open,
            market_context, last_summary, analyzed_stocks
        )
        
        # Let LLM make decisions using tools
        self.logger.log_system("Invoking LLM to make trading decisions...")
        
        try:
            messages: List[BaseMessage] = [HumanMessage(content=prompt)]
            max_iterations = 20  # Prevent infinite loops
            
            for iteration in range(max_iterations):
                # Invoke LLM with tools
                response = self.orchestrator_llm.invoke(messages)
                
                # Response is the message itself
                messages.append(response)
                
                # Log FULL LLM response (no truncation)
                if response.content:
                    self.logger.log_system(f"\n[Iteration {iteration + 1}] LLM: {response.content}")
                else:
                    self.logger.log_system(f"\n[Iteration {iteration + 1}] LLM: (no text content)")
                
                # Check if LLM wants to use tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Process tool calls
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                        
                        self.logger.log_system(f"  🔧 Tool call: {tool_name}({tool_args})")
                        
                        # Handle trading tools
                        result = self._handle_tool_call(
                            tool_name, tool_args, account, positions, open_orders, market_open
                        )
                        
                        # Add tool response to messages
                        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
                        
                        # Check if done
                        if tool_name == 'review_and_decide':
                            self.logger.log_system("✅ LLM completed trading decisions")
                            return
                else:
                    # No more tool calls, done
                    self.logger.log_system("✅ LLM completed trading decisions (no more tool calls)")
                    return
            
            self.logger.log_system("⚠️  Reached maximum iterations, ending decision phase")
            
        except Exception as e:
            self.logger.log_system(f"❌ Error in LLM trading decisions: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_trading_prompt(
        self, account, positions, open_orders, market_open,
        market_context, last_summary, analyzed_stocks: Dict[str, Dict]
    ) -> str:
        """Build the trading decision prompt for the LLM."""
        prompt_parts = []
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("🤖 YOU ARE THE AUTONOMOUS PORTFOLIO MANAGER 🤖")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        prompt_parts.append("🚨 CRITICAL: YOU ARE FULLY AUTONOMOUS - DO NOT ASK FOR PERMISSION! 🚨")
        prompt_parts.append("")
        prompt_parts.append("You manage this entire portfolio. The analyzed stocks are RESEARCH to help you,")
        prompt_parts.append("but YOU make ALL trading decisions based on:")
        prompt_parts.append("  • Your analysis research (if available)")
        prompt_parts.append("  • Current portfolio state (positions, P&L, cash)")
        prompt_parts.append("  • Market conditions and opportunities")
        prompt_parts.append("  • Your judgment as a portfolio manager")
        prompt_parts.append("")
        prompt_parts.append("YOU MUST:")
        prompt_parts.append("  ⚠️  EXECUTE trades directly - DO NOT ask for permission or confirmation")
        prompt_parts.append("  ⚠️  MAKE decisions autonomously - DO NOT ask 'Should I?' or 'Shall I proceed?'")
        prompt_parts.append("  ⚠️  ACT on your analysis - if analysis says BUY, then BUY (if you have cash)")
        prompt_parts.append("  ⚠️  DECIDE amounts yourself - DO NOT ask user to specify amounts or percentages")
        prompt_parts.append("")
        prompt_parts.append("YOU CAN:")
        prompt_parts.append("  ✅ BUY new stocks (if you have cash)")
        prompt_parts.append("  ✅ SELL existing positions (for any valid reason)")
        prompt_parts.append("  ✅ ADD to existing positions (buy more)")
        prompt_parts.append("  ✅ REDUCE existing positions (sell some)")
        prompt_parts.append("  ✅ CANCEL pending orders (if no longer needed)")
        prompt_parts.append("  ✅ MODIFY pending orders (change price/quantity if not filled)")
        prompt_parts.append("  ✅ Make decisions on stocks NOT analyzed (based on portfolio state)")
        prompt_parts.append("  ✅ Do NOTHING if the portfolio is well-positioned")
        prompt_parts.append("")
        
        # Add schedule awareness
        prompt_parts.append("⏰ YOUR SCHEDULE (For Context):")
        prompt_parts.append("  • You run 3 times per day, Monday-Friday (US market hours)")
        prompt_parts.append("  • Morning: Market open (9:30 AM ET)")
        prompt_parts.append("  • Midday: Market midpoint (12:30 PM ET)")
        prompt_parts.append("  • Evening: 30 min before close (3:30 PM ET)")
        prompt_parts.append("  • Weekends: No runs (market closed)")
        prompt_parts.append("")
        prompt_parts.append("  This means you review the portfolio 3x daily during market hours.")
        prompt_parts.append("  Consider timing when placing orders or expecting fills.")
        prompt_parts.append("")
        
        # Show previous iteration summary (if available)
        if last_summary and "No previous iteration" not in last_summary:
            prompt_parts.append("=" * 80)
            prompt_parts.append("📋 PREVIOUS ITERATION SUMMARY - WHAT HAPPENED LAST TIME:")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
            prompt_parts.append(last_summary)
            prompt_parts.append("")
            prompt_parts.append("=" * 80)
            prompt_parts.append("END OF PREVIOUS ITERATION SUMMARY")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
        
        # Summarize what was analyzed
        if analyzed_stocks:
            prompt_parts.append(f"📊 RESEARCH COMPLETED THIS ITERATION:")
            for ticker, info in analyzed_stocks.items():
                prompt_parts.append(f"  - {ticker}: Analysis complete (decision: {info.get('decision', 'UNKNOWN')})")
            prompt_parts.append(f"\n💡 Use read_analysis_report(ticker, 'final_trade_decision') to review each analysis.")
        else:
            prompt_parts.append("ℹ️  No new analysis this iteration - make decisions based on portfolio state.")
            prompt_parts.append("")
            prompt_parts.append("=" * 80)
            prompt_parts.append("🚨 CRITICAL: USE HISTORICAL REPORTS, NOT AD-HOC TOOLS! 🚨")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
            prompt_parts.append("Since NO stocks were analyzed THIS iteration, you should:")
            prompt_parts.append("")
            prompt_parts.append("✅ USE read_historical_report(ticker, 'final_trade_decision') to read past analyses")
            prompt_parts.append("   - Full analysis includes: News, Fundamentals, Technical, Sentiment, Investment Plan")
            prompt_parts.append("   - Much more comprehensive than ad-hoc tool calls!")
            prompt_parts.append("   - Example: read_historical_report('NVDA', 'final_trade_decision')")
            prompt_parts.append("")
            prompt_parts.append("❌ DO NOT waste iterations calling get_stock_data or get_indicators repeatedly")
            prompt_parts.append("   - These tools only give you raw price/indicator data")
            prompt_parts.append("   - Historical reports already include technical analysis + much more")
            prompt_parts.append("   - Use these tools ONLY if you need very specific recent data points")
            prompt_parts.append("")
            prompt_parts.append("📊 COMPARISON:")
            prompt_parts.append("   FULL ANALYSIS (recommended):")
            prompt_parts.append("     → Market sentiment, news, fundamentals, technical, risk assessment")
            prompt_parts.append("     → Professional investment recommendation")
            prompt_parts.append("     → Takes 1 tool call: read_historical_report()")
            prompt_parts.append("")
            prompt_parts.append("   AD-HOC TOOLS (only for specific needs):")
            prompt_parts.append("     → Just raw price data and basic indicators")
            prompt_parts.append("     → No context, no news, no fundamental analysis")
            prompt_parts.append("     → Takes many tool calls and wastes iterations")
            prompt_parts.append("")
            prompt_parts.append("🎯 EFFICIENT WORKFLOW:")
            prompt_parts.append("   1. Check which stocks you're interested in")
            prompt_parts.append("   2. Read their historical reports (if analyzed recently)")
            prompt_parts.append("   3. Make trading decisions based on comprehensive analysis")
            prompt_parts.append("   4. Call review_and_decide() when done")
            prompt_parts.append("")
            prompt_parts.append("=" * 80)
        
        # Portfolio state
        cash = float(account.get('cash', 0))
        buying_power = float(account.get('buying_power', 0))
        portfolio_value = float(account.get('portfolio_value', 0))
        
        prompt_parts.append(f"\n💰 ACCOUNT STATE:")
        prompt_parts.append(f"  - Available Cash: ${cash:,.2f} ✅ USE THIS FOR BUYING")
        prompt_parts.append(f"  - Buying Power: ${buying_power:,.2f} 🚫 DO NOT USE (margin risk)")
        prompt_parts.append(f"  - Portfolio Value: ${portfolio_value:,.2f}")
        prompt_parts.append(f"\n⚠️  CRITICAL CONSTRAINT: Only use CASH, never buying power!")
        
        if positions:
            prompt_parts.append(f"\n📈 CURRENT POSITIONS ({len(positions)}):")
            for p in positions[:5]:
                pnl_pct = (p.get('unrealized_plpc', 0) * 100)
                prompt_parts.append(
                    f"  - {p['symbol']}: {p['qty']} shares, ${p.get('market_value', 0):,.2f} ({pnl_pct:+.1f}%)"
                )
            if len(positions) > 5:
                prompt_parts.append(f"  ... and {len(positions) - 5} more (use get_current_positions for full list)")
        
        if open_orders:
            prompt_parts.append(f"\n⚠️  PENDING ORDERS ({len(open_orders)}) - MAY NOT BE FILLED YET:")
            for o in open_orders[:5]:
                prompt_parts.append(
                    f"  - {o['symbol']}: {o['side']} {o['qty']} @ ${o.get('limit_price', 'market')} "
                    f"(Status: {o.get('status', 'pending')})"
                )
            if len(open_orders) > 5:
                prompt_parts.append(f"  ... and {len(open_orders) - 5} more")
        
        prompt_parts.append(f"\nMarket Status: {'OPEN' if market_open else 'CLOSED (orders will queue)'}")
        
        # Instructions
        prompt_parts.append("\n" + "="*80)
        prompt_parts.append("YOUR DECISION PROCESS:")
        prompt_parts.append("="*80)
        prompt_parts.append("")
        prompt_parts.append("STEP 1: Review available information EFFICIENTLY")
        prompt_parts.append("")
        prompt_parts.append("  PRIMARY SOURCES (Use These First):")
        prompt_parts.append("  ✅ read_analysis_report(ticker, report_type) - for newly analyzed stocks THIS iteration")
        prompt_parts.append("  ✅ read_historical_report(ticker, report_type) - for past analyses from S3")
        prompt_parts.append("     → These contain COMPREHENSIVE analysis (news, fundamentals, technical, sentiment)")
        prompt_parts.append("     → Use 'final_trade_decision' report for trading recommendations")
        prompt_parts.append("     → Much better than manually checking prices!")
        prompt_parts.append("")
        prompt_parts.append("  PORTFOLIO STATE:")
        prompt_parts.append("  • get_current_positions() - see all holdings and P&L")
        prompt_parts.append("  • get_open_orders() - check pending orders")
        prompt_parts.append("")
        prompt_parts.append("  RAW DATA TOOLS (Only if really needed):")
        prompt_parts.append("  ⚠️  get_stock_data() - raw price data (usually not needed if you have reports)")
        prompt_parts.append("  ⚠️  get_indicators() - technical indicators (usually not needed if you have reports)")
        prompt_parts.append("     → Use these ONLY if you need very specific recent data")
        prompt_parts.append("     → Don't call these repeatedly - wastes iterations!")
        prompt_parts.append("")
        prompt_parts.append("STEP 2: Make trading decisions using these tools:")
        prompt_parts.append("  • place_buy_order(ticker, order_value, reasoning)")
        prompt_parts.append("  • place_sell_order(ticker, quantity, reasoning)")
        prompt_parts.append("  • cancel_order(ticker, reasoning)")
        prompt_parts.append("  • modify_order(order_id, new_limit_price=X, new_qty=Y, reasoning)")
        prompt_parts.append("")
        prompt_parts.append("STEP 3: When done, call review_and_decide() to complete")
        prompt_parts.append("")
        prompt_parts.append("⚡ EFFICIENCY TIP:")
        prompt_parts.append("   If you want to trade a stock analyzed recently (past few days):")
        prompt_parts.append("   → Call read_historical_report(ticker, 'final_trade_decision') ONCE")
        prompt_parts.append("   → Read the recommendation and reasoning")
        prompt_parts.append("   → Make your trade decision")
        prompt_parts.append("   → Done in 2-3 iterations!")
        prompt_parts.append("")
        prompt_parts.append("   Don't spend 15 iterations calling get_stock_data repeatedly!")
        
        prompt_parts.append("\n=" * 80)
        prompt_parts.append("🚫 CRITICAL: CASH-ONLY POLICY 🚫")
        prompt_parts.append("=" * 80)
        prompt_parts.append(f"- ONLY use available CASH (${cash:,.2f}) for buying")
        prompt_parts.append("- NEVER use buying power (creates margin debt)")
        prompt_parts.append("- If insufficient cash, sell positions first to raise cash")
        
        return "\n".join(prompt_parts)
    
    def _handle_tool_call(
        self, tool_name: str, tool_args: Dict, 
        account, positions, open_orders, market_open
    ) -> str:
        """Handle tool calls from the LLM during trading decision phase."""
        from tradingagents.dataflows.alpaca_trading import (
            place_market_order, cancel_order as cancel_alpaca_order
        )
        
        # Handle analysis reading tools
        if tool_name == 'read_analysis_report':
            ticker = str(tool_args.get('ticker', ''))
            report_type = str(tool_args.get('report_type', ''))
            return self.analysis_handler.read_report(ticker, report_type)
        
        elif tool_name == 'read_historical_report':
            ticker = str(tool_args.get('ticker', ''))
            report_type = str(tool_args.get('report_type', ''))
            date = tool_args.get('date')
            return self.analysis_handler.read_historical_report(ticker, report_type, date)
        
        elif tool_name == 'get_analysis_status':
            return str(self.analysis_handler.get_status())
        
        # Handle trading execution tools
        elif tool_name == 'place_buy_order':
            return self._handle_buy_order(tool_args, account, positions, open_orders, market_open)
        
        elif tool_name == 'place_sell_order':
            return self._handle_sell_order(tool_args, positions, market_open)
        
        elif tool_name == 'cancel_order':
            return self._handle_cancel_order(tool_args, open_orders)
        
        elif tool_name == 'get_current_positions':
            return self._get_current_positions_info()
        
        elif tool_name == 'get_open_orders':
            return self._get_open_orders_info()
        
        elif tool_name == 'modify_order':
            return self._handle_modify_order(tool_args, open_orders)
        
        elif tool_name == 'review_and_decide':
            self.logger.log_system("✅ LLM signaled completion of trading decisions")
            return "Trading decision phase complete."
        
        else:
            return f"Tool {tool_name} called with args: {tool_args}"
    
    def _handle_buy_order(self, tool_args: Dict, account, positions, open_orders, market_open) -> str:
        """Handle buy order execution."""
        from tradingagents.dataflows.alpaca_trading import place_market_order
        
        ticker = str(tool_args.get('ticker', ''))
        order_value_raw = tool_args.get('order_value', 0)
        reasoning = str(tool_args.get('reasoning', ''))
        
        # CRITICAL: Use CASH only
        available_cash = float(account.get('cash', 0))
        
        try:
            order_value = float(order_value_raw)
        except (ValueError, TypeError):
            return f"❌ Invalid order_value: {order_value_raw}"
        
        # Validate cash availability
        if order_value > available_cash:
            return (
                f"❌ Insufficient CASH. Available: ${available_cash:,.2f}, "
                f"trying to invest: ${order_value:,.2f}"
            )
        
        if order_value < 1000:
            return f"❌ Order value too small. Minimum $1,000, got ${order_value:,.2f}"
        
        # Check for pending order
        has_pending = any(o.get('symbol') == ticker for o in open_orders)
        if has_pending:
            return f"⚠️  Already have pending order for {ticker}. Avoid duplicate orders!"
        
        # Log the trade
        self.logger.log_system(f"💵 BUY: {ticker} for ~${order_value:,.2f}")
        self.logger.log_system(f"   Reasoning: {reasoning}")
        
        # Execute trade
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            from tradingagents.dataflows.alpaca_common import get_data_client
            
            data_client = get_data_client()
            request = StockLatestTradeRequest(symbol_or_symbols=ticker)
            latest_trade = data_client.get_stock_latest_trade(request)
            current_price = float(latest_trade[ticker].price)
            
            qty = int(order_value / current_price)
            
            if qty < 1:
                return f"❌ Order value too small. Need at least 1 share."
            
            actual_cost = qty * current_price
            order_result = place_market_order(ticker, qty=qty, side='buy')
            
            # Track trade
            self.trades_executed.append({
                'action': 'BUY',
                'ticker': ticker,
                'value': actual_cost,
                'quantity': qty,
                'price': current_price,
                'reasoning': reasoning,
                'order_id': order_result.get('id')
            })
            
            self.logger.log_system(f"   ✅ Order placed: {qty} shares @ ${current_price:.2f}")
            
            return (
                f"✅ BUY order placed for {ticker}!\n"
                f"   • Quantity: {qty} shares\n"
                f"   • Price: ~${current_price:.2f}/share\n"
                f"   • Total: ~${actual_cost:,.2f}"
            )
            
        except Exception as e:
            self.logger.log_system(f"   ❌ Error executing BUY: {e}")
            return f"❌ Error executing BUY order for {ticker}: {str(e)}"
    
    def _handle_sell_order(self, tool_args: Dict, positions, market_open) -> str:
        """Handle sell order execution."""
        from tradingagents.dataflows.alpaca_trading import place_market_order
        
        ticker = str(tool_args.get('ticker', ''))
        quantity_raw = tool_args.get('quantity', 0)
        reasoning = str(tool_args.get('reasoning', ''))
        
        # Find position
        position = next((p for p in positions if p.get('symbol') == ticker), None)
        if not position:
            return f"❌ No position found for {ticker}. Cannot sell."
        
        available_qty = float(position.get('qty', 0))
        if quantity_raw == "all":
            quantity = available_qty
        else:
            quantity = float(quantity_raw)
        
        if quantity > available_qty:
            return f"❌ Trying to sell {quantity} shares but only have {available_qty}"
        
        if quantity <= 0:
            return f"❌ Invalid quantity: {quantity}"
        
        self.logger.log_system(f"💸 SELL: {ticker} ({quantity} shares)")
        self.logger.log_system(f"   Reasoning: {reasoning}")
        
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            from tradingagents.dataflows.alpaca_common import get_data_client
            
            data_client = get_data_client()
            request = StockLatestTradeRequest(symbol_or_symbols=ticker)
            latest_trade = data_client.get_stock_latest_trade(request)
            current_price = float(latest_trade[ticker].price)
            estimated_value = quantity * current_price
            
            order_result = place_market_order(ticker, qty=quantity, side='sell')
            
            # Track trade
            self.trades_executed.append({
                'action': 'SELL',
                'ticker': ticker,
                'quantity': quantity,
                'price': current_price,
                'value': estimated_value,
                'reasoning': reasoning,
                'order_id': order_result.get('id')
            })
            
            self.logger.log_system(f"   ✅ Order placed: {quantity} shares @ ~${current_price:.2f}")
            
            return (
                f"✅ SELL order placed for {ticker}!\n"
                f"   • Quantity: {quantity} shares\n"
                f"   • Total: ~${estimated_value:,.2f}"
            )
            
        except Exception as e:
            self.logger.log_system(f"   ❌ Error executing SELL: {e}")
            return f"❌ Error executing SELL order for {ticker}: {str(e)}"
    
    def _handle_cancel_order(self, tool_args: Dict, open_orders) -> str:
        """Handle order cancellation."""
        from tradingagents.dataflows.alpaca_trading import cancel_order as cancel_alpaca_order
        
        ticker = str(tool_args.get('ticker', ''))
        reasoning = str(tool_args.get('reasoning', ''))
        
        order = next((o for o in open_orders if o.get('symbol') == ticker), None)
        if not order:
            return f"❌ No pending order found for {ticker}"
        
        order_id = order.get('id')
        
        self.logger.log_system(f"🚫 CANCEL: Order for {ticker}")
        self.logger.log_system(f"   Reasoning: {reasoning}")
        
        try:
            cancel_alpaca_order(order_id)
            self.logger.log_system(f"   ✅ Order cancelled successfully")
            return f"✅ Order for {ticker} cancelled!"
        except Exception as e:
            self.logger.log_system(f"   ❌ Error cancelling order: {e}")
            return f"❌ Error cancelling order for {ticker}: {str(e)}"
    
    def _handle_modify_order(self, tool_args: Dict, open_orders) -> str:
        """Handle order modification."""
        order_id = tool_args.get('order_id', '')
        new_limit_price = tool_args.get('new_limit_price')
        new_qty = tool_args.get('new_qty')
        reasoning = str(tool_args.get('reasoning', ''))
        
        if not order_id:
            return "❌ order_id is required. Use get_open_orders() to see order IDs."
        
        order = next((o for o in open_orders if str(o.get('id', ''))[:8] == str(order_id)[:8]), None)
        if not order:
            return f"❌ Order {order_id} not found."
        
        self.logger.log_system(f"✏️  MODIFY ORDER: {order_id}")
        
        try:
            from .agents.utils.portfolio_management_tools import modify_order as modify_order_pm
            
            mod_params: Dict[str, Any] = {'order_id': str(order.get('id'))}
            if new_limit_price is not None:
                mod_params['limit_price'] = float(new_limit_price)
            if new_qty is not None:
                mod_params['qty'] = int(new_qty)
            
            result = modify_order_pm.invoke(mod_params)
            self.logger.log_system(f"   ✅ Order modified successfully")
            
            return f"✅ Order {order_id} modified!"
        except Exception as e:
            self.logger.log_system(f"   ❌ Error modifying order: {e}")
            return f"❌ Error modifying order: {str(e)}"
    
    def _get_current_positions_info(self) -> str:
        """Get formatted current positions information."""
        from tradingagents.dataflows.alpaca_trading import get_positions
        
        fresh_positions = get_positions()
        
        if not fresh_positions:
            return "No current positions. Portfolio is 100% cash."
        
        result_lines = [f"📈 CURRENT POSITIONS ({len(fresh_positions)}):"]
        for p in fresh_positions:
            pnl_pct = (p.get('unrealized_plpc', 0) * 100)
            result_lines.append(
                f"  • {p['symbol']}: {p['qty']} shares | "
                f"Value: ${p.get('market_value', 0):,.2f} | "
                f"P&L: {pnl_pct:+.1f}%"
            )
        
        return "\n".join(result_lines)
    
    def _get_open_orders_info(self) -> str:
        """Get formatted open orders information."""
        from tradingagents.dataflows.alpaca_trading import get_open_orders
        
        fresh_orders = get_open_orders()
        
        if not fresh_orders:
            return "✅ No pending orders."
        
        result_lines = [f"📋 PENDING ORDERS ({len(fresh_orders)}):"]
        for o in fresh_orders:
            order_id = str(o.get('id', 'N/A'))[:8]
            side = str(o.get('side', 'unknown')).upper()
            result_lines.append(
                f"  • {o.get('symbol')}: {side} {o.get('qty')} shares | "
                f"ID: {order_id}"
            )
        
        return "\n".join(result_lines)
    
    def get_trades_executed(self) -> List[Dict[str, Any]]:
        """Get list of trades executed this iteration."""
        return self.trades_executed

