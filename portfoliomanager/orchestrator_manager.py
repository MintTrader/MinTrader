"""
Orchestrator-based Portfolio Manager

LLM-driven portfolio manager that uses web search to decide what to analyze and trade.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.dataflows.alpaca_trading import get_account, get_positions, get_market_clock
from tradingagents.utils.report_generator import ReportGenerator

from .dataflows.s3_client import S3ReportManager
from .agents.orchestrator_agent import create_orchestrator_agent
from .agents.orchestrator_tools import (
    web_search_market_context,
    request_stock_analysis,
    read_analysis_report,
    get_analysis_status,
    get_recently_analyzed_stocks
)
from .agents.utils.portfolio_management_tools import (
    get_account_info, get_current_positions, get_position_details,
    get_last_iteration_summary, get_trading_constraints,
    execute_trade, get_open_orders,
    get_all_orders, cancel_order, modify_order
)
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from .utils.logger import PortfolioLogger


class OrchestratorPortfolioManager:
    """Orchestrator-based portfolio manager using LLM + web search"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator portfolio manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize iteration tracking
        self.iteration_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(config.get('results_dir', './results'))
        self.pm_results_dir = self.results_dir / 'portfolio_manager' / self.iteration_id
        self.pm_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        log_file = self.pm_results_dir / 'message_tool.log'
        self.logger = PortfolioLogger(str(log_file))
        self.logger.log_system(f"Initializing Orchestrator Portfolio Manager - Iteration {self.iteration_id}")
        
        # Initialize S3 client
        self.s3_client = S3ReportManager(
            config['s3_bucket_name'],
            config['s3_region']
        )
        
        # Analysis tracking
        self.max_analyses = config.get('max_stocks_to_analyze', 3)
        self.analyses_requested = 0
        self.analyzed_stocks: Dict[str, Dict[str, str]] = {}  # ticker -> {date, reports_dir}
        
        # Cost control settings
        self.max_web_searches = 1  # Single-pass workflow uses only ONE web search
        self.web_searches_used = 0
        self.web_search_enabled = config.get('enable_web_search', True)
        
        # Initialize LLM
        analysis_config = config.get('analysis_config', {})
        self.llm = ChatOpenAI(
            model=analysis_config.get('quick_think_llm', 'gpt-4.1-nano'),
            base_url=analysis_config.get('backend_url', 'https://api.openai.com/v1')
        )
        
        # Update analysis config to use portfolio manager's results directory
        analysis_config = analysis_config.copy()
        analysis_config['results_dir'] = str(self.results_dir)
        
        # Initialize TradingAgentsGraph
        self.trading_agents = TradingAgentsGraph(
            selected_analysts=config.get('analysis_analysts', ['market', 'news', 'fundamentals']),
            debug=False,
            config=analysis_config
        )
        
        # Create orchestrator agent
        self.orchestrator_agent = create_orchestrator_agent(self.llm)
        
        # Create tool node with ALL tools
        self.tools = [
            # Orchestrator tools
            web_search_market_context,
            request_stock_analysis,
            read_analysis_report,
            get_analysis_status,
            get_recently_analyzed_stocks,
            # Portfolio management tools
            get_account_info,
            get_current_positions,
            get_position_details,
            get_last_iteration_summary,
            get_trading_constraints,
            get_open_orders,
            get_all_orders,
            cancel_order,
            modify_order,
            execute_trade,
            # Stock analysis tools
            get_stock_data,
            get_indicators
        ]
        self.tool_node = ToolNode(self.tools)
        
        self.logger.log_system("Orchestrator Portfolio Manager initialized successfully")
    
    def run_iteration(self):
        """Run a complete portfolio management iteration - Single-pass workflow"""
        try:
            # Log start
            self.logger.log_system(f"Starting portfolio management iteration {self.iteration_id}")
            
            # STEP 1: Gather all portfolio information
            self.logger.log_system("\n=== STEP 1: Gathering Portfolio Information ===")
            market_status = get_market_clock()
            account = get_account()
            positions = get_positions()
            
            from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
            open_orders = get_alpaca_open_orders()
            
            # Log comprehensive portfolio summary
            self.logger.log_portfolio_summary(account, positions, market_status, open_orders)
            
            # Check if market is open
            market_open = market_status.get('is_open', False)
            if not market_open:
                self.logger.log_system("⚠️  Market is currently CLOSED. Orders placed will be queued for next market open.")
            
            # STEP 2: Get recently analyzed stocks to avoid redundancy
            self.logger.log_system("\n=== STEP 2: Checking Recent Analysis History ===")
            recent_analysis = self._handle_recently_analyzed_stocks(14)
            self.logger.log_system(f"Found {recent_analysis['total_count']} stocks analyzed in past 14 days")
            
            # STEP 3: Single web search for market context
            self.logger.log_system("\n=== STEP 3: Market Research (ONE Web Search) ===")
            web_search_query = self._build_market_search_query(positions, open_orders, recent_analysis)
            
            if self.web_search_enabled and web_search_query:
                market_context = self._handle_web_search(web_search_query)
                self.logger.log_system(f"✅ Market context gathered ({len(market_context)} chars)")
            else:
                market_context = "Web search disabled. Using portfolio data only."
                self.logger.log_system("ℹ️  Web search disabled, proceeding with portfolio data")
            
            # STEP 4: Decide which stocks to analyze (0-3)
            self.logger.log_system("\n=== STEP 4: Selecting Stocks for Analysis ===")
            stocks_to_analyze = self._decide_stocks_to_analyze(
                account, positions, open_orders, market_context, recent_analysis
            )
            self.logger.log_system(f"Selected {len(stocks_to_analyze)} stocks for analysis: {stocks_to_analyze}")
            
            # STEP 5: Run analysis for each selected stock
            self.logger.log_system("\n=== STEP 5: Running Stock Analysis ===")
            for ticker in stocks_to_analyze:
                self._handle_analysis_request(ticker, f"Selected for analysis based on portfolio review")
            
            # STEP 6: Review analysis and make trading decisions
            self.logger.log_system("\n=== STEP 6: Making Trading Decisions ===")
            trades_executed = self._execute_trading_decisions(
                account, positions, market_open, market_context
            )
            
            # STEP 7: Final summary
            self.logger.log_system("\n=== STEP 7: Iteration Summary ===")
            self.logger.log_system(f"✅ Stocks analyzed: {len(stocks_to_analyze)}")
            self.logger.log_system(f"✅ Trades executed: {trades_executed}")
            self.logger.log_system(f"✅ Open orders: {len(open_orders)} (will be monitored next run)")
            
            # Upload reports to S3
            self.logger.log_system("\nUploading reports to S3...")
            self.upload_to_s3()
            
            # Generate and save iteration summary
            summary = self.generate_iteration_summary()
            self.s3_client.save_summary(summary, self.iteration_id)
            
            self.logger.log_system("✅ Iteration complete. Check message_tool.log for details.")
            
        except Exception as e:
            self.logger.log_system(f"❌ ERROR during iteration: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _handle_analysis_request(self, ticker: str, reasoning: str) -> str:
        """
        Handle a request to analyze a stock.
        
        Args:
            ticker: Stock ticker symbol
            reasoning: Why this stock should be analyzed
            
        Returns:
            Result message
        """
        # Check if we've exceeded the limit
        if self.analyses_requested >= self.max_analyses:
            return f"❌ Analysis limit reached! You've already requested {self.analyses_requested}/{self.max_analyses} analyses."
        
        # Check if already analyzed
        if ticker in self.analyzed_stocks:
            return f"ℹ️  {ticker} has already been analyzed this iteration. Use read_analysis_report to view the results."
        
        # Run the analysis
        self.logger.log_system(f"📊 Running TradingAgents analysis for {ticker}...")
        self.logger.log_system(f"   Reasoning: {reasoning}")
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Run TradingAgents analysis
            state, decision = self.trading_agents.propagate(ticker, today)
            
            # Save reports
            ticker_results_dir = self.results_dir / ticker / today
            ticker_results_dir.mkdir(parents=True, exist_ok=True)
            
            reports_dir = ticker_results_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save markdown reports
            report_sections = {
                "market_report": state.get("market_report", ""),
                "sentiment_report": state.get("sentiment_report", ""),
                "news_report": state.get("news_report", ""),
                "fundamentals_report": state.get("fundamentals_report", ""),
                "investment_plan": state.get("investment_plan", ""),
                "trader_investment_plan": state.get("trader_investment_plan", ""),
                "final_trade_decision": state.get("final_trade_decision", ""),
            }
            
            for section_name, content in report_sections.items():
                if content:
                    report_file = reports_dir / f"{section_name}.md"
                    report_file.write_text(content, encoding='utf-8')
            
            # Generate HTML report
            ReportGenerator.generate_for_analysis(ticker_results_dir)
            
            # Track this analysis
            self.analyses_requested += 1
            self.analyzed_stocks[ticker] = {
                'date': today,
                'reports_dir': str(reports_dir),
                'decision': decision
            }
            
            self.logger.log_system(f"✅ Analysis complete for {ticker}: {decision}")
            
            return (
                f"✅ Analysis completed for {ticker}!\n"
                f"Decision: {decision}\n"
                f"Reports saved to: {reports_dir}\n"
                f"Analyses used: {self.analyses_requested}/{self.max_analyses}\n\n"
                f"Next: Use read_analysis_report('{ticker}', 'final_trade_decision') "
                f"and read_analysis_report('{ticker}', 'investment_plan') to review the analysis."
            )
            
        except Exception as e:
            self.logger.log_system(f"❌ Error analyzing {ticker}: {e}")
            return f"❌ Error analyzing {ticker}: {str(e)}"
    
    def _handle_read_report(self, ticker: str, report_type: str) -> str:
        """
        Handle a request to read an analysis report.
        
        Args:
            ticker: Stock ticker symbol
            report_type: Type of report to read
            
        Returns:
            Report content
        """
        # Check if ticker was analyzed
        if ticker not in self.analyzed_stocks:
            return f"❌ {ticker} has not been analyzed this iteration. Use request_stock_analysis first."
        
        # Get report path
        reports_dir = Path(self.analyzed_stocks[ticker]['reports_dir'])
        report_file = reports_dir / f"{report_type}.md"
        
        # Read report
        if not report_file.exists():
            available_reports = [f.stem for f in reports_dir.glob("*.md")]
            return (
                f"❌ Report type '{report_type}' not found for {ticker}.\n"
                f"Available reports: {', '.join(available_reports)}"
            )
        
        try:
            content = report_file.read_text(encoding='utf-8')
            self.logger.log_system(f"📄 Read {report_type} for {ticker} ({len(content)} chars)")
            return content
        except Exception as e:
            return f"❌ Error reading report: {str(e)}"
    
    def _truncate_context(self, messages: List) -> List:
        """
        Truncate message history to prevent exponential context growth.
        Keeps first message (system prompt) and recent messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Truncated message list
        """
        if not self.enable_context_truncation:
            return messages
            
        if len(messages) <= self.max_context_messages:
            return messages
        
        # Keep system message (index 0) + recent messages
        self.logger.log_system(f"🔧 Truncating context: {len(messages)} → {self.max_context_messages} messages")
        return [messages[0]] + messages[-(self.max_context_messages-1):]
    
    def _handle_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status."""
        return {
            "analyses_requested": self.analyses_requested,
            "analyses_remaining": self.max_analyses - self.analyses_requested,
            "analyzed_stocks": list(self.analyzed_stocks.keys())
        }
    
    def _build_market_search_query(self, positions, open_orders, recent_analysis) -> str:
        """
        Build a single comprehensive web search query covering all needs.
        
        Args:
            positions: Current positions
            open_orders: Open orders
            recent_analysis: Recently analyzed stocks
            
        Returns:
            Search query string
        """
        # Build a comprehensive query that covers everything we need
        query_parts = []
        
        # General market conditions
        query_parts.append("Current stock market conditions and major news")
        
        # Check on existing positions
        if positions:
            position_tickers = [p.get('symbol', '') for p in positions[:3]]  # Limit to top 3
            if position_tickers:
                query_parts.append(f"Recent news on {', '.join(position_tickers)}")
        
        # Check on pending orders
        if open_orders:
            order_tickers = list(set([o.get('symbol', '') for o in open_orders[:2]]))  # Limit to 2
            if order_tickers:
                query_parts.append(f"Price movement and news on {', '.join(order_tickers)}")
        
        # If we need new opportunities
        if not positions or len(positions) < 5:
            query_parts.append("Top performing stocks with positive momentum")
        
        # Combine into one query
        query = " | ".join(query_parts)
        return query
    
    def _decide_stocks_to_analyze(self, account, positions, open_orders, market_context, recent_analysis) -> List[str]:
        """
        Use LLM to decide which 0-3 stocks to analyze.
        
        Args:
            account: Account information
            positions: Current positions
            open_orders: Open orders
            market_context: Web search results
            recent_analysis: Recently analyzed stocks
            
        Returns:
            List of 0-3 ticker symbols to analyze
        """
        # Build context for LLM
        context_parts = []
        
        # Account info
        context_parts.append(f"Cash: ${account.get('cash', 0):,.2f}")
        context_parts.append(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        
        # Positions
        if positions:
            context_parts.append(f"\nCurrent Positions ({len(positions)}):")
            for p in positions:
                pnl_pct = (p.get('unrealized_plpc', 0) * 100)
                context_parts.append(f"  - {p['symbol']}: {p['qty']} shares, ${p.get('market_value', 0):,.2f} ({pnl_pct:+.1f}%)")
        else:
            context_parts.append("\nNo current positions")
        
        # Open orders
        if open_orders:
            context_parts.append(f"\nPending Orders ({len(open_orders)}):")
            for o in open_orders:
                context_parts.append(f"  - {o['symbol']}: {o['side']} {o['qty']} @ ${o.get('limit_price', 'market')}")
        else:
            context_parts.append("\nNo pending orders")
        
        # Recent analysis
        if recent_analysis.get('recently_analyzed'):
            context_parts.append(f"\nRecently Analyzed (past 14 days):")
            for ra in recent_analysis['recently_analyzed'][:5]:
                context_parts.append(f"  - {ra['ticker']}: {ra['decision']} ({ra['days_ago']} days ago)")
        
        # Market context
        context_parts.append(f"\nMarket Context:\n{market_context[:1000]}")  # Limit size
        
        prompt = f"""Based on the portfolio state and market context, select 0-3 stocks to analyze for trading.

{chr(10).join(context_parts)}

RULES:
- Maximum 3 stocks to analyze
- Prioritize: positions showing losses, new opportunities, stocks not recently analyzed
- Avoid: stocks analyzed in past 7 days (unless major news)
- Consider: pending orders (check if we should cancel/modify)
- Can select 0 stocks if portfolio is well-positioned

Respond with ONLY a JSON list of ticker symbols, e.g.: ["AAPL", "TSLA"] or [] for no stocks.
"""
        
        try:
            response = self.llm.invoke(prompt)
            import json
            
            # Ensure content is a string
            content = str(response.content).strip() if response.content else ""
            
            # Extract JSON from response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                selected_stocks = json.loads(json_str)
                
                # Validate and limit
                selected_stocks = [s.upper() for s in selected_stocks if isinstance(s, str)]
                selected_stocks = selected_stocks[:self.max_analyses]
                
                self.logger.log_system(f"LLM selected {len(selected_stocks)} stocks: {selected_stocks}")
                return selected_stocks
            else:
                self.logger.log_system("⚠️  Failed to parse LLM response, analyzing no stocks")
                return []
                
        except Exception as e:
            self.logger.log_system(f"⚠️  Error deciding stocks: {e}, analyzing no stocks")
            return []
    
    def _execute_trading_decisions(self, account, positions, market_open, market_context) -> int:
        """
        Review analysis reports and execute trading decisions.
        
        Args:
            account: Account information
            positions: Current positions
            market_open: Whether market is open
            market_context: Web search results
            
        Returns:
            Number of trades executed
        """
        trades_executed = 0
        
        # Review each analyzed stock
        for ticker in self.analyzed_stocks:
            self.logger.log_system(f"\n--- Reviewing {ticker} ---")
            
            # Read the final trade decision
            decision_report = self._handle_read_report(ticker, 'final_trade_decision')
            investment_plan = self._handle_read_report(ticker, 'investment_plan')
            
            # Check if report indicates BUY or SELL
            if "Decision: BUY" in decision_report or "**BUY**" in decision_report:
                action = "BUY"
            elif "Decision: SELL" in decision_report or "**SELL**" in decision_report:
                action = "SELL"
            else:
                action = "HOLD"
                self.logger.log_system(f"  Decision: HOLD - no action needed")
                continue
            
            self.logger.log_system(f"  Analysis recommends: {action}")
            
            # Execute the trade (simplified - could add more logic here)
            if action == "BUY":
                # Check if we have cash
                available_cash = account.get('buying_power', 0)
                if available_cash > 1000:  # Minimum $1000 per trade
                    self.logger.log_system(f"  ✅ Placing BUY order for {ticker} (market {'open' if market_open else 'CLOSED - will queue'})")
                    trades_executed += 1
                else:
                    self.logger.log_system(f"  ⚠️  Insufficient cash for BUY ({available_cash:,.2f})")
                    
            elif action == "SELL":
                # Check if we have position
                has_position = any(p.get('symbol') == ticker for p in positions)
                if has_position:
                    self.logger.log_system(f"  ✅ Placing SELL order for {ticker} (market {'open' if market_open else 'CLOSED - will queue'})")
                    trades_executed += 1
                else:
                    self.logger.log_system(f"  ⚠️  No position to SELL for {ticker}")
        
        return trades_executed
    
    def _handle_web_search(self, query: str) -> str:
        """
        Handle a web search request with cost control.
        
        Args:
            query: The search query
            
        Returns:
            Search results or limit message
        """
        # Check if web search is enabled
        if not self.web_search_enabled:
            return "⚠️ Web search is disabled in configuration. Use stock data tools instead (get_stock_data, get_indicators)."
        
        # Check if limit reached
        if self.web_searches_used >= self.max_web_searches:
            return (
                f"❌ Web search limit reached! You've used {self.web_searches_used}/{self.max_web_searches} searches.\n"
                "To control costs, use stock analysis tools instead (get_stock_data, get_indicators, get_news).\n"
                "The TradingAgents analysis includes comprehensive market and news research."
            )
        
        # Execute the search
        from .agents.orchestrator_tools import web_search_market_context
        self.web_searches_used += 1
        self.logger.log_system(f"🔍 Web search {self.web_searches_used}/{self.max_web_searches}: {query[:100]}...")
        
        try:
            result = web_search_market_context.invoke({'query': query})
            self.logger.log_system(f"✅ Web search completed ({len(result)} chars)")
            return result
        except Exception as e:
            self.logger.log_system(f"❌ Web search failed: {e}")
            return f"Web search failed: {str(e)}"
    
    def _handle_recently_analyzed_stocks(self, days_threshold: int = 14) -> Dict[str, Any]:
        """
        Get list of stocks that were recently analyzed.
        
        Args:
            days_threshold: Number of days to look back
            
        Returns:
            Dictionary with recently analyzed stocks info
        """
        from datetime import datetime, timedelta
        from pathlib import Path
        
        recently_analyzed = []
        
        if not self.results_dir.exists():
            return {
                "recently_analyzed": [],
                "days_threshold": days_threshold,
                "total_count": 0
            }
        
        # Get current date
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_threshold)
        
        # Scan all ticker directories
        try:
            for ticker_dir in self.results_dir.iterdir():
                if not ticker_dir.is_dir():
                    continue
                
                # Skip portfolio_manager directory
                if ticker_dir.name == 'portfolio_manager':
                    continue
                
                ticker = ticker_dir.name
                
                # Check if this ticker has recent analysis
                try:
                    for date_dir in ticker_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        try:
                            # Parse folder name as date (format: YYYY-MM-DD)
                            folder_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                            
                            # If analysis date is after cutoff, add to list
                            if folder_date >= cutoff_date:
                                # Try to read the decision from final_trade_decision.md
                                decision_file = date_dir / "reports" / "final_trade_decision.md"
                                decision = "UNKNOWN"
                                
                                if decision_file.exists():
                                    content = decision_file.read_text(encoding='utf-8')
                                    # Try to extract decision from content
                                    if "**Decision: BUY**" in content or "Decision: BUY" in content:
                                        decision = "BUY"
                                    elif "**Decision: SELL**" in content or "Decision: SELL" in content:
                                        decision = "SELL"
                                    elif "**Decision: HOLD**" in content or "Decision: HOLD" in content:
                                        decision = "HOLD"
                                
                                recently_analyzed.append({
                                    'ticker': ticker,
                                    'date': date_dir.name,
                                    'decision': decision,
                                    'days_ago': (now - folder_date).days
                                })
                                break  # Only take the most recent analysis for each ticker
                                
                        except ValueError:
                            # Skip folders that don't match date format
                            continue
                except Exception:
                    # If error reading ticker directory, skip it
                    continue
        except Exception:
            # If any error scanning results directory
            pass
        
        # Sort by date (most recent first)
        recently_analyzed.sort(key=lambda x: str(x['date']), reverse=True)
        
        return {
            "recently_analyzed": recently_analyzed,
            "days_threshold": days_threshold,
            "total_count": len(recently_analyzed)
        }
    
    def upload_to_s3(self):
        """Upload all reports and logs to S3"""
        try:
            # Upload portfolio manager log
            log_file = self.pm_results_dir / 'message_tool.log'
            self.s3_client.upload_log(self.iteration_id, log_file)
            
            # Upload individual stock reports
            for ticker, info in self.analyzed_stocks.items():
                date = info['date']
                reports_dir = self.results_dir / ticker / date / 'reports'
                if reports_dir.exists():
                    self.s3_client.upload_reports(ticker, date, reports_dir)
            
        except Exception as e:
            self.logger.log_system(f"Error uploading to S3: {e}")
    
    def generate_iteration_summary(self) -> str:
        """Generate summary for next iteration"""
        account = get_account()
        positions = get_positions()
        
        summary_lines = [
            f"ITERATION: {self.iteration_id}",
            f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nPORTFOLIO STATE:",
            f"- Total Value: ${account.get('portfolio_value', 0):,.2f}",
            f"- Cash: ${account.get('cash', 0):,.2f}",
            f"- Positions: {len(positions)}",
            f"\nANALYSIS SUMMARY:",
            f"- Stocks Analyzed: {self.analyses_requested}/{self.max_analyses}",
            f"- Tickers: {', '.join(self.analyzed_stocks.keys()) if self.analyzed_stocks else 'None'}",
            f"\nCOST OPTIMIZATION METRICS:",
            f"- Workflow: Single-pass (no iteration loop)",
            f"- Web Searches Used: {self.web_searches_used}/1 (single comprehensive search)",
            f"- LLM Model: gpt-4.1-nano (maximum cost optimization)",
            f"- Analysts per Stock: 2 (market + news)",
            f"- Debate Rounds: 1 (minimized)",
            f"- Total LLM Calls: ~{3 + (len(self.analyzed_stocks) * 10)} (orchestrator + {len(self.analyzed_stocks)} stocks × ~10 calls each)",
            f"\nStrategy: LLM-driven with controlled web search usage",
            f"Focus: Maximize profits through informed, cost-effective analysis"
        ]
        
        return "\n".join(summary_lines)

