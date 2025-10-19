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
        
        # Initialize LLM
        analysis_config = config.get('analysis_config', {})
        self.llm = ChatOpenAI(
            model=analysis_config.get('quick_think_llm', 'gpt-4o-mini'),
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
        """Run a complete portfolio management iteration"""
        try:
            # Log start
            self.logger.log_system(f"Starting orchestrator iteration {self.iteration_id}")
            
            # Check market status and show portfolio summary
            market_status = get_market_clock()
            account = get_account()
            positions = get_positions()
            
            from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
            open_orders = get_alpaca_open_orders()
            
            # Log comprehensive portfolio summary
            self.logger.log_portfolio_summary(account, positions, market_status, open_orders)
            
            # Check if market is open
            if not market_status.get('is_open', False):
                self.logger.log_system("⚠️  Market is currently CLOSED. Analysis will proceed but trades may not execute immediately.")
            
            # Create initial state for orchestrator
            state = {
                'messages': [
                    HumanMessage(content=(
                        "Manage the portfolio to maximize profits. "
                        "Use web search to understand market conditions, "
                        "decide which stocks to analyze (up to 3), "
                        "review their analysis reports, and execute trading decisions."
                    ))
                ]
            }
            
            # Run orchestrator in a loop
            self.logger.log_action("Starting orchestrator agent...")
            max_iterations = 30  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                self.logger.log_system(f"\n--- Orchestrator Iteration {iteration} ---")
                
                # Run orchestrator agent
                result = self.orchestrator_agent(state)
                
                # Add agent's message to state
                state['messages'].append(result['messages'][0])
                
                # Check if agent made tool calls
                if hasattr(result['messages'][0], 'tool_calls') and result['messages'][0].tool_calls:
                    tool_calls = result['messages'][0].tool_calls
                    
                    self.logger.log_system(f"Agent requested {len(tool_calls)} tool calls")
                    
                    # Process each tool call
                    tool_messages = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', '')
                        
                        self.logger.log_tool_call(tool_name, tool_args)
                        
                        # Intercept special tools
                        if tool_name == 'request_stock_analysis':
                            result_content = self._handle_analysis_request(
                                tool_args.get('ticker', ''),
                                tool_args.get('reasoning', '')
                            )
                            tool_messages.append(
                                ToolMessage(content=result_content, tool_call_id=tool_call_id)
                            )
                        elif tool_name == 'read_analysis_report':
                            result_content = self._handle_read_report(
                                tool_args.get('ticker', ''),
                                tool_args.get('report_type', '')
                            )
                            tool_messages.append(
                                ToolMessage(content=result_content, tool_call_id=tool_call_id)
                            )
                        elif tool_name == 'get_analysis_status':
                            result_content = self._handle_analysis_status()
                            tool_messages.append(
                                ToolMessage(content=str(result_content), tool_call_id=tool_call_id)
                            )
                        elif tool_name == 'get_recently_analyzed_stocks':
                            days = tool_args.get('days', 14)
                            result_content = self._handle_recently_analyzed_stocks(days)
                            tool_messages.append(
                                ToolMessage(content=str(result_content), tool_call_id=tool_call_id)
                            )
                        else:
                            # Execute normal tools through tool node
                            # Create a temporary state for this tool call
                            temp_state = {
                                'messages': state['messages'] + [result['messages'][0]]
                            }
                            tool_result = self.tool_node.invoke(temp_state)
                            
                            # Find the matching tool message
                            for msg in tool_result['messages']:
                                if hasattr(msg, 'tool_call_id') and msg.tool_call_id == tool_call_id:
                                    tool_messages.append(msg)
                                    break
                    
                    # Add all tool messages to state
                    state['messages'].extend(tool_messages)
                    
                    # Log tool results
                    for msg in tool_messages:
                        if hasattr(msg, 'content'):
                            content_preview = msg.content[:500] if len(msg.content) > 500 else msg.content
                            self.logger.log_tool_result(content_preview)
                else:
                    # No more tool calls, agent is done
                    self.logger.log_agent("Orchestrator", result.get('orchestrator_decision', 'Complete'))
                    break
            
            if iteration >= max_iterations:
                self.logger.log_system("WARNING: Reached maximum iterations. Agent may not have completed all tasks.")
            
            # Upload reports to S3
            self.logger.log_system("Uploading reports to S3...")
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
    
    def _handle_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status."""
        return {
            "analyses_requested": self.analyses_requested,
            "analyses_remaining": self.max_analyses - self.analyses_requested,
            "analyzed_stocks": list(self.analyzed_stocks.keys())
        }
    
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
            f"- Stocks Analyzed: {self.analyses_requested}",
            f"- Tickers: {', '.join(self.analyzed_stocks.keys()) if self.analyzed_stocks else 'None'}",
            f"\nStrategy: LLM-driven with web search for market context",
            f"Focus: Maximize profits through informed, analysis-backed decisions"
        ]
        
        return "\n".join(summary_lines)

