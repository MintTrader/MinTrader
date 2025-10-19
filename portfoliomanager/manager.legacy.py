"""
Portfolio Manager

Main orchestrator for autonomous portfolio management.
Refactored to follow TradingAgents pattern with agents using prompts and tools.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.dataflows.alpaca_trading import get_account, get_positions, get_market_clock
from tradingagents.utils.report_generator import ReportGenerator

from .dataflows.s3_client import S3ReportManager
from .agents.portfolio_agent import create_portfolio_agent
from .agents.utils.portfolio_management_tools import (
    get_account_info, get_current_positions, get_position_details,
    get_last_iteration_summary, get_trading_constraints,
    execute_trade, get_watchlist_stocks, get_open_orders,
    get_all_orders, cancel_order, modify_order
)
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from .agents.stock_screener import StockScreener
from .agents.watchlist_manager import WatchlistManager
from .agents.news_stock_discovery import NewsStockDiscovery
from .agents.portfolio_strategy_agent import create_portfolio_strategy_agent
from .utils.logger import PortfolioLogger


class PortfolioManager:
    """Main portfolio management orchestrator using agent-tool pattern"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio manager.
        
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
        self.logger.log_system(f"Initializing Portfolio Manager - Iteration {self.iteration_id}")
        
        # Initialize components
        self.s3_client = S3ReportManager(
            config['s3_bucket_name'],
            config['s3_region']
        )
        self.watchlist_manager = WatchlistManager(config.get('watchlist', []))
        
        # Max stocks to analyze per iteration
        self.max_stocks_to_analyze = config.get('max_stocks_to_analyze', 3)
        
        # Track stocks analyzed in this iteration
        self.analyzed_stocks: list[dict[str, str]] = []
        
        # Initialize LLM for portfolio agent (needed by sector analyzer)
        analysis_config = config.get('analysis_config', {})
        self.llm = ChatOpenAI(
            model=analysis_config.get('quick_think_llm', 'gpt-4o-mini'),
            base_url=analysis_config.get('backend_url', 'https://api.openai.com/v1')
        )
        
        # Initialize news-based stock discovery
        self.news_discovery = NewsStockDiscovery(config)
        
        # Initialize portfolio strategy agent (requires self.llm)
        self.strategy_agent = create_portfolio_strategy_agent(self.llm)
        
        # Update analysis config to use portfolio manager's results directory
        analysis_config = analysis_config.copy()
        analysis_config['results_dir'] = str(self.results_dir)
        
        # Initialize TradingAgentsGraph
        self.trading_agents = TradingAgentsGraph(
            selected_analysts=config.get('analysis_analysts', ['market', 'news', 'fundamentals']),
            debug=False,
            config=analysis_config
        )
        
        # Create portfolio agent
        self.portfolio_agent = create_portfolio_agent(self.llm)
        
        # Create tool node for portfolio agent
        # Include both portfolio tools and stock analysis tools
        # NOTE: screen_new_opportunities is NOT included because stock discovery
        # is handled by the manager before invoking the portfolio agent
        self.tool_node = ToolNode([
            # Portfolio management tools
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
            modify_order,
            # Stock analysis tools (from TradingAgents)
            get_stock_data,
            get_indicators
        ])
        
        self.logger.log_system("Portfolio Manager initialized successfully")
    
    def run_iteration(self):
        """Run a complete portfolio management iteration"""
        try:
            # Step 0: Check market status and show portfolio summary
            self.logger.log_system(f"Starting portfolio management iteration {self.iteration_id}")
            
            market_status = get_market_clock()
            account = get_account()
            positions = get_positions()
            
            # Get open orders
            from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
            open_orders = get_alpaca_open_orders()
            
            # Log comprehensive portfolio summary
            self.logger.log_portfolio_summary(account, positions, market_status, open_orders)
            
            # Check if market is open
            if not market_status.get('is_open', False):
                self.logger.log_system("⚠️  Market is currently CLOSED. Analysis will proceed but trades may not execute immediately.")
            
            # Step 1: Decide portfolio strategy (expand, review for sell, review for buy, or hold)
            self.logger.log_action("Analyzing portfolio and deciding strategy...")
            strategy_decision = self.strategy_agent(
                positions=positions,
                account=account,
                max_stocks=self.max_stocks_to_analyze
            )
            
            # Log strategy decision
            action = strategy_decision.get('action', 'HOLD')
            reasoning = strategy_decision.get('reasoning', 'N/A')
            stocks_to_analyze = strategy_decision.get('stocks_to_analyze', [])
            preferred_sectors = strategy_decision.get('preferred_sectors', [])
            
            self.logger.log_system(f"📊 Strategy Decision:")
            self.logger.log_system(f"   Action: {action}")
            self.logger.log_system(f"   Reasoning: {reasoning}")
            
            # Exit if HOLD decision
            if action == 'HOLD':
                self.logger.log_system("✅ Strategy: HOLD - No action needed at this time.")
                return
            
            # Step 2: Get stocks to analyze based on strategy
            if action == 'EXPAND_PORTFOLIO':
                # Discover new stocks from news
                self.logger.log_action(f"Discovering new stocks from news...")
                if preferred_sectors:
                    self.logger.log_system(f"   Preferred sectors: {', '.join(preferred_sectors)}")
                
                stocks_to_analyze = self.discover_stocks_from_news(preferred_sectors=preferred_sectors)
                
                if not stocks_to_analyze:
                    self.logger.log_system(f"No stocks found. Skipping this iteration.")
                    return
                
                self.logger.log_system(f"📰 Discovered {len(stocks_to_analyze)} stocks: {', '.join(stocks_to_analyze)}")
            
            elif action in ['REVIEW_FOR_SELL', 'REVIEW_FOR_BUY']:
                # Use stocks from strategy decision (existing positions)
                if not stocks_to_analyze:
                    self.logger.log_system(f"No stocks identified for review. Skipping this iteration.")
                    return
                
                action_text = "potential SELL" if action == 'REVIEW_FOR_SELL' else "potential additional BUY"
                self.logger.log_system(f"📊 Re-analyzing {len(stocks_to_analyze)} existing stocks for {action_text}: {', '.join(stocks_to_analyze)}")
            
            # Step 3: Run TradingAgents analysis on selected stocks
            self.logger.log_action(f"Analyzing {len(stocks_to_analyze)} stocks with TradingAgents...")
            stock_recommendations = self.run_stock_analyses(stocks_to_analyze)
            
            # Step 4: Handle recommendations based on action type
            if action == 'EXPAND_PORTFOLIO' or action == 'REVIEW_FOR_BUY':
                # Filter for BUY recommendations
                buy_recommendations = {
                    ticker: rec for ticker, rec in stock_recommendations.items()
                    if rec.get('decision') == 'BUY'
                }
                
                if not buy_recommendations:
                    self.logger.log_system("⚠️  No BUY recommendations found. Skipping portfolio agent execution.")
                    self.logger.log_system(f"   Analysis Results: {', '.join([f'{t}={r.get('decision')}' for t, r in stock_recommendations.items()])}")
                else:
                    self.logger.log_system(f"✅ Found {len(buy_recommendations)} BUY recommendation(s): {', '.join(buy_recommendations.keys())}")
                    
                    # Let portfolio agent make decisions
                    self.logger.log_action("Portfolio Agent evaluating BUY opportunities...")
                    self.run_portfolio_agent(buy_recommendations)
            
            elif action == 'REVIEW_FOR_SELL':
                # Filter for SELL recommendations
                sell_recommendations = {
                    ticker: rec for ticker, rec in stock_recommendations.items()
                    if rec.get('decision') == 'SELL'
                }
                
                if not sell_recommendations:
                    self.logger.log_system("⚠️  No SELL recommendations found. Positions holding steady.")
                    self.logger.log_system(f"   Analysis Results: {', '.join([f'{t}={r.get('decision')}' for t, r in stock_recommendations.items()])}")
                else:
                    self.logger.log_system(f"✅ Found {len(sell_recommendations)} SELL recommendation(s): {', '.join(sell_recommendations.keys())}")
                    
                    # Let portfolio agent make decisions
                    self.logger.log_action("Portfolio Agent evaluating SELL opportunities...")
                    self.run_portfolio_agent(sell_recommendations)
            
            # Step 5: Upload reports to S3
            self.logger.log_system("Uploading reports to S3...")
            self.upload_to_s3()
            
            # Step 6: Generate and save iteration summary
            summary = self.generate_iteration_summary()
            self.s3_client.save_summary(summary, self.iteration_id)
            
            self.logger.log_system("✅ Iteration complete. Check message_tool.log for details.")
            
        except Exception as e:
            self.logger.log_system(f"❌ ERROR during iteration: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _was_recently_analyzed(self, ticker: str, days_threshold: int = 14) -> bool:
        """
        Check if a stock was analyzed within the past N days.
        
        Args:
            ticker: Stock ticker symbol
            days_threshold: Number of days to look back (default: 14)
        
        Returns:
            True if stock was analyzed within threshold, False otherwise
        """
        ticker_dir = self.results_dir / ticker
        
        if not ticker_dir.exists():
            return False
        
        # Get current date
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_threshold)
        
        # Check all date folders
        try:
            for date_dir in ticker_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                try:
                    # Parse folder name as date (format: YYYY-MM-DD)
                    folder_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    
                    # If analysis date is after cutoff, stock was recently analyzed
                    if folder_date >= cutoff_date:
                        return True
                        
                except ValueError:
                    # Skip folders that don't match date format
                    continue
        except Exception:
            # If any error, assume not recently analyzed
            return False
        
        return False
    
    def _get_recently_analyzed_stocks(self, days_threshold: int = 14) -> List[str]:
        """
        Get list of all stocks analyzed within the past N days.
        
        Args:
            days_threshold: Number of days to look back (default: 14)
        
        Returns:
            List of ticker symbols that were recently analyzed
        """
        recently_analyzed: List[str] = []
        
        if not self.results_dir.exists():
            return recently_analyzed
        
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
                                recently_analyzed.append(ticker)
                                break  # No need to check other dates for this ticker
                                
                        except ValueError:
                            # Skip folders that don't match date format
                            continue
                except Exception:
                    # If error reading ticker directory, skip it
                    continue
        except Exception:
            # If any error scanning results directory
            pass
        
        return recently_analyzed
    
    def discover_stocks_from_news(self, preferred_sectors: List[str] | None = None) -> List[str]:
        """
        Discover NEW stocks from news sources using LLM web search.
        Excludes: stocks we already own (but NOT recently analyzed stocks)
        
        Args:
            preferred_sectors: Optional list of sectors to focus on
        
        Returns:
            List of NEW stock tickers to analyze
        """
        # Build exclusion list - only exclude stocks we currently own
        exclude_tickers = []
        
        # Get list of stocks we already own
        positions = get_positions()
        existing_tickers = [position['symbol'] for position in positions]
        exclude_tickers.extend(existing_tickers)
        
        # Remove duplicates
        exclude_tickers = list(set(exclude_tickers))
        
        # Log what we're excluding
        if existing_tickers:
            self.logger.log_system(f"   ├─ Excluding {len(existing_tickers)} owned stock(s): {', '.join(existing_tickers)}")
        
        # Discover stocks using LLM with web search
        news_stocks = self.news_discovery.discover_from_news(
            preferred_sectors=preferred_sectors,
            exclude_tickers=exclude_tickers
        )
        
        # Log results
        if news_stocks:
            sector_info = f" in {', '.join(preferred_sectors)}" if preferred_sectors else ""
            self.logger.log_system(f"   └─ {len(news_stocks)} new stock(s) to analyze{sector_info}: {', '.join(news_stocks)}")
        else:
            self.logger.log_system(f"   └─ No new stocks discovered")
        
        return news_stocks[:self.max_stocks_to_analyze]
    
    def run_stock_analyses(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run TradingAgents analysis on selected stocks"""
        recommendations = {}
        today = datetime.now().strftime("%Y-%m-%d")
        
        for i, ticker in enumerate(tickers, 1):
            try:
                self.logger.log_system(f"   [{i}/{len(tickers)}] Analyzing {ticker}...")
                
                # Run TradingAgents analysis
                state, decision = self.trading_agents.propagate(ticker, today)
                
                # Save to standard location: results/{ticker}/{date}/
                # This allows the "recently analyzed" check to work properly
                ticker_results_dir = self.results_dir / ticker / today
                ticker_results_dir.mkdir(parents=True, exist_ok=True)
                
                reports_dir = ticker_results_dir / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                # Track this stock for S3 upload
                self.analyzed_stocks.append({'ticker': ticker, 'date': today})
                
                # Save markdown reports from state
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
                
                recommendations[ticker] = {
                    'decision': decision,
                    'summary': state.get('final_trade_decision', ''),
                    'market_report': state.get('market_report', ''),
                    'investment_plan': state.get('investment_plan', ''),
                    'results_dir': str(ticker_results_dir)
                }
                
                # More user-friendly decision logging
                decision_emoji = {
                    'BUY': '🟢',
                    'SELL': '🔴',
                    'HOLD': '🟡'
                }.get(decision, '⚪')
                
                self.logger.log_analysis(f"   ✓ {ticker}: {decision_emoji} {decision}")
                
            except Exception as e:
                self.logger.log_system(f"   ✗ Error analyzing {ticker}: {e}")
                recommendations[ticker] = {
                    'decision': 'HOLD',
                    'summary': f'Error during analysis: {e}',
                    'error': str(e)
                }
        
        return recommendations
    
    def run_portfolio_agent(self, stock_recommendations: Dict[str, Dict[str, Any]]):
        """
        Run portfolio agent with tools to make trading decisions.
        The agent will use tools to fetch data, analyze, and execute trades.
        """
        # Create initial state
        state = {
            'messages': [
                HumanMessage(content="Analyze the portfolio and make strategic trading decisions to maximize profits.")
            ],
            'stock_recommendations': stock_recommendations
        }
        
        # Run agent in a loop to handle tool calls
        max_iterations = 20  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Run portfolio agent
            result = self.portfolio_agent(state)
            
            # Add agent's message to state
            state['messages'].append(result['messages'][0])  # type: ignore
            
            # Check if agent made tool calls
            if hasattr(result['messages'][0], 'tool_calls') and result['messages'][0].tool_calls:
                tool_calls = result['messages'][0].tool_calls
                
                self.logger.log_system(f"Agent requested {len(tool_calls)} tool calls")
                
                # Log each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    self.logger.log_tool_call(tool_name, tool_args)
                
                # Execute tools
                tool_result = self.tool_node.invoke(state)
                
                # Add tool results to state
                state['messages'].extend(tool_result['messages'])  # type: ignore
                
                # Log tool results
                for msg in tool_result['messages']:
                    if hasattr(msg, 'content'):
                        self.logger.log_tool_result(msg.content[:500] if len(msg.content) > 500 else msg.content)
            else:
                # No more tool calls, agent is done
                self.logger.log_agent("Portfolio Agent", result.get('portfolio_decision', 'Analysis complete'))
                break
        
        if iteration >= max_iterations:
            self.logger.log_system("WARNING: Reached maximum iterations. Agent may not have completed all tasks.")
    
    def upload_to_s3(self):
        """Upload all reports and logs to S3"""
        try:
            # Upload portfolio manager log
            log_file = self.pm_results_dir / 'message_tool.log'
            self.s3_client.upload_log(self.iteration_id, log_file)
            
            # Upload individual stock reports for stocks analyzed in this iteration
            for stock_info in self.analyzed_stocks:
                ticker = stock_info['ticker']
                date = stock_info['date']
                
                # Check if reports directory exists
                reports_dir = self.results_dir / ticker / date / 'reports'
                if reports_dir.exists():
                    self.s3_client.upload_reports(ticker, date, reports_dir)
            
        except Exception as e:
            self.logger.log_system(f"Error uploading to S3: {e}")
    
    def generate_iteration_summary(self) -> str:
        """Generate summary for next iteration"""
        # Get portfolio state
        account = get_account()
        positions = get_positions()
        
        summary_lines = [
            f"ITERATION: {self.iteration_id}",
            f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nPORTFOLIO STATE:",
            f"- Total Value: ${account.get('portfolio_value', 0):,.2f}",
            f"- Cash: ${account.get('cash', 0):,.2f}",
            f"- Positions: {len(positions)}",
            f"\nStrategy: Maximize profits through selective, medium-term trading",
            f"Focus: Quality positions with strong fundamentals and positive momentum"
        ]
        
        return "\n".join(summary_lines)

