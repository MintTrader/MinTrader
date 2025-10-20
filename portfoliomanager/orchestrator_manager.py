"""
Orchestrator-based Portfolio Manager

LLM-driven portfolio manager that uses web search to decide what to analyze and trade.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
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
    get_recently_analyzed_stocks,
    read_historical_report,
    place_buy_order,
    place_sell_order,
    cancel_order as cancel_order_tool,
    review_and_decide
)
from .agents.utils.portfolio_management_tools import (
    get_account_info, get_current_positions, get_position_details,
    get_last_iteration_summary, get_trading_constraints,
    execute_trade, get_open_orders,
    get_all_orders, cancel_order as cancel_order_pm, modify_order
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
        
        # Create tool node with ALL tools
        self.tools = [
            # Orchestrator tools
            web_search_market_context,
            request_stock_analysis,
            read_analysis_report,
            get_analysis_status,
            get_recently_analyzed_stocks,
            read_historical_report,
            # Trading execution tools
            place_buy_order,
            place_sell_order,
            cancel_order_tool,
            review_and_decide,
            # Portfolio management tools
            get_account_info,
            get_current_positions,
            get_position_details,
            get_last_iteration_summary,
            get_trading_constraints,
            get_open_orders,
            get_all_orders,
            cancel_order_pm,
            modify_order,
            execute_trade,
            # Stock analysis tools
            get_stock_data,
            get_indicators
        ]
        self.tool_node = ToolNode(self.tools)
        
        # Bind tools to LLM for orchestrator agent
        self.orchestrator_llm = self.llm.bind_tools(self.tools)
        
        # Track trades executed by LLM
        self.trades_executed: List[Dict[str, Any]] = []
        
        self.logger.log_system("Orchestrator Portfolio Manager initialized successfully")
    
    def run_iteration(self):
        """Run a complete portfolio management iteration - Single-pass workflow"""
        try:
            # Log start
            self.logger.log_system(f"Starting portfolio management iteration {self.iteration_id}")
            
            # STEP 0: Retrieve last iteration summary
            self.logger.log_system("\n=== STEP 0: Retrieving Last Iteration Summary ===")
            last_summary = self.s3_client.get_last_summary()
            if last_summary:
                self.logger.log_system("🧠💭 Retrieved previous iteration summary from S3")
            else:
                self.logger.log_system("ℹ️  No previous iteration summary found (first run)")
                last_summary = "No previous iteration data available."
            
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
                # Log the FULL content (no truncation) - user wants to see everything
                self.logger.log_system(f"   Full Market Context:\n{market_context}")
            else:
                market_context = "Web search disabled. Using portfolio data only."
                self.logger.log_system("ℹ️  Web search disabled, proceeding with portfolio data")
            
            # STEP 4: Decide which stocks to analyze (0-3)
            self.logger.log_system("\n=== STEP 4: Selecting Stocks for Analysis ===")
            stocks_to_analyze = self._decide_stocks_to_analyze(
                account, positions, open_orders, market_context, recent_analysis, last_summary
            )
            self.logger.log_system(f"Selected {len(stocks_to_analyze)} stocks for analysis: {stocks_to_analyze}")
            
            # STEP 5: Run analysis for each selected stock
            self.logger.log_system("\n=== STEP 5: Running Stock Analysis ===")
            for ticker in stocks_to_analyze:
                self._handle_analysis_request(ticker, f"Selected for analysis based on portfolio review")
            
            # STEP 6: LLM reviews analysis and makes trading decisions using tools
            self.logger.log_system("\n=== STEP 6: LLM Making Trading Decisions ===")
            self._llm_trading_decisions(
                account, positions, open_orders, market_open, market_context, last_summary
            )
            
            # Refresh open orders after trades
            from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
            updated_open_orders = get_alpaca_open_orders()
            
            # STEP 7: Final summary
            self.logger.log_system("\n=== STEP 7: Iteration Summary ===")
            self.logger.log_system(f"✅ Stocks analyzed: {len(stocks_to_analyze)}")
            self.logger.log_system(f"✅ Trades executed by LLM: {len(self.trades_executed)}")
            self.logger.log_system(f"✅ Open orders: {len(updated_open_orders)}")
            
            if updated_open_orders:
                self.logger.log_system("\n📋 Pending Orders (to be monitored next run):")
                for order in updated_open_orders[:5]:  # Show first 5
                    self.logger.log_system(
                        f"   - {order.get('symbol')}: {order.get('side')} {order.get('qty')} "
                        f"@ ${order.get('limit_price', 'market')} (Status: {order.get('status')})"
                    )
                if len(updated_open_orders) > 5:
                    self.logger.log_system(f"   ... and {len(updated_open_orders) - 5} more")
            
            # Upload reports to S3
            self.logger.log_system("\nUploading reports to S3...")
            self.upload_to_s3()
            
            # Generate and save iteration summary
            summary = self.generate_iteration_summary(updated_open_orders)
            
            # LOG THE SUMMARY - so user can see what agent is writing to itself for next iteration
            self.logger.log_system("\n" + "=" * 80)
            self.logger.log_system("📝 SUMMARY FOR NEXT ITERATION (Agent's Memory)")
            self.logger.log_system("=" * 80)
            self.logger.log_system("\nThis is what the agent will read when it wakes up next time:\n")
            self.logger.log_system(summary)
            self.logger.log_system("\n" + "=" * 80)
            self.logger.log_system("END OF SUMMARY FOR NEXT ITERATION")
            self.logger.log_system("=" * 80 + "\n")
            
            # Save to S3 for next iteration
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
        Handle a request to read an analysis report from current iteration.
        
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
    
    def _handle_read_historical_report(self, ticker: str, report_type: str, date: Optional[str] = None) -> str:
        """
        Handle a request to read a historical analysis report from S3.
        
        Args:
            ticker: Stock ticker symbol
            report_type: Type of report to read
            date: Optional specific date (YYYY-MM-DD)
            
        Returns:
            Report content from S3
        """
        try:
            content = self.s3_client.get_report_from_s3(ticker, report_type, date)
            
            if content is None:
                return (
                    f"❌ Report not found: {ticker} / {report_type}" + 
                    (f" / {date}" if date else " (latest)") + "\n" +
                    "Available report types: final_trade_decision, investment_plan, market_report, " +
                    "fundamentals_report, news_report, trader_investment_plan"
                )
            
            date_used = date or "latest"
            self.logger.log_system(f"📄 Read historical {report_type} for {ticker} from S3 ({date_used}, {len(content)} chars)")
            return f"=== Historical Report: {ticker} / {report_type} / {date_used} ===\n\n{content}"
            
        except Exception as e:
            self.logger.log_system(f"❌ Error reading historical report for {ticker}: {e}")
            return f"❌ Error reading historical report: {str(e)}"
    
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
        
        # If we need new opportunities - Focus on GROWTH POTENTIAL, not just best performers
        if not positions or len(positions) < 5:
            query_parts.append("List specific stock tickers: 5-10 stocks with strong growth potential and catalysts - undervalued companies with improving fundamentals, upcoming catalysts, or emerging market opportunities (not just recent top performers)")
        
        # Combine into one query
        query = " | ".join(query_parts)
        return query
    
    def _decide_stocks_to_analyze(self, account, positions, open_orders, market_context, recent_analysis, last_summary) -> List[str]:
        """
        Use LLM to decide which 0-3 stocks to analyze.
        
        Args:
            account: Account information
            positions: Current positions
            open_orders: Open orders
            market_context: Web search results
            recent_analysis: Recently analyzed stocks
            last_summary: Previous iteration summary
            
        Returns:
            List of 0-3 ticker symbols to analyze
        """
        # Build context for LLM
        context_parts = []
        
        # ========== CRITICAL: Show exclusion list FIRST ==========
        # Build exclusion list BEFORE showing anything else
        recently_analyzed_tickers = set()
        stocks_to_exclude = set()
        stocks_recently_analyzed_warning = []
        stocks_ok_to_reanalyze = []
        
        if recent_analysis.get('recently_analyzed'):
            for ra in recent_analysis['recently_analyzed']:
                ticker = ra['ticker']
                days_ago = ra['days_ago']
                date = ra['date']
                
                # Track all recently analyzed (for position marking)
                if days_ago < 14:
                    recently_analyzed_tickers.add(ticker)
                
                # Categorize by recency
                if days_ago < 3:
                    # Exclude stocks analyzed in past 3 days
                    stocks_to_exclude.add(ticker)
                    stocks_recently_analyzed_warning.append(f"{ticker} (analyzed {days_ago} days ago on {date})")
                elif days_ago < 7:
                    # Show warning but allow if major news
                    stocks_recently_analyzed_warning.append(f"{ticker} (analyzed {days_ago} days ago on {date} - only if major news)")
                else:
                    # Older analyses, can re-analyze
                    stocks_ok_to_reanalyze.append(f"{ticker} (last analyzed {days_ago} days ago)")
        
        # ========== START PROMPT: EXCLUSION LIST FIRST AND PROMINENT ==========
        context_parts.append("=" * 80)
        context_parts.append("🚨 CRITICAL: DO NOT SELECT THESE STOCKS 🚨")
        context_parts.append("=" * 80)
        
        if stocks_to_exclude:
            # Show simple ticker list first
            excluded_tickers_str = ", ".join(sorted(stocks_to_exclude))
            context_parts.append(f"❌ EXCLUDED TICKERS: {excluded_tickers_str}")
            context_parts.append(f"❌ DO NOT SELECT: {excluded_tickers_str}")
            context_parts.append(f"❌ NEVER CHOOSE: {excluded_tickers_str}")
            context_parts.append("")
            context_parts.append(f"These {len(stocks_to_exclude)} stocks were recently analyzed:")
            for item in sorted(stocks_recently_analyzed_warning):
                if any(ticker in item for ticker in stocks_to_exclude):
                    context_parts.append(f"   • {item}")
            context_parts.append("")
            context_parts.append(f"💡 You can still TRADE these stocks using: read_historical_report(ticker, 'final_trade_decision')")
            context_parts.append(f"💡 But DO NOT select them for NEW analysis!")
        else:
            context_parts.append("✅ No recently analyzed stocks to exclude (all stocks available)")
        
        context_parts.append("")
        
        # Previous iteration context
        if last_summary and "No previous iteration" not in last_summary:
            context_parts.append("=== PREVIOUS ITERATION SUMMARY ===")
            # Extract key info from last summary
            summary_lines = last_summary.split('\n')
            for line in summary_lines[:10]:  # First 10 lines usually have key info
                if line.strip():
                    context_parts.append(line)
            context_parts.append("")
        
        # Account info
        context_parts.append("=== CURRENT ACCOUNT STATE ===")
        context_parts.append(f"Cash: ${account.get('cash', 0):,.2f}")
        context_parts.append(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        context_parts.append(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        
        # Positions - mark which ones need analysis
        if positions:
            context_parts.append(f"\nCurrent Positions ({len(positions)}):")
            positions_needing_analysis = []
            for p in positions:
                ticker = p['symbol']
                pnl_pct = (p.get('unrealized_plpc', 0) * 100)
                
                # Check if this position was recently analyzed
                if ticker in stocks_to_exclude:
                    status = "🚫 Recently analyzed - DO NOT SELECT (use historical report)"
                elif ticker in recently_analyzed_tickers:
                    status = "✅ Analyzed recently"
                else:
                    status = "⚠️  NEEDS ANALYSIS - GOOD CANDIDATE"
                    positions_needing_analysis.append(ticker)
                
                context_parts.append(
                    f"  - {ticker}: {p['qty']} shares, ${p.get('market_value', 0):,.2f} "
                    f"({pnl_pct:+.1f}%) {status}"
                )
            
            if positions_needing_analysis:
                context_parts.append(f"\n✅ {len(positions_needing_analysis)} position(s) AVAILABLE for analysis: {', '.join(positions_needing_analysis)}")
        else:
            context_parts.append("\nNo current positions")
        
        # Open orders (IMPORTANT: may not fill immediately)
        if open_orders:
            context_parts.append(f"\nPending Orders ({len(open_orders)}) - MAY NOT BE FILLED YET:")
            for o in open_orders:
                context_parts.append(
                    f"  - {o['symbol']}: {o['side']} {o['qty']} @ ${o.get('limit_price', 'market')} "
                    f"(Status: {o.get('status', 'pending')})"
                )
            context_parts.append("⚠️  These orders may fill in the future - consider this when making decisions")
        else:
            context_parts.append("\nNo pending orders")
        
        # Market context - INCREASED LIMIT to capture stock recommendations
        # Web search often puts recommendations at the end, so we need more context
        context_parts.append(f"\n=== MARKET CONTEXT ===\n{market_context[:3000]}")  # Increased from 1000 to 3000 chars
        
        # Extract potential stock tickers from market context (simple regex)
        import re
        potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', market_context[:3000])
        # Filter out common words that look like tickers
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'NOT', 'BUT', 'WITH', 'FROM', 'THIS', 'THAT', 'NYSE', 'NASDAQ', 'ETF', 'IPO', 'CEO', 'CFO', 'USA', 'USD'}
        potential_tickers = [t for t in potential_tickers if t not in common_words and len(t) <= 5]
        
        if potential_tickers:
            # Show unique tickers (limit to first 20)
            unique_tickers = list(dict.fromkeys(potential_tickers))[:20]
            context_parts.append(f"\n💡 STOCK TICKERS FOUND IN MARKET CONTEXT: {', '.join(unique_tickers)}")
            context_parts.append("   ⬆️  These are potential candidates extracted from the market research above")
            # Log for debugging
            self.logger.log_system(f"📊 Extracted {len(unique_tickers)} potential tickers from market context: {', '.join(unique_tickers[:10])}{'...' if len(unique_tickers) > 10 else ''}")
        
        # Build the exclusion reminder for the task section
        if stocks_to_exclude:
            excluded_tickers_str = ", ".join(sorted(stocks_to_exclude))
            exclusion_reminder = f"""
🚨🚨🚨 CRITICAL EXCLUSION LIST 🚨🚨🚨
❌ DO NOT SELECT: {excluded_tickers_str}
❌ NEVER CHOOSE: {excluded_tickers_str}
These stocks were analyzed in the past 3 days. DO NOT select them!
"""
        else:
            exclusion_reminder = "✅ No exclusions - all stocks are available for analysis"
        
        prompt = f"""Based on the portfolio state, pending orders, and market context, select 0-3 stocks to analyze for trading.

{chr(10).join(context_parts)}

================================================================================
YOUR TASK: SELECT 0-3 STOCKS FOR ANALYSIS
================================================================================

{exclusion_reminder}

CRITICAL RULES (READ FIRST):
1. 🚫 NEVER select stocks from the EXCLUSION LIST shown above and at the top
2. ✅ Maximum 3 stocks total
3. 💡 You can trade excluded stocks using read_historical_report - no need to re-analyze them

SELECTION STRATEGY - Balance these objectives:

1️⃣ EXISTING POSITIONS marked "⚠️ NEEDS ANALYSIS - GOOD CANDIDATE":
   - These haven't been analyzed recently and need fresh analysis
   - After analysis you can:
     * 📈 BUY MORE: Add to winning positions or average down
     * 🤝 HOLD: Maintain current position
     * 📉 SELL: Exit or take profits
   - Prioritize positions with large gains/losses

2️⃣ NEW INVESTMENT OPPORTUNITIES (FOCUS ON GROWTH POTENTIAL):
   - USE MARKET CONTEXT BELOW to find specific stock tickers mentioned
   - Look for stocks with REAL GROWTH POTENTIAL, not just recent performance:
     * Companies with improving fundamentals (revenue growth, margin expansion)
     * Stocks with upcoming catalysts (new products, market expansion, regulatory approval)
     * Undervalued companies with strong business models
     * NOT just "best performers" or momentum stocks
   - The market context may list specific tickers - EXTRACT THEM
   - Consider diversification (avoid sector over-concentration)
   - Pay special attention to the end of market context (contains stock recommendations)
   - THINK LONG-TERM VALUE, not short-term price movements

SUGGESTED APPROACH:
• If 3+ positions NEED ANALYSIS → Select 2-3 of them (portfolio health first)
• If 1-2 positions NEED ANALYSIS → Mix: 1-2 existing + 1 new opportunity
• If all positions recently analyzed → Focus on 2-3 NEW opportunities
• If portfolio well-positioned → Select 0 stocks (patience is a strategy)

💡 REMEMBER:
- Positions marked with 🚫 can still be traded using historical reports
- Don't re-analyze what's already been analyzed
- Consider pending orders (may fill later)
- New opportunities come from market context research

{f"⚠️  FINAL REMINDER - DO NOT INCLUDE THESE TICKERS: {excluded_tickers_str}" if stocks_to_exclude else ""}

Respond with ONLY a JSON list of ticker symbols (must NOT include excluded tickers).
Examples:
- ["NVDA", "MSFT"]: Two new stocks to analyze
- ["AAPL"]: One existing position needing analysis (only if NOT excluded!)
- []: No stocks (if nothing compelling or all positions recently analyzed)
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
                
                # HARD FILTER: Remove stocks from exclusion list
                original_selection = selected_stocks.copy()
                selected_stocks = [s for s in selected_stocks if s not in stocks_to_exclude]
                
                # Log what happened
                if len(original_selection) > len(selected_stocks):
                    excluded = [s for s in original_selection if s in stocks_to_exclude]
                    self.logger.log_system(
                        f"⚠️  LLM ignored exclusion list and selected: {excluded}. "
                        f"Filtered them out (exclusion list: {sorted(list(stocks_to_exclude))}). "
                        f"The LLM saw the exclusion list 3 times in the prompt but still selected these stocks. "
                        f"This might indicate the model ({self.llm.model_name}) is not following instructions well."
                    )
                elif stocks_to_exclude:
                    self.logger.log_system(
                        f"✅ LLM successfully avoided {len(stocks_to_exclude)} recently analyzed stocks: "
                        f"{sorted(list(stocks_to_exclude))}"
                    )
                
                # Limit to max analyses
                selected_stocks = selected_stocks[:self.max_analyses]
                
                if selected_stocks:
                    self.logger.log_system(f"LLM selected {len(selected_stocks)} stocks: {selected_stocks}")
                else:
                    self.logger.log_system(
                        f"LLM selected 0 stocks. This could mean:\n"
                        f"  • Portfolio is well-positioned (all positions recently analyzed)\n"
                        f"  • No compelling opportunities in market context\n"
                        f"  • LLM being cautious - which is OK!"
                    )
                
                return selected_stocks
            else:
                self.logger.log_system("⚠️  Failed to parse LLM response, analyzing no stocks")
                return []
                
        except Exception as e:
            self.logger.log_system(f"⚠️  Error deciding stocks: {e}, analyzing no stocks")
            return []
    
    def _llm_trading_decisions(self, account, positions, open_orders, market_open, market_context, last_summary):
        """
        Let LLM review analysis and make trading decisions using tools.
        
        The LLM has access to:
        1. TradingAgents analysis reports (via read_analysis_report tool)
        2. Portfolio state (via get_account_info, get_current_positions tools)
        3. Previous iteration summary
        4. Trading execution tools (place_buy_order, place_sell_order, cancel_order)
        
        The LLM will:
        - Review all analyzed stocks
        - Consider current portfolio state and pending orders
        - Use tools to execute trades
        - Signal completion with review_and_decide tool
        
        Args:
            account: Account information
            positions: Current positions
            open_orders: Current open orders
            market_open: Whether market is open
            market_context: Web search results  
            last_summary: Previous iteration summary
        """
        # Build prompt for LLM
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
        prompt_parts.append("EXAMPLE DECISIONS (what you SHOULD do):")
        prompt_parts.append("  ✅ 'Analysis recommends BUY for NVDA. I have $86k cash. I will invest $30k in NVDA.'")
        prompt_parts.append("  ✅ 'Analysis says SELL MT but we don't own it. Skipping MT. Proceeding with NVDA buy.'")
        prompt_parts.append("  ✅ 'Portfolio has 3 positions performing well. No action needed. Calling review_and_decide.'")
        prompt_parts.append("")
        prompt_parts.append("WRONG BEHAVIOR (what you MUST AVOID):")
        prompt_parts.append("  ❌ 'Shall I proceed with buying NVDA?' - NO! Just do it!")
        prompt_parts.append("  ❌ 'Please specify the amount to invest' - NO! You decide!")
        prompt_parts.append("  ❌ 'Should I sell MT?' - NO! Analyze the situation and decide!")
        prompt_parts.append("")
        prompt_parts.append("The analysis is a TOOL, not a constraint. You're in charge! ACT DECISIVELY!")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        # Show previous iteration summary (if available) - CRITICAL for context
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
            prompt_parts.append("💡 USE THIS CONTEXT:")
            prompt_parts.append("  • Check if previous pending orders are now filled (compare with current positions)")
            prompt_parts.append("  • Review action items from previous iteration")
            prompt_parts.append("  • Consider what was analyzed last time (avoid re-analyzing too soon)")
            prompt_parts.append("  • Build on previous decisions (don't contradict recent actions without good reason)")
            prompt_parts.append("")
        
        # Summarize what was analyzed
        if self.analyzed_stocks:
            prompt_parts.append(f"📊 RESEARCH COMPLETED THIS ITERATION:")
            for ticker, info in self.analyzed_stocks.items():
                prompt_parts.append(f"  - {ticker}: Analysis complete (decision: {info.get('decision', 'UNKNOWN')})")
            prompt_parts.append(f"\n💡 Use read_analysis_report(ticker, 'final_trade_decision') to review each analysis.")
            prompt_parts.append("   This research can inform your decisions, but you're not limited to trading only these stocks!")
        else:
            prompt_parts.append("ℹ️  No new analysis this iteration - make decisions based on portfolio state.")
        
        # Portfolio state
        cash = float(account.get('cash', 0))
        buying_power = float(account.get('buying_power', 0))
        portfolio_value = float(account.get('portfolio_value', 0))
        
        prompt_parts.append(f"\n💰 ACCOUNT STATE:")
        prompt_parts.append(f"  - Available Cash: ${cash:,.2f} ✅ USE THIS FOR BUYING")
        prompt_parts.append(f"  - Buying Power: ${buying_power:,.2f} 🚫 DO NOT USE (margin risk)")
        prompt_parts.append(f"  - Portfolio Value: ${portfolio_value:,.2f}")
        prompt_parts.append(f"\n⚠️  CRITICAL CONSTRAINT: Only use CASH, never buying power!")
        prompt_parts.append(f"   • Available for new positions: ${cash:,.2f}")
        prompt_parts.append(f"   • This prevents margin trading and negative balances")
        
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
                prompt_parts.append(f"  ... and {len(open_orders) - 5} more (use get_open_orders for full list)")
            prompt_parts.append("\n⚠️  Don't place duplicate orders for stocks with pending orders!")
        
        prompt_parts.append(f"\nMarket Status: {'OPEN' if market_open else 'CLOSED (orders will queue)'}")
        
        # Instructions
        prompt_parts.append("\n" + "="*80)
        prompt_parts.append("YOUR DECISION PROCESS:")
        prompt_parts.append("="*80)
        prompt_parts.append("")
        prompt_parts.append("STEP 1: Review available information")
        prompt_parts.append("  • read_analysis_report(ticker, report_type) - for newly analyzed stocks")
        prompt_parts.append("  • read_historical_report(ticker, report_type) - for past analyses from S3")
        prompt_parts.append("  • get_current_positions() - see all holdings and P&L")
        prompt_parts.append("  • get_open_orders() - check pending orders")
        prompt_parts.append("")
        prompt_parts.append("STEP 2: Assess your portfolio")
        prompt_parts.append("  • Which positions are performing well/poorly?")
        prompt_parts.append("  • Do you have available cash to invest?")
        prompt_parts.append("  • Are there positions you should exit or reduce?")
        prompt_parts.append("  • Should you add to winning positions?")
        prompt_parts.append("")
        prompt_parts.append("STEP 3: Make trading decisions using these tools:")
        prompt_parts.append("  • place_buy_order(ticker, order_value, reasoning) - Buy new or add to positions")
        prompt_parts.append("  • place_sell_order(ticker, quantity, reasoning) - Exit or reduce positions")
        prompt_parts.append("  • cancel_order(ticker, reasoning) - Cancel pending orders that are no longer needed")
        prompt_parts.append("  • modify_order(order_id, new_limit_price=X, new_qty=Y, reasoning) - Edit pending orders")
        prompt_parts.append("    💡 Check get_open_orders() to see order IDs and current status")
        prompt_parts.append("")
        prompt_parts.append("STEP 4: When done, call review_and_decide() to complete")
        prompt_parts.append("")
        prompt_parts.append("REMEMBER: You're managing the entire portfolio, not just the analyzed stocks!")
        
        prompt_parts.append("\nIMPORTANT RULES:")
        prompt_parts.append("=" * 60)
        prompt_parts.append("🚫 CRITICAL: CASH-ONLY POLICY 🚫")
        prompt_parts.append("=" * 60)
        prompt_parts.append(f"- ONLY use available CASH (${cash:,.2f}) for buying")
        prompt_parts.append("- NEVER use buying power (creates margin debt)")
        prompt_parts.append("- NEVER go negative or into margin")
        prompt_parts.append("- If insufficient cash, you MUST sell positions first to raise cash")
        prompt_parts.append("")
        prompt_parts.append("=" * 80)
        prompt_parts.append("TRADING GUIDELINES - YOU HAVE FULL AUTONOMY:")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        prompt_parts.append("🔥 YOU ARE NOT LIMITED TO ANALYZED STOCKS:")
        prompt_parts.append("  • Analysis is research, not a requirement")
        prompt_parts.append("  • You can trade ANY position in your portfolio")
        prompt_parts.append("  • You can buy stocks even if not analyzed (if you have good reason)")
        prompt_parts.append("  • You can sell stocks even if not analyzed (if you have good reason)")
        prompt_parts.append("")
        prompt_parts.append("📈 STRATEGIC PORTFOLIO ACTIONS:")
        prompt_parts.append("")
        prompt_parts.append("BUY/ADD DECISIONS (based on conviction, not hype):")
        prompt_parts.append("  • BUY NEW STOCKS: When analysis shows strong fundamentals + good entry point")
        prompt_parts.append("  • ADD TO WINNERS: When fundamentals remain strong and position size allows")
        prompt_parts.append("  • AVERAGE DOWN: Only for quality stocks with intact fundamentals (not falling knives!)")
        prompt_parts.append("")
        prompt_parts.append("SELL/TRIM DECISIONS (strategic, not emotional):")
        prompt_parts.append("  • REBALANCE: When one position exceeds target allocation (e.g., >10-15% of portfolio)")
        prompt_parts.append("  • EXIT ON FUNDAMENTALS: When analysis shows deteriorating business/competitive position")
        prompt_parts.append("  • STOP-LOSS: When position hits meaningful loss threshold (e.g., -15% to -20%)")
        prompt_parts.append("  • BETTER OPPORTUNITY: Sell weaker position to fund stronger opportunity")
        prompt_parts.append("  • NOT: 'Stock down 10%' or 'Market red today' - these are poor reasons!")
        prompt_parts.append("")
        prompt_parts.append("ORDER MANAGEMENT:")
        prompt_parts.append("  • MODIFY ORDERS: Adjust price if stock moved but thesis still valid")
        prompt_parts.append("  • CANCEL ORDERS: When original thesis no longer holds or strategy changed")
        prompt_parts.append("")
        prompt_parts.append("💡 STRATEGIC DECISION FRAMEWORK:")
        prompt_parts.append("  • Base decisions on FUNDAMENTALS first (analysis, business strength, competitive position)")
        prompt_parts.append("  • Use P&L as a SIGNAL, not a trigger:")
        prompt_parts.append("    - Large loss → Re-evaluate fundamentals, not automatic sell")
        prompt_parts.append("    - Large gain → Check if over-concentrated, not automatic profit-taking")
        prompt_parts.append("  • Portfolio allocation matters: Rebalance when positions exceed 10-15%")
        prompt_parts.append("  • Market volatility is NORMAL - don't react emotionally")
        prompt_parts.append("  • Check pending orders to avoid duplicates")
        prompt_parts.append("  • Provide clear, strategic reasoning for each trade")
        prompt_parts.append("  • PATIENCE IS A STRATEGY: It's OK to make 0 trades if nothing is actionable")
        prompt_parts.append("")
        prompt_parts.append("")
        prompt_parts.append("🎯 SMART INVESTING PRINCIPLES:")
        prompt_parts.append("  • DON'T panic sell just because a stock is down - markets fluctuate!")
        prompt_parts.append("  • DON'T chase arbitrary profit targets - sell strategically")
        prompt_parts.append("  • DO hold positions you believe in through normal volatility")
        prompt_parts.append("  • DO sell when fundamentals deteriorate or strategy changes")
        prompt_parts.append("  • DO buy based on conviction from thorough analysis")
        prompt_parts.append("  • DO rebalance when portfolio gets over-concentrated")
        prompt_parts.append("")
        prompt_parts.append("EXAMPLES OF SMART DECISIONS:")
        prompt_parts.append("  ✅ 'Analysis shows GOOGL fundamentals are strong - I'll add to this winning position'")
        prompt_parts.append("  ✅ 'ABT is 30% of portfolio (over-concentrated) - I'll trim and diversify'")
        prompt_parts.append("  ✅ 'Portfolio has $80k cash and analysis shows MSFT has strong fundamentals - I'll invest'")
        prompt_parts.append("  ✅ 'AMD down 15% but analysis shows fundamentals deteriorated - now I'll exit strategically'")
        prompt_parts.append("  ✅ 'TSLA hit my -20% stop-loss threshold - I'll sell to limit losses'")
        prompt_parts.append("  ✅ 'NVDA order at $500 not filling, stock at $510 with strong momentum - I'll modify to $510'")
        prompt_parts.append("  ✅ 'Pending AAPL order no longer aligns with strategy - I'll cancel it'")
        prompt_parts.append("  ✅ 'Positions performing well, no fundamental changes - I'll hold through volatility'")
        prompt_parts.append("")
        prompt_parts.append("❌ BAD DECISIONS (avoid these):")
        prompt_parts.append("  ❌ 'Stock down 5% today - I'll panic sell' (emotional, not strategic)")
        prompt_parts.append("  ❌ 'Stock up 10% - I'll take profits' (arbitrary, no real reason)")
        prompt_parts.append("  ❌ 'Market is red today - I'll sell everything' (panic, not investing)")
        prompt_parts.append("  ❌ 'I'll buy this stock with no analysis' (no conviction/research)")
        
        prompt = "\n".join(prompt_parts)
        
        # Let LLM make decisions using tools
        self.logger.log_system("Invoking LLM to make trading decisions...")
        
        try:
            messages = [HumanMessage(content=prompt)]
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
                        result = self._handle_llm_tool_call(tool_name, tool_args, account, positions, open_orders, market_open)
                        
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
    
    def _handle_llm_tool_call(self, tool_name: str, tool_args: Dict, account, positions, open_orders, market_open) -> str:
        """
        Handle tool calls from the LLM during trading decision phase.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments for the tool
            account: Account information
            positions: Current positions
            open_orders: Current open orders
            market_open: Whether market is open
            
        Returns:
            Tool result message
        """
        from tradingagents.dataflows.alpaca_trading import place_market_order, cancel_order as cancel_alpaca_order
        
        # Handle analysis reading tools
        if tool_name == 'read_analysis_report':
            ticker = str(tool_args.get('ticker', ''))
            report_type = str(tool_args.get('report_type', ''))
            return self._handle_read_report(ticker, report_type)
        
        elif tool_name == 'read_historical_report':
            ticker = str(tool_args.get('ticker', ''))
            report_type = str(tool_args.get('report_type', ''))
            date = tool_args.get('date')  # Optional
            return self._handle_read_historical_report(ticker, report_type, date)
        
        elif tool_name == 'get_analysis_status':
            return str(self._handle_analysis_status())
        
        # Handle trading execution tools
        elif tool_name == 'place_buy_order':
            ticker = str(tool_args.get('ticker', ''))
            order_value_raw = tool_args.get('order_value', 0)
            reasoning = str(tool_args.get('reasoning', ''))
            
            # CRITICAL: Use CASH only, never buying power (to avoid margin/negative balance)
            available_cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            
            try:
                order_value = float(order_value_raw)
            except (ValueError, TypeError):
                return f"❌ Invalid order_value: {order_value_raw}"
            
            # STRICT VALIDATION: Only use available cash, never go into margin
            if order_value > available_cash:
                return (
                    f"❌ Insufficient CASH. Available cash: ${available_cash:,.2f}, "
                    f"trying to invest: ${order_value:,.2f}\n"
                    f"   ⚠️  POLICY: We only use CASH, not buying power (${buying_power:,.2f}), "
                    f"to avoid margin and negative balances."
                )
            
            if order_value < 1000:
                return f"❌ Order value too small. Minimum $1,000, got ${order_value:,.2f}"
            
            # Check for existing position (note: adding to positions is allowed!)
            existing_position = next((p for p in positions if p.get('symbol') == ticker), None)
            position_note = ""
            if existing_position:
                current_qty = existing_position.get('qty', 0)
                current_value = existing_position.get('market_value', 0)
                position_note = f"\n   📊 Current position: {current_qty} shares (${current_value:,.2f})"
            
            # Check for pending order
            has_pending = any(o.get('symbol') == ticker for o in open_orders)
            if has_pending:
                return f"⚠️  Already have pending order for {ticker}. Avoid duplicate orders!"
            
            # Log the trade
            self.logger.log_system(f"💵 BUY: {ticker} for ~${order_value:,.2f}")
            self.logger.log_system(f"   Reasoning: {reasoning}")
            self.logger.log_system(f"   Market: {'OPEN' if market_open else 'CLOSED - will queue'}")
            
            # Execute actual trade
            try:
                # Get current price and calculate quantity
                from alpaca.data.requests import StockLatestTradeRequest
                from tradingagents.dataflows.alpaca_common import get_data_client
                
                data_client = get_data_client()
                request = StockLatestTradeRequest(symbol_or_symbols=ticker)
                latest_trade = data_client.get_stock_latest_trade(request)
                current_price = float(latest_trade[ticker].price)
                
                # Calculate quantity (round down to avoid over-spending)
                qty = int(order_value / current_price)
                
                if qty < 1:
                    return f"❌ Order value too small. ${order_value:,.2f} only buys {qty} shares at ${current_price:.2f}/share. Need at least 1 share."
                
                actual_cost = qty * current_price
                
                # Place the market order
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
                
                self.logger.log_system(f"   ✅ Order placed: {qty} shares @ ${current_price:.2f} = ${actual_cost:,.2f}")
                
                return (
                    f"✅ BUY order placed for {ticker}!\n"
                    f"   • Quantity: {qty} shares\n"
                    f"   • Price: ~${current_price:.2f}/share\n"
                    f"   • Total: ~${actual_cost:,.2f}\n"
                    f"   • Order ID: {order_result.get('id')}\n"
                    f"   {'• Will execute when market opens' if not market_open else '• Executing now'}"
                )
                
            except Exception as e:
                self.logger.log_system(f"   ❌ Error executing BUY: {e}")
                return f"❌ Error executing BUY order for {ticker}: {str(e)}"
        
        elif tool_name == 'place_sell_order':
            ticker = str(tool_args.get('ticker', ''))
            quantity = tool_args.get('quantity', 0)
            reasoning = str(tool_args.get('reasoning', ''))
            
            # Find position
            position = next((p for p in positions if p.get('symbol') == ticker), None)
            if not position:
                return f"❌ No position found for {ticker}. Cannot sell."
            
            # Check quantity
            available_qty = float(position.get('qty', 0))
            if quantity == "all":
                quantity = available_qty
            else:
                quantity = float(quantity)
            
            if quantity > available_qty:
                return f"❌ Trying to sell {quantity} shares but only have {available_qty}"
            
            if quantity <= 0:
                return f"❌ Invalid quantity: {quantity}. Must be positive."
            
            # Log the trade
            self.logger.log_system(f"💸 SELL: {ticker} ({quantity} shares)")
            self.logger.log_system(f"   Reasoning: {reasoning}")
            self.logger.log_system(f"   Market: {'OPEN' if market_open else 'CLOSED - will queue'}")
            
            # Execute actual trade
            try:
                # Get current price for logging
                from alpaca.data.requests import StockLatestTradeRequest
                from tradingagents.dataflows.alpaca_common import get_data_client
                
                data_client = get_data_client()
                request = StockLatestTradeRequest(symbol_or_symbols=ticker)
                latest_trade = data_client.get_stock_latest_trade(request)
                current_price = float(latest_trade[ticker].price)
                estimated_value = quantity * current_price
                
                # Place the market order
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
                
                self.logger.log_system(f"   ✅ Order placed: {quantity} shares @ ~${current_price:.2f} = ~${estimated_value:,.2f}")
                
                return (
                    f"✅ SELL order placed for {ticker}!\n"
                    f"   • Quantity: {quantity} shares\n"
                    f"   • Price: ~${current_price:.2f}/share\n"
                    f"   • Total: ~${estimated_value:,.2f}\n"
                    f"   • Order ID: {order_result.get('id')}\n"
                    f"   {'• Will execute when market opens' if not market_open else '• Executing now'}"
                )
                
            except Exception as e:
                self.logger.log_system(f"   ❌ Error executing SELL: {e}")
                return f"❌ Error executing SELL order for {ticker}: {str(e)}"
        
        elif tool_name == 'cancel_order':
            ticker = str(tool_args.get('ticker', ''))
            reasoning = str(tool_args.get('reasoning', ''))
            
            # Find pending order
            order = next((o for o in open_orders if o.get('symbol') == ticker), None)
            if not order:
                return f"❌ No pending order found for {ticker}"
            
            order_id = order.get('id')
            order_side = order.get('side')
            order_qty = order.get('qty')
            
            # Log the cancellation
            self.logger.log_system(f"🚫 CANCEL: Order for {ticker}")
            self.logger.log_system(f"   Order: {order_side} {order_qty} shares")
            self.logger.log_system(f"   Reasoning: {reasoning}")
            
            # Execute actual cancellation
            try:
                cancel_alpaca_order(order_id)
                
                self.logger.log_system(f"   ✅ Order cancelled successfully")
                
                return (
                    f"✅ Order for {ticker} cancelled!\n"
                    f"   • Order ID: {order_id}\n"
                    f"   • Type: {order_side} {order_qty} shares\n"
                    f"   • Reason: {reasoning}"
                )
                
            except Exception as e:
                self.logger.log_system(f"   ❌ Error cancelling order: {e}")
                return f"❌ Error cancelling order for {ticker}: {str(e)}"
        
        elif tool_name == 'get_current_positions':
            # Refresh positions and return formatted info
            from tradingagents.dataflows.alpaca_trading import get_positions as get_alpaca_positions
            fresh_positions = get_alpaca_positions()
            
            if not fresh_positions:
                return "No current positions. Portfolio is 100% cash."
            
            result_lines = [f"📈 CURRENT POSITIONS ({len(fresh_positions)}):"]
            for p in fresh_positions:
                pnl_pct = (p.get('unrealized_plpc', 0) * 100)
                result_lines.append(
                    f"  • {p['symbol']}: {p['qty']} shares | "
                    f"Value: ${p.get('market_value', 0):,.2f} | "
                    f"P&L: {pnl_pct:+.1f}% | "
                    f"Avg Cost: ${p.get('avg_entry_price', 0):.2f}"
                )
            
            return "\n".join(result_lines)
        
        elif tool_name == 'get_open_orders':
            # Refresh orders and return formatted info
            from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
            fresh_orders = get_alpaca_open_orders()
            
            if not fresh_orders:
                return "✅ No pending orders. All previous orders filled or cancelled."
            
            result_lines = [f"📋 PENDING ORDERS ({len(fresh_orders)}):"]
            for o in fresh_orders:
                order_id = str(o.get('id', 'N/A'))[:8]
                result_lines.append(
                    f"  • {o.get('symbol')}: {o.get('side').upper()} {o.get('qty')} shares | "
                    f"Price: ${o.get('limit_price', 'market')} | "
                    f"Status: {o.get('status', 'unknown')} | "
                    f"ID: {order_id}"
                )
            
            return "\n".join(result_lines)
        
        elif tool_name == 'modify_order':
            # Modify/edit a pending order
            order_id = tool_args.get('order_id', '')
            new_limit_price = tool_args.get('new_limit_price')
            new_qty = tool_args.get('new_qty')
            reasoning = str(tool_args.get('reasoning', ''))
            
            if not order_id:
                return "❌ order_id is required. Use get_open_orders() to see order IDs."
            
            # Find the order
            order = next((o for o in open_orders if str(o.get('id', ''))[:8] == str(order_id)[:8]), None)
            if not order:
                return f"❌ Order {order_id} not found. Use get_open_orders() to see current orders."
            
            self.logger.log_system(f"✏️  MODIFY ORDER: {order_id}")
            self.logger.log_system(f"   Original: {order.get('side')} {order.get('qty')} {order.get('symbol')} @ ${order.get('limit_price', 'market')}")
            if new_limit_price:
                self.logger.log_system(f"   New price: ${new_limit_price}")
            if new_qty:
                self.logger.log_system(f"   New quantity: {new_qty}")
            self.logger.log_system(f"   Reasoning: {reasoning}")
            
            # Execute modification
            try:
                from .agents.utils.portfolio_management_tools import modify_order as modify_order_pm
                
                # Build modification parameters
                mod_params = {'order_id': str(order.get('id'))}
                if new_limit_price is not None:
                    mod_params['limit_price'] = float(new_limit_price)
                if new_qty is not None:
                    mod_params['qty'] = int(new_qty)
                
                result = modify_order_pm.invoke(mod_params)
                
                self.logger.log_system(f"   ✅ Order modified successfully")
                
                return (
                    f"✅ Order {order_id} modified!\n"
                    f"   • Symbol: {order.get('symbol')}\n"
                    f"   • Side: {order.get('side')}\n"
                    + (f"   • New Price: ${new_limit_price}\n" if new_limit_price else "")
                    + (f"   • New Quantity: {new_qty}\n" if new_qty else "")
                    + f"   • Reason: {reasoning}"
                )
                
            except Exception as e:
                self.logger.log_system(f"   ❌ Error modifying order: {e}")
                return f"❌ Error modifying order {order_id}: {str(e)}"
        
        elif tool_name == 'review_and_decide':
            self.logger.log_system("✅ LLM signaled completion of trading decisions")
            return "Trading decision phase complete."
        
        # If tool not recognized, return generic message
        else:
            return f"Tool {tool_name} called with args: {tool_args}"
    
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
        Get list of stocks that were recently analyzed from S3 and local storage.
        
        This method fetches the stock evaluation history before running the main LLM call,
        creating a mapping of which stocks were analyzed recently to help the LLM make
        informed decisions about what to analyze next.
        
        Args:
            days_threshold: Number of days to look back
            
        Returns:
            Dictionary with recently analyzed stocks info
        """
        from datetime import datetime, timedelta
        from pathlib import Path
        
        recently_analyzed = []
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_threshold)
        
        # STEP 1: Fetch from S3 (primary source)
        self.logger.log_system("Fetching stock analysis history from S3...")
        try:
            s3_history = self.s3_client.get_analyzed_stocks_history(days_threshold)
            
            for ticker, date_list in s3_history.items():
                if date_list:
                    # Get the most recent date for this ticker
                    most_recent_date = date_list[0]  # Already sorted in descending order
                    
                    try:
                        folder_date = datetime.strptime(most_recent_date, "%Y-%m-%d")
                        days_ago = (now - folder_date).days
                        
                        # Just track that analysis exists - LLM can use read_historical_report to see it
                        recently_analyzed.append({
                            'ticker': ticker,
                            'date': most_recent_date,
                            'days_ago': days_ago,
                            'source': 'S3'
                        })
                        
                    except ValueError:
                        continue
                        
            self.logger.log_system(f"Found {len(recently_analyzed)} stocks from S3")
            
        except Exception as e:
            self.logger.log_system(f"⚠️  Error fetching from S3: {e}")
        
        # STEP 2: Supplement with local data (fallback/additional)
        # This catches any analyses that haven't been uploaded to S3 yet
        if self.results_dir.exists():
            try:
                local_tickers = set()
                
                for ticker_dir in self.results_dir.iterdir():
                    if not ticker_dir.is_dir():
                        continue
                    
                    # Skip portfolio_manager directory
                    if ticker_dir.name == 'portfolio_manager':
                        continue
                    
                    ticker = ticker_dir.name
                    
                    # Skip if already found in S3
                    if any(item['ticker'] == ticker for item in recently_analyzed):
                        continue
                    
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
                                    # Just track that analysis exists - LLM can use read_historical_report
                                    recently_analyzed.append({
                                        'ticker': ticker,
                                        'date': date_dir.name,
                                        'days_ago': (now - folder_date).days,
                                        'source': 'Local'
                                    })
                                    local_tickers.add(ticker)
                                    break  # Only take the most recent analysis for each ticker
                                    
                            except ValueError:
                                # Skip folders that don't match date format
                                continue
                    except Exception:
                        # If error reading ticker directory, skip it
                        continue
                        
                if local_tickers:
                    self.logger.log_system(f"Found {len(local_tickers)} additional stocks from local storage")
                    
            except Exception as e:
                self.logger.log_system(f"⚠️  Error scanning local storage: {e}")
        
        # Sort by date (most recent first)
        recently_analyzed.sort(key=lambda x: str(x['date']), reverse=True)
        
        # Log summary for LLM context
        summary_lines = []
        summary_lines.append(f"\n📊 Stock Analysis History (past {days_threshold} days):")
        summary_lines.append(f"   Total stocks analyzed: {len(recently_analyzed)}")
        
        if recently_analyzed:
            summary_lines.append(f"\n   Recent analyses (use read_historical_report to view):")
            for item in recently_analyzed[:10]:  # Show top 10
                summary_lines.append(
                    f"   - {item['ticker']}: {item['date']} "
                    f"({item['days_ago']} days ago, from {item['source']})"
                )
            if len(recently_analyzed) > 10:
                summary_lines.append(f"   ... and {len(recently_analyzed) - 10} more")
            summary_lines.append(f"\n   💡 Use read_historical_report(ticker, 'final_trade_decision') to see past recommendations")
        
        self.logger.log_system('\n'.join(summary_lines))
        
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
    
    def generate_iteration_summary(self, open_orders=None) -> str:
        """
        Generate summary for next iteration focused on actionable information.
        
        This summary is saved to S3 and retrieved by the next iteration to help
        the agent understand what happened and what needs to be checked/done next.
        
        Args:
            open_orders: Current open orders to include in summary
            
        Returns:
            Summary string optimized for agent context
        """
        account = get_account()
        positions = get_positions()
        
        summary_lines = [
            f"ITERATION: {self.iteration_id}",
            f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"=" * 80,
            f"WHAT HAPPENED IN THIS ITERATION:",
            f"=" * 80,
            f"",
        ]
        
        # What did we research?
        if self.analyzed_stocks:
            summary_lines.append("📊 RESEARCH COMPLETED:")
            for ticker, info in self.analyzed_stocks.items():
                summary_lines.append(f"  • {ticker}: {info.get('decision', 'UNKNOWN')} (analyzed on {info.get('date', 'unknown date')})")
            summary_lines.append("")
        else:
            summary_lines.append("📊 No new research this iteration")
            summary_lines.append("")
        
        # What trades did we make?
        if self.trades_executed:
            summary_lines.append(f"💼 TRADES EXECUTED ({len(self.trades_executed)}):")
            for trade in self.trades_executed:
                if trade['action'] == 'BUY':
                    summary_lines.append(
                        f"  • BUY {trade['ticker']}: {trade.get('quantity', 0)} shares @ ${trade.get('price', 0):.2f} "
                        f"(Total: ${trade.get('value', 0):,.2f})"
                    )
                elif trade['action'] == 'SELL':
                    summary_lines.append(
                        f"  • SELL {trade['ticker']}: {trade.get('quantity', 0)} shares @ ${trade.get('price', 0):.2f} "
                        f"(Total: ${trade.get('value', 0):,.2f})"
                    )
                summary_lines.append(f"    Reason: {trade.get('reasoning', 'N/A')}")
            summary_lines.append("")
        else:
            summary_lines.append("💼 No trades executed this iteration")
            summary_lines.append("")
        
        # Current portfolio state
        summary_lines.extend([
            f"=" * 80,
            f"CURRENT PORTFOLIO STATE:",
            f"=" * 80,
            f"",
            f"💰 ACCOUNT:",
            f"  • Total Value: ${account.get('portfolio_value', 0):,.2f}",
            f"  • Cash Available: ${account.get('cash', 0):,.2f} ⚠️ Use this for buying",
            f"  • Buying Power: ${account.get('buying_power', 0):,.2f} (DO NOT USE - margin)",
            f"",
        ])
        
        # Add position details with actionable insights
        if positions:
            summary_lines.append(f"📈 POSITIONS ({len(positions)}):")
            for p in positions[:15]:  # Show more positions for better context
                pnl_pct = (p.get('unrealized_plpc', 0) * 100)
                market_value = p.get('market_value', 0)
                
                # Add performance indicators
                if pnl_pct > 10:
                    indicator = "🟢 Strong performer"
                elif pnl_pct > 5:
                    indicator = "🟢 Good"
                elif pnl_pct > -5:
                    indicator = "⚪ Neutral"
                elif pnl_pct > -10:
                    indicator = "🟡 Underperforming"
                else:
                    indicator = "🔴 Significant loss"
                
                summary_lines.append(
                    f"  • {p['symbol']}: {p['qty']} shares | ${market_value:,.2f} | "
                    f"P&L: {pnl_pct:+.1f}% | {indicator}"
                )
            if len(positions) > 15:
                summary_lines.append(f"  ... and {len(positions) - 15} more (check next iteration)")
            summary_lines.append("")
        else:
            summary_lines.append("📈 No positions (100% cash)")
            summary_lines.append("")
        
        # Add pending orders (CRITICAL for next iteration)
        summary_lines.append(f"=" * 80)
        if open_orders:
            summary_lines.append(f"⚠️  PENDING ORDERS - ACTION REQUIRED:")
            summary_lines.append(f"=" * 80)
            summary_lines.append(f"You have {len(open_orders)} order(s) that may or may not be filled.")
            summary_lines.append("")
            
            for order in open_orders[:10]:
                order_id = order.get('id', 'N/A')
                order_id_str = str(order_id)[:8] if order_id != 'N/A' else 'N/A'
                summary_lines.append(
                    f"  • {order.get('symbol')}: {order.get('side').upper()} {order.get('qty')} shares "
                    f"@ ${order.get('limit_price', 'market')} | Status: {order.get('status', 'pending')} | ID: {order_id_str}"
                )
            if len(open_orders) > 10:
                summary_lines.append(f"  ... and {len(open_orders) - 10} more")
            summary_lines.append("")
        else:
            summary_lines.append(f"✅ NO PENDING ORDERS")
            summary_lines.append(f"=" * 80)
            summary_lines.append("All previous orders have been filled or cancelled.")
            summary_lines.append("")
        
        # Action items for next iteration
        summary_lines.append(f"=" * 80)
        summary_lines.append(f"🎯 ACTION ITEMS FOR NEXT ITERATION:")
        summary_lines.append(f"=" * 80)
        summary_lines.append("")
        
        action_items = []
        
        # Check pending orders
        if open_orders:
            action_items.append(f"1. CHECK PENDING ORDERS:")
            action_items.append(f"   • {len(open_orders)} order(s) may have filled - use get_current_positions() to check")
            action_items.append(f"   • Cancel any orders that are no longer relevant")
            action_items.append(f"   • Don't place duplicate orders for same stocks")
            action_items.append("")
        
        # Review position performance
        if positions:
            # Find positions that need attention
            strong_performers = [p for p in positions if (p.get('unrealized_plpc', 0) * 100) > 10]
            poor_performers = [p for p in positions if (p.get('unrealized_plpc', 0) * 100) < -10]
            
            action_num = 2 if open_orders else 1
            action_items.append(f"{action_num}. REVIEW POSITION PERFORMANCE:")
            
            if strong_performers:
                strong_tickers = [p['symbol'] for p in strong_performers[:5]]
                action_items.append(f"   • Strong performers (>10% gain): {', '.join(strong_tickers)}")
                action_items.append(f"     Consider: Taking profits or adding more")
            
            if poor_performers:
                poor_tickers = [p['symbol'] for p in poor_performers[:5]]
                action_items.append(f"   • Poor performers (<-10% loss): {', '.join(poor_tickers)}")
                action_items.append(f"     Consider: Cutting losses or averaging down if fundamentals strong")
            
            if not strong_performers and not poor_performers:
                action_items.append(f"   • All positions performing within -10% to +10% range")
            
            action_items.append("")
        
        # Cash utilization
        cash = account.get('cash', 0)
        portfolio_value = account.get('portfolio_value', 0)
        cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0
        
        action_num = len([item for item in action_items if item and item[0].isdigit()]) + 1
        action_items.append(f"{action_num}. CASH UTILIZATION:")
        action_items.append(f"   • Available cash: ${cash:,.2f} ({cash_pct:.1f}% of portfolio)")
        
        if cash_pct > 50:
            action_items.append(f"   • High cash allocation - consider finding investment opportunities")
        elif cash_pct > 20:
            action_items.append(f"   • Moderate cash allocation - can make new investments")
        elif cash_pct < 5:
            action_items.append(f"   • Low cash - may need to sell positions before buying new ones")
        else:
            action_items.append(f"   • Balanced cash allocation")
        action_items.append("")
        
        # Recent analysis
        if self.analyzed_stocks:
            action_num = len([item for item in action_items if item and item[0].isdigit()]) + 1
            action_items.append(f"{action_num}. RECENT ANALYSIS AVAILABLE:")
            action_items.append(f"   • Analyzed in this iteration: {', '.join(self.analyzed_stocks.keys())}")
            action_items.append(f"   • Use read_historical_report(ticker, 'final_trade_decision') to review")
            action_items.append("")
        
        # General reminder
        action_num = len([item for item in action_items if item and item[0].isdigit()]) + 1
        action_items.append(f"{action_num}. REMEMBER YOUR AUTONOMY:")
        action_items.append(f"   • You manage the ENTIRE portfolio, not just analyzed stocks")
        action_items.append(f"   • You can buy, sell, or hold ANY position based on:")
        action_items.append(f"     - Performance (P&L)")
        action_items.append(f"     - Market conditions")
        action_items.append(f"     - Portfolio allocation")
        action_items.append(f"     - Your judgment")
        action_items.append(f"   • Analysis is helpful but NOT required for every decision")
        
        summary_lines.extend(action_items)
        summary_lines.append("")
        summary_lines.append(f"=" * 80)
        summary_lines.append(f"END OF ITERATION SUMMARY")
        summary_lines.append(f"=" * 80)
        
        return "\n".join(summary_lines)

