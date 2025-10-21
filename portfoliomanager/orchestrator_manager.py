"""
Orchestrator-based Portfolio Manager

LLM-driven portfolio manager that uses web search to decide what to analyze and trade.
This is the main orchestrator that coordinates all portfolio management operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.dataflows.alpaca_trading import get_account, get_positions, get_market_clock

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

# Import new modular components
from .analysis_handler import AnalysisHandler
from .trading_executor import TradingExecutor
from .stock_selector import StockSelector
from .report_manager import ReportManager
from .portfolio_helpers import PortfolioHelpers


class OrchestratorPortfolioManager:
    """
    Orchestrator-based portfolio manager using LLM + web search.
    
    This is the main coordinator that delegates to specialized handlers:
    - AnalysisHandler: Stock analysis operations
    - TradingExecutor: Trading decisions and execution
    - StockSelector: Stock selection logic
    - ReportManager: Report generation and S3 uploads
    - PortfolioHelpers: Utility functions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator portfolio manager.
        
        Args:
            config: Configuration dictionary with settings for:
                - results_dir: Where to save results
                - s3_bucket_name: S3 bucket for reports
                - s3_region: AWS region
                - max_stocks_to_analyze: Max stocks per iteration (default: 3)
                - enable_web_search: Enable/disable web search (default: True)
                - analysis_config: Configuration for TradingAgents
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
        
        # Initialize modular components
        max_analyses = config.get('max_stocks_to_analyze', 3)
        
        self.analysis_handler = AnalysisHandler(
            self.trading_agents, self.results_dir, self.s3_client, 
            self.logger, max_analyses
        )
        
        self.stock_selector = StockSelector(
            self.llm, self.logger, max_analyses
        )
        
        self.report_manager = ReportManager(
            self.s3_client, self.logger, self.results_dir, self.pm_results_dir
        )
        
        self.portfolio_helpers = PortfolioHelpers(
            self.s3_client, self.results_dir, self.logger
        )
        self.portfolio_helpers.set_web_search_config(
            enabled=config.get('enable_web_search', True),
            max_searches=1
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
        
        # Initialize trading executor (needs orchestrator_llm)
        self.trading_executor = TradingExecutor(
            self.orchestrator_llm, self.logger, self.analysis_handler, self.s3_client
        )
        
        self.logger.log_system("Orchestrator Portfolio Manager initialized successfully")
    
    def run_iteration(self):
        """
        Run a complete portfolio management iteration using single-pass workflow.
        
        Steps:
        1. Retrieve last iteration summary from S3
        2. Gather portfolio information (account, positions, orders)
        3. Check recent analysis history
        4. Perform web search for market context (if enabled)
        5. Select stocks to analyze (0-3)
        6. Run analysis for selected stocks
        7. LLM makes trading decisions using tools
        8. Upload reports to S3
        9. Generate iteration summary for next time
        """
        try:
            # Log start
            self.logger.log_system(f"Starting portfolio management iteration {self.iteration_id}")
            
            # STEP 0: Retrieve last iteration summary
            self.logger.log_system("\n=== STEP 0: Retrieving Last Iteration Summary ===")
            last_summary = self._get_last_summary()
            
            # STEP 1: Gather portfolio information
            self.logger.log_system("\n=== STEP 1: Gathering Portfolio Information ===")
            market_status, account, positions, open_orders = self._gather_portfolio_info()
            market_open = market_status.get('is_open', False)
            
            # STEP 2: Get recently analyzed stocks
            self.logger.log_system("\n=== STEP 2: Checking Recent Analysis History ===")
            recent_analysis = self.portfolio_helpers.get_recently_analyzed_stocks(14)
            self.logger.log_system(f"Found {recent_analysis['total_count']} stocks analyzed in past 14 days")
            
            # STEP 3: Web search for market context
            self.logger.log_system("\n=== STEP 3: Market Research ===")
            market_context = self._perform_market_research(positions, open_orders, recent_analysis)
            
            # STEP 4: Decide which stocks to analyze
            self.logger.log_system("\n=== STEP 4: Selecting Stocks for Analysis ===")
            stocks_to_analyze = self.stock_selector.decide_stocks_to_analyze(
                account, positions, open_orders, market_context, recent_analysis, last_summary
            )
            self.logger.log_system(f"Selected {len(stocks_to_analyze)} stocks: {stocks_to_analyze}")
            
            # STEP 5: Run analysis for selected stocks
            self.logger.log_system("\n=== STEP 5: Running Stock Analysis ===")
            for ticker in stocks_to_analyze:
                self.analysis_handler.request_analysis(
                    ticker, f"Selected for analysis based on portfolio review"
                )
            
            # STEP 6: LLM makes trading decisions
            self.logger.log_system("\n=== STEP 6: LLM Making Trading Decisions ===")
            self.trading_executor.make_trading_decisions(
                account, positions, open_orders, market_open, 
                market_context, last_summary, 
                self.analysis_handler.get_analyzed_stocks()
            )
            
            # Refresh orders after trades
            from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
            updated_open_orders = get_alpaca_open_orders()
            
            # STEP 7: Final summary
            self._log_iteration_summary(stocks_to_analyze, updated_open_orders)
            
            # Upload reports and save summary
            self._finalize_iteration(updated_open_orders, last_summary)
            
            self.logger.log_system("✅ Iteration complete. Check message_tool.log for details.")
            
        except Exception as e:
            self.logger.log_system(f"❌ ERROR during iteration: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _get_last_summary(self) -> str:
        """Retrieve last iteration summary from S3."""
        last_summary = self.s3_client.get_last_summary()
        if last_summary:
            self.logger.log_system("🧠💭 Retrieved previous iteration summary from S3")
        else:
            self.logger.log_system("ℹ️  No previous iteration summary found (first run)")
            last_summary = "No previous iteration data available."
        return last_summary
    
    def _gather_portfolio_info(self):
        """Gather current portfolio information."""
        market_status = get_market_clock()
        account = get_account()
        positions = get_positions()
        
        from tradingagents.dataflows.alpaca_trading import get_open_orders as get_alpaca_open_orders
        open_orders = get_alpaca_open_orders()
        
        # Log comprehensive portfolio summary
        self.logger.log_portfolio_summary(account, positions, market_status, open_orders)
        
        # Check if market is open
        if not market_status.get('is_open', False):
            self.logger.log_system("⚠️  Market is currently CLOSED. Orders will be queued.")
        
        return market_status, account, positions, open_orders
    
    def _perform_market_research(self, positions, open_orders, recent_analysis) -> str:
        """Perform market research via web search."""
        web_search_query = self.stock_selector.build_market_search_query(
            positions, open_orders, recent_analysis
        )
        
        if web_search_query:
            market_context = self.portfolio_helpers.perform_web_search(web_search_query)
            self.logger.log_system(f"✅ Market context gathered ({len(market_context)} chars)")
            self.logger.log_system(f"   Full Market Context:\n{market_context}")
        else:
            market_context = "Web search disabled. Using portfolio data only."
            self.logger.log_system("ℹ️  Web search disabled")
        
        return market_context
    
    def _log_iteration_summary(self, stocks_to_analyze, updated_open_orders):
        """Log iteration summary."""
        self.logger.log_system("\n=== STEP 7: Iteration Summary ===")
        self.logger.log_system(f"✅ Stocks analyzed: {len(stocks_to_analyze)}")
        self.logger.log_system(
            f"✅ Trades executed by LLM: {len(self.trading_executor.get_trades_executed())}"
        )
        self.logger.log_system(f"✅ Open orders: {len(updated_open_orders)}")
        
        if updated_open_orders:
            self.logger.log_system("\n📋 Pending Orders (to be monitored next run):")
            for order in updated_open_orders[:5]:
                self.logger.log_system(
                    f"   - {order.get('symbol')}: {order.get('side')} {order.get('qty')} "
                    f"@ ${order.get('limit_price', 'market')} (Status: {order.get('status')})"
                )
            if len(updated_open_orders) > 5:
                self.logger.log_system(f"   ... and {len(updated_open_orders) - 5} more")
    
    def _finalize_iteration(self, updated_open_orders, last_summary: str = ""):
        """Finalize iteration: upload reports and save summary."""
        # Upload reports to S3
        self.logger.log_system("\nUploading reports to S3...")
        self.report_manager.upload_reports(
            self.iteration_id, 
            self.analysis_handler.get_analyzed_stocks()
        )
        
        # Generate and save iteration summary
        account = get_account()
        positions = get_positions()
        
        summary = self.report_manager.generate_iteration_summary(
            self.iteration_id,
            self.llm.model_name if hasattr(self.llm, 'model_name') else 'gpt-4o-mini',
            self.analysis_handler.get_analyzed_stocks(),
            self.trading_executor.get_trades_executed(),
            account,
            positions,
            updated_open_orders,
            last_summary
        )
        
        # Log the summary
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
