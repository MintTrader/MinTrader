"""
Analysis Handler Module

Handles stock analysis requests, report reading, and analysis status tracking.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from tradingagents.utils.report_generator import ReportGenerator


class AnalysisHandler:
    """Handles stock analysis operations and report management"""
    
    def __init__(self, trading_agents, results_dir: Path, s3_client, logger, max_analyses: int = 3):
        """
        Initialize analysis handler.
        
        Args:
            trading_agents: TradingAgentsGraph instance
            results_dir: Results directory path
            s3_client: S3ReportManager instance
            logger: PortfolioLogger instance
            max_analyses: Maximum number of analyses per iteration
        """
        self.trading_agents = trading_agents
        self.results_dir = results_dir
        self.s3_client = s3_client
        self.logger = logger
        self.max_analyses = max_analyses
        
        # Analysis tracking
        self.analyses_requested = 0
        self.analyzed_stocks: Dict[str, Dict[str, str]] = {}  # ticker -> {date, reports_dir, decision}
    
    def request_analysis(self, ticker: str, reasoning: str) -> str:
        """
        Request analysis for a stock.
        
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
    
    def read_report(self, ticker: str, report_type: str) -> str:
        """
        Read an analysis report from current iteration.
        
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
    
    def read_historical_report(self, ticker: str, report_type: str, date: Optional[str] = None) -> str:
        """
        Read a historical analysis report from S3.
        
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get current analysis status."""
        return {
            "analyses_requested": self.analyses_requested,
            "analyses_remaining": self.max_analyses - self.analyses_requested,
            "analyzed_stocks": list(self.analyzed_stocks.keys())
        }
    
    def get_analyzed_stocks(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of analyzed stocks."""
        return self.analyzed_stocks

