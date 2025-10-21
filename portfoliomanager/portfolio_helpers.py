"""
Portfolio Helpers Module

Utility functions for web search, stock history tracking, and helper operations.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any


class PortfolioHelpers:
    """Utility functions for portfolio management"""
    
    def __init__(self, s3_client, results_dir: Path, logger):
        """
        Initialize portfolio helpers.
        
        Args:
            s3_client: S3ReportManager instance
            results_dir: Results directory path
            logger: PortfolioLogger instance
        """
        self.s3_client = s3_client
        self.results_dir = results_dir
        self.logger = logger
        self.web_searches_used = 0
        self.max_web_searches = 1
        self.web_search_enabled = True
    
    def set_web_search_config(self, enabled: bool = True, max_searches: int = 1):
        """
        Configure web search settings.
        
        Args:
            enabled: Whether web search is enabled
            max_searches: Maximum number of web searches allowed
        """
        self.web_search_enabled = enabled
        self.max_web_searches = max_searches
    
    def perform_web_search(self, query: str) -> str:
        """
        Perform a web search with cost control.
        
        Args:
            query: The search query
            
        Returns:
            Search results or limit message
        """
        # Check if web search is enabled
        if not self.web_search_enabled:
            return "⚠️ Web search is disabled in configuration. Use stock data tools instead."
        
        # Check if limit reached
        if self.web_searches_used >= self.max_web_searches:
            return (
                f"❌ Web search limit reached! Used {self.web_searches_used}/{self.max_web_searches} searches.\n"
                "To control costs, use stock analysis tools instead."
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
    
    def get_recently_analyzed_stocks(self, days_threshold: int = 14) -> Dict[str, Any]:
        """
        Get list of stocks recently analyzed from S3 and local storage.
        
        Args:
            days_threshold: Number of days to look back
            
        Returns:
            Dictionary with recently analyzed stocks info
        """
        recently_analyzed = []
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_threshold)
        
        # STEP 1: Fetch from S3 (primary source)
        self.logger.log_system("Fetching stock analysis history from S3...")
        try:
            s3_history = self.s3_client.get_analyzed_stocks_history(days_threshold)
            
            for ticker, date_list in s3_history.items():
                if date_list:
                    most_recent_date = date_list[0]  # Already sorted
                    
                    try:
                        folder_date = datetime.strptime(most_recent_date, "%Y-%m-%d")
                        days_ago = (now - folder_date).days
                        
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
        
        # STEP 2: Supplement with local data
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
                    
                    # Check for recent analysis
                    try:
                        for date_dir in ticker_dir.iterdir():
                            if not date_dir.is_dir():
                                continue
                            
                            try:
                                folder_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                                
                                if folder_date >= cutoff_date:
                                    recently_analyzed.append({
                                        'ticker': ticker,
                                        'date': date_dir.name,
                                        'days_ago': (now - folder_date).days,
                                        'source': 'Local'
                                    })
                                    local_tickers.add(ticker)
                                    break
                                    
                            except ValueError:
                                continue
                    except Exception:
                        continue
                        
                if local_tickers:
                    self.logger.log_system(f"Found {len(local_tickers)} additional stocks from local storage")
                    
            except Exception as e:
                self.logger.log_system(f"⚠️  Error scanning local storage: {e}")
        
        # Sort by date (most recent first)
        recently_analyzed.sort(key=lambda x: str(x['date']), reverse=True)
        
        # Log summary
        self._log_analysis_history(recently_analyzed, days_threshold)
        
        return {
            "recently_analyzed": recently_analyzed,
            "days_threshold": days_threshold,
            "total_count": len(recently_analyzed)
        }
    
    def _log_analysis_history(self, recently_analyzed: list, days_threshold: int):
        """Log analysis history summary."""
        summary_lines = []
        summary_lines.append(f"\n📊 Stock Analysis History (past {days_threshold} days):")
        summary_lines.append(f"   Total stocks analyzed: {len(recently_analyzed)}")
        
        if recently_analyzed:
            summary_lines.append(f"\n   Recent analyses (use read_historical_report to view):")
            for item in recently_analyzed[:10]:
                summary_lines.append(
                    f"   - {item['ticker']}: {item['date']} "
                    f"({item['days_ago']} days ago, from {item['source']})"
                )
            if len(recently_analyzed) > 10:
                summary_lines.append(f"   ... and {len(recently_analyzed) - 10} more")
            summary_lines.append(
                f"\n   💡 Use read_historical_report(ticker, 'final_trade_decision') "
                f"to see past recommendations"
            )
        
        self.logger.log_system('\n'.join(summary_lines))

