"""
Watchlist Manager

Manages a watchlist of stocks for potential investment.
"""

from typing import List


class WatchlistManager:
    """Manages stock watchlist"""
    
    def __init__(self, watchlist: List[str]):
        """
        Initialize watchlist manager.
        
        Args:
            watchlist: Initial list of ticker symbols
        """
        self.watchlist = list(set(watchlist))  # Remove duplicates
    
    def get_watchlist(self) -> List[str]:
        """
        Get current watchlist.
        
        Returns:
            List of ticker symbols
        """
        return self.watchlist.copy()
    
    def add_ticker(self, ticker: str):
        """
        Add a ticker to the watchlist.
        
        Args:
            ticker: Stock ticker symbol
        """
        if ticker not in self.watchlist:
            self.watchlist.append(ticker)
    
    def remove_ticker(self, ticker: str):
        """
        Remove a ticker from the watchlist.
        
        Args:
            ticker: Stock ticker symbol
        """
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
    
    def filter_existing(self, existing_tickers: List[str]) -> List[str]:
        """
        Filter out tickers that are already in portfolio.
        
        Args:
            existing_tickers: List of tickers already held
            
        Returns:
            Filtered watchlist
        """
        return [t for t in self.watchlist if t not in existing_tickers]

