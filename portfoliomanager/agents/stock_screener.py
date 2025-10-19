"""
Stock Screener

Provides candidate pool of stocks for the agent to analyze.
The agent uses stock tools (get_stock_data, get_indicators) to perform actual analysis.
"""

from typing import List, Dict, Optional


class StockScreener:
    """Provides candidate stocks for portfolio agent to analyze"""
    
    def __init__(self, config: Dict):
        """
        Initialize stock screener.
        
        Args:
            config: Screener configuration
        """
        self.config = config
        self.max_picks = config.get('max_screener_picks', 3)
        
    def screen_opportunities(self, existing_tickers: Optional[List[str]] = None) -> List[str]:
        """
        Return candidate stocks for the agent to analyze.
        
        The agent will use stock tools to:
        - Check price momentum with get_stock_data()
        - Analyze technicals with get_indicators()
        - Make informed decisions based on the data
        
        Args:
            existing_tickers: Tickers to exclude (already in portfolio)
            
        Returns:
            List of ticker symbols to analyze
        """
        existing_tickers = existing_tickers or []
        
        # Candidate pool of liquid, well-known stocks
        # In production, this could use Alpaca's screener API or other sources
        candidate_pool = [
            # Large-cap tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
            "AMD", "INTC", "ORCL", "ADBE", "CRM", "NFLX", "CSCO",
            
            # Finance
            "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
            
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "ABT", "LLY",
            
            # Consumer
            "WMT", "HD", "DIS", "NKE", "SBUX", "MCD", "TGT", "COST",
            
            # Industrial
            "BA", "CAT", "GE", "HON", "MMM", "UPS", "LMT", "RTX",
            
            # Communication
            "T", "VZ", "CMCSA",
            
            # Energy
            "XOM", "CVX", "COP", "SLB",
            
            # Semiconductors
            "AVGO", "QCOM", "TXN", "AMAT", "MU",
            
            # Software
            "PLTR", "SNOW", "DDOG", "CRWD", "ZS",
            
            # E-commerce / Retail
            "SHOP", "ETSY", "EBAY",
            
            # Biotech
            "GILD", "BIIB", "REGN", "VRTX",
            
            # Automotive
            "F", "GM", "RIVN", "LCID",
            
            # Cloud/Data
            "SNOW", "NET", "DDOG", "MDB",
        ]
        
        # Remove duplicates and filter out existing positions
        candidate_pool = list(set(candidate_pool))
        candidates = [t for t in candidate_pool if t not in existing_tickers]
        
        # Return subset for agent to analyze
        # Agent will use stock tools to find the best opportunities
        return candidates[:self.max_picks]
