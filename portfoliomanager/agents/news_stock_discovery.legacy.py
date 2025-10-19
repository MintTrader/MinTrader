"""
News-Based Stock Discovery

Identifies trending stocks using LLM with web search capabilities.
"""

import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Set
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class NewsStockDiscovery:
    """Discovers trending stocks using LLM with web search"""
    
    def __init__(self, config: Dict):
        """
        Initialize news-based stock discovery.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_stocks = config.get('max_news_stocks', 3)
        
        # Initialize LLM with web search capability
        analysis_config = config.get('analysis_config', {})
        self.llm = ChatOpenAI(
            model=analysis_config.get('quick_think_llm', 'gpt-4o-mini'),
            base_url=analysis_config.get('backend_url', 'https://api.openai.com/v1'),
            temperature=0.3  # Lower temperature for more consistent results
        )
    
    def discover_from_news(self, preferred_sectors: List[str] | None = None, exclude_tickers: List[str] | None = None) -> List[str]:
        """
        Discover trending stocks using LLM with web search.
        
        Args:
            preferred_sectors: Optional list of sectors to filter by (e.g., ["Technology", "Healthcare"])
            exclude_tickers: Optional list of tickers to exclude (e.g., existing positions + recently analyzed)
        
        Returns:
            List of ticker symbols (up to max_stocks)
        """
        try:
            # Build the prompt for the LLM
            exclude_list = exclude_tickers or []
            sectors_text = ", ".join(preferred_sectors) if preferred_sectors else "any sector"
            
            prompt = f"""You are a stock market analyst. Use web search to find the most TRENDING stocks right now.

TASK: Find up to {self.max_stocks} MOST TRENDING stocks in these sectors: {sectors_text}

CRITERIA:
- Focus on stocks with significant positive momentum, news coverage, or market attention in the last 24-48 hours
- Must be publicly traded on major US exchanges (NYSE, NASDAQ)
- Prioritize stocks with:
  * Breaking news or major announcements
  * Strong price momentum or volume spikes
  * High social media/analyst attention
  * Upcoming catalysts (earnings, product launches, etc.)

EXCLUDE these stocks (already owned or recently analyzed):
{', '.join(exclude_list) if exclude_list else 'None'}

SECTORS TO FOCUS ON: {sectors_text}

FORMAT YOUR RESPONSE AS A JSON OBJECT:
{{
    "stocks": [
        {{
            "ticker": "STOCK_SYMBOL",
            "reason": "Brief reason why this stock is trending (1-2 sentences)"
        }}
    ]
}}

IMPORTANT:
- Return ONLY the JSON object, no other text
- Include ONLY ticker symbols that are different from the excluded list
- Focus on REAL-TIME trending stocks based on current web search results
- Limit to {self.max_stocks} stocks maximum"""

            # Call LLM with web search enabled
            # Note: OpenAI's ChatGPT models automatically use web search when needed
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse response - handle various response formats
            response_content = response.content
            if isinstance(response_content, str):
                response_text = response_content.strip()
            elif isinstance(response_content, list) and response_content:
                # If content is a list, convert first element to string
                first_item = response_content[0]
                if isinstance(first_item, dict):
                    response_text = json.dumps(first_item)
                else:
                    response_text = str(first_item).strip()
            else:
                response_text = str(response_content)
            
            # Extract JSON from response (handle cases where LLM adds markdown formatting)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Extract ticker symbols
            tickers = [stock['ticker'].upper() for stock in result.get('stocks', [])]
            
            # Filter out any tickers that are in the exclude list (double-check)
            exclude_set = set(ticker.upper() for ticker in exclude_list)
            tickers = [ticker for ticker in tickers if ticker not in exclude_set]
            
            # Log the discoveries with reasons
            if tickers and 'stocks' in result:
                print("\n💡 LLM Web Search Results:")
                for stock in result['stocks']:
                    ticker = stock['ticker'].upper()
                    if ticker in tickers:
                        print(f"   • {ticker}: {stock.get('reason', 'N/A')}")
            
            return tickers[:self.max_stocks]
            
        except Exception as e:
            print(f"❌ Error discovering stocks with LLM web search: {e}")
            import traceback
            traceback.print_exc()
            return []

