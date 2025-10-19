from datetime import datetime
import json
from alpaca.data.requests import NewsRequest
from .alpaca_common import get_news_client, validate_alpaca_date


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """
    Returns news articles for a stock from Alpaca News API.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search (yyyy-mm-dd).
        end_date: End date for news search (yyyy-mm-dd).

    Returns:
        JSON string containing news data, formatted similarly to Alpha Vantage output.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Validate end date for Alpaca free tier (15-minute restriction)
        end_dt = validate_alpaca_date(end_dt, "end date")
        
        # Get news client
        client = get_news_client()
        
        # Create request for news
        request_params = NewsRequest(
            symbols=ticker,
            start=start_dt,
            end=end_dt,
            limit=50,  # Match Alpha Vantage's default limit
            sort="DESC"  # Most recent first
        )
        
        # Fetch news
        news_response = client.get_news(request_params)
        
        # Handle different response formats from Alpaca API
        # The response might be a dict, object with attributes, or iterable (NewsSet)
        news_data = []
        
        if isinstance(news_response, dict):
            news_data = news_response.get('news', [])
        elif hasattr(news_response, 'news'):
            news_data = news_response.news
        elif isinstance(news_response, tuple) and len(news_response) > 0:
            # Handle tuple response - first element might be the data
            news_data = news_response[0] if hasattr(news_response[0], '__iter__') else []
        elif hasattr(news_response, '__iter__') and not isinstance(news_response, str):
            try:
                news_data = list(news_response)
            except:
                news_data = []
        
        # Convert to format similar to Alpha Vantage
        articles = []
        for article in news_data:
            try:
                # Handle different article formats
                article_dict = {
                    "title": getattr(article, 'headline', getattr(article, 'title', 'No title')),
                    "url": getattr(article, 'url', getattr(article, 'link', '')),
                    "time_published": getattr(article, 'created_at', datetime.now()).strftime("%Y%m%dT%H%M%S") if hasattr(article, 'created_at') else datetime.now().strftime("%Y%m%dT%H%M%S"),
                    "authors": getattr(article, 'author', []) if hasattr(article, 'author') else [],
                    "summary": getattr(article, 'summary', '') if hasattr(article, 'summary') else "",
                    "source": getattr(article, 'source', '') if hasattr(article, 'source') else "",
                    "symbols": getattr(article, 'symbols', [ticker]) if hasattr(article, 'symbols') else [ticker],
                }
                articles.append(article_dict)
            except Exception as e:
                # Skip articles that can't be processed
                continue
        
        # Format as JSON
        result = {
            "feed": articles,
            "items": str(len(articles)),
            "source": "Alpaca News API"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        print(f"Error fetching news from Alpaca for {ticker}: {e}")
        # Re-raise exception so fallback mechanism can try next vendor
        raise

