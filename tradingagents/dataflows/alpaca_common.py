import os
import urllib3
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.trading.client import TradingClient

# Disable SSL warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Base URLs
PAPER_API_URL = "https://paper-api.alpaca.markets"
LIVE_API_URL = "https://api.alpaca.markets"
DATA_API_URL = "https://data.alpaca.markets"


def get_api_key() -> str:
    """Retrieve the Alpaca API key from environment variables."""
    api_key = os.getenv("ALPACA_API_KEY")
    if not api_key:
        raise ValueError("ALPACA_API_KEY environment variable is not set.")
    return api_key


def get_secret_key() -> str:
    """Retrieve the Alpaca secret key from environment variables."""
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not secret_key:
        raise ValueError("ALPACA_SECRET_KEY environment variable is not set.")
    return secret_key


def is_paper_mode() -> bool:
    """Check if we're running in paper mode (default: True for safety)."""
    paper_mode = os.getenv("ALPACA_PAPER_MODE", "true").lower()
    return paper_mode in ("true", "1", "yes", "on")


def get_trading_client() -> TradingClient:
    """
    Get an Alpaca Trading Client instance.
    Always uses paper trading by default for safety.
    """
    api_key = get_api_key()
    secret_key = get_secret_key()
    paper = is_paper_mode()
    
    # Only print warning for live mode, not info for paper mode
    if not paper:
        print("⚠️  WARNING: Running in LIVE trading mode! Set ALPACA_PAPER_MODE=true for paper trading.")
    
    # Create client with SSL verification disabled
    client = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        url_override=PAPER_API_URL if paper else LIVE_API_URL
    )
    
    return client


def get_data_client() -> StockHistoricalDataClient:
    """
    Get an Alpaca Historical Stock Data Client instance.
    Used for fetching historical OHLCV data.
    """
    api_key = get_api_key()
    secret_key = get_secret_key()
    
    # Create client
    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
        url_override=DATA_API_URL
    )
    
    return client


def get_news_client() -> NewsClient:
    """
    Get an Alpaca News Client instance.
    Used for fetching news data.
    """
    api_key = get_api_key()
    secret_key = get_secret_key()
    
    # Create client
    client = NewsClient(
        api_key=api_key,
        secret_key=secret_key
    )
    
    return client


def disable_ssl_verification():
    """
    Disable SSL verification for requests library.
    This is called globally to ensure all requests skip SSL verification.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.ssl_ import create_urllib3_context
    
    # Create a custom SSL context that doesn't verify
    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            context = create_urllib3_context()
            context.check_hostname = False
            context.verify_mode = False
            kwargs['ssl_context'] = context
            return super().init_poolmanager(*args, **kwargs)
    
    # Monkey patch requests to use our adapter
    requests.packages.urllib3.disable_warnings()
    
    # Set default verify to False
    requests.Session.verify = False


def validate_alpaca_date(date_value, date_name="end date"):
    """
    Validate and adjust dates for Alpaca API free tier limitations.
    
    Alpaca free tier cannot query data from the past 15 minutes.
    This function checks if a date is too recent and adjusts it if necessary.
    
    Args:
        date_value: datetime object or datetime string in YYYY-MM-DD format
        date_name: name of the date parameter for error messages
        
    Returns:
        Adjusted datetime object that complies with Alpaca's 15-minute restriction
    """
    # Convert string to datetime if needed
    if isinstance(date_value, str):
        date_value = datetime.strptime(date_value, "%Y-%m-%d")
    
    # Calculate the cutoff time (15 minutes + 1 minute buffer = 16 minutes ago)
    cutoff_time = datetime.now() - timedelta(minutes=16)
    
    # If the date is too recent, adjust it silently
    if date_value > cutoff_time:
        return cutoff_time
    
    return date_value


# Initialize SSL settings on module import
disable_ssl_verification()

