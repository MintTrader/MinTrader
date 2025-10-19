from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from .alpaca_common import get_data_client, validate_alpaca_date


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Returns raw daily OHLCV values for a stock from Alpaca.

    Args:
        symbol: The ticker symbol. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily time series data filtered to the date range.
        Format matches Alpha Vantage output: timestamp,open,high,low,close,volume
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Validate end date for Alpaca free tier (15-minute restriction)
        end_dt = validate_alpaca_date(end_dt, "end date")
        
        # Add one day to end_date to make it inclusive
        end_dt = end_dt + timedelta(days=1)
        
        # Get data client
        client = get_data_client()
        
        # Create request for daily bars
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt
        )
        
        # Fetch bars
        bars = client.get_stock_bars(request_params)
        
        # Convert to DataFrame
        # Check if data exists in the .data attribute (the raw data storage)
        if hasattr(bars, 'data') and symbol in bars.data and bars.data[symbol]:
            # BarSet.__getitem__ returns a list of Bar objects, not a DataFrame
            # We need to manually convert to DataFrame
            bar_data = bars[symbol]
            if hasattr(bar_data, 'df'):
                df = bar_data.df
            else:
                # Manually convert list of bars to DataFrame
                df = pd.DataFrame([vars(bar) for bar in bar_data])
        else:
            # No data available - raise exception to trigger fallback
            available = list(bars.data.keys()) if hasattr(bars, 'data') else 'N/A'
            raise ValueError(f"No data returned from Alpaca for symbol {symbol} in date range {start_date} to {end_date}. Available symbols: {available}")
        
        # Reset index to get timestamp as column
        df = df.reset_index()
        
        # Rename columns to match Alpha Vantage format
        df = df.rename(columns={
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Select only the columns we need
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Format timestamp as date string (YYYY-MM-DD)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
        
        # Sort by date descending (most recent first) to match Alpha Vantage
        df = df.sort_values('timestamp', ascending=False)
        
        # Convert to CSV string
        csv_string = df.to_csv(index=False)
        
        return csv_string
        
    except Exception as e:
        print(f"Error fetching stock data from Alpaca for {symbol}: {e}")
        # Re-raise exception so fallback mechanism can try next vendor
        raise

