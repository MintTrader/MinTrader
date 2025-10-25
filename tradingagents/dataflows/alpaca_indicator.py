from datetime import datetime, timedelta
import os
import logging
import pandas as pd
import numpy as np
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from .alpaca_common import get_data_client, validate_alpaca_date

# Get logger for this module
logger = logging.getLogger(__name__)

# Debug logging control
DEBUG_LOGGING = os.getenv('TRADINGAGENTS_DEBUG_LOGGING', 'false').lower() in ('true', '1', 'yes')


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close"
) -> str:
    """
    Returns technical indicator values calculated from Alpaca bar data.

    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to calculate
        curr_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back
        interval: Time interval (daily, weekly, monthly) - currently only daily supported
        time_period: Number of data points for calculation
        series_type: The desired price type (close, open, high, low)

    Returns:
        String containing indicator values and description
    """
    
    supported_indicators = {
        "close_50_sma": ("50 SMA", "close"),
        "close_200_sma": ("200 SMA", "close"),
        "close_10_ema": ("10 EMA", "close"),
        "macd": ("MACD", "close"),
        "macds": ("MACD Signal", "close"),
        "macdh": ("MACD Histogram", "close"),
        "rsi": ("RSI", "close"),
        "boll": ("Bollinger Middle", "close"),
        "boll_ub": ("Bollinger Upper Band", "close"),
        "boll_lb": ("Bollinger Lower Band", "close"),
        "atr": ("ATR", None),
        "vwma": ("VWMA", "close")
    }

    indicator_descriptions = {
        "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.",
        "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.",
        "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.",
        "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.",
        "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.",
        "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.",
        "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.",
        "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.",
        "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.",
        "vwma": "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    }

    if indicator not in supported_indicators:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(supported_indicators.keys())}"
        )

    try:
        # Parse date and calculate lookback
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        
        # Validate current date for Alpaca free tier (15-minute restriction)
        curr_date_dt = validate_alpaca_date(curr_date_dt, "current date")
        
        # Fetch extra data for indicator calculation (need more history for proper calculation)
        extra_days = 250  # Fetch extra days for accurate indicator calculation
        start_date = curr_date_dt - timedelta(days=look_back_days + extra_days)
        end_date = curr_date_dt + timedelta(days=1)  # Make end date inclusive
        
        # Validate end date as well (important when curr_date is today)
        end_date = validate_alpaca_date(end_date, "end date")
        
        # Get data client
        client = get_data_client()
        
        # Create request for daily bars
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        # Fetch bars
        bars = client.get_stock_bars(request_params)
        
        if DEBUG_LOGGING:
            logger.debug(f"INDICATOR: Type of bars: {type(bars)}")
            logger.debug(f"INDICATOR: hasattr data: {hasattr(bars, 'data')}")
            if hasattr(bars, 'data'):
                logger.debug(f"INDICATOR: symbol in bars.data: {symbol in bars.data}")
                logger.debug(f"INDICATOR: bars.data keys: {list(bars.data.keys())}")
                if symbol in bars.data:
                    logger.debug(f"INDICATOR: bars.data[{symbol}] length: {len(bars.data[symbol])}")
        
        # Convert to DataFrame
        # Check if data exists in the .data attribute (same as stock file)
        if hasattr(bars, 'data') and symbol in bars.data and bars.data[symbol]:
            # BarSet.__getitem__ returns a list of Bar objects, not a DataFrame
            bar_data = bars[symbol]
            if hasattr(bar_data, 'df'):
                df = bar_data.df.reset_index()
            else:
                # Manually convert list of bars to DataFrame
                if DEBUG_LOGGING:
                    logger.debug(f"INDICATOR: bar_data is a list, converting manually")
                df = pd.DataFrame([vars(bar) for bar in bar_data])
                if 'timestamp' not in df.columns:
                    df = df.reset_index()
        else:
            available = list(bars.data.keys()) if hasattr(bars, 'data') else 'N/A'
            raise ValueError(f"No data returned from Alpaca for symbol {symbol}. Available: {available}")
        
        # Ensure we have data
        if df.empty:
            raise ValueError(f"No data returned from Alpaca for symbol {symbol} (empty dataframe)")
        
        # Calculate the requested indicator
        if indicator == "close_50_sma":
            df['indicator'] = df['close'].rolling(window=50).mean()
        elif indicator == "close_200_sma":
            df['indicator'] = df['close'].rolling(window=200).mean()
        elif indicator == "close_10_ema":
            df['indicator'] = df['close'].ewm(span=10, adjust=False).mean()
        elif indicator in ["macd", "macds", "macdh"]:
            # Calculate MACD components
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            if indicator == "macd":
                df['indicator'] = macd_line
            elif indicator == "macds":
                df['indicator'] = signal_line
            else:  # macdh
                df['indicator'] = macd_hist
        elif indicator == "rsi":
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=time_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=time_period).mean()
            rs = gain / loss
            df['indicator'] = 100 - (100 / (1 + rs))
        elif indicator in ["boll", "boll_ub", "boll_lb"]:
            # Calculate Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            
            if indicator == "boll":
                df['indicator'] = sma_20
            elif indicator == "boll_ub":
                df['indicator'] = sma_20 + (2 * std_20)
            else:  # boll_lb
                df['indicator'] = sma_20 - (2 * std_20)
        elif indicator == "atr":
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['indicator'] = true_range.rolling(window=time_period).mean()
        elif indicator == "vwma":
            # Calculate Volume Weighted Moving Average
            df['indicator'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Filter to the requested date range
        before_date = curr_date_dt - timedelta(days=look_back_days)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        mask = (pd.to_datetime(df['date']) >= pd.to_datetime(before_date.date())) & \
               (pd.to_datetime(df['date']) <= pd.to_datetime(curr_date_dt.date()))
        df_filtered = df[mask].copy()
        
        # Format output
        ind_string = ""
        for _, row in df_filtered.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            value = row['indicator']
            if pd.notna(value):
                ind_string += f"{date_str}: {value:.4f}\n"
        
        if not ind_string:
            ind_string = "No data available for the specified date range.\n"
        
        result_str = (
            f"## {indicator.upper()} values from {before_date.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n\n"
            + indicator_descriptions.get(indicator, "No description available.")
        )
        
        return result_str
        
    except Exception as e:
        logger.error(f"Error calculating indicator {indicator} from Alpaca for {symbol}: {e}")
        # Re-raise exception so fallback mechanism can try next vendor
        raise

