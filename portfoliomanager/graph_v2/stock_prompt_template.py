"""
Stock Portfolio Prompt Template Generator

Generates a comprehensive market state prompt for LLM-based stock portfolio management,
similar to the crypto trading prompt structure.

This module fetches:
- Current market data for all portfolio stocks
- Technical indicators (EMA, MACD, RSI, Volume, ATR)
- Intraday price series
- Longer-term context (daily timeframe)
- Account performance and positions
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_stock_portfolio_prompt(
    state: Dict[str, Any],
    iteration_count: int = 0,
    start_time: Optional[datetime] = None
) -> str:
    """
    Generate a comprehensive prompt for LLM with current portfolio state and market data.
    
    Args:
        state: Portfolio state containing account, positions, market data
        iteration_count: Number of times the portfolio manager has been invoked
        start_time: When the portfolio manager started running
        
    Returns:
        Formatted prompt string with all market state and portfolio data
    """
    
    if start_time is None:
        start_time = datetime.now()
    
    minutes_since_start = int((datetime.now() - start_time).total_seconds() / 60)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Extract state data
    account = state.get("account", {})
    positions = state.get("positions", [])
    market_clock = state.get("market_clock", {})
    
    # Calculate portfolio metrics
    portfolio_value = account.get("portfolio_value", 0)
    cash = account.get("cash", 0)
    equity = account.get("equity", 0)
    
    # Calculate total return (assuming starting value - you may want to track this)
    # For now, using a placeholder calculation
    last_equity = account.get("last_equity", portfolio_value)
    total_return_pct = ((portfolio_value - last_equity) / last_equity * 100) if last_equity > 0 else 0.0
    
    # Start building the prompt
    prompt_parts = []
    
    # ==================== Header ====================
    prompt_parts.append(
        f"It has been {minutes_since_start} minutes since you started managing the portfolio. "
        f"The current time is {current_time} and you've been invoked {iteration_count} times. "
        f"Below, we are providing you with a variety of state data, price data, and technical signals "
        f"so you can discover alpha. Below that is your current account information, value, performance, positions, etc.\n"
    )
    
    prompt_parts.append(
        "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n"
    )
    
    prompt_parts.append(
        "Timeframes note: Unless stated otherwise in a section title, "
        "intraday series are provided at 5-minute intervals. "
        "If a stock uses a different interval, it is explicitly stated in that stock's section.\n"
    )
    
    prompt_parts.append("=" * 80)
    
    # ==================== Market Status ====================
    is_open = market_clock.get("is_open", False)
    next_open = market_clock.get("next_open", "N/A")
    next_close = market_clock.get("next_close", "N/A")
    
    prompt_parts.append(
        f"\nMARKET STATUS\n"
        f"{'=' * 80}"
    )
    prompt_parts.append(
        f"Market is currently: {'OPEN' if is_open else 'CLOSED'}"
    )
    if not is_open:
        prompt_parts.append(f"Next open: {next_open}")
    else:
        prompt_parts.append(f"Next close: {next_close}")
    
    prompt_parts.append("=" * 80)
    
    # ==================== Individual Stock Data ====================
    prompt_parts.append(
        f"\nCURRENT MARKET STATE FOR ALL STOCKS IN PORTFOLIO\n"
        f"{'=' * 80}"
    )
    
    # Get market data for each position
    # Import alpaca data functions
    from portfoliomanager.dataflows.alpaca_portfolio import get_intraday_bars, get_daily_bars
    
    for position in positions:
        symbol = position.get("symbol", "UNKNOWN")
        current_price = position.get("current_price", 0)
        
        # Initialize variables for this symbol
        intraday_bars = []
        daily_bars = []
        intraday_closes = []
        intraday_volumes = []
        daily_closes = []
        daily_highs = []
        daily_lows = []
        daily_volumes = []
        ema20_intraday = []
        macd_intraday = []
        rsi14_intraday = []
        
        try:
            # Fetch intraday data (5-minute bars, last 2 hours = 24 bars)
            intraday_bars = get_intraday_bars(symbol, timeframe="5Min", limit=24)
            
            # Fetch daily data for longer-term context (60 days)
            daily_bars = get_daily_bars(symbol, limit=60)
            
            # Calculate indicators if we have enough data
            if intraday_bars and daily_bars:
                # Extract price series
                intraday_closes = [bar['close'] for bar in intraday_bars]
                intraday_volumes = [bar['volume'] for bar in intraday_bars]
                
                daily_closes = [bar['close'] for bar in daily_bars]
                daily_highs = [bar['high'] for bar in daily_bars]
                daily_lows = [bar['low'] for bar in daily_bars]
                daily_volumes = [bar['volume'] for bar in daily_bars]
                
                # Calculate intraday indicators
                ema20_intraday = calculate_ema(intraday_closes, period=20) if len(intraday_closes) >= 20 else []
                macd_intraday = calculate_macd(intraday_closes) if len(intraday_closes) >= 26 else []
                rsi14_intraday = calculate_rsi(intraday_closes, period=14) if len(intraday_closes) >= 15 else []
                
                # Calculate daily indicators
                atr14_daily = calculate_atr(daily_highs, daily_lows, daily_closes, period=14) if len(daily_closes) >= 15 else 0.0
                avg_volume_20d = sum(daily_volumes[-20:]) / 20 if len(daily_volumes) >= 20 else 0
                
                # Get current values
                current_ema20 = ema20_intraday[-1] if ema20_intraday else current_price
                current_macd = macd_intraday[-1] if macd_intraday else 0
                current_rsi14 = rsi14_intraday[-1] if rsi14_intraday else 50
                current_volume = intraday_volumes[-1] if intraday_volumes else 0
                
                # Calculate spread (approximate as percentage of price)
                current_spread = current_price * 0.001  # Approximate 0.1% spread
                
                # Add stock section
                prompt_parts.append(
                    f"\nALL {symbol} DATA\n"
                    f"{'-' * 80}"
                )
                
                # Current state with real data
                prompt_parts.append(
                    f"current_price = {current_price:.2f}, "
                    f"current_ema20 = {current_ema20:.2f}, "
                    f"current_macd = {current_macd:.3f}, "
                    f"current_rsi (14 period) = {current_rsi14:.2f}"
                )
                
                # Volume and volatility
                prompt_parts.append(
                    f"\nIn addition, here is the latest {symbol} volume and volatility metrics:"
                )
                prompt_parts.append(
                    f"Average Volume (20-day): {avg_volume_20d:,.0f}  Current Volume: {current_volume:,.0f}"
                )
                prompt_parts.append(
                    f"ATR (14-period): {atr14_daily:.2f}  Current Spread: {current_spread:.3f}"
                )
            else:
                # Fallback if data fetch failed
                prompt_parts.append(
                    f"\nALL {symbol} DATA\n"
                    f"{'-' * 80}"
                )
                prompt_parts.append(
                    f"current_price = {current_price:.2f}, "
                    f"current_ema20 = N/A (insufficient data), "
                    f"current_macd = N/A (insufficient data), "
                    f"current_rsi (14 period) = N/A (insufficient data)"
                )
                prompt_parts.append(
                    f"\nIn addition, here is the latest {symbol} volume and volatility metrics:"
                )
                prompt_parts.append(
                    f"Average Volume (20-day): N/A  Current Volume: N/A"
                )
                prompt_parts.append(
                    f"ATR (14-period): N/A  Current Spread: N/A"
                )
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            # Fallback if exception occurred
            prompt_parts.append(
                f"\nALL {symbol} DATA\n"
                f"{'-' * 80}"
            )
            prompt_parts.append(
                f"current_price = {current_price:.2f}, "
                f"Error fetching additional data: {str(e)}"
            )
        
        # Intraday series and longer-term context (only if we have the data)
        try:
            if intraday_bars and daily_bars and intraday_closes and daily_closes:
                # Intraday series (last 10 bars)
                last_10_prices = intraday_closes[-10:] if len(intraday_closes) >= 10 else intraday_closes
                last_10_ema20 = ema20_intraday[-10:] if len(ema20_intraday) >= 10 else ema20_intraday
                last_10_macd = macd_intraday[-10:] if len(macd_intraday) >= 10 else macd_intraday
                last_10_rsi = rsi14_intraday[-10:] if len(rsi14_intraday) >= 10 else rsi14_intraday
                last_10_volumes = intraday_volumes[-10:] if len(intraday_volumes) >= 10 else intraday_volumes
                
                prompt_parts.append(
                    "\nIntraday series (5-minute intervals, oldest → latest):"
                )
                prompt_parts.append(
                    f"{symbol} prices: {[round(p, 2) for p in last_10_prices]}"
                )
                if last_10_ema20:
                    prompt_parts.append(
                        f"EMA indicators (20-period): {[round(e, 2) for e in last_10_ema20]}"
                    )
                if last_10_macd:
                    prompt_parts.append(
                        f"MACD indicators: {[round(m, 3) for m in last_10_macd]}"
                    )
                if last_10_rsi:
                    prompt_parts.append(
                        f"RSI indicators (14-Period): {[round(r, 2) for r in last_10_rsi]}"
                    )
                prompt_parts.append(
                    f"Volume series: {[int(v) for v in last_10_volumes]}"
                )
                
                # Calculate daily indicators for longer-term context
                sma20_daily = sum(daily_closes[-20:]) / 20 if len(daily_closes) >= 20 else 0
                sma50_daily = sum(daily_closes[-50:]) / 50 if len(daily_closes) >= 50 else 0
                atr30_daily = calculate_atr(daily_highs, daily_lows, daily_closes, period=30) if len(daily_closes) >= 31 else 0.0
                macd_daily = calculate_macd(daily_closes) if len(daily_closes) >= 26 else []
                rsi14_daily = calculate_rsi(daily_closes, period=14) if len(daily_closes) >= 15 else []
                
                last_10_macd_daily = macd_daily[-10:] if len(macd_daily) >= 10 else macd_daily
                last_10_rsi_daily = rsi14_daily[-10:] if len(rsi14_daily) >= 10 else rsi14_daily
                
                # Longer-term context (daily timeframe)
                prompt_parts.append(
                    "\nLonger-term context (daily timeframe):"
                )
                if sma20_daily and sma50_daily:
                    prompt_parts.append(
                        f"20-Day SMA: {sma20_daily:.2f} vs. 50-Day SMA: {sma50_daily:.2f}"
                    )
                if atr14_daily and atr30_daily:
                    prompt_parts.append(
                        f"14-Day ATR: {atr14_daily:.2f} vs. 30-Day ATR: {atr30_daily:.2f}"
                    )
                if len(daily_volumes) >= 20:
                    prompt_parts.append(
                        f"Current Volume: {daily_volumes[-1]:,.0f} vs. Average Volume (20-day): {avg_volume_20d:,.0f}"
                    )
                if last_10_macd_daily:
                    prompt_parts.append(
                        f"MACD indicators (daily): {[round(m, 3) for m in last_10_macd_daily]}"
                    )
                if last_10_rsi_daily:
                    prompt_parts.append(
                        f"RSI indicators (14-Period daily): {[round(r, 2) for r in last_10_rsi_daily]}"
                    )
                
                # Fundamental metrics (placeholder for now)
                prompt_parts.append(
                    "\nFundamental metrics:"
                )
                prompt_parts.append(
                    "Market Cap: N/A, P/E Ratio: N/A, Dividend Yield: N/A%, Beta: N/A"
                )
        except Exception as e:
            logger.error(f"Error processing indicators for {symbol}: {e}")
            prompt_parts.append(
                "\nIntraday series: Data processing error"
            )
        
        prompt_parts.append("=" * 80)
    
    # ==================== Account Information ====================
    prompt_parts.append(
        f"\nHERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE\n"
        f"{'=' * 80}"
    )
    
    prompt_parts.append(
        f"Current Total Return (percent): {total_return_pct:.2f}%"
    )
    prompt_parts.append(
        f"Available Cash: ${cash:,.2f}"
    )
    prompt_parts.append(
        f"Current Account Value: ${portfolio_value:,.2f}"
    )
    prompt_parts.append(
        f"Total Equity: ${equity:,.2f}"
    )
    
    # Current positions
    if positions:
        prompt_parts.append(
            "\nCurrent live positions & performance:"
        )
        for pos in positions:
            position_str = (
                f"{{'symbol': '{pos.get('symbol')}', "
                f"'quantity': {pos.get('qty', 0):.2f}, "
                f"'entry_price': {pos.get('avg_entry_price', 0):.2f}, "
                f"'current_price': {pos.get('current_price', 0):.2f}, "
                f"'market_value': {pos.get('market_value', 0):.2f}, "
                f"'unrealized_pnl': {pos.get('unrealized_pl', 0):.2f}, "
                f"'unrealized_pnl_pct': {(pos.get('unrealized_plpc', 0) * 100):.2f}%, "
                f"'cost_basis': {pos.get('cost_basis', 0):.2f}, "
                f"'change_today': {pos.get('change_today', 0):.2f}}}"
            )
            prompt_parts.append(position_str)
    else:
        prompt_parts.append(
            "\nNo current positions"
        )
    
    # Performance metrics
    # You can add Sharpe ratio, max drawdown, etc. if calculated
    prompt_parts.append(
        "\nPortfolio Beta: [CALCULATE IF NEEDED]"
    )
    prompt_parts.append(
        "Sharpe Ratio: [CALCULATE IF NEEDED]"
    )
    prompt_parts.append(
        "Max Drawdown: [CALCULATE IF NEEDED]"
    )
    
    return "\n".join(prompt_parts)


def generate_stock_trading_prompt_with_live_data(
    state: Dict[str, Any],
    market_data_fetcher: Any,  # Your market data API client
    iteration_count: int = 0,
    start_time: Optional[datetime] = None
) -> str:
    """
    Generate a comprehensive prompt with LIVE market data for all stocks.
    
    This version actually fetches real market data instead of using placeholders.
    
    Args:
        state: Portfolio state
        market_data_fetcher: Object with methods to fetch market data
        iteration_count: Number of iterations
        start_time: Start time
        
    Returns:
        Formatted prompt with live market data
    """
    
    if start_time is None:
        start_time = datetime.now()
    
    minutes_since_start = int((datetime.now() - start_time).total_seconds() / 60)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    account = state.get("account", {})
    positions = state.get("positions", [])
    market_clock = state.get("market_clock", {})
    
    portfolio_value = account.get("portfolio_value", 0)
    cash = account.get("cash", 0)
    equity = account.get("equity", 0)
    last_equity = account.get("last_equity", portfolio_value)
    total_return_pct = ((portfolio_value - last_equity) / last_equity * 100) if last_equity > 0 else 0.0
    
    prompt_parts = []
    
    # Header
    prompt_parts.append(
        f"It has been {minutes_since_start} minutes since you started managing the portfolio. "
        f"The current time is {current_time} and you've been invoked {iteration_count} times. "
        f"Below, we are providing you with a variety of state data, price data, and technical signals "
        f"so you can discover alpha. Below that is your current account information, value, performance, positions, etc.\n"
    )
    prompt_parts.append(
        "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n"
    )
    prompt_parts.append(
        "Timeframes note: Unless stated otherwise, "
        "intraday series are provided at 5-minute intervals.\n"
    )
    prompt_parts.append("=" * 80)
    
    # Market Status
    is_open = market_clock.get("is_open", False)
    next_open = market_clock.get("next_open", "N/A")
    next_close = market_clock.get("next_close", "N/A")
    
    prompt_parts.append(
        f"\nMARKET STATUS\n"
        f"{'=' * 80}\n"
        f"Market is currently: {'OPEN' if is_open else 'CLOSED'}"
    )
    if not is_open:
        prompt_parts.append(f"Next open: {next_open}")
    else:
        prompt_parts.append(f"Next close: {next_close}")
    
    prompt_parts.append("=" * 80)
    
    # Individual Stock Data with LIVE fetching
    prompt_parts.append(
        f"\nCURRENT MARKET STATE FOR ALL STOCKS IN PORTFOLIO\n"
        f"{'=' * 80}"
    )
    
    for position in positions:
        symbol = position.get("symbol", "UNKNOWN")
        current_price = position.get("current_price", 0)
        
        try:
            # Fetch intraday data (5-minute bars, last 2 hours = 24 bars, show last 10)
            intraday_bars = market_data_fetcher.get_intraday_bars(
                symbol, 
                timeframe="5Min",
                limit=24
            )
            
            # Fetch daily data for longer-term context
            daily_bars = market_data_fetcher.get_daily_bars(
                symbol,
                limit=60  # 60 days for indicators
            )
            
            # Calculate indicators
            from .indicators import calculate_ema, calculate_macd, calculate_rsi, calculate_atr
            
            # Intraday indicators (on 5-min bars)
            intraday_closes = [bar['close'] for bar in intraday_bars]
            intraday_volumes = [bar['volume'] for bar in intraday_bars]
            
            ema20_intraday = calculate_ema(intraday_closes, period=20)
            macd_intraday = calculate_macd(intraday_closes)
            rsi14_intraday = calculate_rsi(intraday_closes, period=14)
            
            # Daily indicators
            daily_closes = [bar['close'] for bar in daily_bars]
            daily_highs = [bar['high'] for bar in daily_bars]
            daily_lows = [bar['low'] for bar in daily_bars]
            daily_volumes = [bar['volume'] for bar in daily_bars]
            
            sma20_daily = sum(daily_closes[-20:]) / 20
            sma50_daily = sum(daily_closes[-50:]) / 50
            atr14_daily = calculate_atr(daily_highs, daily_lows, daily_closes, period=14)
            atr30_daily = calculate_atr(daily_highs, daily_lows, daily_closes, period=30)
            macd_daily = calculate_macd(daily_closes)
            rsi14_daily = calculate_rsi(daily_closes, period=14)
            
            avg_volume_20d = sum(daily_volumes[-20:]) / 20
            
            # Current state
            current_ema20 = ema20_intraday[-1] if ema20_intraday else current_price
            current_macd = macd_intraday[-1] if macd_intraday else 0
            current_rsi14 = rsi14_intraday[-1] if rsi14_intraday else 50
            
            prompt_parts.append(
                f"\nALL {symbol} DATA\n"
                f"{'-' * 80}\n"
                f"current_price = {current_price:.2f}, "
                f"current_ema20 = {current_ema20:.2f}, "
                f"current_macd = {current_macd:.3f}, "
                f"current_rsi (14 period) = {current_rsi14:.3f}"
            )
            
            # Volume and volatility
            current_volume = intraday_volumes[-1] if intraday_volumes else 0
            prompt_parts.append(
                f"\nIn addition, here is the latest {symbol} volume and volatility metrics:\n"
                f"Average Volume (20-day): {avg_volume_20d:,.0f}  "
                f"Current Volume: {current_volume:,.0f}\n"
                f"ATR (14-day): {atr14_daily:.2f}  "
                f"ATR (30-day): {atr30_daily:.2f}"
            )
            
            # Intraday series (last 10 bars)
            last_10_prices = intraday_closes[-10:]
            last_10_ema20 = ema20_intraday[-10:] if len(ema20_intraday) >= 10 else ema20_intraday
            last_10_macd = macd_intraday[-10:] if len(macd_intraday) >= 10 else macd_intraday
            last_10_rsi = rsi14_intraday[-10:] if len(rsi14_intraday) >= 10 else rsi14_intraday
            last_10_volumes = intraday_volumes[-10:]
            
            prompt_parts.append(
                "\nIntraday series (5-minute intervals, oldest → latest):\n"
                f"{symbol} prices: {[round(p, 2) for p in last_10_prices]}\n"
                f"EMA indicators (20-period): {[round(e, 3) for e in last_10_ema20]}\n"
                f"MACD indicators: {[round(m, 3) for m in last_10_macd]}\n"
                f"RSI indicators (14-Period): {[round(r, 3) for r in last_10_rsi]}\n"
                f"Volume series: {[int(v) for v in last_10_volumes]}"
            )
            
            # Longer-term context (daily)
            last_10_macd_daily = macd_daily[-10:] if len(macd_daily) >= 10 else macd_daily
            last_10_rsi_daily = rsi14_daily[-10:] if len(rsi14_daily) >= 10 else rsi14_daily
            
            prompt_parts.append(
                "\nLonger-term context (daily timeframe):\n"
                f"20-Day SMA: {sma20_daily:.2f} vs. 50-Day SMA: {sma50_daily:.2f}\n"
                f"14-Day ATR: {atr14_daily:.2f} vs. 30-Day ATR: {atr30_daily:.2f}\n"
                f"Current Volume: {daily_volumes[-1]:,.0f} vs. Average Volume (20-day): {avg_volume_20d:,.0f}\n"
                f"MACD indicators (daily): {[round(m, 3) for m in last_10_macd_daily]}\n"
                f"RSI indicators (14-Period daily): {[round(r, 3) for r in last_10_rsi_daily]}"
            )
            
            # Fundamental metrics (if available from fetcher)
            if hasattr(market_data_fetcher, 'get_fundamentals'):
                fundamentals = market_data_fetcher.get_fundamentals(symbol)
                prompt_parts.append(
                    "\nFundamental metrics:\n"
                    f"Market Cap: ${fundamentals.get('market_cap', 0):,.0f}, "
                    f"P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}, "
                    f"Dividend Yield: {fundamentals.get('dividend_yield', 0):.2f}%, "
                    f"Beta: {fundamentals.get('beta', 'N/A')}"
                )
            
            prompt_parts.append("=" * 80)
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            prompt_parts.append(
                f"\nALL {symbol} DATA\n"
                f"{'-' * 80}\n"
                f"Error fetching market data: {str(e)}\n"
                f"{'=' * 80}"
            )
    
    # Account Information
    prompt_parts.append(
        f"\nHERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE\n"
        f"{'=' * 80}\n"
        f"Current Total Return (percent): {total_return_pct:.2f}%\n"
        f"Available Cash: ${cash:,.2f}\n"
        f"Current Account Value: ${portfolio_value:,.2f}\n"
        f"Total Equity: ${equity:,.2f}"
    )
    
    if positions:
        prompt_parts.append(
            "\nCurrent live positions & performance:"
        )
        for pos in positions:
            position_dict = {
                'symbol': pos.get('symbol'),
                'quantity': round(pos.get('qty', 0), 2),
                'entry_price': round(pos.get('avg_entry_price', 0), 2),
                'current_price': round(pos.get('current_price', 0), 2),
                'market_value': round(pos.get('market_value', 0), 2),
                'unrealized_pnl': round(pos.get('unrealized_pl', 0), 2),
                'unrealized_pnl_pct': round(pos.get('unrealized_plpc', 0) * 100, 2),
                'cost_basis': round(pos.get('cost_basis', 0), 2),
                'change_today': round(pos.get('change_today', 0), 2)
            }
            prompt_parts.append(str(position_dict))
    else:
        prompt_parts.append("\nNo current positions")
    
    return "\n".join(prompt_parts)


# ==================== Helper Functions for Indicators ====================

def calculate_ema(prices: List[float], period: int = 20) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return []
    
    ema_values = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    # Calculate EMA
    for i in range(period, len(prices)):
        ema = (prices[i] - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(ema)
    
    return ema_values


def calculate_macd(
    prices: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> List[float]:
    """Calculate MACD (just the MACD line, not signal or histogram)."""
    if len(prices) < slow_period:
        return []
    
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    # Align the EMAs
    offset = slow_period - fast_period
    macd_line = [fast_ema[i + offset] - slow_ema[i] for i in range(len(slow_ema))]
    
    return macd_line


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        return []
    
    rsi_values = []
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    for i in range(period - 1, len(changes)):
        window = changes[i - period + 1:i + 1]
        gains = [max(0, change) for change in window]
        losses = [abs(min(0, change)) for change in window]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    return rsi_values


def calculate_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> float:
    """Calculate Average True Range."""
    if len(highs) < period + 1:
        return 0.0
    
    true_ranges = []
    
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        true_range = max(high_low, high_close, low_close)
        true_ranges.append(true_range)
    
    # Return the average of the last 'period' true ranges
    if len(true_ranges) >= period:
        return sum(true_ranges[-period:]) / period
    else:
        return sum(true_ranges) / len(true_ranges)

