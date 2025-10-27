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


# ==================== Exit Strategy Guidance ====================

EXIT_STRATEGY_GUIDANCE = """
BRACKET ORDER REQUIREMENTS:
===========================
Every BUY must use place_buy_bracket_order with mandatory stop-loss & take-profit.

Example:
place_buy_bracket_order(
    symbol="AAPL",
    notional=5000,                # Dollar amount to invest
    type="market",
    stop_loss_price=171.00,       # Entry × 0.95 (-5%)
    take_profit_price=198.00      # Entry × 1.10 (+10%)
)

CALCULATION:
Entry: $180.00
Stop-Loss (-5%): $180 × 0.95 = $171.00
Take-Profit (+10%): $180 × 1.10 = $198.00

RISK MANAGEMENT:
- Position size: Use 5-10% of available cash per position
- Stop-loss: Typically 3-7% below entry
- Take-profit: Typically 10-20% above entry (aim for 2:1 reward:risk)
- NEVER place orders without exits
"""


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
        f"PORTFOLIO MANAGER STATUS\n"
        f"{'=' * 80}\n"
        f"Current Time: {current_time}\n"
        f"Run #{iteration_count} ({minutes_since_start} minutes since start)\n"
        f"{'=' * 80}"
    )
    
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
    
    # ==================== Current Positions Overview ====================
    if positions:
        prompt_parts.append(
            f"\nCURRENT POSITIONS\n"
            f"{'=' * 80}"
        )
        for pos in positions:
            symbol = pos.get("symbol", "UNKNOWN")
            qty = pos.get("qty", 0)
            current_price = pos.get("current_price", 0)
            avg_entry = pos.get("avg_entry_price", 0)
            unrealized_pl_pct = pos.get("unrealized_plpc", 0) * 100
            market_value = pos.get("market_value", 0)
            
            prompt_parts.append(
                f"  {symbol}: {qty:.2f} shares @ ${current_price:.2f} "
                f"(entry: ${avg_entry:.2f}, P&L: {unrealized_pl_pct:+.1f}%, value: ${market_value:,.2f})"
            )
        prompt_parts.append("=" * 80)
    else:
        prompt_parts.append(
            f"\nCURRENT POSITIONS\n"
            f"{'=' * 80}\n"
            f"No positions currently held\n"
            f"{'=' * 80}"
        )
    
    # ==================== Account Information ====================
    prompt_parts.append(
        f"\nACCOUNT SUMMARY\n"
        f"{'=' * 80}\n"
        f"Available Cash: ${cash:,.2f}\n"
        f"Portfolio Value: ${portfolio_value:,.2f}\n"
        f"Total Equity: ${equity:,.2f}\n"
        f"Total Return: {total_return_pct:+.2f}%\n"
        f"{'=' * 80}"
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
            # Functions are defined at the bottom of this file
            
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
    
    ema_values: list[float] = []
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
    
    rsi_values: list[float] = []
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    for i in range(period - 1, len(changes)):
        window = changes[i - period + 1:i + 1]
        gains = [max(0, change) for change in window]
        losses = [abs(min(0, change)) for change in window]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            rsi = 100.0
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
    
    true_ranges: list[float] = []
    
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

