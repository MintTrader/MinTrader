"""
Trading Constraints

Defines and validates trading constraints for portfolio management.
"""

from typing import Dict, Tuple, Any


class TradingConstraints:
    """Manages trading constraints and validation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trading constraints from config.
        
        Args:
            config: Configuration dictionary
        """
        self.max_position_size_pct = config.get('max_position_size_pct', 10)
        self.max_portfolio_concentration = config.get('max_portfolio_concentration', 30)
        self.max_trades_per_day = config.get('max_trades_per_day', 10)
        self.min_cash_reserve_pct = config.get('min_cash_reserve_pct', 5)
        self.stop_loss_pct = config.get('stop_loss_pct', 5)
        self.min_holding_days = config.get('min_holding_days', 7)
        self.min_conviction_score = config.get('min_conviction_score', 7)
        
        # Track trades executed today
        self.trades_today = 0
    
    def validate_trade(self, trade: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a trade against constraints.
        
        Args:
            trade: Trade dictionary with keys: ticker, action, quantity, conviction_score
            portfolio_state: Current portfolio state with positions, cash, portfolio_value
            
        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        # Check conviction score
        if trade.get('conviction_score', 0) < self.min_conviction_score:
            return False, f"Conviction score {trade.get('conviction_score')} below minimum {self.min_conviction_score}"
        
        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day ({self.max_trades_per_day}) reached"
        
        action = trade.get('action', '').upper()
        ticker = trade.get('ticker', '')
        quantity = trade.get('quantity', 0)
        
        # For BUY trades
        if action == 'BUY':
            # Check if we have enough cash
            cash_available = portfolio_state.get('cash', 0)
            estimated_cost = quantity * trade.get('estimated_price', 0)
            
            if estimated_cost > cash_available:
                return False, f"Insufficient cash: ${cash_available:.2f} available, ${estimated_cost:.2f} needed"
            
            # Check max position size
            portfolio_value = portfolio_state.get('portfolio_value', 0)
            if portfolio_value > 0:
                position_size_pct = (estimated_cost / portfolio_value) * 100
                if position_size_pct > self.max_position_size_pct:
                    return False, f"Position size {position_size_pct:.1f}% exceeds max {self.max_position_size_pct}%"
            
            # Check cash reserve
            remaining_cash = cash_available - estimated_cost
            min_cash_needed = portfolio_value * (self.min_cash_reserve_pct / 100)
            if remaining_cash < min_cash_needed:
                return False, f"Would violate minimum cash reserve requirement ({self.min_cash_reserve_pct}%)"
        
        # For SELL trades
        elif action == 'SELL':
            # Check if we have the position
            positions = portfolio_state.get('positions', {})
            if ticker not in positions:
                return False, f"No position in {ticker} to sell"
            
            current_qty = positions[ticker].get('qty', 0)
            if quantity > current_qty:
                return False, f"Trying to sell {quantity} shares but only have {current_qty}"
            
            # Check holding period (discourage quick flips unless stop-loss)
            holding_days = positions[ticker].get('holding_days', 0)
            unrealized_pl_pct = positions[ticker].get('unrealized_pl_pct', 0)
            
            # Allow selling if stop-loss triggered
            if unrealized_pl_pct <= -self.stop_loss_pct:
                pass  # Stop-loss override
            elif holding_days < self.min_holding_days:
                return False, f"Position held for only {holding_days} days (min: {self.min_holding_days}), " \
                              f"not in stop-loss territory (P&L: {unrealized_pl_pct:.1f}%)"
        
        return True, "Valid trade"
    
    def increment_trade_count(self):
        """Increment the daily trade counter"""
        self.trades_today += 1
    
    def reset_daily_counter(self):
        """Reset the daily trade counter (call at start of each trading day)"""
        self.trades_today = 0
    
    def get_prompt_text(self) -> str:
        """
        Format constraints for LLM prompt.
        
        Returns:
            Formatted constraints text
        """
        return f"""TRADING CONSTRAINTS:
1. Maximum position size: {self.max_position_size_pct}% of portfolio value
2. Maximum portfolio concentration: {self.max_portfolio_concentration}%
3. Maximum trades per day: {self.max_trades_per_day}
4. Minimum cash reserve: {self.min_cash_reserve_pct}% of portfolio value
5. Stop-loss threshold: {self.stop_loss_pct}% loss
6. Minimum holding period: {self.min_holding_days} days (unless stop-loss triggered)
7. Minimum conviction score: {self.min_conviction_score}/10 to execute trade

IMPORTANT: Only propose trades that meet these constraints. Quality over quantity."""

