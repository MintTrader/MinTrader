"""
Portfolio Logger

Logging utility that matches the tradingagents message_tool.log format
for consistency and transparency.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PortfolioLogger:
    """Logger for portfolio management activities"""
    
    def __init__(self, log_file: str):
        """
        Initialize the portfolio logger.
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear existing log file if it exists
        if self.log_file.exists():
            self.log_file.unlink()
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in HH:MM:SS format"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _write_log(self, message: str, print_to_console: bool = True):
        """Write a message to the log file and optionally print to console"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
        if print_to_console:
            print(message)
    
    def log_portfolio_summary(self, account: Dict, positions: List[Dict], 
                            market_status: Dict, open_orders: Optional[List[Dict]] = None):
        """
        Log comprehensive portfolio summary at the start.
        
        Args:
            account: Account information dictionary
            positions: List of position dictionaries
            market_status: Market clock information
            open_orders: List of open order dictionaries (optional)
        """
        timestamp = self._get_timestamp()
        
        # Header
        self._write_log("\n" + "="*70)
        self._write_log(f"{timestamp} [PORTFOLIO SUMMARY]")
        self._write_log("="*70)
        
        # Market Status
        is_open = market_status.get('is_open', False)
        market_status_str = "🟢 OPEN" if is_open else "🔴 CLOSED"
        self._write_log(f"\n📊 Market Status: {market_status_str}")
        if not is_open:
            next_open = market_status.get('next_open', 'Unknown')
            self._write_log(f"   Next Open: {next_open}")
        
        # Account Balances
        self._write_log(f"\n💰 Account Balance:")
        self._write_log(f"   Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        self._write_log(f"   Cash Available:  ${account.get('cash', 0):,.2f}")
        self._write_log(f"   Buying Power:    ${account.get('buying_power', 0):,.2f}")
        
        # Open Orders Summary
        if open_orders is None:
            open_orders = []
        
        self._write_log(f"\n📋 Open Orders: {len(open_orders)}")
        
        if open_orders:
            self._write_log("   Order Details:")
            for order in open_orders:
                if 'error' in order:
                    continue
                    
                order_id = order.get('id', order.get('order_id', 'N/A'))
                ticker = order.get('symbol', order.get('ticker', 'N/A'))
                side = order.get('side', '').upper()
                qty = order.get('qty', 0)
                order_type = order.get('type', order.get('order_type', 'N/A'))
                status = order.get('status', 'N/A')
                
                # Format prices if available
                limit_price = order.get('limit_price')
                stop_price = order.get('stop_price')
                price_str = ""
                if limit_price:
                    price_str = f" @ ${float(limit_price):.2f}"
                elif stop_price:
                    price_str = f" stop @ ${float(stop_price):.2f}"
                
                # Side emoji
                side_emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "⚪"
                
                self._write_log(
                    f"   {side_emoji} {ticker:6s} | {side:4s} {float(qty):>8.2f} shares{price_str} | "
                    f"Type: {order_type:8s} | Status: {status:10s} | ID: {order_id}"
                )
        else:
            self._write_log("   No open orders")
        
        # Positions Summary
        self._write_log(f"\n📈 Open Positions: {len(positions)}")
        
        if positions:
            total_pl = sum(pos.get('unrealized_pl', 0) for pos in positions)
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            total_pl_pct = (total_pl / (total_value - total_pl) * 100) if (total_value - total_pl) > 0 else 0
            
            self._write_log(f"   Total Position Value: ${total_value:,.2f}")
            self._write_log(f"   Total Unrealized P/L: ${total_pl:,.2f} ({total_pl_pct:+.2f}%)")
            self._write_log("\n   Position Details:")
            
            for pos in sorted(positions, key=lambda x: x.get('market_value', 0), reverse=True):
                ticker = pos.get('symbol', pos.get('ticker', 'N/A'))  # Try 'symbol' first, then 'ticker'
                qty = pos.get('qty', 0)
                current_price = pos.get('current_price', 0)
                market_value = pos.get('market_value', 0)
                unrealized_pl = pos.get('unrealized_pl', 0)
                unrealized_pl_pct = pos.get('unrealized_pl_pct', 0)
                
                pl_symbol = "📈" if unrealized_pl >= 0 else "📉"
                self._write_log(
                    f"   {pl_symbol} {ticker:6s} | {qty:>8.2f} shares @ ${current_price:>8.2f} | "
                    f"Value: ${market_value:>10,.2f} | P/L: ${unrealized_pl:>10,.2f} ({unrealized_pl_pct:>+6.2f}%)"
                )
        else:
            self._write_log("   No open positions")
        
        self._write_log("\n" + "="*70 + "\n")
    
    def log_system(self, message: str):
        """
        Log a system message.
        
        Args:
            message: System message to log
        """
        timestamp = self._get_timestamp()
        log_msg = f"{timestamp} [SYSTEM] {message}"
        self._write_log(log_msg)
    
    def log_agent(self, agent_name: str, message: str):
        """
        Log an agent message.
        
        Args:
            agent_name: Name of the agent
            message: Agent message to log
        """
        timestamp = self._get_timestamp()
        log_msg = f"{timestamp} [{agent_name}] {message}"
        self._write_log(log_msg)
        print(log_msg)
    
    def log_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """
        Log a tool call.
        
        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
        """
        timestamp = self._get_timestamp()
        args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        log_msg = f"{timestamp} [TOOL CALL] {tool_name}({args_str})"
        self._write_log(log_msg)
        print(log_msg)
    
    def log_tool_result(self, result: Any):
        """
        Log a tool result.
        
        Args:
            result: Result from tool execution
        """
        timestamp = self._get_timestamp()
        # Truncate long results
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "..."
        log_msg = f"{timestamp} [TOOL RESULT] {result_str}"
        self._write_log(log_msg)
    
    def log_reasoning(self, message: str):
        """
        Log reasoning or thinking process.
        
        Args:
            message: Reasoning message to log
        """
        timestamp = self._get_timestamp()
        log_msg = f"{timestamp} [REASONING] {message}"
        self._write_log(log_msg)
    
    def log_trade(self, action: str, ticker: str, quantity: float, conviction: Optional[int] = None, price: Optional[float] = None, reasoning: Optional[str] = None):
        """
        Log a trade execution.
        
        Args:
            action: Trade action (BUY/SELL)
            ticker: Stock ticker
            quantity: Number of shares
            conviction: Conviction score (1-10)
            price: Execution price (optional)
            reasoning: Trade reasoning (optional)
        """
        timestamp = self._get_timestamp()
        price_str = f" @ ${price:.2f}" if price else ""
        conviction_str = f" (conviction: {conviction}/10)" if conviction else ""
        
        # Trade emoji
        trade_emoji = "🟢 BUY" if action.upper() == "BUY" else "🔴 SELL"
        
        log_msg = f"{timestamp} [TRADE] {trade_emoji} {quantity} shares of {ticker}{price_str}{conviction_str}"
        self._write_log(log_msg)
        
        if reasoning:
            self._write_log(f"{timestamp}         Reason: {reasoning}")
    
    def log_action(self, action: str, details: Optional[str] = None):
        """
        Log an action in process.
        
        Args:
            action: Action being taken
            details: Additional details (optional)
        """
        timestamp = self._get_timestamp()
        log_msg = f"{timestamp} [ACTION] {action}"
        self._write_log(log_msg)
        if details:
            self._write_log(f"{timestamp}         {details}")
    
    def log_analysis(self, message: str):
        """
        Log an analysis completion.
        
        Args:
            message: Analysis message to log
        """
        timestamp = self._get_timestamp()
        log_msg = f"{timestamp} [ANALYSIS] {message}"
        self._write_log(log_msg)
        print(log_msg)

