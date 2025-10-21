"""
Report Manager Module

Handles summary generation, S3 uploads, and iteration reporting.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, cast, Optional
from langchain_openai import ChatOpenAI
import re


class ReportManager:
    """Handles report generation and S3 uploads"""
    
    def __init__(self, s3_client, logger, results_dir: Path, pm_results_dir: Path):
        """
        Initialize report manager.
        
        Args:
            s3_client: S3ReportManager instance
            logger: PortfolioLogger instance
            results_dir: Main results directory
            pm_results_dir: Portfolio manager results directory
        """
        self.s3_client = s3_client
        self.logger = logger
        self.results_dir = results_dir
        self.pm_results_dir = pm_results_dir
    
    def extract_tracking_metadata(self, last_summary: str) -> Dict[str, Any]:
        """
        Extract tracking metadata from previous summary.
        
        Args:
            last_summary: Previous iteration summary
            
        Returns:
            Dictionary with tracking data (iterations, days_operating, first_start_date)
        """
        metadata: Dict[str, Any] = {
            'total_iterations': 0,
            'days_operating': 0,
            'first_start_date': cast(Optional[str], None)
        }
        
        if not last_summary or "No previous iteration" in last_summary:
            return metadata
        
        try:
            # Extract total iterations
            iterations_match = re.search(r'Total Iterations:\s*(\d+)', last_summary)
            if iterations_match:
                metadata['total_iterations'] = int(iterations_match.group(1))
            
            # Extract days operating
            days_match = re.search(r'Days Operating:\s*(\d+)', last_summary)
            if days_match:
                metadata['days_operating'] = int(days_match.group(1))
            
            # Extract first start date
            date_match = re.search(r'First Start Date:\s*(\d{4}-\d{2}-\d{2})', last_summary)
            if date_match:
                metadata['first_start_date'] = date_match.group(1)
                
        except Exception as e:
            self.logger.log_system(f"Warning: Could not parse tracking metadata: {e}")
        
        return metadata
    
    
    def upload_reports(self, iteration_id: str, analyzed_stocks: Dict[str, Dict]):
        """
        Upload all reports and logs to S3.
        
        Args:
            iteration_id: Current iteration ID
            analyzed_stocks: Dictionary of analyzed stocks
        """
        try:
            # Upload portfolio manager log
            log_file = self.pm_results_dir / 'message_tool.log'
            self.s3_client.upload_log(iteration_id, log_file)
            
            # Upload individual stock reports
            for ticker, info in analyzed_stocks.items():
                date = info['date']
                reports_dir = self.results_dir / ticker / date / 'reports'
                if reports_dir.exists():
                    self.s3_client.upload_reports(ticker, date, reports_dir)
            
        except Exception as e:
            self.logger.log_system(f"Error uploading to S3: {e}")
    
    def generate_iteration_summary(
        self, iteration_id: str, llm_model_name: str,
        analyzed_stocks: Dict[str, Dict], trades_executed: list,
        account, positions, open_orders, last_summary: str = ""
    ) -> str:
        """
        Generate agent-written iteration summary for next wake-up.
        
        Args:
            iteration_id: Current iteration ID
            llm_model_name: Name of the LLM model to use
            analyzed_stocks: Dictionary of analyzed stocks
            trades_executed: List of trades executed
            account: Account information
            positions: Current positions
            open_orders: Current open orders
            
        Returns:
            Agent-generated summary text
        """
        # Build context about what happened
        iteration_context: Dict[str, Any] = {
            "iteration_id": iteration_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "analyzed_stocks": analyzed_stocks,
            "trades_executed": trades_executed,
            "account": {
                "portfolio_value": account.get('portfolio_value', 0),
                "cash": account.get('cash', 0),
                "buying_power": account.get('buying_power', 0)
            },
            "positions": [
                {
                    "symbol": p['symbol'],
                    "qty": p['qty'],
                    "market_value": p.get('market_value', 0),
                    "pnl_pct": (p.get('unrealized_plpc', 0) * 100)
                }
                for p in positions[:20]
            ],
            "open_orders": [
                {
                    "symbol": o.get('symbol'),
                    "side": o.get('side'),
                    "qty": o.get('qty'),
                    "status": o.get('status'),
                    "limit_price": o.get('limit_price')
                }
                for o in (open_orders[:10] if open_orders else [])
            ]
        }
        
        # Build formatted strings
        analyzed_str = self._format_analyzed_stocks(analyzed_stocks)
        trades_str = self._format_trades(trades_executed)
        
        # Extract positions and orders from context with proper typing
        positions_from_context = cast(List[Any], iteration_context['positions'])
        orders_from_context = cast(List[Any], iteration_context['open_orders'])
        
        positions_str = self._format_positions(positions_from_context)
        orders_str = self._format_orders(orders_from_context)
        
        # Extract account info
        account_data = cast(Dict[str, Any], iteration_context['account'])
        portfolio_value = account_data['portfolio_value']
        cash = account_data['cash']
        num_positions = len(positions_from_context)
        timestamp = str(iteration_context['timestamp'])
        
        # Extract and update tracking metadata
        prev_metadata = self.extract_tracking_metadata(last_summary)
        total_iterations = prev_metadata['total_iterations'] + 1
        
        # Calculate days operating
        if prev_metadata['first_start_date']:
            first_start = datetime.strptime(prev_metadata['first_start_date'], '%Y-%m-%d')
            current_date = datetime.strptime(timestamp.split()[0], '%Y-%m-%d')
            days_operating = (current_date - first_start).days
        else:
            # This is the first iteration
            first_start_date = timestamp.split()[0]
            days_operating = 0
            prev_metadata['first_start_date'] = first_start_date
        
        # Build prompt for agent
        prompt = f"""You are a portfolio manager who just completed an iteration. 
Write a natural, conversational summary for yourself to read when you wake up for the next iteration.

This is YOUR memory - write it like you're leaving notes for your future self. Be specific about:
- What you researched and why
- What you learned about the market or specific stocks
- What trades you made and your reasoning
- What you're concerned about or watching
- What you're planning to check or do next time
- Any thoughts, observations, or reminders to yourself

Don't use emojis, headers, or numbered lists. Just write naturally like you're journaling.

📅 YOUR SCHEDULE (for context):
   - You run 3 times per day, Monday-Friday (US market hours)
   - Morning: Market open (9:30 AM ET)
   - Midday: Market midpoint (12:30 PM ET)  
   - Evening: 30 min before close (3:30 PM ET)
   - Weekends: No runs (market closed)
   
⏱️  YOUR OPERATION HISTORY:
   - Total Iterations: {total_iterations}
   - Days Operating: {days_operating}
   - First Start Date: {prev_metadata['first_start_date']}
   
   This helps you understand your experience level and history.

CURRENT ITERATION DETAILS:

ITERATION: {iteration_id}
TIME: {timestamp}

STOCKS I ANALYZED:
{analyzed_str}

TRADES I EXECUTED:
{trades_str}

CURRENT PORTFOLIO STATE:
- Total Value: ${portfolio_value:,.2f}
- Cash: ${cash:,.2f}
- Number of Positions: {num_positions}

TOP POSITIONS (by value):
{positions_str}

PENDING ORDERS:
{orders_str}

Now write your summary for your next wake-up. Be specific, honest, and thoughtful. Remember, you'll wake up at one of your 3 daily scheduled times (morning, midday, or evening)."""

        try:
            llm = ChatOpenAI(model=llm_model_name, temperature=0.7)
            response = llm.invoke(prompt)
            agent_summary = str(response.content) if response.content else ""
            
            # Add metadata header with tracking information
            header = f"ITERATION {iteration_id} - {timestamp}\n"
            header += "=" * 80 + "\n"
            header += f"Total Iterations: {total_iterations}\n"
            header += f"Days Operating: {days_operating}\n"
            header += f"First Start Date: {prev_metadata['first_start_date']}\n"
            header += "=" * 80 + "\n\n"
            
            return header + agent_summary
            
        except Exception as e:
            self.logger.log_system(f"Error generating agent summary: {e}")
            return self._generate_basic_summary(iteration_context)
    
    def _format_analyzed_stocks(self, analyzed_stocks: Dict[str, Dict]) -> str:
        """Format analyzed stocks for summary."""
        if not analyzed_stocks or not isinstance(analyzed_stocks, dict):
            return "None this iteration"
        
        analyzed_list = []
        for ticker, info in analyzed_stocks.items():
            if isinstance(info, dict):
                decision = info.get('decision', 'UNKNOWN')
                reasoning = info.get('reasoning', 'N/A')
                analyzed_list.append(f"- {ticker}: {decision} ({reasoning})")
        
        return "\n".join(analyzed_list) if analyzed_list else "None this iteration"
    
    def _format_trades(self, trades_executed: list) -> str:
        """Format trades for summary."""
        if not trades_executed or not isinstance(trades_executed, list):
            return "No trades this iteration"
        
        trades_list = []
        for t in trades_executed:
            if isinstance(t, dict):
                action = t.get('action', 'UNKNOWN')
                ticker = t.get('ticker', 'N/A')
                quantity = t.get('quantity', 0)
                price = t.get('price', 0)
                reasoning = t.get('reasoning', 'N/A')
                trades_list.append(f"- {action} {ticker}: {quantity} shares @ ${price:.2f} - {reasoning}")
        
        return "\n".join(trades_list) if trades_list else "No trades this iteration"
    
    def _format_positions(self, positions: list) -> str:
        """Format positions for summary."""
        if not positions or not isinstance(positions, list):
            return "No positions"
        
        sorted_positions = sorted(
            positions, 
            key=lambda x: x.get('market_value', 0) if isinstance(x, dict) else 0, 
            reverse=True
        )[:10]
        
        positions_list = []
        for p in sorted_positions:
            if isinstance(p, dict):
                symbol = p.get('symbol', 'N/A')
                qty = p.get('qty', 0)
                market_value = p.get('market_value', 0)
                pnl_pct = p.get('pnl_pct', 0)
                positions_list.append(f"- {symbol}: {qty} shares, ${market_value:,.2f}, P&L {pnl_pct:+.1f}%")
        
        return "\n".join(positions_list) if positions_list else "No positions"
    
    def _format_orders(self, orders: list) -> str:
        """Format orders for summary."""
        if not orders or not isinstance(orders, list):
            return "No pending orders"
        
        orders_list = []
        for o in orders:
            if isinstance(o, dict):
                side = str(o.get('side', 'unknown')).upper()
                symbol = o.get('symbol', 'N/A')
                qty = o.get('qty', 0)
                limit_price = o.get('limit_price', 'market')
                status = o.get('status', 'unknown')
                orders_list.append(f"- {side} {symbol}: {qty} shares @ ${limit_price}, status: {status}")
        
        return "\n".join(orders_list) if orders_list else "No pending orders"
    
    def _generate_basic_summary(self, context: Dict[str, Any]) -> str:
        """Fallback basic summary if agent generation fails."""
        lines = [
            f"ITERATION {context.get('iteration_id', 'unknown')} - {context.get('timestamp', 'unknown')}",
            "=" * 80,
            "",
        ]
        
        # Analyzed stocks
        analyzed_stocks = context.get('analyzed_stocks', {})
        if analyzed_stocks and isinstance(analyzed_stocks, dict):
            lines.append("Analyzed this iteration:")
            for ticker, info in analyzed_stocks.items():
                if isinstance(info, dict):
                    lines.append(f"  {ticker}: {info.get('decision', 'UNKNOWN')}")
            lines.append("")
        
        # Trades
        trades_executed = context.get('trades_executed', [])
        if trades_executed and isinstance(trades_executed, list):
            lines.append("Trades executed:")
            for trade in trades_executed:
                if isinstance(trade, dict):
                    action = trade.get('action', 'UNKNOWN')
                    ticker = trade.get('ticker', 'N/A')
                    quantity = trade.get('quantity', 0)
                    lines.append(f"  {action} {ticker}: {quantity} shares")
            lines.append("")
        
        # Account state
        account = context.get('account', {})
        if isinstance(account, dict):
            portfolio_value = account.get('portfolio_value', 0)
            cash = account.get('cash', 0)
            lines.append(f"Portfolio value: ${portfolio_value:,.2f}")
            lines.append(f"Cash available: ${cash:,.2f}")
        
        positions = context.get('positions', [])
        if isinstance(positions, list):
            lines.append(f"Positions: {len(positions)}")
        
        open_orders = context.get('open_orders', [])
        if open_orders and isinstance(open_orders, list):
            lines.append(f"Pending orders: {len(open_orders)}")
        
        return "\n".join(lines)

