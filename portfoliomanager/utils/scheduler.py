"""
Trading Scheduler

Schedules portfolio management iterations at specified times during trading days.
"""

import time
from datetime import datetime, time as dt_time, date
from typing import List, Optional
try:
    import pytz  # type: ignore
except ImportError:
    pytz = None  # type: ignore


class TradingScheduler:
    """Manages scheduled execution of portfolio management iterations"""
    
    def __init__(self, schedule_times: List[str], timezone: str = 'America/New_York'):
        """
        Initialize trading scheduler.
        
        Args:
            schedule_times: List of times in HH:MM format (e.g., ['09:35', '12:00'])
            timezone: Timezone for scheduling (default: US Eastern for NYSE)
        """
        self.schedule_times = self._parse_times(schedule_times)
        if pytz:
            self.timezone = pytz.timezone(timezone)
        else:
            self.timezone = None  # Will use local time
        self.last_run_date: Optional[date] = None
    
    def _parse_times(self, time_strings: List[str]) -> List[dt_time]:
        """
        Parse time strings to time objects.
        
        Args:
            time_strings: List of time strings in HH:MM format
            
        Returns:
            List of time objects
        """
        times = []
        for time_str in time_strings:
            hour, minute = map(int, time_str.split(':'))
            times.append(dt_time(hour=hour, minute=minute))
        return sorted(times)
    
    def _is_trading_day(self) -> bool:
        """
        Check if today is a trading day (Monday-Friday).
        
        Returns:
            True if trading day, False otherwise
        """
        if self.timezone:
            now = datetime.now(self.timezone)
        else:
            now = datetime.now()
        # Monday = 0, Sunday = 6
        return now.weekday() < 5  # Monday through Friday
    
    def _should_run_now(self) -> bool:
        """
        Check if we should run an iteration now.
        
        Returns:
            True if we should run, False otherwise
        """
        if not self._is_trading_day():
            return False
        
        if self.timezone:
            now = datetime.now(self.timezone)
        else:
            now = datetime.now()
        current_date = now.date()
        current_time = now.time()
        
        # Check if we've already run today
        if self.last_run_date == current_date:
            return False
        
        # Check if current time matches any scheduled time (within 1 minute)
        for scheduled_time in self.schedule_times:
            # Create datetime objects for comparison
            scheduled_dt = datetime.combine(current_date, scheduled_time)
            current_dt = datetime.combine(current_date, current_time)
            
            # Check if we're within 1 minute of scheduled time
            diff = abs((current_dt - scheduled_dt).total_seconds())
            if diff < 60:  # Within 1 minute
                return True
        
        return False
    
    def run_scheduled(self, portfolio_manager, check_interval: int = 30):
        """
        Run portfolio manager at scheduled times.
        
        Args:
            portfolio_manager: PortfolioManager instance
            check_interval: Seconds between schedule checks (default: 30)
        """
        print(f"Trading Scheduler started. Schedule times: {[t.strftime('%H:%M') for t in self.schedule_times]}")
        print(f"Timezone: {self.timezone}")
        print(f"Checking every {check_interval} seconds...")
        
        try:
            while True:
                if self.timezone:
                    now = datetime.now(self.timezone)
                else:
                    now = datetime.now()
                
                if self._should_run_now():
                    print(f"\n{'='*60}")
                    print(f"Scheduled run triggered at {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}\n")
                    
                    try:
                        # Run portfolio management iteration
                        portfolio_manager.run_iteration()
                        
                        # Mark that we've run today
                        self.last_run_date = now.date()
                        
                        print(f"\n{'='*60}")
                        if self.timezone:
                            now_str = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"Iteration completed at {now_str}")
                        print(f"{'='*60}\n")
                        
                    except Exception as e:
                        print(f"Error during scheduled run: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
    
    def run_once(self, portfolio_manager):
        """
        Run portfolio manager once (for manual/testing execution).
        
        Args:
            portfolio_manager: PortfolioManager instance
        """
        if self.timezone:
            now = datetime.now(self.timezone)
        else:
            now = datetime.now()
        print(f"\nManual run triggered at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            portfolio_manager.run_iteration()
            if self.timezone:
                now_str = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')
            else:
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\nIteration completed at {now_str}")
        except Exception as e:
            print(f"Error during manual run: {e}")
            import traceback
            traceback.print_exc()

