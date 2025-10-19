"""
Orchestrator Portfolio Manager Main Entry Point

LLM-driven autonomous portfolio management with web search.
"""

import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .orchestrator_manager import OrchestratorPortfolioManager
from .config import PORTFOLIO_CONFIG
from .utils.scheduler import TradingScheduler


def main():
    """Main entry point for orchestrator portfolio manager"""
    parser = argparse.ArgumentParser(
        description="Orchestrator-based Autonomous Portfolio Management System"
    )
    parser.add_argument(
        '--mode',
        choices=['scheduled', 'once'],
        default='once',
        help='Run mode: scheduled (recurring) or once (single iteration)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = PORTFOLIO_CONFIG.copy()
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    try:
        # Initialize portfolio manager
        print("\n" + "="*60)
        print("ORCHESTRATOR PORTFOLIO MANAGEMENT SYSTEM")
        print("="*60)
        print(f"\nStrategy: LLM-driven with web search")
        print(f"Max Analyses per Iteration: {config.get('max_stocks_to_analyze', 3)}")
        print(f"Min Conviction: {config['min_conviction_score']}/10")
        print(f"Min Holding Days: {config['min_holding_days']}")
        print("="*60 + "\n")
        
        pm = OrchestratorPortfolioManager(config)
        
        if args.mode == 'scheduled':
            # Run on schedule
            print("Starting scheduler...")
            scheduler = TradingScheduler(config['schedule_times'])
            scheduler.run_scheduled(pm)
        else:
            # Run once
            print("Running single iteration...\n")
            scheduler = TradingScheduler(config['schedule_times'])
            scheduler.run_once(pm)
            print("\nIteration complete!")
    
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

