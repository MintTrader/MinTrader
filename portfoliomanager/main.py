"""
Portfolio Manager Main Entry Point

LangGraph-based autonomous portfolio management system.
Uses graph architecture with MCP tools for trading operations.
"""

import sys
import os
import argparse
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .config import PORTFOLIO_CONFIG
from .graph_v2 import run_portfolio_iteration
from .graph_v2.portfolio_graph import stream_portfolio_iteration


def main():
    """Main entry point for portfolio manager"""
    parser = argparse.ArgumentParser(
        description="LangGraph-based Autonomous Portfolio Management System"
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
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming mode (shows real-time progress)'
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
        run_portfolio_manager(config, args.mode, args.stream)
    
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_portfolio_manager(config, mode, stream=False):
    """Run LangGraph portfolio manager"""
    
    print("\n" + "="*60)
    print("🤖 LANGGRAPH PORTFOLIO MANAGER")
    print("="*60)
    print(f"\n✨ Architecture: Graph-based with MCP tools")
    print(f"🤖 Decision Making: Fully autonomous")
    print(f"📊 Max Analyses per Iteration: {config.get('max_stocks_to_analyze', 3)}")
    print("="*60 + "\n")
    
    if mode == 'scheduled':
        print("⚠️  Scheduled mode not yet implemented")
        print("Running single iteration instead...\n")
    
    if stream:
        # Streaming mode
        print("🌊 Streaming mode enabled - showing real-time progress\n")
        asyncio.run(stream_portfolio_iteration(config))
    else:
        # Standard mode
        print("Running single iteration...\n")
        result = run_portfolio_iteration(config)
        
        # Show results
        print("\n" + "="*60)
        print("📊 ITERATION RESULTS")
        print("="*60)
        print(f"✅ Iteration ID: {result['iteration_id']}")
        print(f"✅ Phase: {result['phase']}")
        print(f"✅ Stocks Analyzed: {len(result.get('analysis_results', {}))}")
        print(f"✅ Trades Executed: {len(result.get('executed_trades', []))}")
        
        if result.get('executed_trades'):
            print("\n📋 Executed Trades:")
            for trade in result['executed_trades']:
                status = trade.get('status', 'unknown')
                ticker = trade.get('ticker')
                action = trade.get('action')
                
                if status == 'submitted':
                    print(f"  ✅ {action} {ticker} - Order ID: {trade.get('order_id')}")
                else:
                    print(f"  ❌ {action} {ticker} - Error: {trade.get('error', 'Unknown')}")
        
        if result.get('error'):
            print(f"\n⚠️  Error: {result['error']}")
        
        print("\n" + "="*60)
        print("✅ Iteration complete!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()

