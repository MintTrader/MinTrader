"""
Portfolio Manager Main Entry Point

LangGraph-based autonomous portfolio management system.
Uses graph architecture with MCP tools for trading operations.
"""

import sys
import os
import argparse
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variable to suppress MCP logs in subprocesses
os.environ.setdefault('MCP_LOG_LEVEL', 'ERROR')

# Setup logging with both console and file output
from pathlib import Path
from datetime import datetime

# Create logs directory
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Generate log filename with timestamp
log_filename = LOGS_DIR / f"portfolio_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Verify directory was created
if not LOGS_DIR.exists():
    print(f"WARNING: Failed to create logs directory at {LOGS_DIR.absolute()}")
else:
    print(f"Logs directory: {LOGS_DIR.absolute()}")

# Configure logging - suppress MCP server logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_filename)  # File output
    ]
)

# Suppress verbose MCP server logs (these are very noisy in production)
# The MCP library logs every tool request at INFO level
logging.getLogger('mcp').setLevel(logging.ERROR)
logging.getLogger('mcp.server').setLevel(logging.ERROR)
logging.getLogger('mcp.client').setLevel(logging.ERROR)
logging.getLogger('mcp.server.stdio').setLevel(logging.ERROR)
logging.getLogger('langchain_mcp_adapters').setLevel(logging.WARNING)

# Suppress other verbose loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Get logger for this module
logger = logging.getLogger(__name__)

from .config import PORTFOLIO_CONFIG
from .graph_v2 import run_portfolio_iteration
from .graph_v2.portfolio_graph import stream_portfolio_iteration

# Store log filename globally so it can be accessed by graph nodes
CURRENT_LOG_FILE = None


def main():
    """Main entry point for portfolio manager"""
    global CURRENT_LOG_FILE
    CURRENT_LOG_FILE = log_filename
    
    # Print startup message before any MCP initialization
    print("\n🚀 Starting Portfolio Manager...")
    print("=" * 60)
    
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
    
    logger.info("Portfolio Manager initialized")
    logger.info(f"Logging to: {log_filename}")
    logger.info(f"Log file exists: {Path(log_filename).exists()}")
    logger.info(f"CURRENT_LOG_FILE: {CURRENT_LOG_FILE}")
    
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
        logger.info("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


def run_portfolio_manager(config, mode, stream=False):
    """Run LangGraph portfolio manager"""
    
    logger.info("="*60)
    logger.info("🤖 LANGGRAPH PORTFOLIO MANAGER")
    logger.info("="*60)
    logger.info(f"✨ Architecture: Graph-based with MCP tools")
    logger.info(f"🤖 Decision Making: Fully autonomous")
    logger.info(f"📊 Max Analyses per Iteration: {config.get('max_stocks_to_analyze', 3)}")
    logger.info("="*60)
    
    if mode == 'scheduled':
        logger.warning("⚠️  Scheduled mode not yet implemented")
        logger.info("Running single iteration instead...")
    
    if stream:
        # Streaming mode
        logger.info("🌊 Streaming mode enabled - showing real-time progress")
        asyncio.run(stream_portfolio_iteration(config))
    else:
        # Standard mode
        logger.info("Running single iteration...")
        result = run_portfolio_iteration(config)
        
        # Show results
        logger.info("="*60)
        logger.info("📊 ITERATION RESULTS")
        logger.info("="*60)
        logger.info(f"✅ Iteration ID: {result['iteration_id']}")
        logger.info(f"✅ Phase: {result['phase']}")
        logger.info(f"✅ Stocks Analyzed: {len(result.get('analysis_results', {}))}")
        logger.info(f"✅ Trades Executed: {len(result.get('executed_trades', []))}")
        
        if result.get('executed_trades'):
            logger.info("📋 Executed Trades:")
            for trade in result['executed_trades']:
                status = trade.get('status', 'unknown')
                ticker = trade.get('ticker')
                action = trade.get('action')
                
                if status == 'submitted':
                    logger.info(f"  ✅ {action} {ticker} - Order ID: {trade.get('order_id')}")
                else:
                    logger.error(f"  ❌ {action} {ticker} - Error: {trade.get('error', 'Unknown')}")
        
        if result.get('error'):
            logger.error(f"⚠️  Error: {result['error']}")
        
        logger.info("="*60)
        logger.info("✅ Iteration complete!")
        logger.info("="*60)


if __name__ == "__main__":
    main()

