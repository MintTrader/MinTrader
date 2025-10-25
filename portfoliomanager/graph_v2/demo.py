"""
Demo script for LangGraph-based Portfolio Manager

This script demonstrates the new architecture with:
1. MCP adapter for Alpaca tools
2. LangGraph state management
3. Graph-based workflow
4. Optional human-in-the-loop
5. Streaming capabilities
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure we have required env vars
# When using Ollama, OPENAI_API_KEY is not required (ChromaDB handles embeddings locally)
required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
if os.getenv("LLM_PROVIDER") not in ["ollama", None]:
    required_vars.append("OPENAI_API_KEY")

missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    print(f"❌ Missing environment variables: {', '.join(missing)}")
    print("Please set them in .env file or environment")
    exit(1)

from portfoliomanager.config import PORTFOLIO_CONFIG
from .portfolio_graph import (
    create_portfolio_graph,
    run_portfolio_iteration,
    stream_portfolio_iteration
)


def demo_basic_workflow():
    """
    Demo 1: Fully autonomous workflow.
    
    This runs the full portfolio management iteration autonomously:
    - Assess portfolio
    - Research market and select stocks (unified)
    - Analyze stocks
    - Make trading decisions autonomously
    - Execute trades automatically
    """
    print("\n" + "="*70)
    print("DEMO 1: Autonomous Workflow (Fully Automated)")
    print("="*70)
    
    result = run_portfolio_iteration(PORTFOLIO_CONFIG)
    
    print("\n📊 RESULTS:")
    print(f"  Iteration ID: {result['iteration_id']}")
    print(f"  Phase: {result['phase']}")
    print(f"  Stocks Analyzed: {len(result['analysis_results'])}")
    print(f"  Trades Executed: {len(result['executed_trades'])}")
    
    if result.get('error'):
        print(f"  ⚠️  Error: {result['error']}")
    
    # Show executed trades
    if result['executed_trades']:
        print("\n  Executed Trades:")
        for trade in result['executed_trades']:
            status = trade.get('status', 'unknown')
            symbol = trade.get('ticker')
            action = trade.get('action')
            
            if status == 'submitted':
                print(f"    ✅ {action} {symbol} - Order ID: {trade.get('order_id')}")
            else:
                print(f"    ❌ {action} {symbol} - Error: {trade.get('error', 'Unknown')}")


def demo_error_recovery():
    """
    Demo 2: Error recovery with checkpointing.
    
    This demonstrates automatic state recovery after errors.
    """
    print("\n" + "="*70)
    print("DEMO 2: Error Recovery and Checkpointing")
    print("="*70)
    
    from datetime import datetime
    
    graph = create_portfolio_graph(PORTFOLIO_CONFIG, enable_checkpointing=True)
    
    iteration_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_state = {
        "iteration_id": iteration_id,
        "config": PORTFOLIO_CONFIG,
        "phase": "init",
        "messages": [],
        "stocks_to_analyze": [],
        "analysis_results": {},
        "recently_analyzed": {},
        "pending_trades": [],
        "executed_trades": [],
        "web_search_used": False,
        "market_context": "",
        "last_summary": "",
        "error": None
    }
    
    print("\n✅ Graph has automatic checkpointing enabled")
    print("   State is saved after each node execution")
    print("   Can recover from errors and resume execution")
    
    # Include thread_id for checkpointing support
    run_config = {"configurable": {"thread_id": iteration_id}}
    result = graph.invoke(initial_state, config=run_config)
    
    print("\n📊 RESULTS:")
    print(f"  Iteration ID: {result['iteration_id']}")
    print(f"  Final Phase: {result['phase']}")
    print(f"  Trades Executed: {len(result['executed_trades'])}")


async def demo_streaming():
    """
    Demo 3: Streaming workflow with real-time updates.
    
    This shows progress as each node executes.
    """
    print("\n" + "="*70)
    print("DEMO 3: Streaming Workflow")
    print("="*70)
    
    await stream_portfolio_iteration(PORTFOLIO_CONFIG)


def demo_graph_visualization():
    """
    Demo 4: Generate graph visualization.
    
    This creates a visualization of the workflow graph.
    """
    print("\n" + "="*70)
    print("DEMO 4: Graph Visualization")
    print("="*70)
    
    graph = create_portfolio_graph(PORTFOLIO_CONFIG, enable_checkpointing=False)
    
    # Generate Mermaid diagram
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
        print("✅ Graph visualization displayed")
    except Exception as e:
        print(f"⚠️  Could not display graph: {e}")
        print("You can visualize the graph in LangGraph Studio instead")
    
    # Print graph structure
    print("\n📊 Graph Structure:")
    print(f"  Nodes: {list(graph.get_graph().nodes.keys())}")
    print(f"  Edges: {len(graph.get_graph().edges)}")


def demo_state_inspection():
    """
    Demo 5: Inspect state at each step.
    
    This shows how to examine the state after each node execution.
    """
    print("\n" + "="*70)
    print("DEMO 5: State Inspection")
    print("="*70)
    
    from datetime import datetime
    
    graph = create_portfolio_graph(PORTFOLIO_CONFIG, enable_checkpointing=True)
    
    # Initial state
    iteration_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_state = {
        "iteration_id": iteration_id,
        "config": PORTFOLIO_CONFIG,
        "phase": "init",
        "messages": [],
        "stocks_to_analyze": [],
        "analysis_results": {},
        "recently_analyzed": {},
        "pending_trades": [],
        "executed_trades": [],
        "web_search_used": False,
        "market_context": "",
        "last_summary": "",
        "error": None
    }
    
    # Stream and inspect
    print("\n📊 Inspecting state at each node:\n")
    
    # Include thread_id for checkpointing support
    run_config = {"configurable": {"thread_id": iteration_id}}
    for event in graph.stream(initial_state, config=run_config):
        node_name = list(event.keys())[0]
        state = event[node_name]
        
        print(f"After {node_name}:")
        print(f"  Phase: {state.get('phase')}")
        print(f"  Stocks to analyze: {state.get('stocks_to_analyze', [])}")
        print(f"  Analysis results: {len(state.get('analysis_results', {}))}")
        print(f"  Pending trades: {len(state.get('pending_trades', []))}")
        print(f"  Executed trades: {len(state.get('executed_trades', []))}")
        print()


def demo_mcp_tools():
    """
    Demo 6: Show available MCP tools.
    
    This lists all tools available from the Alpaca MCP server.
    """
    print("\n" + "="*70)
    print("DEMO 6: Alpaca MCP Tools")
    print("="*70)
    
    from .mcp_adapter import get_alpaca_mcp_tools
    
    tools = get_alpaca_mcp_tools()
    
    print(f"\n📦 Available tools: {len(tools)}\n")
    
    # Group tools by category
    categories = {
        "Account": [],
        "Orders": [],
        "Positions": [],
        "Market Data": [],
        "Assets": [],
        "Other": []
    }
    
    for tool in tools:
        name = tool.name
        if "account" in name:
            categories["Account"].append(name)
        elif "order" in name:
            categories["Orders"].append(name)
        elif "position" in name:
            categories["Positions"].append(name)
        elif any(x in name for x in ["bar", "quote", "trade", "snapshot", "clock"]):
            categories["Market Data"].append(name)
        elif "asset" in name:
            categories["Assets"].append(name)
        else:
            categories["Other"].append(name)
    
    for category, tool_names in categories.items():
        if tool_names:
            print(f"{category}:")
            for name in sorted(tool_names):
                print(f"  - {name}")
            print()


def main():
    """
    Main demo runner.
    """
    print("\n" + "="*70)
    print("LangGraph Portfolio Manager - Demonstration")
    print("="*70)
    
    print("\nAvailable demos:")
    print("1. Autonomous workflow (fully automated)")
    print("2. Error recovery with checkpointing")
    print("3. Streaming workflow")
    print("4. Graph visualization")
    print("5. State inspection")
    print("6. MCP tools listing")
    print("7. Run all demos")
    print("0. Exit")
    
    choice = input("\nSelect demo (0-7): ").strip()
    
    if choice == "1":
        demo_basic_workflow()
    elif choice == "2":
        demo_error_recovery()
    elif choice == "3":
        asyncio.run(demo_streaming())
    elif choice == "4":
        demo_graph_visualization()
    elif choice == "5":
        demo_state_inspection()
    elif choice == "6":
        demo_mcp_tools()
    elif choice == "7":
        demo_mcp_tools()
        demo_graph_visualization()
        demo_state_inspection()
        demo_basic_workflow()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

