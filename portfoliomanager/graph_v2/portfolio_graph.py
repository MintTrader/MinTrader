"""
Portfolio Management Graph

Main LangGraph workflow for autonomous portfolio management.
Connects all nodes into a coherent workflow with:
- Conditional routing
- Automatic checkpointing
- Human-in-the-loop support
- Streaming capabilities
"""

from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import PortfolioState
from .nodes import (
    assess_portfolio_node,
    make_decisions_node,
    update_summary_node
)


def create_portfolio_graph(config: dict, enable_checkpointing: bool = True):
    """
    Create and compile the portfolio management graph.
    
    Smart 3-Step Workflow with Market Check:
    =========================================
    
    1. GET PORTFOLIO STATE (assess_portfolio)
       - Fetch current account info, positions, orders, market clock
       - Load last summary from S3 (agent's memory from previous run)
       - Provides context: portfolio state + memory
       - **KEY**: Checks if market is open first - returns early if closed
    
    2. DECIDE & EXECUTE BRACKET ORDERS (make_decisions) [SKIPPED IF MARKET CLOSED]
       - LLM reviews portfolio state and available cash
       - Makes autonomous trading decisions
       - Executes bracket orders (with stop-loss & take-profit)
       - Uses OpenAI function calling with safe Alpaca MCP tools
       - Only runs when market is open
    
    3. UPDATE SUMMARY (update_summary)
       - Generate summary of what happened this run
       - Track run count, timing, trades executed
       - Save to S3 as agent's memory for next run
       - Memory provides continuity across runs
       - Creates special "market closed" summary when appropriate
    
    Memory System:
    ==============
    The last_summary serves as the agent's memory:
    - Pulled at start of each run (step 1)
    - Updated at end of each run (step 3)
    - Tracks: run count, portfolio changes, market context, next steps
    - Enables context generation for stock analysis
    
    Conditional Routing:
    ====================
    - If market is OPEN: assess_portfolio -> make_decisions -> update_summary -> END
    - If market is CLOSED: assess_portfolio -> update_summary -> END
    
    Args:
        config: Portfolio configuration dict
        enable_checkpointing: Whether to enable state persistence
        
    Returns:
        Compiled graph ready for execution
        
    Example:
        >>> graph = create_portfolio_graph(PORTFOLIO_CONFIG)
        >>> initial_state = {
        ...     "iteration_id": "20251026_120000",
        ...     "config": PORTFOLIO_CONFIG,
        ...     "phase": "init",
        ...     "messages": []
        ... }
        >>> result = graph.invoke(initial_state)
    """
    
    # Create graph with state schema
    workflow = StateGraph(PortfolioState)
    
    # Add 3 nodes
    workflow.add_node("assess_portfolio", assess_portfolio_node)
    workflow.add_node("make_decisions", make_decisions_node)
    workflow.add_node("update_summary", update_summary_node)
    
    # Define conditional routing function
    def should_trade(state: PortfolioState) -> str:
        """
        Decide whether to proceed with trading based on market status.
        
        Returns:
            - "make_decisions" if market is open
            - "update_summary" if market is closed or error occurred
        """
        phase = state.get("phase", "")
        
        if phase == "market_closed":
            return "update_summary"  # Skip trading, go directly to summary
        elif phase == "error":
            return "update_summary"  # Skip trading on error, save error summary
        else:
            return "make_decisions"  # Market is open, proceed with trading
    
    # Define edges with conditional routing
    workflow.add_edge(START, "assess_portfolio")
    
    # Conditional: only trade if market is open
    workflow.add_conditional_edges(
        "assess_portfolio",
        should_trade,
        {
            "make_decisions": "make_decisions",
            "update_summary": "update_summary"
        }
    )
    
    workflow.add_edge("make_decisions", "update_summary")
    workflow.add_edge("update_summary", END)
    
    # Compile with optional checkpointing
    if enable_checkpointing:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


# HITL functionality removed - system is fully autonomous


def run_portfolio_iteration(config: dict):
    """
    Run a complete autonomous portfolio management iteration.
    
    This is a convenience function that:
    1. Creates the graph
    2. Sets up initial state
    3. Runs the workflow autonomously
    4. Returns results
    
    Args:
        config: Portfolio configuration
        
    Returns:
        Final state after iteration completes
        
    Example:
        >>> from portfoliomanager.config import PORTFOLIO_CONFIG
        >>> result = run_portfolio_iteration(PORTFOLIO_CONFIG)
        >>> print(f"Executed {len(result['executed_trades'])} trades")
    """
    
    # Create graph with checkpointing for error recovery
    graph = create_portfolio_graph(config, enable_checkpointing=True)
    
    # Create initial state
    iteration_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    initial_state = {
        "iteration_id": iteration_id,
        "config": config,
        "phase": "init",
        "messages": [],
        "last_summary": "",
        "error": None
    }
    
    # Run graph to completion (fully autonomous)
    # Include thread_id for checkpointing support
    run_config = {"configurable": {"thread_id": iteration_id}}
    return graph.invoke(initial_state, config=run_config)


async def stream_portfolio_iteration(config: dict):
    """
    Run portfolio iteration with streaming for real-time updates.
    
    This shows progress as each node executes, giving better visibility
    into what the system is doing.
    
    Args:
        config: Portfolio configuration
        
    Example:
        >>> import asyncio
        >>> from portfoliomanager.config import PORTFOLIO_CONFIG
        >>> asyncio.run(stream_portfolio_iteration(PORTFOLIO_CONFIG))
    """
    
    graph = create_portfolio_graph(config)
    
    # Create initial state
    iteration_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_state = {
        "iteration_id": iteration_id,
        "config": config,
        "phase": "init",
        "messages": [],
        "pending_trades": [],
        "executed_trades": [],
        "last_summary": "",
        "error": None
    }
    
    print("\n" + "="*60)
    print(f"🚀 Starting Portfolio Iteration: {iteration_id}")
    print("="*60)
    
    # Stream events with thread_id for checkpointing
    run_config = {"configurable": {"thread_id": iteration_id}}
    async for event in graph.astream(initial_state, config=run_config):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        
        phase = node_output.get("phase", "unknown")
        print(f"\n✓ Completed: {node_name} (phase: {phase})")
        
        # Show key information per node
        if node_name == "assess_portfolio":
            if phase == "market_closed":
                error_msg = node_output.get("error", "Market is closed")
                print(f"  🚫 {error_msg}")
                print(f"  ⏸️  Trading suspended")
            else:
                account = node_output.get("account", {})
                last_summary = node_output.get("last_summary", "")
                print(f"  💰 Cash: ${account.get('cash', 0):,.2f}")
                print(f"  📈 Portfolio: ${account.get('portfolio_value', 0):,.2f}")
                if last_summary:
                    print(f"  📜 Loaded memory from previous run")
        
        elif node_name == "make_decisions":
            if phase == "market_closed":
                print(f"  🚫 Skipped (market closed)")
            else:
                executed = node_output.get("executed_trades", [])
                print(f"  ⚡ Executed {len(executed)} trades")
        
        elif node_name == "update_summary":
            if phase == "complete":
                run_count = node_output.get("run_count", 1)
                if run_count > 0:
                    print(f"  📝 Updated agent memory (Run #{run_count})")
                else:
                    print(f"  📝 Market closed summary saved")
    
    print("\n" + "="*60)
    print("✅ Iteration Complete!")
    print("="*60)

