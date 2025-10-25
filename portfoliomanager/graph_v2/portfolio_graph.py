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
    select_stocks_node,
    analyze_stocks_node,
    make_decisions_node,
    execute_trades_node,
    upload_results_to_s3_node,
    should_analyze_stocks,
    should_execute_trades
)


def create_portfolio_graph(config: dict, enable_checkpointing: bool = True):
    """
    Create and compile the portfolio management graph.
    
    Workflow:
    1. START → assess_portfolio: Get current portfolio state + fetch recently analyzed from S3
    2. assess_portfolio → select_stocks: Research market & select stocks (unified)
    3. select_stocks → [conditional]:
       - If stocks selected → analyze_stocks
       - If no stocks → make_decisions
    4. analyze_stocks → make_decisions: LLM decides on trades
    5. make_decisions → [conditional]:
       - If trades pending → execute_trades
       - If no trades → upload_to_s3
    6. execute_trades → upload_to_s3: Upload results, summary, and logs to S3
    7. upload_to_s3 → END
    
    Args:
        config: Portfolio configuration dict
        enable_checkpointing: Whether to enable state persistence
        
    Returns:
        Compiled graph ready for execution
        
    Example:
        >>> graph = create_portfolio_graph(PORTFOLIO_CONFIG)
        >>> initial_state = {
        ...     "iteration_id": "20251022_120000",
        ...     "config": PORTFOLIO_CONFIG,
        ...     "phase": "init",
        ...     "messages": []
        ... }
        >>> result = graph.invoke(initial_state)
    """
    
    # Create graph with state schema
    workflow = StateGraph(PortfolioState)
    
    # Add all nodes
    workflow.add_node("assess_portfolio", assess_portfolio_node)
    workflow.add_node("select_stocks", select_stocks_node)
    workflow.add_node("analyze_stocks", analyze_stocks_node)
    workflow.add_node("make_decisions", make_decisions_node)
    workflow.add_node("execute_trades", execute_trades_node)
    workflow.add_node("upload_to_s3", upload_results_to_s3_node)
    
    # Define edges
    # START → assess portfolio
    workflow.add_edge(START, "assess_portfolio")
    
    # assess → select (select now includes market research)
    workflow.add_edge("assess_portfolio", "select_stocks")
    
    # select → [conditional: analyze or decide]
    workflow.add_conditional_edges(
        "select_stocks",
        should_analyze_stocks,
        {
            "analyze": "analyze_stocks",
            "decide": "make_decisions"
        }
    )
    
    # analyze → decide
    workflow.add_edge("analyze_stocks", "make_decisions")
    
    # decide → [conditional: execute or upload]
    workflow.add_conditional_edges(
        "make_decisions",
        should_execute_trades,
        {
            "execute": "execute_trades",
            "complete": "upload_to_s3"
        }
    )
    
    # execute → upload
    workflow.add_edge("execute_trades", "upload_to_s3")
    
    # upload → end
    workflow.add_edge("upload_to_s3", END)
    
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
        "stocks_to_analyze": [],
        "analysis_results": {},
        "recently_analyzed": {},
        "pending_trades": [],
        "executed_trades": [],
        "web_search_used": False,
        "market_context": "",
        "promising_sectors": [],
        "growth_stocks": [],
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
        "stocks_to_analyze": [],
        "analysis_results": {},
        "recently_analyzed": {},
        "pending_trades": [],
        "executed_trades": [],
        "web_search_used": False,
        "market_context": "",
        "promising_sectors": [],
        "growth_stocks": [],
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
            account = node_output.get("account", {})
            print(f"  💰 Cash: ${account.get('cash', 0):,.2f}")
            print(f"  📈 Portfolio: ${account.get('portfolio_value', 0):,.2f}")
        
        elif node_name == "select_stocks":
            stocks = node_output.get("stocks_to_analyze", [])
            sectors = node_output.get("promising_sectors", [])
            growth = node_output.get("growth_stocks", [])
            
            if sectors:
                print(f"  📈 Identified {len(sectors)} promising sectors")
            if growth:
                print(f"  🎯 Identified {len(growth)} growth opportunities")
            
            print(f"  ✅ Selected: {stocks if stocks else 'None'}")
        
        elif node_name == "analyze_stocks":
            results = node_output.get("analysis_results", {})
            for ticker, result in results.items():
                print(f"  📊 {ticker}: {result.get('recommendation', 'N/A')}")
        
        elif node_name == "make_decisions":
            trades = node_output.get("pending_trades", [])
            print(f"  💭 Generated {len(trades)} trade decisions")
        
        elif node_name == "execute_trades":
            executed = node_output.get("executed_trades", [])
            print(f"  ⚡ Executed {len(executed)} trades")
    
    print("\n" + "="*60)
    print("✅ Iteration Complete!")
    print("="*60)

