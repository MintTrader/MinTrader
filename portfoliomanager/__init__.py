"""
Portfolio Management Module

LLM-driven autonomous portfolio management system with web search.
Single smart orchestrator that handles stock discovery, analysis, and trading decisions.
"""

from .orchestrator_manager import OrchestratorPortfolioManager
from .config import PORTFOLIO_CONFIG

# Backward compatibility alias
PortfolioManager = OrchestratorPortfolioManager

__all__ = ['PortfolioManager', 'OrchestratorPortfolioManager', 'PORTFOLIO_CONFIG']

