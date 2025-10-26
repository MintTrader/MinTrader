"""
Shared LLM Factory for MinTrader

Unified LLM factory that supports multiple providers and automatically enables LangSmith tracing.

Providers Supported:
- OpenAI (cloud)
- Ollama (local - zero cost)
- Anthropic (cloud)
- Google (cloud)
- OpenRouter (cloud)

Environment Variables:
- LANGSMITH_TRACING: Enable LangSmith tracing (true/false)
- LANGSMITH_API_KEY: Your LangSmith API key
- LANGSMITH_WORKSPACE_ID: Optional workspace ID
- LANGSMITH_PROJECT: Optional project name
- LLM_MODEL: Default model name
- LLM_PROVIDER: Explicit provider (auto-detected if not set)
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
- PYTHONHTTPSVERIFY: Set to 0 to disable SSL verification (development only)

Usage:
    >>> from shared.llm_factory import get_llm
    >>> llm = get_llm("gpt-4o-mini")  # OpenAI
    >>> llm = get_llm("llama3")        # Ollama (auto-detected)
"""

import os
from typing import Optional, Any
from langchain_core.language_models import BaseChatModel

# Configure LangSmith SSL settings on module import
from shared.langsmith_config import configure_langsmith_ssl
configure_langsmith_ssl()


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """
    Get an LLM instance based on configuration.
    
    Automatically detects provider from model name or uses explicit provider.
    LangSmith tracing is automatically enabled if LANGSMITH_TRACING=true.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o-mini", "llama3", "mistral")
        temperature: Temperature for generation (default: 0)
        provider: Explicit provider ("openai", "ollama", "anthropic", "google", "openrouter")
        base_url: Base URL for API (optional, for OpenAI-compatible endpoints)
        **kwargs: Additional arguments passed to the LLM constructor
    
    Returns:
        BaseChatModel: Configured LLM instance
    
    Examples:
        >>> # OpenAI (default)
        >>> llm = get_llm("gpt-4o-mini")
        
        >>> # Local Ollama
        >>> llm = get_llm("llama3", provider="ollama")
        
        >>> # Auto-detect from model name
        >>> llm = get_llm("llama3")  # Detects Ollama
        
        >>> # OpenRouter or custom endpoint
        >>> llm = get_llm("gpt-4", base_url="https://openrouter.ai/api/v1")
    """
    # Default to environment variable or gpt-4o-mini
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Auto-detect provider if not specified
    if provider is None:
        provider = _detect_provider(model_name)
    
    provider = provider.lower()
    
    # Create LLM based on provider
    if provider == "openai":
        return _create_openai_llm(model_name, temperature, base_url, **kwargs)
    elif provider == "ollama":
        return _create_ollama_llm(model_name, temperature, base_url, **kwargs)
    elif provider == "anthropic":
        return _create_anthropic_llm(model_name, temperature, base_url, **kwargs)
    elif provider == "google":
        return _create_google_llm(model_name, temperature, **kwargs)
    elif provider == "openrouter":
        # OpenRouter uses OpenAI-compatible API
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        return _create_openai_llm(model_name, temperature, base_url, **kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: openai, ollama, anthropic, google, openrouter"
        )


def _detect_provider(model_name: str) -> str:
    """
    Auto-detect LLM provider from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Provider name
    """
    model_lower = model_name.lower()
    
    # Check for explicit provider in environment
    env_provider = os.getenv("LLM_PROVIDER")
    if env_provider:
        return env_provider.lower()
    
    # OpenAI models
    if any(x in model_lower for x in ["gpt-", "o1-", "o3-"]):
        return "openai"
    
    # Anthropic models
    if "claude" in model_lower:
        return "anthropic"
    
    # Google models
    if any(x in model_lower for x in ["gemini", "palm"]):
        return "google"
    
    # Common local/Ollama models
    if any(x in model_lower for x in [
        "llama", "mistral", "mixtral", "phi", "gemma", 
        "qwen", "vicuna", "wizardlm", "orca", "deepseek",
        "gpt-oss"  # User's local model
    ]):
        return "ollama"
    
    # Default to OpenAI for unknown models
    return "openai"


def _create_openai_llm(
    model_name: str,
    temperature: float,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """Create OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY") and not kwargs.get("api_key"):
        raise ValueError(
            "OpenAI API key is required but not found. "
            "Please set the OPENAI_API_KEY environment variable or "
            "set LLM_PROVIDER=ollama to use Ollama instead (requires Ollama to be running locally)."
        )
    
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        **kwargs
    }
    
    # Use custom base_url if provided (for OpenRouter, custom endpoints, etc.)
    if base_url:
        llm_kwargs["base_url"] = base_url
    
    return ChatOpenAI(**llm_kwargs)


def _create_ollama_llm(
    model_name: str,
    temperature: float,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """
    Create Ollama LLM instance.
    
    Requires Ollama to be running locally: `ollama serve`
    
    Note: In LangChain v1, use langchain-ollama package (not langchain-community)
    """
    from langchain_ollama import ChatOllama
    
    # Get Ollama base URL from environment or parameter
    if not base_url:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        **kwargs
    )


def _create_anthropic_llm(
    model_name: str,
    temperature: float,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """Create Anthropic LLM instance."""
    from langchain_anthropic import ChatAnthropic
    
    llm_kwargs = {
        "model_name": model_name,
        "temperature": temperature,
        **kwargs
    }
    
    if base_url:
        llm_kwargs["base_url"] = base_url
    
    return ChatAnthropic(**llm_kwargs)


def _create_google_llm(
    model_name: str,
    temperature: float,
    **kwargs: Any
) -> BaseChatModel:
    """Create Google Generative AI LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        **kwargs
    )


# ==================== Config-Based Helpers ====================

def get_llm_from_config(
    config: dict,
    model_key: str = "deep_think_llm",
    temperature: float = 0
) -> BaseChatModel:
    """
    Create an LLM from a configuration dictionary.
    
    Primarily for TradingAgents compatibility.
    
    Args:
        config: Configuration dictionary with llm_provider, model names, and backend_url
        model_key: Key to look up in config for model name (e.g., "deep_think_llm", "quick_think_llm")
        temperature: Temperature for generation
        
    Returns:
        BaseChatModel: Configured LLM instance
        
    Examples:
        >>> config = {
        ...     "llm_provider": "openai",
        ...     "deep_think_llm": "gpt-4o-mini",
        ...     "backend_url": "https://api.openai.com/v1"
        ... }
        >>> llm = get_llm_from_config(config, "deep_think_llm")
    """
    # Get model name from config or environment
    model_name = config.get(model_key) or os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Get provider from config or environment
    provider = config.get("llm_provider") or os.getenv("LLM_PROVIDER")
    
    # Get base URL from config
    base_url = config.get("backend_url")
    
    return get_llm(
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        base_url=base_url
    )


def get_quick_llm(config: dict) -> BaseChatModel:
    """
    Get a fast LLM for quick decisions and selections.
    
    Primarily for PortfolioManager compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BaseChatModel: Configured LLM instance
    """
    # Try multiple config paths for compatibility
    model = (
        config.get("analysis_config", {}).get("quick_think_llm") or
        config.get("quick_think_llm") or
        os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    
    provider = (
        config.get("llm_provider") or
        config.get("analysis_config", {}).get("llm_provider") or
        os.getenv("LLM_PROVIDER")
    )
    base_url = (
        config.get("backend_url") or
        config.get("analysis_config", {}).get("backend_url") or
        None
    )
    
    return get_llm(model, temperature=0, provider=provider, base_url=base_url)


def get_deep_llm(config: dict) -> BaseChatModel:
    """
    Get a powerful LLM for deep analysis.
    
    Primarily for PortfolioManager compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BaseChatModel: Configured LLM instance
    """
    # Try multiple config paths for compatibility
    model = (
        config.get("analysis_config", {}).get("deep_think_llm") or
        config.get("deep_think_llm") or
        os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    
    provider = (
        config.get("llm_provider") or
        config.get("analysis_config", {}).get("llm_provider") or
        os.getenv("LLM_PROVIDER")
    )
    base_url = (
        config.get("backend_url") or
        config.get("analysis_config", {}).get("backend_url") or
        None
    )
    
    return get_llm(model, temperature=0, provider=provider, base_url=base_url)


def get_agent_llm(config: dict) -> BaseChatModel:
    """
    Get an LLM for agent operations (ReAct, tool calling).
    
    Primarily for PortfolioManager compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BaseChatModel: Configured LLM instance
    """
    model = (
        config.get("llm_model") or
        config.get("analysis_config", {}).get("deep_think_llm") or
        os.getenv("LLM_MODEL", "gpt-4")
    )
    
    provider = (
        config.get("llm_provider") or
        config.get("analysis_config", {}).get("llm_provider") or
        os.getenv("LLM_PROVIDER")
    )
    base_url = (
        config.get("backend_url") or
        config.get("analysis_config", {}).get("backend_url") or
        None
    )
    
    return get_llm(model, temperature=0, provider=provider, base_url=base_url)

