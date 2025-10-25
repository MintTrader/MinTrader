"""
LangSmith Configuration for MinTrader

Handles LangSmith setup including SSL verification settings.
"""

import os
import warnings


def configure_langsmith_ssl():
    """
    Configure SSL settings for LangSmith connections.
    
    Supports multiple configuration methods:
    1. Custom CA bundle (recommended for corporate proxies like Cato Networks)
    2. Disable SSL verification (development only)
    
    Environment Variables:
    - REQUESTS_CA_BUNDLE: Path to custom CA bundle (e.g., /path/to/cato-cert.pem)
    - SSL_CERT_FILE: Alternative path to CA bundle
    - PYTHONHTTPSVERIFY: Set to '0' to disable SSL verification (not recommended)
    
    For Cato Networks or corporate proxies:
    1. Export certificate: security find-certificate -a -p | grep -B 5 -A 5 "Cato" > cato-cert.pem
    2. Set REQUESTS_CA_BUNDLE=/path/to/cato-cert.pem
    
    WARNING: Only disable SSL verification in development/testing environments.
    """
    # Check for custom CA bundle first (proper way)
    ca_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    
    if ca_bundle and os.path.exists(ca_bundle):
        # Use custom CA bundle (for Cato Networks, corporate proxies, etc.)
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle
        os.environ["CURL_CA_BUNDLE"] = ca_bundle
        
        print(f"✅ Using custom CA bundle: {ca_bundle}")
        return True  # SSL verification enabled with custom CA
    
    # Check if SSL verification should be disabled (fallback for development)
    verify_ssl = os.getenv("PYTHONHTTPSVERIFY", "1")
    
    if verify_ssl == "0":
        # Disable SSL warnings
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except:
            pass
        
        # Disable SSL warnings for httpx (used by langsmith)
        try:
            import httpx
            # Monkey-patch httpx to disable SSL verification globally
            original_client_init = httpx.Client.__init__
            def patched_init(self, *args, **kwargs):
                kwargs['verify'] = False
                original_client_init(self, *args, **kwargs)
            httpx.Client.__init__ = patched_init
        except:
            pass
        
        # Set environment variables for various libraries
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["SSL_CERT_FILE"] = ""
        
        # Disable SSL in standard library
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
        
        warnings.warn(
            "⚠️  SSL verification is DISABLED. This should only be used in development/testing environments.",
            UserWarning
        )
        
        return False  # SSL verification disabled
    
    return True  # SSL verification enabled


def get_langsmith_client():
    """
    Get a configured LangSmith client with proper SSL settings.
    
    Returns:
        Client: Configured LangSmith client, or None if not configured
    """
    # Configure SSL first
    verify_ssl = configure_langsmith_ssl()
    
    # Check if LangSmith is enabled
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        return None
    
    try:
        from langsmith import Client
        
        # Create client with SSL settings
        client = Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com"),
        )
        
        # Monkey-patch the session to disable SSL verification if needed
        if not verify_ssl:
            import httpx
            # Create a new client with SSL verification disabled
            client._client = httpx.Client(verify=False)
        
        return client
    except Exception as e:
        warnings.warn(f"Failed to initialize LangSmith client: {e}", UserWarning)
        return None


# Configure SSL on module import if LangSmith is enabled
if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    configure_langsmith_ssl()

