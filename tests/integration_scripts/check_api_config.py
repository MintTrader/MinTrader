#!/usr/bin/env python3
"""
Quick API configuration checker.
Verifies that all necessary API keys and configurations are in place.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def check_env_var(var_name, required=True):
    """Check if an environment variable is set."""
    value = os.getenv(var_name)
    if value:
        # Mask the value for security
        if len(value) > 8:
            masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
        else:
            masked = "*" * len(value)
        return True, masked
    else:
        return False, "Not set"

def check_config_file():
    """Check if .env file exists."""
    env_file = Path(__file__).parent.parent / ".env"
    return env_file.exists()

def main():
    """Check API configuration."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*24 + "API CONFIGURATION CHECK" + " "*31 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Check .env file
    print("Configuration File:")
    print("-" * 80)
    if check_config_file():
        print("  ✓ .env file found")
    else:
        print("  ✗ .env file NOT found")
        print("    Create a .env file in the project root")
    print()
    
    # Define required and optional API keys
    api_keys = {
        "Required for most tests": [
            ("ALPACA_API_KEY", True),
            ("ALPACA_SECRET_KEY", True),
            ("ALPACA_PAPER_MODE", True),
        ],
        "Optional (but recommended)": [
            ("OPENAI_API_KEY", False),
        ],
        "Yahoo Finance": [
            # Yahoo Finance doesn't require API keys
        ]
    }
    
    all_ok = True
    
    for category, keys in api_keys.items():
        if not keys:
            continue
            
        print(f"{category}:")
        print("-" * 80)
        
        for key_name, required in keys:
            is_set, value = check_env_var(key_name, required)
            
            if is_set:
                print(f"  ✓ {key_name}: {value}")
            else:
                if required:
                    print(f"  ✗ {key_name}: Not set (REQUIRED)")
                    all_ok = False
                else:
                    print(f"  ⚠ {key_name}: Not set (optional)")
        print()
    
    # Check specific configurations
    print("Configuration Settings:")
    print("-" * 80)
    
    # Check paper trading mode
    paper_mode = os.getenv("ALPACA_PAPER_MODE", "").lower()
    if paper_mode == "true":
        print("  ✓ ALPACA_PAPER_MODE: True (Safe - Paper Trading)")
    elif paper_mode == "false":
        print("  ⚠ ALPACA_PAPER_MODE: False (WARNING - LIVE TRADING!)")
        print("    For testing, set to True")
    else:
        print("  ✗ ALPACA_PAPER_MODE: Not properly configured")
        all_ok = False
    
    print()
    
    # Test imports
    print("Python Package Imports:")
    print("-" * 80)
    
    packages = [
        ("alpaca", "Alpaca API"),
        ("yfinance", "Yahoo Finance"),
        ("pandas", "Pandas"),
        ("requests", "Requests"),
        ("bs4", "BeautifulSoup4 (Google News)"),
        ("openai", "OpenAI (optional)"),
    ]
    
    for package, description in packages:
        try:
            __import__(package)
            print(f"  ✓ {description}: Installed")
        except ImportError:
            if "optional" in description.lower():
                print(f"  ⚠ {description}: Not installed (optional)")
            else:
                print(f"  ✗ {description}: Not installed (REQUIRED)")
                all_ok = False
    
    print()
    
    # Final summary
    print("="*80)
    if all_ok:
        print("✓ Configuration looks good! You can run the integration tests.")
        print("\nTo run all tests:")
        print("  python integration_scripts/run_all_tests.py")
        print("\nTo run individual tests:")
        print("  python integration_scripts/test_core_stock_tools.py")
        print("  python integration_scripts/test_technical_indicator_tools.py")
        print("  python integration_scripts/test_news_tools.py")
        print("  python integration_scripts/test_fundamental_tools.py")
        print("  python integration_scripts/test_trading_tools.py")
        return 0
    else:
        print("⚠️  Some required configurations are missing.")
        print("\nPlease:")
        print("  1. Create or update your .env file")
        print("  2. Add required API keys")
        print("  3. Install missing packages: pip install -r requirements.txt")
        print("  4. Run this script again to verify")
        return 1

if __name__ == "__main__":
    exit(main())

