import os
import json
import pandas as pd
import urllib3
import warnings
from datetime import date, timedelta, datetime
from typing import Annotated, Optional

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

# Disable SSL warnings and verification globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def disable_ssl_verification():
    """
    Disable SSL verification globally across all HTTP libraries.
    This function should be called at the start of the application.
    """
    import ssl
    import requests
    from urllib3.exceptions import InsecureRequestWarning
    
    # Disable urllib3 warnings
    urllib3.disable_warnings(InsecureRequestWarning)
    
    # Disable Python warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    # Set requests to not verify SSL by default
    try:
        # Monkey patch requests Session class
        original_request = requests.Session.request
        
        def patched_request(self, *args, **kwargs):
            kwargs.setdefault('verify', False)
            return original_request(self, *args, **kwargs)
        
        requests.Session.request = patched_request
        
        # Also patch the module-level functions
        original_get = requests.get
        original_post = requests.post
        
        def patched_get(*args, **kwargs):
            kwargs.setdefault('verify', False)
            return original_get(*args, **kwargs)
        
        def patched_post(*args, **kwargs):
            kwargs.setdefault('verify', False)
            return original_post(*args, **kwargs)
        
        requests.get = patched_get
        requests.post = patched_post
        
    except Exception as e:
        print(f"Warning: Could not fully disable SSL verification: {e}")
    
    print("INFO: SSL verification disabled globally")


# Initialize SSL settings on module import
disable_ssl_verification()

def save_output(data: pd.DataFrame, tag: str, save_path: Optional[str] = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
