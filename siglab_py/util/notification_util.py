import json
from typing import Any, List, Dict, Union, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from tabulate import tabulate

from siglab_py.util.discord_notification_util import discord_dispatch_notification
from siglab_py.util.collection_util import recursive_clean_dict
from siglab_py.constants import LogLevel

def dispatch_notification(
    title: str,
    message: Union[str, Dict, pd.DataFrame],
    footer: str,
    params: Dict[str, Any],
    files: Optional[List[tuple]] = None,
    log_level: LogLevel = LogLevel.INFO,
    logger=None,
    param_webhooks_config_section: str = 'notification'
):
    try:
        if isinstance(message, Dict):
            cleaned = recursive_clean_dict(message)
            _message = json.dumps(cleaned, indent=2, separators=(' ', ':'))
        elif isinstance(message, pd.DataFrame):
            _message = message
        else:
            _message = message

        utc_time = datetime.now(timezone.utc)
        footer = f"UTC {utc_time} {footer}"

        discord_dispatch_notification(
            title=title,
            message=_message,
            footer=footer,
            params=params,
            log_level=log_level,
            files=files,                                 # ← pass through
            param_webhooks_config_section=param_webhooks_config_section
        )

    except Exception as e:
        if logger:
            logger.error(f"Failed to dispatch notification for {title}: {e}")
            logger.error(message)

if __name__ == '__main__':
    param_webhooks_config_section="notification"

    params : Dict[str, Any] = {
        param_webhooks_config_section : {
            "info" : {
                "webhook_url" : "https://discord.com/api/webhooks/xxx/xxx"
            },
            "critical" : {
                "webhook_url" : "https://discord.com/api/webhooks/xxx/xxx"
            },
            "alert" : {
                "webhook_url" : "https://discord.com/api/webhooks/xxx/xxx"
            }
      },
    }

    log_level : LogLevel = LogLevel.CRITICAL

    title : str = "Test message"
    footer : str = "... some footer .."


    # Test 1:
    message1 : str = "Testing 1"
    dispatch_notification(title=title, message=message1, footer=footer, params=params, log_level=log_level, param_webhooks_config_section=param_webhooks_config_section)


    # Test 2: Send a Dict
    message2 : Dict[str, Any] = {
        'aaa' : 123,
        'bbb' : 456,
        'ccc' : {
            'ddd' : 789
        }
    }
    dispatch_notification(title=title, message=message2, footer=footer, params=params, log_level=log_level, param_webhooks_config_section=param_webhooks_config_section)
    

    # Test 3: Send DataFrame 
    '''
    For Discord, sending DataFrame is problematic:
    a. When look at messages from browser from desktop: If more three columns, subsequent columns will be wrapped around.
    b. When look at messages from mobile: It's max two columns or wrapping will happenning.
    '''
    np.random.seed(42)
    NUM_SAMPLES = 5
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    rng = np.random.default_rng()
    timestamps = rng.integers(
        int(start_time.timestamp() * 1000),
        int(end_time.timestamp() * 1000),
        size=NUM_SAMPLES,
        dtype=np.int64
    )

    close_prices = 70000 + np.random.normal(0, 50, size=NUM_SAMPLES).round(2)

    data : pd.DataFrame = pd.DataFrame({
        'timestamp_ms': timestamps,
        'close': close_prices
    })
    data['timestamp_ms'] = data['timestamp_ms'].astype('int64')

    # Send as embed without attachment
    message3 = data
    dispatch_notification(
            title=title, 
            message=message3,
            footer=footer, 
            params=params, 
            log_level=log_level, 
            param_webhooks_config_section=param_webhooks_config_section
        )

    # Send as csv attachment
    data.to_csv("dummy_data.csv")
    with open("dummy_data.csv", "rb") as csv:
        dispatch_notification(
            title=title, 
            message=message3, 
            files=[
                (f"dummy_data.csv", csv)
            ],
            footer=footer, 
            params=params, 
            log_level=log_level, 
            param_webhooks_config_section=param_webhooks_config_section
        )
