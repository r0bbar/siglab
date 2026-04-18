import json
from typing import Any, Dict, Union
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from tabulate import tabulate

from siglab_py.util.discord_notification_util import discord_dispatch_notification
from siglab_py.util.collection_util import recursive_clean_dict
from siglab_py.constants import LogLevel

def dispatch_notification(
        title : str,
        message : Union[str, Dict, pd.DataFrame],
        footer : str,
        params : Dict[str, Any],
        log_level : LogLevel = LogLevel.INFO,
        logger = None,
        param_webhooks_config_section : str = 'notification'
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
            param_webhooks_config_section=param_webhooks_config_section
        )
    except Exception as any_notification_error:
        if logger:
            logger.error(f"Failed to dispatch notification for {str(title)}: {any_notification_error}")
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
    np.random.seed(42)


    # Test 3: Send DataFrame
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
    message3 = data
    dispatch_notification(title=title, message=message3, footer=footer, params=params, log_level=log_level, param_webhooks_config_section=param_webhooks_config_section)
