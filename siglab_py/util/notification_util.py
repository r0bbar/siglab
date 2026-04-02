import json
from typing import Any, Dict, Union
from datetime import datetime, timezone
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
        param_webhooks_config_section : str = 'slack'
    ):
    try:
        if isinstance(message, Dict):
            cleaned = recursive_clean_dict(message)
            _message = json.dumps(cleaned, indent=2, separators=(' ', ':'))
        elif isinstance(message, pd.DataFrame):
            # _message = tabulate(message, headers='keys', tablefmt='orgtbl') # type: ignore
            _message = message.to_markdown(index=False) # index=False removes the row numbers
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
    param_webhooks_config_section="slack"

    params : Dict[str, Any] = {
        param_webhooks_config_section : {
            "info" : {
                "webhook_url" : "https://discord.com/api/webhooks/1489048454968905909/hsTZDGBzLZWwtqHr85vRx1kSDAKDTwnDA89AkoiLCZyrtojPz7CaOe4APXmiOIi3-p6S"
            },
            "critical" : {
                "webhook_url" : "https://discord.com/api/webhooks/1489047810996310107/bDfGMtUH0d6uoy94y352eikhD6p1sOAVXVPLY7uLOTWmZ8kN1jhObsJjVXwuqTObFRR7"
            },
            "alert" : {
                "webhook_url" : "https://discord.com/api/webhooks/1489048026659160164/er0WNABzP_kgtdmRYaFGUq89gP5WL8dZFWqRG2axDPQLqGNVssaBakAlAqfgwxP0E6Lr"
            }
      },
    }

    log_level : LogLevel = LogLevel.CRITICAL

    title : str = "Test message"
    footer : str = "... some footer .."

    message1 : str = "Testing 1"
    dispatch_notification(title=title, message=message1, footer=footer, params=params, log_level=log_level, param_webhooks_config_section=param_webhooks_config_section)

    message2 : Dict[str, Any] = {
        'aaa' : 123,
        'bbb' : 456,
        'ccc' : {
            'ddd' : 789
        }
    }
    dispatch_notification(title=title, message=message2, footer=footer, params=params, log_level=log_level, param_webhooks_config_section=param_webhooks_config_section)

    start_date = pd.to_datetime('2024-01-01 00:00:00')
    datetimes = pd.date_range(start=start_date, periods=20, freq='H')
    np.random.seed(42)
    close_prices = np.random.uniform(80000, 90000, size=20).round(2)
    data : pd.DataFrame = pd.DataFrame({
        'datetime': datetimes,
        'close': close_prices
    })
    data['timestamp_ms'] = data['datetime'].astype('int64')
    message3 = data
    dispatch_notification(title=title, message=message3, footer=footer, params=params, log_level=log_level, param_webhooks_config_section=param_webhooks_config_section)
