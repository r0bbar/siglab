import json
from typing import Any, Dict, Union
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from tabulate import tabulate

from util.slack_notification_util import slack_dispatch_notification

from siglab_py.constants import LogLevel

def dispatch_notification(
        title : str,
        message : Union[str, Dict, pd.DataFrame],
        footer : str,
        params : Dict[str, Any],
        log_level : LogLevel = LogLevel.INFO
        ):
    if isinstance(message, Dict):
        _message = json.dumps(message, indent=2, separators=(' ', ':'))
    elif isinstance(message, pd.DataFrame):
        _message = tabulate(message, headers='keys', tablefmt='orgtbl') # type: ignore
    else:
        _message = message

    utc_time = datetime.now(timezone.utc)
    footer = f"UTC {utc_time} {footer}"

    slack_dispatch_notification(title, _message, footer, params, log_level)

if __name__ == '__main__':
    params : Dict[str, Any] = {
        "slack" : {
            "info" : {
                "webhook_url" : "https://hooks.slack.com/services/xxx"
            },
            "critical" : {
                "webhook_url" : "https://hooks.slack.com/services/xxx"
            },
            "alert" : {
                "webhook_url" : "https://hooks.slack.com/services/xxx"
            }
      },
    }

    log_level : LogLevel = LogLevel.CRITICAL

    title : str = "Test message"
    footer : str = "... some footer .."

    message1 : str = "Testing 1"
    dispatch_notification(title=title, message=message1, footer=footer, params=params, log_level=log_level)

    message2 : Dict[str, Any] = {
        'aaa' : 123,
        'bbb' : 456,
        'ccc' : {
            'ddd' : 789
        }
    }
    dispatch_notification(title=title, message=message2, footer=footer, params=params, log_level=log_level)

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
    dispatch_notification(title=title, message=message3, footer=footer, params=params, log_level=log_level)
