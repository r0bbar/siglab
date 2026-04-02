import sys
from typing import Any, Dict
import json
from  datetime import datetime,  timezone
import requests

from siglab_py.constants import LogLevel

def discord_dispatch_notification(
        title : str,
        message : str,
        footer : str,
        params : Dict[str, Any],
        log_level : LogLevel = LogLevel.INFO,
        max_message_len : int = 1800,

        param_webhooks_config_section : str = 'webhooks'
):
    _params = params[param_webhooks_config_section]

    if log_level.value==LogLevel.INFO.value or log_level.value==LogLevel.DEBUG.value:
        webhook_url = _params['info']['webhook_url']
    elif log_level.value==LogLevel.CRITICAL.value:
        webhook_url = _params['critical']['webhook_url']
    elif log_level.value==LogLevel.ERROR.value:
        webhook_url = _params['alert']['webhook_url']
    else:
        webhook_url = _params['info']['webhook_url']

    if not webhook_url:
        return

    embed = {
        "title": title,
        "description": message,
        "footer": {"text": footer},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    payload = {
        "username": "siglab_py",
        "embeds": [embed]
    }

    rsp = requests.post(webhook_url, json=payload)
    if rsp.status_code != 204:
        raise Exception(rsp.status_code, rsp.text)

if __name__ == '__main__':
    params : Dict[str, Any] = {
        "webhooks" : {
            "info" : {
                "webhook_url" : "https://discord.com/api/webhooks/xxx"
            },
            "critical" : {
                "webhook_url" : "https://discord.com/api/webhooks/xxx"
            },
            "alert" : {
                "webhook_url" : "https://discord.com/api/webhooks/xxx"
            }
      },
    }

    log_level : LogLevel = LogLevel.CRITICAL

    title : str = "Test message"
    footer : str = "... some footer .."

    message : Dict[str, Any] = json.dumps(
        {
            'aaa' : 123,
            'bbb' : 456,
            'ccc' : {
                'ddd' : 789
            }
        }, indent=2
    )
    discord_dispatch_notification(title=title, message=message, footer=footer, params=params, log_level=log_level)