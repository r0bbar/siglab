'''
https://medium.com/@natalia_assad/how-send-a-table-to-slack-using-python-d1a20b08abe0
'''
import sys
from typing import Any, Dict
import json
import requests

from siglab_py.constants import LogLevel

def slack_dispatch_notification(
        title : str,
        message : str,
        footer : str,
        params : Dict[str, Any],
        log_level : LogLevel = LogLevel.INFO
):
    slack_params = params['slack']

    if log_level.value==LogLevel.INFO.value or log_level.value==LogLevel.DEBUG.value:
        webhook_url = slack_params['info']['webhook_url']
    elif log_level.value==LogLevel.CRITICAL.value:
        webhook_url = slack_params['critical']['webhook_url']
    elif log_level.value==LogLevel.ERROR.value:
        webhook_url = slack_params['alert']['webhook_url']
    else:
        webhook_url = slack_params['info']['webhook_url']

    if not webhook_url:
        return

    data = {
        "username": "",
        "type": "section",
        "blocks": [
            {
                "type": "header",
                "text": { "type": "plain_text", "text": f"{title}" }
            },
            {
                "type": "section",
                "text": { "type": "mrkdwn", "text": message }
            },
            {
                "type": "section",
                "text": { "type": "plain_text", "text": footer }
            }
        ]
    }

    byte_size = str(sys.getsizeof(data, 2000))
    req_headers = { 'Content-Length': byte_size, 'Content-Type': "application/json"}
    rsp = requests.post(webhook_url, headers=req_headers, data=json.dumps(data))
    if rsp.status_code != 200:
        raise Exception(rsp.status_code, rsp.text)