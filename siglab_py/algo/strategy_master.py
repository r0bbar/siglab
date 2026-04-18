import asyncio
import os
import sys
import traceback
import logging
from typing import Dict, Optional
from enum import Enum
from datetime import datetime, timedelta
import time
import arrow
from dateutil import parser
import argparse
import json
import re
import pandas as pd
import hashlib
from tabulate import tabulate
import matplotlib.pyplot as plt
from redis import StrictRedis

from siglab_py.util.notification_util import dispatch_notification

current_filename = os.path.basename(__file__)
current_dir : str = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

logging.Formatter.converter = time.gmtime
logger: logging.Logger = logging.getLogger()
log_level = logging.INFO # DEBUG --> INFO --> WARNING --> ERROR
logger.setLevel(log_level)
format_str: str = '%(asctime)s %(message)s'
formatter: logging.Formatter = logging.Formatter(format_str)
sh: logging.StreamHandler = logging.StreamHandler()
sh.setLevel(log_level)
sh.setFormatter(formatter)
fh = logging.FileHandler(f"strategy_master.log")
fh.setLevel(log_level)
fh.setFormatter(formatter)     
logger.addHandler(fh)

'''
Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python strategy_master.py --loop_freq_ms 1000 --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx
    
    --slack_info_url/slack_critial_url/slack_alert_url: optional. Lookup how to configure "Incoming WebHooks" (a slack app) under Slack's "Browse Apps"

launch.json for Debugging from VSCode:
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python Debugger: Current File",
                "type": "debugpy",
                "request": "launch",
                "justMyCode": false,
                "program": "${file}",
                "console": "integratedTerminal",
                "args": [
                    "--loop_freq_ms", "1000",

                    "--slack_info_url", "https://hooks.slack.com/services/xxx",
                    "--slack_critial_url", "https://hooks.slack.com/services/xxx",
                    "--slack_alert_url", "https://hooks.slack.com/services/xxx",
                ],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            }
        ]
'''
param : Dict = {
    'loop_freq_ms' : 10000, 
    'current_filename' : current_filename,

    # regex corresponding to position_topic.
    "position_topic_regex" : r"^position_.*", 
    "selected_fields_for_notification" : [ "key",  "pos_side", "pos_status" ],

    'notification' : {
        'footer' : None,

        # notification webhook url's for notifications
        'notification' : {
            'info' : { 'webhook_url' : None },
            'critical' : { 'webhook_url' : None },
            'alert' : { 'webhook_url' : None },
        }
    },
    
    'mds': {
        'topics': {
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl_ms': 1000 * 60 * 15
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore

    parser.add_argument("--loop_freq_ms", help="Loop delays. Default: 60000ms or 60sec", default=60000)

    parser.add_argument("--notification_info_url", help="Webhook url for INFO", default=None)
    parser.add_argument("--notification_critical_url", help="Webhook url for CRITICAL", default=None)
    parser.add_argument("--notification_alert_url", help="Webhook url for ALERT", default=None)

    args = parser.parse_args()

    param['loop_freq_ms'] = int(args.loop_freq_ms)
    
    param['notification']['notification']['info']['webhook_url'] = args.notification_info_url
    param['notification']['notification']['critical']['webhook_url'] = args.notification_critical_url
    param['notification']['notification']['alert']['webhook_url'] = args.notification_alert_url

    param['notification']['footer'] = f"From {param['current_filename']}"

    logger.info(f"Startup args: {args}") # Dont use logger, not yet setup yet.
    logger.info(f"param: {json.dumps(param, indent=2)}")

def init_redis_client() -> StrictRedis:
    redis_client : StrictRedis = StrictRedis(
                    host = param['mds']['redis']['host'],
                    port = param['mds']['redis']['port'],
                    db = 0,
                    ssl = False
                )
    try:
        redis_client.keys()
    except ConnectionError as redis_conn_error:
        err_msg = f"Failed to connect to redis: {param['mds']['redis']['host']}, port: {param['mds']['redis']['port']}"
        logger.info(f"Failed to init redis connection. Will skip publishes to redis. {err_msg}")
        redis_client = None # type: ignore
    
    return redis_client

async def main() -> None:
    parse_args()

    notification_params : Dict[str, Any] = param['notification']

    position_topic_regex : str = param['position_topic_regex']
    position_topic_regex_pattern : Pattern = re.compile(position_topic_regex)

    try:        
        redis_client : Optional[StrictRedis] = init_redis_client()
    except Exception as redis_err:
        redis_client = None
        logger.info(f"Failed to connect to redis. Still run but not publishing to it. {redis_err}")

    loop_counter : int = 0
    prev_message_hash = None
    position_summaries : List[Dict[str, Union[str, int, float, None]]] = []
    while True:
        try:
            start_ts_sec = time.time()

            position_summaries.clear()

            keys = redis_client.keys()
            for key in keys:
                try:
                    s_key : str = key.decode("utf-8")
                    if position_topic_regex_pattern.match(s_key):
                        print(s_key)
                        message = redis_client.get(key)
                        if message:
                            message = message.decode('utf-8')
                            position_summary = json.loads(message)
                            _position_summary = {}
                            for field in position_summary:
                                if position_summary[field]!="---":
                                    _position_summary[field] = position_summary[field]

                            _position_summary["key"] = f"{position_summary['gateway_id']} {position_summary['ticker']}"
                            position_summaries.append(_position_summary)

                except Exception as key_err:
                    logger.error(f"{key_err}")
        
            pd_position_summaries = pd.DataFrame(position_summaries)
            _pd_position_summaries = pd_position_summaries[param["selected_fields_for_notification"]]

            s_position_summaries = tabulate(pd_position_summaries, headers='keys', tablefmt='psql', showindex=False)
            logger.info(s_position_summaries)
            
            row_hashes = pd.util.hash_pandas_object(_pd_position_summaries, index=False)
            message_hash = hashlib.sha256(row_hashes.values).hexdigest()
            logger.info(f"message_hash: {message_hash}, prev_message_hash: {prev_message_hash}. Change? {message_hash!=prev_message_hash}")
            if message_hash!=prev_message_hash:
                prev_message_hash = message_hash

                dispatch_notification(
                                    title=f"#position {param['current_filename']}", 
                                    message=_pd_position_summaries, 
                                    footer=param['notification']['footer'], 
                                    params=notification_params, 
                                    log_level=LogLevel.INFO, 
                                    logger=logger
                                )
            
            elapsed_ms = int((time.time() - start_ts_sec) *1000)
            logger.info(f"[loop# {loop_counter}] end to end elapsed_ms: {elapsed_ms:,}")

        except Exception as loop_err:
            err_msg = f'loop error {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}'
            logger.error(err_msg)
            dispatch_notification(title=f"{param['current_filename']} error.", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)
                
        finally:
            await asyncio.sleep(int(param['loop_freq_ms'] / 1000))

            loop_counter += 1

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")