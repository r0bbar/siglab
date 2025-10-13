import asyncio
import sys
import traceback
import os
import argparse
import json
import hashlib
import re
from datetime import datetime, timedelta, timezone
import time
import pytz
import arrow
from enum import Enum
import logging
import requests
from typing import Dict, Optional, Set, Any, Union, List
from redis import StrictRedis

from siglab_py.util.notification_util import dispatch_notification

current_filename = os.path.basename(__file__)

'''
google_monitor fetches messages from particular query. Then:
    a. Save (and accumulate) messages to message cache file (No duplicates) for further analysis.
        message_cache_file: str = f"google_search_messages.json"

    b. If any of keywords in message_keywords_filter matches words in message (--message_keywords_filter):
        - Publish to redis for strategy consumption, topic: param['mds']['topics']['google_alert']
        - Dispatch slack alert
        - If scripts runs on Windows, play a wav file (Feels free make modification play sounds on Ubuntu for example)

Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python google_monitor.py --apikey xxx --search_engine_id yyy --query "site:twitter.com @user_id1 @user_id2 some topic" --slack_info_url=https://hooks.slack.com/services/xxx --slack_critial_url=https://hooks.slack.com/services/xxx --slack_alert_url=https://hooks.slack.com/services/xxx

    alert_wav_path
        Point it to wav file for alert notification. It's using 'winsound', i.e. Windows only.
        Set to None otherwise.
        
Google API: https://console.cloud.google.com/apis/credentials?project=YOUR_PROJECT
		name: YOUR_API_KEY_NAME
		apikey: ?????

Google Search Engine
	To create
		name: siglab_py_search: https://programmablesearchengine.google.com/controlpanel/create
			<script async src="https://cse.google.com/cse.js?cx=YOUR_SEARCH_ENGINE_ID">
			</script>
			<div class="gcse-search"></div>
	Then enable it: https://console.developers.google.com/apis/api/customsearch.googleapis.com/overview?project=?????

launch.json for Debugging from VSCode:
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "justMyCode": false,
                "args" : [
                        "--apikey", "xxx",
                        "--search_engine_id", "yyy",
                        "--query", "site:twitter.com @user_id1 @user_id2 some topic",
                        "--slack_info_url", "https://hooks.slack.com/services/xxx",
                        "--slack_critial_url", "https://hooks.slack.com/services/xxx",
                        "--slack_alert_url", "https://hooks.slack.com/services/xxx",
                    ],
            }
        ]
    }
'''

param: Dict[str, Any] = {
    'apikey': os.getenv('GOOGLE_APIKEY', 'xxx'),
    'search_engine_id': os.getenv('GOOGLE_SEARCH_ENGINE_ID', 'xxx'),
    'num_results' : 10,
    'query' : '',
    'alert_wav_path' : r"d:\sounds\terrible.wav",
    "num_shouts" : 5, # How many times 'alert_wav_path' is played
    "loop_freq_ms" : 1000*60*15, # Google allow max 100 calls per day free.
    'current_filename' : current_filename,

    'notification' : {
        'footer' : None,

        # slack webhook url's for notifications
        'slack' : {
            'info' : { 'webhook_url' : None },
            'critical' : { 'webhook_url' : None },
            'alert' : { 'webhook_url' : None },
        }
    },
    
    'mds': {
        'topics': {
            'tg_alert': 'tg_alert'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl_ms': 1000 * 60 * 15
        }
    }
}

class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

logging.Formatter.converter = time.gmtime
logger: logging.Logger = logging.getLogger()
log_level: int = logging.INFO
logger.setLevel(log_level)
format_str: str = '%(asctime)s %(message)s'
formatter: logging.Formatter = logging.Formatter(format_str)
sh: logging.StreamHandler = logging.StreamHandler()
sh.setLevel(log_level)
sh.setFormatter(formatter)
logger.addHandler(sh)

def log(message: str, log_level: LogLevel = LogLevel.INFO) -> None:
    if log_level.value < LogLevel.WARNING.value:
        logger.info(f"{datetime.now()} {message}")
    elif log_level.value == LogLevel.WARNING.value:
        logger.warning(f"{datetime.now()} {message}")
    elif log_level.value == LogLevel.ERROR.value:
        logger.error(f"{datetime.now()} {message}")

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore
    parser.add_argument("--apikey", help="API key", default=None)
    parser.add_argument("--search_engine_id", help="Google search engine ID", default=None)
    parser.add_argument("--num_results", help="Max number items to fetch", default=10)
    parser.add_argument("--query", help="Query - what are you looking for?", default=None)
    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()
    
    param['apikey'] = args.apikey
    param['search_engine_id'] = args.search_engine_id
    param['num_results'] = args.num_results
    param['query'] = args.query

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

    param['notification']['footer'] = f"From {param['current_filename']}"

    print(f"Startup args: {args}") # Dont use logger, not yet setup yet.
    print(f"param: {print(json.dumps(param, indent=2))}")

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
        log(f"Failed to init redis connection. Will skip publishes to redis. {err_msg}")
        redis_client = None # type: ignore
    
    return redis_client

def search_google_custom(query, api_key, search_engine_id, num_results=10):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'num': num_results,
        'sort': 'date',
        'dateRestrict': 'd1'  # Restrict to most recent (adjust as needed: d1=day, m1=month, etc.)
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        log(f"Query error: {response.status_code} - {response.text}")
        return None
    
async def main() -> None:
    parse_args()

    message_cache_file: str = f"google_search_messages.json"
    log(f"message_cache_file: {message_cache_file}")

    notification_params : Dict[str, Any] = param['notification']

    processed_messages : List[Dict[str, Any]] = []
    seen_hashes : Set[str] = set()
    if os.path.exists(message_cache_file):
        with open(message_cache_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                message_data = json.loads(line)
                message_hash: str = hashlib.sha256(message_data['message'].encode('utf-8')).hexdigest()

                message_data['datetime'] = pytz.UTC.localize(arrow.get(message_data['datetime']).datetime.replace(tzinfo=None))

                if message_hash not in seen_hashes:
                    seen_hashes.add(message_hash)
                    processed_messages.append(message_data)

            processed_messages = sorted(processed_messages, key=lambda m: m['datetime'])
    
    try:        
        redis_client: Optional[StrictRedis] = init_redis_client()
    except Exception as redis_err:
        redis_client = None
        log(f"Failed to connect to redis. Still run but not publishing to it. {redis_err}")

    while True:
        try:
            results = search_google_custom(param['query'], param['apikey'], param['search_engine_id'], param['num_results'])

            if results:
                if 'items' in results:
                    for item in results['items']:
                        title = item.get('title', 'No title')
                        snippet = item.get('snippet', 'No snippet')
                        link = item.get('link', 'No link')
                        published_date = item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', 'No date')

                        dt_message = datetime.now()
                        timestamp_ms  = int(dt_message.timestamp() * 1000)
                        message_data: Dict[str, Any] = {
                            "timestamp_ms": timestamp_ms,
                            "datetime": dt_message.isoformat(), # Always in UTC
                            "message": snippet
                        }
                        json_str: str = json.dumps(message_data, ensure_ascii=False, sort_keys=True)
                        message_hash: str = hashlib.sha256(snippet.encode('utf-8')).hexdigest()
                        if (message_hash not in seen_hashes):
                            seen_hashes.add(message_hash)
                            processed_messages.append(message_data)

                            log(f"{message_data}")

                            dispatch_notification(title=f"{param['current_filename']} Incoming!", message=message_data, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger) # type: ignore

                            with open(message_cache_file, 'a', encoding='utf-8') as f:
                                json.dump(message_data, f, ensure_ascii=False)
                                f.write('\n')

                            if param['alert_wav_path']and sys.platform == 'win32':
                                import winsound
                                for _ in range(param['num_shouts']):
                                    winsound.PlaySound(param['alert_wav_path'], winsound.SND_FILENAME)

                            if redis_client:
                                try:
                                    publish_topic = f"google_search"
                                    redis_client.publish(publish_topic, json_str)
                                    redis_client.setex(message_hash, param['mds']['redis']['ttl_ms'] // 1000, json_str)
                                    log(f"Published message {json_str} to Redis topic {publish_topic}", LogLevel.INFO)
                                except Exception as e:
                                    log(f"Failed to publish to Redis: {str(e)}", LogLevel.ERROR)

                await asyncio.sleep(int(param['loop_freq_ms'] / 1000))

            if processed_messages:
                oldest_message: Dict[str, Any] = min(processed_messages, key=lambda x: x['timestamp_ms'])
                newest_message: Dict[str, Any] = max(processed_messages, key=lambda x: x['timestamp_ms'])
                log(
                    json.dumps(
                        {
                            'num_messages': len(processed_messages),
                            'oldest': {
                                'timestamp_ms': oldest_message['timestamp_ms'],
                                'datetime': datetime.fromtimestamp(int(oldest_message['timestamp_ms']/1000),tz=timezone.utc).isoformat()
                            },
                            'latest': {
                                'timestamp_ms': newest_message['timestamp_ms'],
                                'datetime': datetime.fromtimestamp(int(newest_message['timestamp_ms']/1000),tz=timezone.utc).isoformat()
                            }
                        }, indent=2
                    ),
                    LogLevel.INFO
                )

        except Exception as e:
            log(f"Oops {str(e)} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}", LogLevel.ERROR)
        finally:
            await asyncio.sleep(int(param['loop_freq_ms'] / 1000))

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Stopped by user", LogLevel.INFO)
    except Exception as e:
        log(f"Unexpected error: {str(e)}", LogLevel.ERROR)