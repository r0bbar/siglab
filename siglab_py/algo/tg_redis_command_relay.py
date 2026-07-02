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
from typing import Dict, Optional, Set, Any, Union, List
from redis import StrictRedis
from pprint import pformat
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from telethon.types import Message

from siglab_py.util.notification_util import dispatch_notification

current_filename = os.path.basename(__file__)

'''
tg_redis_commander fetches messages from particular private TG channel and publish it to redis.

Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python tg_redis_command_relay.py.py --api_id xxx --api_hash yyy --hash_key zzz --channel_name xxx --channel_invite_url "https://t.me/xxx" --commands_filter "status, block" --slack_info_url=https://hooks.slack.com/services/xxx --slack_critial_url=https://hooks.slack.com/services/xxx --slack_alert_url=https://hooks.slack.com/services/xxx

api_id and api_hash
    Go to https://my.telegram.org/
    It's under "API development tools"

hash_key
    [part] of secret used to generate expected hash which should be included as first token of incoming message.

    full_hash_key = f"{param['hash_key']}{datetime.now().strftime('%Y%m%d')}"
    
    "%Y%m%d %H%M%S" is format string if you want yyyyMMdd HH:MM:SS. Here we take only "yyyyMMdd", avoiding HH so there's no confusion if it's running on UTC machine, or otherwise.
                    
    This is for sender/message validation.

alert_wav_path
    Point it to wav file for alert notification. It's using 'winsound', i.e. Windows only.
    Set to None otherwise.

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
                        "--api_id", "xxx",
                        "--api_hash", "yyy",
                        "--hash_key", "zzz",
                        "--channel_name", "optional",
                        "--channel_invite_url", "https://t.me/xxx",
                        "--commands_filter", "status, block",
                        "--notification_info_url", "https://hooks.slack.com/services/xxx",
                        "--notification_critical_url", "https://hooks.slack.com/services/xxx",
                        "--notification_alert_url", "https://hooks.slack.com/services/xxx",
                    ],
            }
        ]
    }

https://norman-lm-fung.medium.com/monitoring-telegram-channel-tg-monitor-from-siglab-py-f7ec30c2c32e
'''

param: Dict[str, Any] = {
    'api_id': os.getenv('TELEGRAM_API_ID', 'xxx'),
    'api_hash': os.getenv('TELEGRAM_API_HASH', 'xxx'),
    'phone': os.getenv('TELEGRAM_PHONE', '+XXXYYYYYYYY'),
    'message_keywords_filter': [],
    'alert_wav_path' : r"", # Example, d:\sounds\terrible.wav. If left blank, no sound will be played.
    "num_shouts" : 5, # How many times 'alert_wav_path' is played
    "loop_freq_ms" : 5000,
    'current_filename' : current_filename,

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
            'command': 'tg_command'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl_ms': 1000 * 15
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
    parser.add_argument("--api_id", help="TG api_id", default=None)
    parser.add_argument("--api_hash", help="TG api_hash", default=None)
    parser.add_argument("--hash_key", help="[part] of secret used to generate expected hash which should be included as first token of incoming message.", default=None)
    parser.add_argument("--channel_name", help="TG channel name, can leave blank.", default=None)
    parser.add_argument("--channel_invite_url", help="TG channel_invite_url, example https://t.me/xxx", default=None)
    parser.add_argument("--commands_filter", help="TG command filter: Comma separated list, case-insensitive. Default: None (i.e. no filter)", default=None)
    
    parser.add_argument("--notification_info_url", help="Webhook url for INFO", default=None)
    parser.add_argument("--notification_critical_url", help="Webhook url for CRITICAL", default=None)
    parser.add_argument("--notification_alert_url", help="Webhook url for ALERT", default=None)

    args = parser.parse_args()
    
    param['api_id'] = args.api_id
    param['api_hash'] = args.api_hash
    param['hash_key'] = args.hash_key
    param['channel_name'] = args.channel_name
    param['channel_invite_url'] = args.channel_invite_url
    if args.commands_filter:
        param['commands_filter'] = args.commands_filter.split(',')
        param['commands_filter'] = [ x.lower().strip() for x in param['commands_filter'] ]

    param['notification']['notification']['info']['webhook_url'] = args.notification_info_url
    param['notification']['notification']['critical']['webhook_url'] = args.notification_critical_url
    param['notification']['notification']['alert']['webhook_url'] = args.notification_alert_url

    param['notification']['footer'] = f"From {param['current_filename']} {param['channel_name'].lstrip('@')}"

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

async def main() -> None:
    parse_args()

    session_file: str = f"{param['channel_name'].lstrip('@')}_session"
    log(f"session_file: {session_file}")

    notification_params : Dict[str, Any] = param['notification']
            
    try:        
        redis_client: Optional[StrictRedis] = init_redis_client()
    except Exception as redis_err:
        redis_client = None
        log(f"Failed to connect to redis. Still run but not publishing to it. {redis_err}")
    
    seen_hashes = []
    async with TelegramClient(session_file, param['api_id'], param['api_hash']) as client:
        try:
            channel_entity = await client.get_entity(param['channel_invite_url']) 
            log(f"channel: {channel_entity.id}")
            
            commands = []
            while True:
                since = datetime.now() - timedelta(minutes=1)
                print(f"{datetime.now()} fetching channel messages ...")
                async for message in client.iter_messages(channel_entity, offset_date=since, limit=10): # Looks TG lags can > 1 minute
                    s_message = f"{message.date} {message.sender.title} {message.text}"

                    full_hash_key = f"{param['hash_key']}{datetime.now().strftime('%Y%m%d')}" # "%Y%m%d %H%M%S" is format string if you want yyyyMMdd HH:MM:SS. Here we take only "yyyyMMdd MM", avoiding HH so there's no confusion if it's running on UTC machine, or otherwise.
                    expected_hash = hashlib.sha256(full_hash_key.encode()).hexdigest()
                    expected_hash = expected_hash[:3] + expected_hash[-3:] # Take only first three and last three char
                    
                    '''
                    Example command format: 
                        xxxxx master1 status
                    Where:
                        'xxxxx' is message hash
                        'master1' is command's target (or intended recipient)
                        'status' is the actual command to execute: report status for example (Do something very safe, read only ops for example assuming TG channels unsecured even for private channels)
                    '''
                    if message.text:
                        message_text = message.text.lower().strip()
                        command_tokens = message_text.split(' ')
                        command_tokens = [ token for token in command_tokens if token ]
                        message_hash = command_tokens[0]
                        
                        if (
                            message_hash==expected_hash # verify message/sender is valid.
                        ):
                            seen_hash = hashlib.sha256(f"{int(message.date.timestamp())}{message.text}".encode()).hexdigest()
                            if seen_hash not in seen_hashes: # Guard against duplicates, REPLAY attacks
                                seen_hashes.append(seen_hash)

                                target = command_tokens[1]
                                command = command_tokens[2]

                                if command in param['commands_filter'] or not param['commands_filter']:
                                    incoming = {
                                        'tg_timestamp_ms' : int(message.date.timestamp() *1000), # TG can lag by > 60 sec
                                        'recv_timestamp_ms' : datetime.now(),
                                        'sender_id' : message.sender.title,
                                        'target' : target,
                                        'command' : command,
                                    }
                                    commands.append(incoming)
                                    
                                    print(f"registered command: {s_message}")

                                else:
                                    print(f"message discarded, command '{command}' not registered in commands_filter: {s_message}")
                                    
                            else:
                                print(f"message discarded, already processed: {s_message}")

                        else:
                            print(f"message discarded, message_hash {message_hash} not matching expected_hash {expected_hash}: {s_message}")

                if commands:
                    log(f"Commands received:")
                    log(f"{pformat(commands, indent=2, width=100)}")
                                        
                    if redis_client:
                        try:
                            publish_topic = f"{param['mds']['topics']['command']}"
                            redis_client.set(name=publish_topic, value=json.dumps(commands).encode('utf-8'), ex=param['mds']['redis']['ttl_ms'] // 1000)
                            log(f"Published {len(commands)} commands to Redis topic {publish_topic}", LogLevel.INFO)
                        except Exception as e:
                            log(f"Failed to publish to Redis: {str(e)}", LogLevel.ERROR)
                        finally:
                            commands.clear()
                    
                    if param['alert_wav_path'] and sys.platform == 'win32':
                        import winsound
                        for _ in range(param['num_shouts']):
                            winsound.PlaySound(param['alert_wav_path'], winsound.SND_FILENAME)
                        
                await asyncio.sleep(int(param['loop_freq_ms']/1000)) # So long you wait one sec, TG wont block your subsequent call 15 sec!

        except Exception as e:
            log(f"Oops {str(e)} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}", LogLevel.ERROR)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Stopped by user", LogLevel.INFO)
    except Exception as e:
        log(f"Unexpected error: {str(e)}", LogLevel.ERROR)