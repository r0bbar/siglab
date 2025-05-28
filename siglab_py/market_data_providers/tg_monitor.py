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
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from telethon.types import Message
from redis import StrictRedis

current_filename = os.path.basename(__file__)

'''
tg_monitor fetches messages from particular TG channel (--channel_username). Then:
    a. Save (and accumulate) messages to message cache file (No duplicates) for further analysis.
        message_cache_file: str = f"{param['channel_username'].lstrip('@')}_messages.json"
       Note, only messages from senders in param['users_filter'] will be included.

    b. If any of keywords in message_keywords_filter matches words in message (--message_keywords_filter):
        - Publish to redis for strategy consumption, topic: param['mds']['topics']['tg_alert']
        - Dispatch slack alert
        - If scripts runs on Windows, play a wav file (Feels free make modification play sounds on Ubuntu for example)

Usage:
    python tg_monitor.py --api_id xxx --api_hash yyy --phone +XXXYYYYYYYY --channel_username @zZz --users_filter "yYy,zZz" --start_date 2025-03-01 --message_keywords_filter "trump,trade war,tariff" --slack_info_url=https://hooks.slack.com/services/xxx --slack_critial_url=https://hooks.slack.com/services/xxx --slack_alert_url=https://hooks.slack.com/services/xxx

api_id and api_hash
    Go to https://my.telegram.org/
    It's under "API development tools"

phone
    Format: +XXXYYYYYYYY where XXX is country/area code.

channel_username and users_filter
    You decide which TG channel, and under that channel what users you're tracking. TG is noisy.

start_date
    tg_monitor relies on TelegramClient.get_messages: https://docs.telethon.dev/en/stable/modules/client.html#telethon.client.messages.MessageMethods.get_messages
    As with most API, they generally limit number of entries you can fetch in one go (~100).
    tg_monitor implemented sliding window technique with cutoff = start_date for the first fetch.
    Note, if you increment your sliding window too quickly, you'd miss some messages. If you scan too slowly, it'd take forever. It's a balancing act.
    Atm, here's the general logic how we move the sliding window foward:
        * If fetches return zero message that matches param['users_filter], we move the window forward by an hour.
        * Otherwise, we move the window by adding 5 minutes to timestamp of last message fetched (whether matches users_filter or not) just fetched.
    A decent specification of 'users_filter' will help minimize chances of missed messages.
    Default: None, cutoff date will be set to tm1 in this case if message_cache_file NOT exists.                
             Otherwise, it'd default to latest message's datetime, plus a couple minute.
    Set to 'yyyy-MM-dd' if you want to collect more history for analysis. 
    Generally, TG allows you fetch around three months history. Anything more, get_messages will return empty array.

    Regardless, messages you collected is accumulated in 'message_cache_file'.

message_keywords_filter
    TG message_keywords_filter: Comma separated list, case-insensitive. Default: None (i.e. no keywords)
    Example, --message_keywords_filter "exploit, attack, hack, breach, compromise, stolen, leak, security incident, phishing, social engineer, withdrawals freeze, frozen"
    
    We play a sound, publish to redis..etc, only if message contain any of the keywords in message_keywords_filter.
    Quote around comma separated list.

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
                        "--api_hash", "xxx",
                        "--phone", "+XXXYYYYYYYY",
                        "--channel_username", "@SomeChannel",
                        "--users_filter", "SomeBody",
                        "--start_date", "2025-03-01",
                        "--message_keywords_filter", "exploit, attack, hack, breach, compromise, stolen, leak, security incident, phishing, social engineer, withdrawals freeze, frozen",
                        "--slack_info_url", "https://hooks.slack.com/services/xxx",
                        "--slack_critial_url", "https://hooks.slack.com/services/xxx",
                        "--slack_alert_url", "https://hooks.slack.com/services/xxx",
                    ],
            }
        ]
    }
'''

param: Dict[str, Any] = {
    'api_id': os.getenv('TELEGRAM_API_ID', 'xxx'),
    'api_hash': os.getenv('TELEGRAM_API_HASH', 'xxx'),
    'phone': os.getenv('TELEGRAM_PHONE', '+XXXYYYYYYYY'),
    'channel_username': '@SomeChannel',
    'users_filter' : None,
    'message_keywords_filter': [],
    'start_date': None,  
    'alert_wav_path' : r"d:\sounds\terrible.wav",

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
    parser.add_argument("--api_id", help="TG api_id", default=None)
    parser.add_argument("--api_hash", help="TG api_hash", default=None)
    parser.add_argument("--phone", help="G Phone tied to TG. Format: +XXXYYYYYYYY where XXX is country/area code.", default=None)
    parser.add_argument("--channel_username", help="TG channel_username", default=None)
    parser.add_argument("--users_filter", help="Comma separated list of TG user names", default=None)
    parser.add_argument("--message_keywords_filter", help="TG message_keywords_filter: Comma separated list, case-insensitive. Default: None (i.e. no keywords)", default=None)
    parser.add_argument("--start_date", help="start_date, format: yyyy-MM-dd. If left to null, cutoff date default to last message's datetime from message cache, or tm1.", default=None)
    
    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()
    
    param['api_id'] = args.api_id
    param['api_hash'] = args.api_hash
    param['channel_username'] = args.channel_username
    if args.users_filter:
        param['users_filter'] = [ user.lower().strip() for user in args.users_filter.split(',') ]
    param['start_date'] = args.start_date
    
    if args.message_keywords_filter:
        param['message_keywords_filter'] = args.message_keywords_filter.split(',')

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

    param['notification']['footer'] = f"From {param['current_filename']} {param['channel_username'].lstrip('@')}"

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

    session_file: str = f"{param['channel_username'].lstrip('@')}_session"
    message_cache_file: str = f"{param['channel_username'].lstrip('@')}_messages.json"
    log(f"session_file: {session_file}")
    log(f"message_cache_file: {message_cache_file}")

    tm1 : datetime = datetime(datetime.now().year, datetime.now().month, datetime.now().day) + timedelta(days=-1)
    tm1 = tm1.astimezone(pytz.UTC)

    last_message_date: datetime = tm1
    processed_messages : List[Dict[str, Any]] = []
    seen_hashes : Set[str] = set()
    if os.path.exists(message_cache_file):
        with open(message_cache_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                message_data = json.loads(line)
                
                # json.dumps before converting datetime to type(datetime)
                json_str: str = json.dumps(message_data, ensure_ascii=False, sort_keys=True)
                message_hash: str = hashlib.sha256(json_str.encode('utf-8')).hexdigest()

                message_data['datetime'] = pytz.UTC.localize(arrow.get(message_data['datetime']).datetime.replace(tzinfo=None))

                if message_hash not in seen_hashes:
                    seen_hashes.add(message_hash)
                    processed_messages.append(message_data)

            processed_messages = sorted(processed_messages, key=lambda m: m['datetime'])
            last_message_date = processed_messages[-1]['datetime']
            
    redis_client: Optional[StrictRedis] = init_redis_client()
        
    start_date: Optional[datetime] = None
    if param.get('start_date'):
        try:
            start_date = datetime.strptime(param['start_date'], '%Y-%m-%d').replace(tzinfo=pytz.UTC)
            log(f"Fetching messages from {param['start_date']} onward", LogLevel.INFO)
        except ValueError as e:
            log(f"Invalid start_date format: {str(e)}. Defaulting to current time.", LogLevel.WARNING)
    
    offset_date : datetime = start_date if start_date else last_message_date + timedelta(minutes=1)

    async with TelegramClient(session_file, param['api_id'], param['api_hash']) as client:
        try:
            if not await client.is_user_authorized():
                try:
                    await client.start(phone=param['phone'])
                except SessionPasswordNeededError:
                    password: str = input("Two-factor authentication enabled. Enter your password: ")
                    await client.start(phone=param['phone'], password=password)
                except FloodWaitError as e:
                    log(f"Flood wait error: Please wait {e.seconds} seconds", LogLevel.ERROR)
                    return
                except Exception as e:
                    log(f"Authorization failed: {str(e)}", LogLevel.ERROR)
                    return
            try:
                channel: Any = await client.get_entity(param['channel_username'])
                log(f"Connected to channel: {channel.title}", LogLevel.INFO)
            except Exception as e:
                log(f"Failed to access channel {param['channel_username']}: {str(e)}", LogLevel.ERROR)
                return
            
            last_message_date: datetime = offset_date
            oldest_message, newest_message = None, None # type: ignore
            while True:
                tm1 = datetime(datetime.now().year, datetime.now().month, datetime.now().day) + timedelta(days=-1)
                tm1 = tm1.astimezone(pytz.UTC)

                messages = []
                for username in param['users_filter']:
                    _messages = await client.get_messages(
                        channel,
                        limit=100,
                        from_user=username,
                        offset_date=last_message_date # offset_date is the cutoff
                    )
                    messages = messages + _messages
                log(f"Fetched {len(messages)} raw messages with offset_date={last_message_date.isoformat()}", LogLevel.INFO)
                
                '''
                Sliding Window: way we increment 'last_message_date' (The cutoff), it's possible we miss some messages.
                However, if we're moving the sliding window too slowly, it'd take forever to scan.
                And if you hit their API too frequently: 
                    Sleeping for 20s (0:00:20) on GetHistoryRequest flood wait
                TG won't ban your account, but still you'd need to wait.
                Adjust how fast you increment below to suit your purpose.
                '''
                realtime_cutoff = (datetime.now() + timedelta(minutes=-3)).astimezone(pytz.UTC)
                relevant_messages = [ msg for msg in messages if (msg.sender.username.lower().strip() if msg.sender and msg.sender.username else str(msg.sender_id)) in param['users_filter'] ]
                if not relevant_messages:
                    last_message_date = last_message_date + timedelta(hours=1) 
                    if last_message_date>realtime_cutoff:
                        last_message_date = realtime_cutoff
                    continue
                else:
                    sorted_messages = sorted(messages, key=lambda m: m.date)
                    last_message_date = sorted_messages[-1].date + timedelta(minutes=5) if sorted_messages[-1].date + timedelta(minutes=5)>last_message_date else last_message_date + timedelta(hours=1)
                    if last_message_date>realtime_cutoff:
                        last_message_date = realtime_cutoff
                
                for message in sorted_messages:  # Process oldest to newest
                    if not isinstance(message, Message):
                        continue
                    sender = await message.get_sender() # type: ignore
                    sender_name: Union[str, int] = sender.username if sender and sender.username else message.sender_id # type: ignore
                    sender_name = str(sender_name).lower().strip()
                    message_date: datetime = message.date # type: ignore
                    if message_date.tzinfo is None:
                        message_date = pytz.UTC.localize(message_date)
                    else:
                        message_date = message_date.astimezone(pytz.UTC)

                    message_text: str = message.message or ""
                    message_text = re.sub(r'[^a-zA-Z0-9\s.!?]', '', message_text)
                    message_text = message_text.replace(',', '')

                    message_data: Dict[str, Any] = {
                        "timestamp_ms": int(message_date.timestamp() * 1000),
                        "datetime": message_date.isoformat(), # Always in UTC
                        "sender": sender_name,
                        "message": message_text
                    }

                    json_str: str = json.dumps(message_data, ensure_ascii=False, sort_keys=True)
                    message_hash: str = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
                    
                    if (
                        (
                            not param['users_filter']
                            or (param['users_filter'] and sender_name in param['users_filter'])
                        )
                        and message_hash not in seen_hashes
                    ):
                        seen_hashes.add(message_hash)
                        processed_messages.append(message_data)

                        with open(message_cache_file, 'a', encoding='utf-8') as f:
                            json.dump(message_data, f, ensure_ascii=False)
                            f.write('\n')

                        if (
                            param['message_keywords_filter']
                            and any(keyword.lower().strip() in message_text.lower() for keyword in param['message_keywords_filter'])
                        ):
                            if param['alert_wav_path'] and message_date>=tm1 and sys.platform == 'win32':
                                import winsound
                                for _ in range(5):
                                    winsound.PlaySound(param['alert_wav_path'], winsound.SND_FILENAME)
                                    log(f"Incoming! {message_data}")

                            if redis_client:
                                try:
                                    publish_topic = f"{param['mds']['topics']['tg_alert']}_{message.id}"
                                    redis_client.publish(publish_topic, json_str)
                                    redis_client.setex(message_hash, param['mds']['redis']['ttl_ms'] // 1000, json_str)
                                    log(f"Published message {message.id} to Redis topic {publish_topic}", LogLevel.INFO)
                                except Exception as e:
                                    log(f"Failed to publish to Redis: {str(e)}", LogLevel.ERROR)
                
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
                else:
                    log(f"No messages processed in this iteration. last_message_date: {last_message_date}", LogLevel.INFO)
                    last_message_date = last_message_date + timedelta(days=1)
                await asyncio.sleep(1)
        except Exception as e:
            log(f"Oops {str(e)} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}", LogLevel.ERROR)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Stopped by user", LogLevel.INFO)
    except Exception as e:
        log(f"Unexpected error: {str(e)}", LogLevel.ERROR)