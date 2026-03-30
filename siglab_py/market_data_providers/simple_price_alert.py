import asyncio
import os
import sys
import traceback
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import time
import arrow
from dateutil import parser
import argparse
import json
from enum import Enum
import feedparser
import pandas as pd
from pprint import pformat
from redis import StrictRedis

from siglab_py.util.market_data_util import instantiate_exchange
from siglab_py.util.market_data_util import fetch_candles
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
log_level: int = logging.INFO
logger.setLevel(log_level)
format_str: str = '%(asctime)s %(message)s'
formatter: logging.Formatter = logging.Formatter(format_str)
sh: logging.StreamHandler = logging.StreamHandler()
sh.setLevel(log_level)
sh.setFormatter(formatter)
logger.addHandler(sh)
fh = logging.FileHandler(f"simple_price_alert.log")
fh.setLevel(log_level)
fh.setFormatter(formatter)     
logger.addHandler(fh)

'''
Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python simple_price_alert.py --tickers "BTC/USDC:USDC,SOL/USDC:USDC,ETH/USDC:USDC,XAU/USDC:USDC,WTI/USDC:USDC,QQQ/USDC:USDC,SPY/USDC:USDC" --exchange_name lighter --candle_size 5m --how_many_candles 288 --num_std 3  --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx
    
    --tickers: comma separated list of tickers you want to monitor
    --exchange_name: defaults to Lighter (https://app.lighter.xyz/trade)
    --candle_size: For example 1m, 5m, 15m, 1h ...
    --how_many_candles: how many candles to fetch. For example a day's worth of 5m candles = 12 5m candles per hr x 24 hours per day = 288 
    --num_std: Default to 3 - this is the threshold before slack notifications dispatched
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
                    "--tickers" , "BTC/USDC:USDC,SOL/USDC:USDC,ETH/USDC:USDC,XAU/USDC:USDC,WTI/USDC:USDC,QQQ/USDC:USDC,SPY/USDC:USDC",
                    "--exchange_name", "lighter",
                    "--candle_size", "5m",
                    "--how_many_candles", "288",
                    "--num_std",  "3",
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
    'loop_freq_ms' : 1000, 
    'current_filename' : current_filename,

    'alert_wav_path' : r"d:\sounds\terrible.wav",
    "num_shouts" : 5, # How many times 'alert_wav_path' is played

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
            'price_alert': 'trading.price.abnormal_price_movement'
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
    parser.add_argument("--tickers", help="Comma seperated list, example: BTC/USDC:USDC,ETH/USDT:USDT,XRP/USDT:USDT ", default="BTC/USDC:USDC,SOL/USDC:USDC,ETH/USDC:USDC,XAU/USDC:USDC,WTI/USDC:USDC,QQQ/USDC:USDC,SPY/USDC:USDC")
    
    parser.add_argument("--exchange_name", help="Exchange to monitor, default is Lighter DEX", default='lighter')
    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
    parser.add_argument("--default_sub_type", help="default_sub_type", default=None)
    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)

    parser.add_argument("--candle_size", help="candle interval: 1m, 1h, 1d... etc", default='1h')
    parser.add_argument("--how_many_candles", help="how_many_candles", default=24*7)
    parser.add_argument("--num_std", help="Number of standard deviation before sending alert? Default: 3", default=3)

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)
    parser.add_argument("--enable_notification", help="Enable notification on new headline arrival? Y or N (default).", default='Y')

    args = parser.parse_args()
    
    param['tickers'] = args.tickers.split(',')

    param['exchange_name'] = args.exchange_name
    param['default_type'] = args.default_type
    param['default_sub_type'] = args.default_sub_type
    param['rate_limit_ms'] = int(args.rate_limit_ms)

    param['candle_size'] = args.candle_size
    param['how_many_candles'] = int(args.how_many_candles)
    param['num_std'] = float(args.num_std)

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

    if args.enable_notification:
        if args.enable_notification=='Y':
            param['enable_notification'] = True
        else:
            param['enable_notification'] = False
    else:
        param['enable_notification'] = True

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

    try:        
        redis_client : Optional[StrictRedis] = init_redis_client()
    except Exception as redis_err:
        redis_client = None
        logger.info(f"Failed to connect to redis. Still run but not publishing to it. {redis_err}")

    exchange : Union[AnyExchange, None] = instantiate_exchange(
        exchange_name=param['exchange_name'],
        api_key=None,
        secret=None,
        passphrase=None,
        default_type=param['default_type'],
        rate_limit_ms=param['rate_limit_ms'],
    )
    markets = exchange.load_markets()

    candle_size = param['candle_size']
    interval = candle_size[-1]
    num_intervals_per_candle = int(candle_size.replace(interval,""))
    number_intervals = param['how_many_candles']

    if interval=="m":
        end_date = datetime.now()
        end_date = datetime(end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, 0)
        start_date = end_date + timedelta(minutes=-num_intervals_per_candle*number_intervals) 

    elif interval=="h":
        end_date = datetime.now()
        end_date = datetime(end_date.year, end_date.month, end_date.day, end_date.hour, 0, 0)
        start_date = end_date + timedelta(hours=-num_intervals_per_candle*number_intervals) 

    elif interval=="d":
        end_date = datetime.now()
        end_date = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0)
        start_date = end_date + timedelta(days=-num_intervals_per_candle*number_intervals)

    cutoff_ts = int(start_date.timestamp()) # in seconds

    loop_counter : int = 0
    while True:
        try:
            start_ts_sec = time.time()
            candles = fetch_candles(
                                                start_ts=cutoff_ts, 
                                                end_ts=int(end_date.timestamp()), 
                                                exchange=exchange, normalized_symbols=param['tickers'], 
                                                candle_size = candle_size, 
                                                num_candles_limit = 100,
                                                logger = None
                                            )
            elapsed_ms = int((time.time() - start_ts_sec) *1000)
            logger.info(f"[loop# {loop_counter}] candles fetch elapsed_ms: {elapsed_ms:,}")
            for ticker in param['tickers']:
                pd_candles = candles[ticker]

                current_candle = pd_candles.iloc[-1]
                dt_local = current_candle['datetime']
                dt_local = dt_local.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_ms = int(current_candle['timestamp_ms'])
                open = float(current_candle['open'])
                close = float(current_candle['close'])

                ob = exchange.fetch_order_book(ticker, limit=1)
                best_bid = ob['bids'][0][0]
                best_ask = ob['asks'][0][0]
                mid = (best_bid + best_ask)/2

                move = mid - open
                move_bps = round(move/open * 1_00_00, 2)

                pd_candles['candle_height'] = pd_candles['high'] - pd_candles['low']
                pd_candles['candle_body_height'] = pd_candles['close'] - pd_candles['open']

                candle_height_std = round(float(pd_candles['candle_height'].std()),  4)
                candle_height_mean = round(float(pd_candles['candle_height'].mean()),  4)
                candle_body_height_std = round(float(pd_candles['candle_body_height'].std()),  4)
                candle_body_height_mean = round(float(pd_candles['candle_body_height'].mean()),  4)

                ticker_price_movement_info = {
                    'ticker' : ticker,
                    'candle_dt_local' : dt_local,
                    'candle_timestamp_ms' : timestamp_ms,
                    'dt_local' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'timestamp_ms' : int(datetime.now().timestamp()),
                    'mid' : mid,
                    'open' : open,
                    'close' : close,
                    'move' : move,
                    'move_bps' : move_bps,
                    'candle_height_std' : candle_height_std,
                    'candle_height_mean' : candle_height_mean,
                    'num_std' : param['num_std']
                }
                logger.info(f"{pformat(ticker_price_movement_info, indent=2, width=100)}")
                if abs(move)>=(param['num_std'] * candle_height_std):
                    if param['enable_notification']:
                        dispatch_notification(title=f'#abnormal_price_move {ticker}', message=ticker_price_movement_info, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                    if redis_client:
                        try:
                            json_str = json.dumps(ticker_price_movement_info)

                            publish_topic = param['mds']['topics']['price_alert']
                            redis_client.publish(publish_topic, json_str)
                            redis_client.setex(publish_topic, param['mds']['redis']['ttl_ms'] // 1000, json_str)
                            logger.info(f"Published filtered price_alert to Redis topic {publish_topic}")

                        except Exception as e:
                            logger.info(f"Failed to publish to Redis: {str(e)}")
                            
                    if param['alert_wav_path'] and sys.platform == 'win32' and os.path.exists(param['alert_wav_path']):
                        import winsound
                        for _ in range(param['num_shouts']):
                            winsound.PlaySound(param['alert_wav_path'], winsound.SND_FILENAME)

            elapsed_ms = int((time.time() - start_ts_sec) *1000)
            logger.info(f"[loop# {loop_counter}] end to end elapsed_ms: {elapsed_ms:,}")

        except Exception as loop_err:
            err_msg = f'loop error {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}'
            logger.error(err_msg)
            dispatch_notification(title=f"{param['current_filename']} error. {_ticker}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)
                
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