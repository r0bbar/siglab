import sys
import traceback
import os
import logging
import argparse
from datetime import datetime, timezone
import time
from typing import Dict, Any, Union
import json
from redis import StrictRedis

from util.market_data_util import instantiate_exchange
from util.notification_util import dispatch_notification

from siglab_py.constants import LogLevel # type: ignore

current_filename = os.path.basename(__file__)

'''
Usage:
    python aggressive_move_alert.py --exchange_name hyperliquid --default_type linear --tickers BTC/USDC:USDC,ETH/USDC:USDC --candle_size 1h --threshold_bps 500 --num_intervals 24 --reversal_num_intervals 2 --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx

    exchange_name: where you source market data from?
    default_type: default_type: spot, linear, inverse, futures ...etc. The convention comes from CCXT https://docs.ccxt.com/en/latest/manual.html#instantiation
    tickers: What do you want to monitor?
    candle_size: 1m, 1h, 1d ... etc
    threshold_bps: Level above which price swing is considered "Aggressive moves"
    num_intervals: Number of intervals. If num_intervals=24 and candle_size=1h, then sliding window size is 1 day.
    reversal_num_intervals: Say reversal_num_intervals=2. If two reversal candles is detected, it'd fire off another alert.

    slack_info_url, slack_critial_url, slack_alert_url: How to get Slack webhook Urls? Please refer to slack_dispatch_notification.py.

    This script is pypy compatible.

Debug from VSCode, launch.json:

    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python Debugger: Current File",
                "type": "debugpy",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "args" : [
                        "--exchange_name", "hyperliquid",
                        "--default_type", "linear",
                        "--rate_limit_ms", "100",

                        "--tickers", "BTC/USDC:USDC,ETH/USDC:USDC",
                        "--candle_size", "1h",
                        "--threshold_bps", "500",
                        "--num_intervals", "24",
                        "--reversal_num_intervals" : , "2",

                        "--slack_info_url", "https://hooks.slack.com/services/xxx",
                        "--slack_critial_url", "https://hooks.slack.com/services/xxx",
                        "--slack_alert_url", "https://hooks.slack.com/services/xxx",
                    ],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            }
        ]
    }
'''
param : Dict = {
    "loop_freq_ms" : 10000, # reduce this if you need trade faster

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

    'mds' : {
        'topics' : {
            
        },
        'redis' : {
            'host' : 'localhost',
            'port' : 6379,
            'db' : 0,
            'ttl_ms' : 1000*60*15 # 15 min?
        }
    }
}

logging.Formatter.converter = time.gmtime
logger = logging.getLogger()
log_level = logging.INFO # DEBUG --> INFO --> WARNING --> ERROR
logger.setLevel(log_level)
format_str = '%(asctime)s %(message)s'
formatter = logging.Formatter(format_str)
sh = logging.StreamHandler()
sh.setLevel(log_level)
sh.setFormatter(formatter)
logger.addHandler(sh)

def log(message : str, log_level : LogLevel = LogLevel.INFO):
    if log_level.value<LogLevel.WARNING.value:
        logger.info(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(f"{datetime.now()} {message}")

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore

    parser.add_argument("--exchange_name", help="exchange_name", default=None)
    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)

    parser.add_argument("--tickers", help="Comma separated list of tickers to monitor. Example BTC/USDC:USDC,ETH/USDC:USDC", default=None)
    parser.add_argument("--candle_size", help="candle_size: 1m, 1h, 1d ...etc supported by the exchange you specified.", default='1h')
    parser.add_argument("--threshold_bps", help="What's 'aggressive move'? threshold defined in bps default 500bps", default=500)
    parser.add_argument("--num_intervals", help="Used in combinations with threshold_bps and candle_size in defining what an 'aggressive move' is.", default=100)
    parser.add_argument("--reversal_num_intervals", help="How many reversal candles to confirm reversal?", default=2)

    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=500)

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()

    param['exchange_name'] = args.exchange_name
    param['default_type'] = args.default_type
    param['rate_limit_ms'] = int(args.rate_limit_ms)

    param['tickers'] = args.tickers.split(',')
    param['candle_size'] = args.candle_size
    param['threshold_bps'] = int(args.threshold_bps)
    param['num_intervals'] = int(args.num_intervals)
    param['reversal_num_intervals'] = int(args.reversal_num_intervals)

    param['loop_freq_ms'] = int(args.loop_freq_ms)

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

    param['notification']['footer'] = f"From {param['current_filename']} {param['exchange_name']}"

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
        raise ConnectionError(err_msg)
    
    return redis_client

def _reversal(
        direction : str,  # up or down
        last_candles
    ) -> bool:
        if direction == "down" and all([ candle[1]<candle[4] for candle in last_candles ]):
            return True
        elif direction == "up" and all([ candle[1]>candle[4] for candle in last_candles ]):
            return True
        else:
            return False
parse_args()

notification_params : Dict[str, Any] = param['notification']

redis_client : StrictRedis = init_redis_client()

exchange = instantiate_exchange(
    exchange_name = param['exchange_name'],
    default_type = param['default_type']
    )

aggressive_move_data_cache : Dict[str, Any] = {}

while True:
    try:
        for ticker in param['tickers']:
            trailing_candles = exchange.fetch_ohlcv( # type: ignore
                symbol=ticker,
                timeframe=param['candle_size'],
                limit=param['num_intervals']
            )
            first_candle = trailing_candles[0]
            first_candle_ts_ms = first_candle[0]
            first_candle_utc_dt = datetime.fromtimestamp(int(first_candle_ts_ms/1000), tz=timezone.utc)
            first_candle_close = first_candle[4]
            last_candle = trailing_candles[-1]
            last_candle_ts_ms = last_candle[0]
            last_candle_utc_dt = datetime.fromtimestamp(int(last_candle_ts_ms/1000), tz=timezone.utc)
            last_candle_close = last_candle[4]

            data = {
                'ticker' : ticker,
                'first_candle_ts_ms' : first_candle_ts_ms,
                'first_candle_utc_dt' : first_candle_utc_dt.strftime("%Y%m%d %H:%M:%S"),
                'first_candle_close' : first_candle_close,
                'last_candle_ts_ms' : last_candle_ts_ms,
                'last_candle_utc_dt' : last_candle_utc_dt.strftime("%Y%m%d %H:%M:%S"),
                'last_candle_close' : last_candle_close
            }

            change_bps = (last_candle_close/first_candle_close-1)*10000 if last_candle_close>first_candle_close else -(first_candle_close/last_candle_close-1)*10000
            data['change_bps'] = change_bps
            if abs(change_bps)>=param['threshold_bps']:
                aggressive_move_trigger_price = first_candle_close
                aggressive_move_price_swing = last_candle_close - first_candle_close
                data['aggressive_move_trigger_price'] = aggressive_move_trigger_price
                data['aggressive_move_price_swing'] = aggressive_move_price_swing
                aggressive_move_data = data
                aggressive_move_data_cache[ticker] = aggressive_move_data

                dispatch_notification(
                    title=f"{param['current_filename']} {param['exchange_name']} Aggressive move detected. {ticker} change_bps: {change_bps}", 
                    message=aggressive_move_data, 
                    footer=param['notification']['footer'], 
                    params=notification_params, 
                    log_level=LogLevel.CRITICAL
                )
            
            log(json.dumps(data, indent=4), log_level=LogLevel.INFO)

            if ticker in aggressive_move_data_cache:
                aggressive_move_data = aggressive_move_data_cache[ticker]
                aggressive_move_trigger_price = aggressive_move_data['aggressive_move_trigger_price']

                reversal : bool = _reversal(
                    direction='up' if aggressive_move_price_swing>0 else 'down',
                    last_candles=trailing_candles[-param['reversal_num_intervals']:]
                )

                if reversal:
                    actual_change_bps = (last_candle_close/aggressive_move_trigger_price-1)*10000 if last_candle_close>aggressive_move_trigger_price else -(aggressive_move_trigger_price/last_candle_close-1)*10000
                    aggressive_move_data['actual_price_swing'] = last_candle_close - aggressive_move_trigger_price
                    aggressive_move_data['actual_change_bps'] = actual_change_bps

                    dispatch_notification(
                        title=f"{param['current_filename']} {param['exchange_name']} Aggressive move reversal detected. {ticker} change_bps: {actual_change_bps}", 
                        message=aggressive_move_data, 
                        footer=param['notification']['footer'], 
                        params=notification_params, 
                        log_level=LogLevel.CRITICAL
                    )

                    # Reset
                    aggressive_move_data_cache.pop(ticker)

    except Exception as loop_err:
        log(f"Error: {loop_err}", log_level=LogLevel.ERROR)
    finally:
        time.sleep(int(param['loop_freq_ms']/1000))
