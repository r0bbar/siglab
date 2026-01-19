import sys
import traceback
from enum import Enum
import argparse
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Union
import hashlib
from collections import deque
import logging
import json
from io import StringIO
import re
from re import Pattern
from tabulate import tabulate
import pandas as pd
import numpy as np
from redis import StrictRedis

from siglab_py.util.analytic_util import compute_candles_stats

'''
candles_provider.py will feed candles to redis. 
    key example: candles-BTC-USDT-SWAP-okx_linear-1h. 
    key format: candles-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$

candles_ta_provider.py will scan for candles-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$
Then read candles from redis. If candle's timestamp hasnt been previously processed, it'd perform TA calculations.
After perform TA calc, it'd publish back to redis under:
    key example: candles_ta-BTC-USDT-SWAP-okx_linear-1h
    key format: candles_ta-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$

From command prompt:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python candles_ta_provider.py --candle_size 1h --ma_long_intervals 24 --ma_short_intervals 8 --boillenger_std_multiples 2 --redis_ttl_ms 3600000 --processed_hash_queue_max_size 999 --pypy_compat N

This script is pypy compatible but you'd need specify --pypy_compat Y so from analyti_util we'd skip import scipy and statsmodels (They are not pypy compatible).
    pypy candles_ta_provider.py --candle_size 1h --ma_long_intervals 24 --ma_short_intervals 8 --boillenger_std_multiples 2 --redis_ttl_ms 3600000 --processed_hash_queue_max_size 999 --pypy_compat Y

Launch.json if you wish to debug from VSCode:
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
                                "--candle_size", "1h",
                                "--ma_long_intervals", "24",
                                "--ma_short_intervals", "8",
                                "--boillenger_std_multiples", "2",
                                "--redis_ttl_ms", "3600000",
                                "--processed_hash_queue_max_size", "999"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
'''
class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

param : Dict = {
    "candle_size" : "1h",   # This instance candles_ta_provider will only process candles with interval specified by "candle_size"

    # For MA (Moving Averages), window sizes are defined by below.
    "ma_long_intervals" : 24,
    'ma_short_intervals' : 8,
    "boillenger_std_multiples" : 2,
    
    # regex corresponding to candles_publish_topic. If you want specific instances to process specific tickers only (performance concerns), you can use this regex filter to do the trick.
    "candles_ta_publish_topic_regex" : r"^candles-[A-Z]+-[A-Z]+-[A-Z]+-[a-z_]+-\d+[smhdwMy]$", 

    # processed_hash_queue is how we avoid reprocess already processed messages. We store hash of candles read in 'processed_hash_queue'.
    # Depending on how many tickers this instance is monitoring, you may want to adjust this queue size.
    "processed_hash_queue_max_size" : 999,

    'job_name' : 'candles_ta_provider',

    # Publish to message bus
    'mds' : {
        'topics' : {
            'candles_publish_topic' : 'candles-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$', # candles_ta_provider will scan redis for matching keys
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

    parser.add_argument("--candle_size", help="candle interval: 1m, 1h, 1d... etc", default='1h')
    parser.add_argument("--ma_long_intervals", help="Window size in number of intervals for higher timeframe", default=24)
    parser.add_argument("--ma_short_intervals", help="Window size in number of intervals for lower timeframe", default=8)
    parser.add_argument("--boillenger_std_multiples", help="Boillenger bands: # std", default=2)
    parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: 1000*60*60 (i.e. 1hr)",default=1000*60*60)
    parser.add_argument("--processed_hash_queue_max_size", help="processed_hash_queue is how we avoid reprocess already processed messages. We store hash of candles read in 'processed_hash_queue'", default=999)

    parser.add_argument("--pypy_compatible", help="pypy_compatible: If Y, analytic_util will import statsmodels.api (slopes and divergence calc). In any case, partition_sliding_window requires scipy.stats.linregress and cannot be used with pypy. Y or N (default).", default='N')

    args = parser.parse_args()
    param['candle_size'] = args.candle_size
    param['ma_long_intervals'] = int(args.ma_long_intervals)
    param['ma_short_intervals'] = int(args.ma_short_intervals)
    param['boillenger_std_multiples'] = int(args.boillenger_std_multiples)

    param['redis_ttl_ms'] = int(args.redis_ttl_ms)
    param['processed_hash_queue_max_size'] = int(args.processed_hash_queue_max_size)

    if args.pypy_compatible:
        if args.pypy_compatible=='Y':
            param['pypy_compatible'] = True
        else:
            param['pypy_compatible'] = False
    else:
        param['pypy_compatible'] = False

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

def work(
    boillenger_std_multiples : float,
    ma_long_intervals : int,
    ma_short_intervals : int,
    candle_size : str,
    redis_client : StrictRedis
):
    candles_ta_publish_topic_regex : str = param['candles_ta_publish_topic_regex']
    candles_ta_publish_topic_regex_pattern : Pattern = re.compile(candles_ta_publish_topic_regex)

    # This is how we avoid reprocess same message twice. We check message hash and cache it.
    processed_hash_queue = deque(maxlen=10)

    while True:
        try:
            keys = redis_client.keys()
            for key in keys:
                try:
                    s_key : str = key.decode("utf-8")
                    if candles_ta_publish_topic_regex_pattern.match(s_key):

                        publish_key : str = s_key.replace('candles-', 'candles_ta-')

                        candles = None
                        message = redis_client.get(key)
                        if message:
                            # When candles_provider.py republish candles to same key (i.e. overwrites it), we'd know.
                            message_hash = hashlib.sha256(message).hexdigest()
                            message = message.decode('utf-8')
                            if message_hash not in processed_hash_queue: # Dont process what's been processed before.
                                processed_hash_queue.append(message_hash)

                                candles = json.loads(message)
                                pd_candles = pd.read_json(StringIO(candles), convert_dates=False)

                                log(f"sliding window size ma_long_intervals: {ma_long_intervals}, ma_short_intervals: {ma_short_intervals}")

                                start = time.time()
                                compute_candles_stats(
                                            pd_candles=pd_candles, 
                                            boillenger_std_multiples=boillenger_std_multiples, 
                                            sliding_window_how_many_candles=ma_long_intervals, 
                                            slow_fast_interval_ratio=(ma_long_intervals/ma_short_intervals),
                                            pypy_compat=param['pypy_compatible']
                                        )
                                compute_candles_stats_elapsed_ms = int((time.time() - start) *1000)
                                data = pd_candles.to_json(orient='records') # type: ignore Otherwise, Error: "to_json" is not a known attribute of "None"

                                start = time.time()
                                if redis_client:
                                    '''
                                    https://redis.io/commands/set/
                                    '''
                                    expiry_sec : int = 0
                                    if candle_size[-1]=="m":
                                        expiry_sec = 60 + 60*15
                                    elif candle_size[-1]=="h":
                                        expiry_sec = 60*60 + 60*15
                                    elif candle_size[-1]=="d":
                                        expiry_sec = 60*60*24 
                                    expiry_sec += 60*15 # additional 15min

                                    redis_client.set(name=publish_key, value=json.dumps(data).encode('utf-8'), ex=expiry_sec)
                                    redis_set_elapsed_ms = int((time.time() - start) *1000)

                                    log(f"published candles {pd_candles.shape[0]} rows. {publish_key} {sys.getsizeof(data, -1)} bytes to mds elapsed {redis_set_elapsed_ms} ms, compute_candles_stats_elapsed_ms: {compute_candles_stats_elapsed_ms}")
                            else:
                                log(f"{s_key} message with hash {message_hash} been processed previously.")


                except Exception as key_error:
                    log(f"Failed to process {key}. Error: {key_error} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")

        except Exception as loop_error:
            log(f"Error: {loop_error} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")

def main():
    parse_args()

    # candles_ta_provider instances go by 'candle_size'
    param['job_name'] = "".join([ param['job_name'], "", param['candle_size'] ])

    # fh = logging.FileHandler(f"{param['job_name']}.log")
    # fh.setLevel(log_level)
    # fh.setFormatter(formatter)     
    # logger.addHandler(fh)

    redis_client : StrictRedis = init_redis_client()
    work(
            boillenger_std_multiples=param['boillenger_std_multiples'],
            ma_long_intervals=param['ma_long_intervals'],
            ma_short_intervals=param['ma_short_intervals'],
            candle_size=param['candle_size'],
            redis_client=redis_client)

main()