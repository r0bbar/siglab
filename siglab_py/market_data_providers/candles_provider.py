import sys
import traceback
from enum import Enum
import argparse
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Union
import logging
import json
from tabulate import tabulate
import asyncio
from threading import Thread
from collections import defaultdict
import pandas as pd
import numpy as np
from redis import StrictRedis
from redis.client import PubSub

from ccxt.binance import binance
from ccxt.okx import okx
from ccxt.bybit import bybit
from ccxt.base.exchange import Exchange

from siglab_py.util.market_data_util import fetch_candles, get_old_ticker, get_ticker_map


'''
To start from command prompt:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python candles_provider.py --provider_id aaa --candle_size 1h --how_many_candles 2169 --redis_ttl_ms 3600000

This script is pypy compatible:
    pypy candles_provider.py --provider_id aaa --candle_size 1h --how_many_candles 2169 --redis_ttl_ms 3600000

Key parameters you may want to modify:
    provider_id: You can trigger this provider instance using trigger_provider.py. Of course, you'd write your own.
    candle_size: 1m, 5m, 15min, 1h, 1d for example.
    how_many_candles: default to 2169 (24 x 90). 
    redis_ttl_ms: This is how long orderbook snapshot will last on redis when provider publishes to it.

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
                            "--provider_id", "YourProviderNameHere",
                            "--candle_size", "1h",
                            "--how_many_candles", "2169",
                            "--redis_ttl_ms", "3600000"
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
    "candle_size" : "1h",
    'how_many_candles' : 24*90,

    'market_type' : 'linear', # For spots, set to "spot". For perpectual, you need to look at ccxt doc, for most exchanges, it's 'linear' or 'swap' for perpetuals. Example, https://github.com/ccxt/ccxt/blob/master/python/ccxt/okx.py?plain=1#L1110
    
    # Provider ID is part of mds publish topic. 
    'provider_id' : 'b0f1b878-c281-43d7-870a-0347f90e6ece',

    # Ticker change
    'ticker_change_map' : '.\\siglab_py\\ticker_change_map.json',

    "loop_freq_ms" : 1000, # reduce this if you need trade faster

    # Publish to message bus
    'mds' : {
        'topics' : {
            'partition_assign_topic' : 'mds_assign_$PROVIDER_ID$',
            'candles_publish_topic' : 'candles-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$'
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
# fh = logging.FileHandler(f"{param['job_name']}.log")
# fh.setLevel(log_level)
# fh.setFormatter(formatter)     
# logger.addHandler(fh)

market_type : str = param['market_type'] 

binance_exchange = binance({
    'defaultType' : market_type
})

okx_exchange = okx({
    'defaultType' : market_type
})

bybit_exchange = bybit({
    'defaultType' : market_type
})

exchanges = {
    f"binance" : binance_exchange,
    f"okx" : okx_exchange,
    f"bybit" : bybit_exchange
}

def log(message : str, log_level : LogLevel = LogLevel.INFO):
    if log_level.value<LogLevel.WARNING.value:
        logger.info(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(f"{datetime.now()} {message}")

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore

    parser.add_argument("--provider_id", help="candle_provider will go to work if from redis a matching topic partition_assign_topic with provider_id in it.", default=None)
    parser.add_argument("--candle_size", help="candle interval: 1m, 1h, 1d... etc", default='1h')
    parser.add_argument("--how_many_candles", help="how_many_candles", default=24*7)

    parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: 1000*60*60 (i.e. 1hr)",default=1000*60*60)
    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=1000)

    args = parser.parse_args()
    if args.provider_id:
        param['provider_id'] = args.provider_id
    param['candle_size'] = args.candle_size
    param['how_many_candles'] = int(args.how_many_candles)

    param['redis_ttl_ms'] = int(args.redis_ttl_ms)
    param['loop_freq_ms'] = int(args.loop_freq_ms)

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
    
def init_redis_channel_subscription(redis_client : StrictRedis, partition_assign_topic : str) -> PubSub:
    redis_client = init_redis_client()
    pubsub = redis_client.pubsub()
    pubsub.subscribe(partition_assign_topic)
    return pubsub

def process_universe(
    universe : pd.DataFrame, 
    ticker_change_map,
    task,
    redis_client : StrictRedis
    ):
    # Key = ticker
    subscribed : Dict[tuple, Dict[str, Any]]  = defaultdict(lambda : {'candles': None, 'num_candles': 0} )

    while task.keep_running:
        start = time.time()
        num_fetches_this_wave = 0

        try:
            i = 1
            for index, row in universe.iterrows():
                exchange_name : str = row['exchange']
                ticker : str = row['ticker']
                
                try:
                    _ticker : str = ticker # In case ticker change
                    if ticker_change_map:
                        old_ticker= get_old_ticker(_ticker, ticker_change_map)
                        if old_ticker:
                            ticker_change_mapping = get_ticker_map(reference_ticker, ticker_change_map)
                            ticker_change_cutoff_sec = int(ticker_change_mapping['cutoff_ms']) / 1000
                            if datetime.now().timestamp()<ticker_change_cutoff_sec:
                                _ticker = old_ticker
                    
                    this_row_header = f'({i} of {universe.shape[0]}) exchange_name: {exchange_name}, ticker: {_ticker}'
                    
                    exchange = exchanges[exchange_name]

                    exchange.load_markets() # in case ticker change after gateway startup

                    fetch_again = False
                    last_fetch = None
                    last_fetch_ts = None
                    if subscribed[(exchange_name, _ticker)]:
                        last_fetch = subscribed[(exchange_name, _ticker)]['candles']
                        if subscribed[(exchange_name, _ticker)]['num_candles']>0:
                            last_fetch_ts = last_fetch.iloc[-1]['timestamp_ms']/1000 # type: ignore Otherwise, Error: Cannot access attribute "iloc" for class "None"
                    candle_size = param['candle_size']
                    interval = candle_size[-1]
                    num_intervals_per_candle = int(candle_size.replace(interval,""))
                    number_intervals = param['how_many_candles']
                    
                    start_date : datetime = datetime.now()
                    end_date : datetime = start_date
                    if interval=="m":
                        end_date = datetime.now()
                        end_date = datetime(end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, 0)
                        start_date = end_date + timedelta(minutes=-num_intervals_per_candle*number_intervals) 

                        num_sec_since_last_fetch = (end_date.timestamp() - last_fetch_ts) if last_fetch_ts else sys.maxsize
                        fetch_again = True if num_sec_since_last_fetch >= 60 / 10 else False

                    elif interval=="h":
                        end_date = datetime.now()
                        end_date = datetime(end_date.year, end_date.month, end_date.day, end_date.hour, 0, 0)
                        start_date = end_date + timedelta(hours=-num_intervals_per_candle*number_intervals) 

                        num_sec_since_last_fetch = (end_date.timestamp() - last_fetch_ts) if last_fetch_ts else sys.maxsize
                        fetch_again = True if num_sec_since_last_fetch >= 60*60 / 10 else False

                    elif interval=="d":
                        end_date = datetime.now()
                        end_date = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0)
                        start_date = end_date + timedelta(days=-num_intervals_per_candle*number_intervals) 

                        num_sec_since_last_fetch = (end_date.timestamp() - last_fetch_ts) if last_fetch_ts else sys.maxsize
                        fetch_again = True if num_sec_since_last_fetch >= 24*60*60 / 10 else False
                    
                    cutoff_ts = int(start_date.timestamp()) # in seconds
                    
                    if fetch_again:
                        if datetime.now().minute==0:
                            time.sleep(10) # Give some time for the exchange

                        candles = fetch_candles(
                                                    start_ts=cutoff_ts, 
                                                    end_ts=int(end_date.timestamp()), 
                                                    exchange=exchange, normalized_symbols=[_ticker], 
                                                    candle_size = candle_size, 
                                                    num_candles_limit = 100,
                                                    logger = None
                                                )
                        subscribed[(exchange_name, _ticker)] = {
                            'candles' : candles[ _ticker ],
                            'num_candles' : candles[ _ticker ].shape[0] # type: ignore Otherwise, Error: "shape" is not a known attribute of "None"
                        }
                        num_fetches_this_wave += 1

                        denormalized_ticker = next(iter([ exchange.markets[x] for x in exchange.markets if exchange.markets[x]['symbol']==_ticker]))['id']

                        publish_key = param['mds']['topics']['candles_publish_topic']
                        publish_key = publish_key.replace('$DENORMALIZED_SYMBOL$', denormalized_ticker)
                        publish_key = publish_key.replace('$EXCHANGE_NAME$', exchange_name)
                        publish_key = publish_key.replace('$INTERVAL$', param['candle_size'])

                        data = candles[_ticker].to_json(orient='records') # type: ignore Otherwise, Error: "to_json" is not a known attribute of "None"
                        
                        start = time.time()
                        if redis_client:
                            '''
                            https://redis.io/commands/set/
                            '''
                            expiry_sec : int = 0
                            if interval=="m":
                                expiry_sec = 60 + 60*15
                            elif interval=="h":
                                expiry_sec = 60*60 + 60*15
                            elif interval=="d":
                                expiry_sec = 60*60*24 
                            expiry_sec += 60*15 # additional 15min

                            redis_client.set(name=publish_key, value=json.dumps(data).encode('utf-8'), ex=expiry_sec)

                            redis_set_elapsed_ms = int((time.time() - start) *1000)

                            log(f"published candles {candles[_ticker].shape[0]} rows. {this_row_header} {publish_key} {sys.getsizeof(data, -1)} bytes to mds elapsed {redis_set_elapsed_ms} ms")

                except Exception as loop_error:
                    log(f"Failed to process {this_row_header}. Error: {loop_error} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")

                if not task.keep_running:
                    break

                i += 1

            if num_fetches_this_wave>0:
                log(f"Fetch candles for whole universe done.  elapsed: {time.time()-start} sec, universe_reload_id: {task.universe_reload_id}. # tickers: {len(subscribed)}")
            else:
                log(f"universe_reload_id: {task.universe_reload_id}, Nothing to fetch this wave. Sleep a bit.")
                
        finally:
            time.sleep(int(param['loop_freq_ms']/1000))
                
    log(f"process_universe exit, universe_reload_id: {task.universe_reload_id}")
    
async def main():
    parse_args()
    
    param['job_name'] = f'candles_provider_{param["provider_id"]}'

    ticker_change_map = None
    with open(param['ticker_change_map'], 'r', encoding='utf-8') as f:
        ticker_change_map : List[Dict[str, Union[str, int]]] = json.load(f)
        log(f"ticker_change_map loaded from {param['ticker_change_map']}")
        log(json.dumps(ticker_change_map))

    redis_client : StrictRedis = init_redis_client()
    partition_assign_topic : str = param['mds']['topics']['partition_assign_topic']
    partition_assign_topic = partition_assign_topic.replace("$PROVIDER_ID$", param['provider_id'])
    redis_pubsub : PubSub = init_redis_channel_subscription(redis_client, partition_assign_topic)

    class ThreadTask:
        def __init__(self, universe_reload_id) -> None:
            self.keep_running = True
            self.universe_reload_id = universe_reload_id
    task = None

    log(f"candles_provider {param['provider_id']} started, waiting for trigger. (Can use trigger_provider.py to trigger it)")

    universe_reload_id = 1
    for message in redis_pubsub.listen():
        if message['type'] == 'message' and message['channel'].decode()==partition_assign_topic:
            if task:
                task.keep_running = False

            tickers = json.loads(message['data'].decode('utf-8'))
            tickers = [ { 'exchange' : x.split('|')[0], 'ticker' : x.split('|')[-1] } for x in tickers ]
            universe = pd.DataFrame(tickers)
            logger.info(f"{partition_assign_topic} {message}")

            task = ThreadTask(universe_reload_id=universe_reload_id)
            t = Thread(target=process_universe, args = (universe, ticker_change_map, task, redis_client))
            t.start()

            universe_reload_id += 1

async def _run_jobs():
    await main()
asyncio.get_event_loop().run_until_complete(_run_jobs())