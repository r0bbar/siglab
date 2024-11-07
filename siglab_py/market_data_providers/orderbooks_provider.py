import sys
import traceback
from enum import Enum
import argparse
import time
from datetime import datetime, timedelta
import operator
from typing import Any, Dict, Union, Mapping
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

from ccxt.base.exchange import Exchange
import ccxt.pro as ccxtpro 

'''
To start from command prompt:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab
    python orderbooks_provider.py --provider_id aaa --instance_capacity 25 --ts_delta_observation_ms_threshold 150 --ts_delta_consecutive_ms_threshold 150 --redis_ttl_ms 3600000

This script is pypy compatible.

Key parameters you may want to modify:
    provider_id: You can trigger this provider instance using test_provider.py. Of course, you'd write your own.
    instance_capacity: max # tickers this provider instance will handle.
    ts_delta_observation_ms_threshold: default to 150ms. "Observation Delta" is clock diff between orderbook timestamp, and your local server clock.
    ts_delta_consecutive_ms_threshold: default to 150ms. "Consecutive Delta" is time elapsed between consecutive orderbook updates.
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
                            "--instance_capacity", 25",
                            "--ts_delta_observation_ms_threshold", "150",
                            "--ts_delta_consecutive_ms_threshold","150",
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
    'market_type' : 'linear', # For spots, set to "spot". For perpectual, you need to look at ccxt doc, for most exchanges, it's 'linear' or 'swap' for perpetuals. Example, https://github.com/ccxt/ccxt/blob/master/python/ccxt/okx.py?plain=1#L1110
    
    # Provider ID is part of mds publish topic. 
    'provider_id' : 'ceaafe1d-e320-44ec-a959-da73edb9c4b1',

    "instance_capacity" : "25",
    
    # Keep track of latency issues
    # a) ts_delta_observation_ms: Keep track of server clock vs timestamp from exchange
    # b) ts_delta_consecutive_ms: Keep track of gap between consecutive updates
    'ts_delta_observation_ms_threshold' : 150,
    'ts_delta_consecutive_ms_threshold' : 150,

    # Publish to message bus
    'mds' : {
        'topics' : {
            'partition_assign_topic' : 'mds_assign_$PROVIDER_ID$',
            'candles_publish_topic' : 'orderbooks_$SYMBOL$_$EXCHANGE$'
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

exchange_params : Dict = {
        'newUpdates': False,
        'options' : {
            'defaultType' : 'swap' # spot, swap
        }
    }

async def instantiate_exhange(
        exchange_name : str,
        old_exchange : Union[Exchange, None]
        ) -> Exchange:
    if old_exchange:
        await old_exchange.close() # type: ignore Otherwise, Error:  Cannot access attribute "close" for class "Exchange  Attribute "close" is unknown
    if exchange_name==f"binance_{market_type}":
        exchange = ccxtpro.binance(exchange_params)
    elif exchange_name==f"okx_{market_type}":
        exchange = ccxtpro.okx(exchange_params)
    elif exchange_name==f"bybit_{market_type}":
        exchange = ccxtpro.bybit(exchange_params)
    else:
        exchange = ccxtpro.binance(exchange_params)
    exchange.name = exchange_name # type: ignore Otherwise, Error: Cannot assign to attribute "name" for class "binance" "str" is not assignable to "None"
    return exchange

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
    parser.add_argument("--instance_capacity", help="Instance capacity in num of tickers it can process. -1: No limit.", default=-1)
    parser.add_argument("--ts_delta_observation_ms_threshold", help="max threshold in ms: server clock vs update timestamp",default=param['ts_delta_observation_ms_threshold'])
    parser.add_argument("--ts_delta_consecutive_ms_threshold", help="max threshold in ms: gap between consecutive updates",default=param['ts_delta_consecutive_ms_threshold'])

    parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: 1000*60*60 (i.e. 1hr)",default=1000*60*60)

    args = parser.parse_args()
    if args.provider_id:
        param['provider_id'] = args.provider_id
    param['instance_capacity'] = int(args.instance_capacity)
    param['ts_delta_observation_ms_threshold'] = int(args.ts_delta_observation_ms_threshold)
    param['ts_delta_consecutive_ms_threshold'] = int(args.ts_delta_consecutive_ms_threshold)
    param['redis_ttl_ms'] = int(args.redis_ttl_ms)

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


class OrderBook:
    def __init__(
            self, 
            ticker : str, 
            exchange_name : str
            ) -> None:
        self.ticker : str = ticker
        self.exchange_name : str = exchange_name
        self.bids : Dict = {}
        self.asks : Dict = {}

        self.last_timestamp_ms : Union[int,None] = None
        self.timestamp : Union[int,None] = None # order book update timestamp in sec
        self.timestamp_ms : Union[int,None] = None # order book update timestamp in ms
        self.ts_delta_consecutive_ms : int = 0
        self.ts_delta_observation_ms : int = 0
        self.is_valid : bool = True

    def update_book(
                self, 
                update : Mapping,
                param : dict
                    ) -> None:
        update_ts_ms = update['timestamp'] 
        if len(str(update_ts_ms))==10:
            update_ts_ms = update_ts_ms*1000
        self.last_timestamp_ms = self.timestamp_ms
        self.timestamp_ms = int(update_ts_ms)
        self.timestamp = int(self.timestamp_ms/1000)

        '''
        Keep track of latency issues
        a) ts_delta_observation_ms: Keep track of server clock vs timestamp from exchange
        b) ts_delta_consecutive_ms: Keep track of gap between consecutive updates
        '''
        self.ts_delta_observation_ms = int(datetime.now().timestamp()*1000) - self.timestamp_ms
        self.ts_delta_consecutive_ms = self.timestamp_ms - self.last_timestamp_ms if self.last_timestamp_ms else 0

        self.is_valid = True
        if self.ts_delta_observation_ms>param['ts_delta_observation_ms_threshold']:
            self.is_valid = False
        if self.ts_delta_consecutive_ms>param['ts_delta_consecutive_ms_threshold']:
            self.is_valid = False

        self.bids.update((float(bid[0]), float(bid[1])) for bid in update.get('bids', []))
        self.asks.update((float(ask[0]), float(ask[1])) for ask in update.get('asks', []))
        self.bids = { key:val for key,val in self.bids.items() if val!=0}
        self.asks = { key:val for key,val in self.asks.items() if val!=0}
        if self.bids and self.asks:
            best_ask = float(min(self.asks.keys()))
            best_bid = float(max(self.bids.keys()))
            if best_ask<best_bid:
                raise ValueError(f"{self.exchange_name} {self.ticker} best bid >= best ask!?! How?")
        self.bids = dict(sorted(self.bids.items(), reverse=True))
        self.asks = dict(sorted(self.asks.items()))
    
    def to_dict(self):
        bids = ([float(price), float(amount)] for price, amount in self.bids.items() if float(amount))
        asks = ([float(price), float(amount)] for price, amount in self.asks.items() if float(amount))
        data = {
            "denormalized_ticker" : self.ticker,
            "exchange_name" : self.exchange_name,
            "bids" : sorted(bids, key=operator.itemgetter(0), reverse=True),
            "asks" : sorted(asks, key=operator.itemgetter(0)),
            "timestamp" : self.timestamp, # in sec
            "timestamp_ms" : self.timestamp_ms, # in ms (timestamp in update from exchange)
            'ts_delta_observation_ms' : self.ts_delta_observation_ms,
            'ts_delta_consecutive_ms' : self.ts_delta_consecutive_ms,
            "is_valid" : self.is_valid
        }

        data['best_ask'] = min(data['asks'])
        data['best_bid'] = min(data['bids'])
        return data
        
class ThreadTask:
        def __init__(self) -> None:
            self.keep_running = True

def handle_ticker(
    exchange_name : str,
    ticker : str,
    candles_publish_topic : str,
    redis_client : StrictRedis,
    task : ThreadTask
):
    asyncio.run(_handle_ticker(
            exchange_name=exchange_name,
            ticker=ticker,
            candles_publish_topic=candles_publish_topic,
            redis_client=redis_client,
            task=task
        )
    )

async def _handle_ticker(
    exchange_name : str,
    ticker : str,
    candles_publish_topic : str,
    redis_client : StrictRedis,
    task : ThreadTask
    ):
    exchange = await instantiate_exhange(exchange_name=exchange_name, old_exchange=None)
    ob = OrderBook(ticker=ticker, exchange_name=exchange_name)
    candles_publish_topic = candles_publish_topic.replace("$SYMBOL$", ticker)
    candles_publish_topic = candles_publish_topic.replace("$EXCHANGE$", exchange_name)
    
    local_server_timestamp_ms = datetime.now().timestamp()*1000
    exchange_timestamp_ms = await exchange.fetch_time() # type: ignore Otherwise, Error: Cannot access attribute "fetch_time" for class "Coroutine[Any, Any, binance | okx | bybit]"
    timestamp_gap_ms = local_server_timestamp_ms - exchange_timestamp_ms
    log(f"{exchange_name} {ticker} local_server_timestamp_ms: {local_server_timestamp_ms} vs exchange_timestamp_ms: {exchange_timestamp_ms}. timestamp_gap_ms: {timestamp_gap_ms}")

    while task.keep_running:
        try:
            update = await exchange.watch_order_book(ticker) # type: ignore Otherwise, Error: Cannot access attribute "watch_order_book" for class "Coroutine[Any, Any, binance | okx | bybit]"
            ob.update_book(update=update, param=param)

            ob_dict = ob.to_dict()

            redis_client.set(name=candles_publish_topic, value=json.dumps(ob_dict), ex=int(param['mds']['redis']['ttl_ms']/1000))

            ob_dict.pop('bids')
            ob_dict.pop('asks')
            pd_ob = pd.DataFrame(ob_dict)
            log(f"{tabulate(pd_ob, headers=pd_ob.columns)}") # type: ignore Otherwise, Error: Argument of type "DataFrame" cannot be assigned to parameter "tabular_data" of type "Mapping[str, Iterable[Any]]

        except ValueError as update_error:
            log(f"Update error! {update_error}")
            exchange = await instantiate_exhange(exchange_name=exchange_name, old_exchange=exchange) # type: ignore Otherwise, Error: Argument of type "Coroutine[Any, Any, Exchange] | Exchange" cannot be assigned to parameter "old_exchange" of type "Exchange | None" in function "instantiate_exhange"
            ob = OrderBook(ticker=ticker, exchange_name=exchange_name)

    
async def main():
    parse_args()
    
    param['job_name'] = f'candles_provider_{param["provider_id"]}'

    redis_client : StrictRedis = init_redis_client()
    partition_assign_topic : str = param['mds']['topics']['partition_assign_topic']
    partition_assign_topic = partition_assign_topic.replace("$PROVIDER_ID$", param['provider_id'])
    candles_publish_topic : str = param['mds']['topics']['candles_publish_topic']
    redis_pubsub : PubSub = init_redis_channel_subscription(redis_client, partition_assign_topic)

    log(f"orderbooks_provider {param['provider_id']} started, waiting for trigger. (Can use test_provider.py to trigger it)")
    
    tasks = []
    for message in redis_pubsub.listen():
        if message['type'] == 'message' and message['channel'].decode()==partition_assign_topic:
            if tasks:
                for task in tasks:
                    task.keep_running = False # Everytime provider get triggered, it'd cancel previous tasks

            tickers = json.loads(message['data'].decode('utf-8'))
            tickers = [ { 'exchange' : x.split('|')[0], 'ticker' : x.split('|')[-1] } for x in tickers ]
            logger.info(f"{partition_assign_topic} {message}")

            ticker_count : int = 1
            for entry in tickers:
                exchange_name : str = entry['exchange']
                ticker : str = entry['ticker']
                task : ThreadTask = ThreadTask()
                t = Thread(target=handle_ticker, args = (exchange_name, ticker, candles_publish_topic, redis_client, task))
                t.start()
                log(f"Task created for {exchange_name}, {ticker}")

                if ticker_count>=param['instance_capacity']:
                    break
                ticker_count+=1

asyncio.run(main())