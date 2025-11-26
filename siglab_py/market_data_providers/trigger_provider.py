from enum import Enum
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Union
import logging
import json
import pandas as pd
import numpy as np
from redis import StrictRedis
from redis.client import PubSub

'''
set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
python trigger_provider.py --provider_id aaa --tickers BTC/USDC:USDC,ETH/USDC:USDC,SOL/USDC:USDC
'''

param : Dict[str, str|List[str]] = {
    'provider_id' : '---'
}

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore
    parser.add_argument("--provider_id", help="candle_provider will go to work if from redis a matching topic partition_assign_topic with provider_id in it.", default=None)
    parser.add_argument("--tickers", help="Ticker(s) you're trading, comma separated list. Example BTC/USDC:USDC,ETH/USDC:USDC,SOL/USDC:USDC", default=None)

    args = parser.parse_args()
    param['provider_id'] = args.provider_id

    tickers = args.tickers.split(',')
    assert(len(tickers)>0)
    param['tickers'] = [ ticker.strip() for ticker in tickers ]

def init_redis_client() -> StrictRedis:
    redis_client : StrictRedis = StrictRedis(
                    host = 'localhost',
                    port = 6379,
                    db = 0,
                    ssl = False
                )
    try:
        redis_client.keys()
    except ConnectionError as redis_conn_error:
        err_msg = f"Failed to connect to redis: {'localhost'}, port: {6379}"
        raise ConnectionError(err_msg)
    
    return redis_client

def trigger_producers(
    redis_client : StrictRedis, 
    exchange_tickers : List, 
    candles_partition_assign_topic : str):
    # https://redis.io/commands/publish/
    redis_client.publish(channel=candles_partition_assign_topic, message=json.dumps(exchange_tickers).encode('utf-8'))

if __name__ == '__main__':
    parse_args()

    provider_id : str = param['provider_id'] # type: ignore
    partition_assign_topic = 'mds_assign_$PROVIDER_ID$'
    candles_partition_assign_topic = partition_assign_topic.replace("$PROVIDER_ID$", provider_id)
    redis_client : StrictRedis = init_redis_client()

    exchange_tickers : List[str] = param['tickers'] # type: ignore
    trigger_producers(
        redis_client=redis_client,
        exchange_tickers=exchange_tickers,
        candles_partition_assign_topic=candles_partition_assign_topic)

    print(f"Sent {exchange_tickers}")

