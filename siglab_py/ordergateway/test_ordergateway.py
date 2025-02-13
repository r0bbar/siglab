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

from ordergateway.client import DivisiblePosition

'''
set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
python test_ordergateway.py --gateway_id bybit_01
'''

param : Dict[str, str] = {
    'gateway_id' : '---'
}

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore
    parser.add_argument("--gateway_id", help="gateway_id: Where are you sending your order?", default=None)

    args = parser.parse_args()
    param['gateway_id'] = args.gateway_id

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

def execute_positions(
    redis_client : StrictRedis, 
    positions : List[DivisiblePosition], 
    ordergateway_topic : str):
    # https://redis.io/commands/publish/
    _positions = [ position.to_dict() for position in positions ]
    redis_client.set(name=ordergateway_topic, value=json.dumps(_positions).encode('utf-8'), ex=60*15)

    print(f"{ordergateway_topic}: Orders sent {_positions}.")

if __name__ == '__main__':
    parse_args()

    gateway_id : str = param['gateway_id']
    ordergateway_topic = 'ordergateway_$GATEWAY_ID$'
    ordergateway_topic = ordergateway_topic.replace("$GATEWAY_ID$", gateway_id)
    redis_client : StrictRedis = init_redis_client()

    # Example, enter into a pair position long SUSHI, short DYDX
    positions : List[DivisiblePosition] = [
        DivisiblePosition(
            ticker = 'SUSHI/USDT:USDT',
            side = 'sell',
            amount = 10,
            leg_room_bps = 5,
            order_type = 'limit',
            slices=5,
            wait_fill_threshold_ms=15000
        ),
        DivisiblePosition(
            ticker = 'DYDX/USDT:USDT',
            side = 'buy',
            amount = 10,
            leg_room_bps = 5,
            order_type = 'limit',
            slices=5,
            wait_fill_threshold_ms=15000
        )
    ]
    execute_positions(
        redis_client=redis_client,
        positions=positions,
        ordergateway_topic=ordergateway_topic)