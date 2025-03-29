from enum import Enum
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Union
import logging
import json
from redis import StrictRedis

from ordergateway.client import DivisiblePosition, execute_positions
from constants import JSON_SERIALIZABLE_TYPES

'''
set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
python test_ordergateway.py --gateway_id hyperliquid_01
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

if __name__ == '__main__':
    parse_args()

    gateway_id : str = param['gateway_id']
    ordergateway_pending_orders_topic = 'ordergateway_pending_orders_$GATEWAY_ID$'
    ordergateway_pending_orders_topic = ordergateway_pending_orders_topic.replace("$GATEWAY_ID$", gateway_id)
    
    ordergateway_executions_topic = "ordergateway_executions_$GATEWAY_ID$"
    ordergateway_executions_topic = ordergateway_executions_topic.replace("$GATEWAY_ID$", gateway_id)

    redis_client : StrictRedis = init_redis_client()

    # Example, enter into a pair position long SUSHI, short DYDX
    positions_1 : List[DivisiblePosition] = [
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
            reduce_only=False,
            order_type = 'limit',
            slices=5,
            wait_fill_threshold_ms=15000
        )
    ]

    positions_2 : List[DivisiblePosition] = [
        DivisiblePosition(
            ticker = 'SUSHI/USDT:USDT',
            side = 'buy',
            amount = 10,
            leg_room_bps = 5,
            reduce_only=False,
            order_type = 'limit',
            slices=5,
            wait_fill_threshold_ms=15000
        ),
    ]

    positions_3 : List[DivisiblePosition] = [
        DivisiblePosition(
            ticker = 'BTC/USDC:USDC',
            side = 'sell',
            amount = 0.00100,
            leg_room_bps = 5,
            reduce_only=True,
            order_type = 'limit',
            slices=1,
            wait_fill_threshold_ms=60000,
            fees_ccy='USDC'
        )
    ]

    
    executed_positions : Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None] = execute_positions(
        redis_client=redis_client,
        positions=positions_3,
        ordergateway_pending_orders_topic=ordergateway_pending_orders_topic,
        ordergateway_executions_topic=ordergateway_executions_topic
        )
    if executed_positions:
        for position in executed_positions:
            print(f"{position['ticker']} {position['side']} amount: {position['amount']} leg_room_bps: {position['leg_room_bps']} slices: {position['slices']}, filled_amount: {position['filled_amount']}, average_cost: {position['average_cost']}, # executions: {len(position['executions'])}, done: {position['done']}, execution_err: {position['execution_err']}") # type: ignore
        
        

    