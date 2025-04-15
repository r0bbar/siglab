import sys
import traceback
import os
import random
from dotenv import load_dotenv
from enum import Enum
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Union, Any
import hashlib
from collections import deque
import logging
import json
import uuid
from io import StringIO
import re
from re import Pattern
from redis import StrictRedis
import asyncio

from util.aws_util import AwsKmsUtil

import ccxt.pro as ccxtpro

from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.ordergateway.client import Order, DivisiblePosition
from siglab_py.constants import LogLevel # type: ignore
from util.notification_util import dispatch_notification

current_filename = os.path.basename(__file__)

'''
Usage:
    python gateway.py --gateway_id hyperliquid_01 --default_type linear --rate_limit_ms 100

    --default_type defaults to linear
    --rate_limit_ms defaults to 100

    --slack_info_url, --slack_critical_url and --slack_alert_url are if you want gateway to dispatch Slack notification on events.
    How to get Slack webhook urls? https://medium.com/@natalia_assad/how-send-a-table-to-slack-using-python-d1a20b08abe0

    Another example:
        python gateway.py --gateway_id hyperliquid_01 --default_type linear --rate_limit_ms 100 --slack_info_url=https://hooks.slack.com/services/xxx --slack_critial_url=https://hooks.slack.com/services/yyy --slack_alert_url=https://hooks.slack.com/services/zzz

This script is pypy compatible:
    pypy gateway.py --gateway_id bybit_01 --default_type linear --rate_limit_ms 100

In above example, $GATEWAY_ID$ is 'bybit_01'.

You should place .env.$GATEWAY_ID$ in same folder as gateway.py. Formmat should be.
    ENCRYPT_DECRYPT_WITH_AWS_KMS=Y
    AWS_KMS_KEY_ID=xxx
    APIKEY=xxx
    SECRET=xxx
    PASSPHRASE=xxx

If ENCRYPT_DECRYPT_WITH_AWS_KMS set to N, APIKEY, SECRET and PASSPHRASE in un-encrypted format(Bad idea in general but if you want to quickly test things out).
If ENCRYPT_DECRYPT_WITH_AWS_KMS set to Y, APIKEY, SECRET and PASSPHRASE are decrypted using AWS KMS (You can use 'encrypt_keys_util.py' to encrypt your credentials)

Optionally, credentials can be passed in as command line arguments, which will override credentials from .env
    python gateway.py --gateway_id bybit_01 --default_type linear --rate_limit_ms 100 --encrypt_decrypt_with_aws_kms Y --aws_kms_key_id xxx --apikey xxx --secret xxx --passphrase xxx

Please lookup 'defaultType' (Whether you're trading spot? Or perpectuals) via ccxt library. It's generally under exchange's method 'describe'. Looks under 'options' tag, look for 'defaultType'.
'Perpetual contracts' are generally referred to as 'linear' or 'swap'.

Examples,
    binance spot, future, margin, delivery, option https://github.com/ccxt/ccxt/blob/master/python/ccxt/binance.py#L1298
    Deribit spot, swap, future https://github.com/ccxt/ccxt/blob/master/python/ccxt/deribit.py#L360
    bybit supports spot, linear, inverse, futures https://github.com/ccxt/ccxt/blob/master/python/ccxt/bybit.py#L1041
    okx supports funding, spot, margin, future, swap, option https://github.com/ccxt/ccxt/blob/master/python/ccxt/okx.py#L1144
    hyperliquid swap only https://github.com/ccxt/ccxt/blob/master/python/ccxt/hyperliquid.py#L225

To add exchange, extend "instantiate_exchange".

To debug from vscode, launch.json:
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
                    "--gateway_id", "bybit_01",
                    "--default_type", "linear",
                    "--rate_limit_ms", "100",

                    "--encrypt_decrypt_with_aws_kms", "N",
                    "--aws_kms_key_id", "",
                    "--apikey", "xxx",
                    "--secret", "xxx",
                    "--passphrase", "xxx",

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

    gateway.py takes orders from redis. Strategies should publish orders under topic specified under param['incoming_orders_topic_regex'].

    Expected order format:
        [
            {
                "ticker": "SUSHI/USDT:USDT",
                "side": "sell",
                "amount": 10,
                "order_type": "limit",
                "leg_room_bps": 5,
                "slices": 5,
                "wait_fill_threshold_ms": 5000,
                "executions": {},
                "filled_amount": 0,
                "average_cost": 0
            }
        ]

    After executions, gateway.py publish back to redis under topic param['executions_publish_topic'].

    Format:
        [
            {
                "ticker": "SUSHI/USDT:USDT",
                "side": "sell",
                "amount": 10,
                "order_type": "limit",
                "leg_room_bps": 5,
                "slices": 5,
                "wait_fill_threshold_ms": 5000,
                "executions": {
                    "xxx": {    <-- order id from exchange
                        "info": { <-- ccxt convention, raw response from exchanges under info tag
                            ...
                        },
                        "id": "xxx", <-- order id from exchange
                        "clientOrderId": "xxx",
                        "timestamp": xxx,
                        "datetime": "xxx",
                        "lastTradeTimestamp": xxx,
                        "lastUpdateTimestamp": xxx,
                        "symbol": "SUSHI/USDT:USDT",
                        "type": "limit",
                        "timeInForce": null,
                        "postOnly": null,
                        "side": "sell",
                        "price": 0.8897,
                        "stopLossPrice": null,
                        "takeProfitPrice": null,
                        "triggerPrice": null,
                        "average": 0.8901,
                        "cost": 1.7802,
                        "amount": 2,
                        "filled": 2,
                        "remaining": 0,
                        "status": "closed",
                        "fee": {
                            "cost": 0.00053406,
                            "currency": "USDT"
                        },
                        "trades": [],
                        "reduceOnly": false,
                        "fees": [
                            {
                                "cost": 0.00053406,
                                "currency": "USDT"
                            }
                        ],
                        "stopPrice": null,
                        "multiplier": 1
                    },
                    "filled_amount": 10,    <-- aggregates computed by gateway.py
                    "average_cost": 0.88979 <-- aggregates computed by gateway.py
                }
                    
                ... more executions ...
        ]
'''
class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

param : Dict = {
    'gateway_id' : '---',
    'dry_run' : False,

    "incoming_orders_topic_regex" : r"ordergateway_pending_orders_$GATEWAY_ID$", 
    "executions_publish_topic" : r"ordergateway_executions_$GATEWAY_ID$",

    "default_fees_ccy" : None,
    "loop_freq_ms" : 500, # reduce this if you need trade faster
    "loops_random_delay_multiplier" : 1, # Add randomness to time between slices are sent off. Set to 1 if no random delay needed.
    "wait_fill_threshold_ms" : 5000,

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

DUMMY_EXECUTION : Dict[str, Any] = {
    "clientOrderId": None,
    "timestamp": None,
    "datetime": None,
    "symbol": None,
    "type": None,
    "side": None,
    "price": None,
    "average": None,
    "cost": None,
    "amount": None,
    "filled": None,
    "remaining": None,
    "status": "closed",
    "fee": {
        "cost": None,
        "currency": "USDT"
    },
    "trades": [],
    "reduceOnly": False,
    "fees": [
        {
            "cost": 0,
            "currency": "USDT"
        }
    ],
    "multiplier": None
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

    parser.add_argument("--gateway_id", help="gateway_id: Where are you sending your order?", default=None)
    parser.add_argument("--dry_run", help="Y or N (default, for testing). If Y, orders won't be dispatched to exchange, gateway will fake reply.", default='N')
    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)

    parser.add_argument("--default_fees_ccy", help="If you're trading crypto, CEX fees USDT, DEX fees USDC in many cases. Default None, in which case gateway won't aggregatge fees from executions for you.", default=None)
    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=500)
    parser.add_argument("--wait_fill_threshold_ms", help="Wait for fills for how long?", default=5000)

    parser.add_argument("--encrypt_decrypt_with_aws_kms", help="Y or N. If encrypt_decrypt_with_aws_kms=N, pass in apikey, secret and passphrase unencrypted (Not recommended, for testing only). If Y, they will be decrypted using AMS KMS key.", default='N')
    parser.add_argument("--aws_kms_key_id", help="AWS KMS key ID", default=None)
    parser.add_argument("--apikey", help="Exchange apikey", default=None)
    parser.add_argument("--secret", help="Exchange secret", default=None)
    parser.add_argument("--passphrase", help="Exchange passphrase", default=None)

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()
    param['gateway_id'] = args.gateway_id

    if args.dry_run:
        if args.dry_run=='Y':
            param['dry_run'] = True
        else:
            param['dry_run'] = False
    else:
        param['dry_run'] = False

    param['default_type'] = args.default_type
    param['rate_limit_ms'] = int(args.rate_limit_ms)
    param['default_fees_ccy'] = args.default_fees_ccy
    param['loop_freq_ms'] = int(args.loop_freq_ms)
    param['wait_fill_threshold_ms'] = int(args.wait_fill_threshold_ms)

    if args.encrypt_decrypt_with_aws_kms:
        if args.encrypt_decrypt_with_aws_kms=='Y':
            param['encrypt_decrypt_with_aws_kms'] = True
        else:
            param['encrypt_decrypt_with_aws_kms'] = False
    else:
        param['encrypt_decrypt_with_aws_kms'] = False

    param['aws_kms_key_id'] = args.aws_kms_key_id
    param['apikey'] = args.apikey
    param['secret'] = args.secret
    param['passphrase'] = args.passphrase

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

    param['notification']['footer'] = f"From {param['current_filename']} {param['gateway_id']}"

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

async def instantiate_exchange(
    gateway_id : str,
    api_key : str,
    secret : str,
    passphrase : str,
    default_type : str,
    rate_limit_ms : float = 100
) -> Union[AnyExchange, None]:
    exchange : Union[AnyExchange, None] = None
    exchange_name : str = gateway_id.split('_')[0]
    exchange_name =exchange_name.lower().strip()

    # Look at ccxt exchange.describe. under 'options' \ 'defaultType' (and 'defaultSubType') for what markets the exchange support.
    # https://docs.ccxt.com/en/latest/manual.html#instantiation
    exchange_params : Dict[str, Any]= {
                        'apiKey' : api_key,
                        'secret' : secret,
                        'enableRateLimit'  : True,
                        'rateLimit' : rate_limit_ms,
                        'options' : {
                            'defaultType' : default_type
                        }
                    }

    if exchange_name=='binance':
        # spot, future, margin, delivery, option
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/binance.py#L1298
        exchange = ccxtpro.binance(exchange_params)  # type: ignore
    elif exchange_name=='bybit':
        # spot, linear, inverse, futures
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/bybit.py#L1041
        exchange = ccxtpro.bybit(exchange_params) # type: ignore
    elif exchange_name=='okx':
        # 'funding', spot, margin, future, swap, option
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/okx.py#L1144
        exchange_params['password'] = passphrase
        exchange = ccxtpro.okx(exchange_params) # type: ignore
    elif exchange_name=='deribit':
        # spot, swap, future
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/deribit.py#L360
        exchange = ccxtpro.deribit(exchange_params)  # type: ignore
    elif exchange_name=='hyperliquid':
        '''
        https://app.hyperliquid.xyz/API
        
        defaultType from ccxt: swap
            https://github.com/ccxt/ccxt/blob/master/python/ccxt/hyperliquid.py#L225
        
        How to integrate? You can skip first 6 min: https://www.youtube.com/watch?v=UuBr331wxr4&t=363s

        Example, 
            API credentials created under "\ More \ API":
                    Ledger Arbitrum Wallet Address: 0xAAAAA <-- This is your Ledger Arbitrum wallet address with which you connect to Hyperliquid. 
                    API Wallet Address 0xBBBBB <-- Generated
                    privateKey 0xCCCCC

        Basic connection via CCXT:
            import asyncio
            import ccxt.pro as ccxtpro

            async def main():
                rate_limit_ms = 100
                exchange_params = {
                    "walletAddress" : "0xAAAAA", # Ledger Arbitrum Wallet Address here! Not the generated address.
                    "privateKey" : "0xCCCCC"
                }
                exchange = ccxtpro.hyperliquid(exchange_params) 
                balances = await exchange.fetch_balance()
                print(balances)

            asyncio.run(main())
        '''
        exchange = ccxtpro.hyperliquid(
            {
                "walletAddress" : api_key,
                "privateKey" : secret,
                'enableRateLimit'  : True,
                'rateLimit' : rate_limit_ms
            }
        )  # type: ignore
    else:
        raise ValueError(f"Unsupported exchange {exchange_name}, check gateway_id {gateway_id}.")

    await exchange.load_markets() # type: ignore

    '''
    Is this necessary? The added trouble is for example bybit.authenticate requires arg 'url'. binance doesn't. And fetch_balance already test credentials.

    try:
        await exchange.authenticate() # type: ignore
    except Exception as swallow_this_error:
        pass
    '''

    return exchange

async def watch_orders_task(
    exchange : AnyExchange,
    executions : Dict[str, Dict[str, Any]]
):
    while True:
        try:
            order_updates = await exchange.watch_orders() # type: ignore
            for order_update in order_updates:
                order_id = order_update['id']
                executions[order_id] = order_update

            log(f"order updates: {json.dumps(order_updates, indent=4)}", log_level=LogLevel.INFO)
        except Exception as loop_err:
            print(f"watch_orders_task error: {loop_err}")
        
        await asyncio.sleep(param['loop_freq_ms']/1000)

async def send_heartbeat(exchange):

    await asyncio.sleep(10)

    while True:
        try:
            first_ws_url = next(iter(exchange.clients))
            client = exchange.clients[first_ws_url]
            message = exchange.ping(client)
            await client.send(message)
            log('Heartbeat sent')
        except Exception as hb_error:
            log(f'Failed to send heartbeat: {hb_error}')
        finally:
            await asyncio.sleep(20)

async def execute_one_position(
    exchange : AnyExchange,
    position : DivisiblePosition,
    param : Dict,
    executions : Dict[str, Dict[str, Any]],
    notification_params : Dict[str, Any]
):
    try:
        market : Dict[str, Any] = exchange.markets[position.ticker] if position.ticker in exchange.markets else None # type: ignore
        if not market:
            raise ArgumentError(f"Market not found for {position.ticker} under {exchange.name}") # type: ignore

        min_amount = float(market['limits']['amount']['min']) if market['limits']['amount']['min'] else 0 # type: ignore
        multiplier = market['contractSize'] if 'contractSize' in market and market['contractSize'] else 1
        position.multiplier = multiplier

        log(f"{position.ticker} min_amount: {min_amount}, multiplier: {multiplier}") 

        slices : List[Order] = position.to_slices()

        # Residual handling in last slice
        if len(slices)>1:
            last_slice = slices[-1]
            last_slice_rounded_amount_in_base_ccy = exchange.amount_to_precision(position.ticker, last_slice.amount/multiplier) # After divided by multiplier, rounded_slice_amount_in_base_ccy in number of contracts actually (Not in base ccy).
            last_slice_rounded_amount_in_base_ccy = float(last_slice_rounded_amount_in_base_ccy) if last_slice_rounded_amount_in_base_ccy else 0
            if last_slice_rounded_amount_in_base_ccy<=min_amount:
                slices.pop()
                slices[-1].amount += last_slice.amount

                log(f"{position.ticker} Last slice residual smaller than min_amount. Amount is added to prev slice instead. last_slice_amount: {last_slice.amount/multiplier}, last_slice_rounded_amount: {last_slice_rounded_amount_in_base_ccy}") 

        i = 0
        for slice in slices:
            try:
                log(f"{position.ticker} sending slice# {i}")

                dt_now : datetime = datetime.now()

                slice_amount_in_base_ccy : float = slice.amount
                rounded_slice_amount_in_base_ccy = slice_amount_in_base_ccy / multiplier # After divided by multiplier, rounded_slice_amount_in_base_ccy in number of contracts actually (Not in base ccy).
                rounded_slice_amount_in_base_ccy = exchange.amount_to_precision(position.ticker, rounded_slice_amount_in_base_ccy) # type: ignore
                rounded_slice_amount_in_base_ccy = float(rounded_slice_amount_in_base_ccy) if rounded_slice_amount_in_base_ccy else 0
                rounded_slice_amount_in_base_ccy = rounded_slice_amount_in_base_ccy if rounded_slice_amount_in_base_ccy>min_amount else min_amount

                if rounded_slice_amount_in_base_ccy==0:
                    log(f"{position.ticker} Slice amount rounded to zero?! slice amount before rounding: {slice.amount}") 
                    continue

                orderbook = await exchange.fetch_order_book(symbol=position.ticker, limit=3) # type: ignore
                if position.side=='buy':
                    asks = [ ask[0] for ask in orderbook['asks'] ]
                    best_asks = min(asks)
                    limit_price : float= best_asks * (1 + position.leg_room_bps/10000)
                else:
                    bids = [ bid[0] for bid in orderbook['bids'] ]
                    best_bid = max(bids)
                    limit_price : float = best_bid * (1 - position.leg_room_bps/10000)
                    
                rounded_limit_price : float = exchange.price_to_precision(position.ticker, limit_price) # type: ignore
                rounded_limit_price = float(rounded_limit_price)
                
                order_params = {
                    'reduceOnly': slice.reduce_only
                }
                if position.order_type=='limit':
                    if not param['dry_run']:
                        executed_order = await exchange.create_order( # type: ignore
                            symbol = position.ticker,
                            type = position.order_type,
                            amount = rounded_slice_amount_in_base_ccy,
                            price = rounded_limit_price,
                            side = position.side,
                            params = order_params
                        )
                    else:
                        executed_order = DUMMY_EXECUTION.copy()
                        executed_order['clientOrderId'] = str(uuid.uuid4())
                        executed_order['timestamp'] = dt_now.timestamp()
                        executed_order['datetime'] = dt_now
                        executed_order['symbol'] = position.ticker
                        executed_order['type'] = position.order_type
                        executed_order['side'] = position.side
                        executed_order['price'] = rounded_limit_price
                        executed_order['average'] = rounded_limit_price
                        executed_order['cost'] = 0
                        executed_order['amount'] = rounded_slice_amount_in_base_ccy
                        executed_order['filled'] = rounded_slice_amount_in_base_ccy
                        executed_order['remaining'] = 0
                        executed_order['status'] = 'closed'
                        executed_order['multiplier'] = position.multiplier

                else:
                    if not param['dry_run']:
                        executed_order = await exchange.create_order( # type: ignore
                            symbol = position.ticker,
                            type = position.order_type,
                            amount = rounded_slice_amount_in_base_ccy,
                            side = position.side,
                            params = order_params
                        )
                    else:
                        executed_order = DUMMY_EXECUTION.copy()
                        executed_order['clientOrderId'] = str(uuid.uuid4())
                        executed_order['timestamp'] = dt_now.timestamp()
                        executed_order['datetime'] = dt_now
                        executed_order['symbol'] = position.ticker
                        executed_order['type'] = position.order_type
                        executed_order['side'] = position.side
                        executed_order['price'] = rounded_limit_price
                        executed_order['average'] = rounded_limit_price
                        executed_order['cost'] = 0
                        executed_order['amount'] = rounded_slice_amount_in_base_ccy
                        executed_order['filled'] = rounded_slice_amount_in_base_ccy
                        executed_order['remaining'] = 0
                        executed_order['status'] = 'closed'
                        executed_order['multiplier'] = position.multiplier

                executed_order['slice_id'] = i

                '''
                Format of executed_order:
                    executed_order
                    {'info': {'clOrdId': 'xxx', 'ordId': '2245241151525347328', 'sCode': '0', 'sMsg': 'Order placed', 'tag': 'xxx', 'ts': '1739415800635'}, 'id': '2245241151525347328', 'clientOrderId': 'xxx', 'timestamp': None, 'datetime': None, 'lastTradeTimestamp': None, 'lastUpdateTimestamp': None, 'symbol': 'SUSHI/USDT:USDT', 'type': 'limit', 'timeInForce': None, 'postOnly': None, 'side': 'buy', 'price': None, 'stopLossPrice': None, 'takeProfitPrice': None, 'triggerPrice': None, 'average': None, 'cost': None, 'amount': None, 'filled': None, 'remaining': None, 'status': None, 'fee': None, 'trades': [], 'reduceOnly': False, 'fees': [], 'stopPrice': None}
                    special variables:
                    function variables:
                    'info': {'clOrdId': 'xxx', 'ordId': '2245241151525347328', 'sCode': '0', 'sMsg': 'Order placed', 'tag': 'xxx', 'ts': '1739415800635'}
                    'id': '2245241151525347328'
                    'clientOrderId': 'xxx'
                    'timestamp': None
                    'datetime': None
                    'lastTradeTimestamp': None
                    'lastUpdateTimestamp': None
                    'symbol': 'SUSHI/USDT:USDT'
                    'type': 'limit'
                    'timeInForce': None
                    'postOnly': None
                    'side': 'buy'
                    'price': None
                    'stopLossPrice': None
                    'takeProfitPrice': None
                    'triggerPrice': None
                    'average': None
                    'cost': None
                    'amount': None
                    'filled': None
                    'remaining': None
                    'status': None
                    'fee': None
                    'trades': []
                    'reduceOnly': False
                    'fees': []
                    'stopPrice': None
                '''
                order_id = executed_order['id']
                order_status = executed_order['status']
                filled_amount = executed_order['filled']
                remaining_amount = executed_order['remaining']
                executed_order['multiplier'] = multiplier
                position.append_execution(order_id, executed_order)

                log(f"Order dispatched: {order_id}. status: {order_status}, filled_amount: {filled_amount}, remaining_amount: {remaining_amount}")

                if not order_status or order_status!='closed':
                    wait_threshold_sec = position.wait_fill_threshold_ms / 1000 
                    
                    start_time = time.time()
                    elapsed_sec = time.time() - start_time
                    while elapsed_sec < wait_threshold_sec:
                        order_update = None
                        if order_id in executions:
                            order_update = executions[order_id]
                        
                        if order_update:
                            order_status = order_update['status']
                            filled_amount = order_update['filled']
                            remaining_amount = order_update['remaining']
                            order_update['multiplier'] = multiplier
                            position.append_execution(order_id, order_update)

                            if remaining_amount <= 0:
                                log(f"Limit order fully filled: {order_id}, order_update: {json.dumps(order_update, indent=4)}", log_level=LogLevel.INFO)
                                break

                        loops_random_delay_multiplier : int = random.randint(1, param['loops_random_delay_multiplier']) if param['loops_random_delay_multiplier']!=1 else 1
                        loop_freq_sec : int = max(1, param['loop_freq_ms']/1000)
                        await asyncio.sleep(loop_freq_sec * loops_random_delay_multiplier)

                        elapsed_sec = time.time() - start_time
                        log(f"{position.ticker} waiting for order update ... elapsed_sec: {elapsed_sec}")
                
                
                # Cancel hung limit order, resend as market
                if order_status!='closed':
                    # If no update from websocket, do one last fetch via REST
                    order_update = await exchange.fetch_order(order_id, position.ticker) # type: ignore 
                    order_status = order_update['status']
                    filled_amount = order_update['filled']
                    remaining_amount = order_update['remaining']
                    order_update['multiplier'] = multiplier

                    position.append_execution(order_id, order_update)

                    if order_status!='closed':
                        log(f"Final order_update before cancel+resend: {json.dumps(order_update, indent=4)}", log_level=LogLevel.INFO)
                        await exchange.cancel_order(order_id, position.ticker)  # type: ignore
                        position.get_execution(order_id)['status'] = 'canceled'
                        log(f"Canceled unfilled/partial filled order: {order_id}. Resending remaining_amount: {remaining_amount} as market order.", log_level=LogLevel.INFO)
                        
                        rounded_slice_amount_in_base_ccy = exchange.amount_to_precision(position.ticker, remaining_amount) # type: ignore
                        rounded_slice_amount_in_base_ccy = float(rounded_slice_amount_in_base_ccy)
                        rounded_slice_amount_in_base_ccy = rounded_slice_amount_in_base_ccy if rounded_slice_amount_in_base_ccy>min_amount else min_amount
                        if rounded_slice_amount_in_base_ccy>0:
                            executed_resent_order = await exchange.create_order(  # type: ignore
                                symbol=position.ticker,
                                type='market',
                                amount=remaining_amount,
                                side=position.side
                            )

                            order_id = executed_resent_order['id']
                            order_status = executed_resent_order['status']
                            executed_resent_order['multiplier'] = multiplier
                            position.append_execution(order_id, executed_resent_order)

                            wait_threshold_sec = position.wait_fill_threshold_ms / 1000 

                            start_time = time.time()
                            elapsed_sec = time.time() - start_time
                            while (not order_status or order_status!='closed') and (elapsed_sec < wait_threshold_sec):
                                order_update = None
                                if order_id in executions:
                                    order_update = executions[order_id]

                                if order_update:
                                    order_id = order_update['id']
                                    order_status = order_update['status']
                                    filled_amount = order_update['filled']
                                    remaining_amount = order_update['remaining']

                                elapsed_sec = time.time() - start_time
                                log(f"Waiting for resent market order to close {order_id} ... elapsed_sec: {elapsed_sec}")

                                loops_random_delay_multiplier : int = random.randint(1, param['loops_random_delay_multiplier']) if param['loops_random_delay_multiplier']!=1 else 1
                                loop_freq_sec : int = max(1, param['loop_freq_ms']/1000)
                                await asyncio.sleep(loop_freq_sec * loops_random_delay_multiplier)

                            if (not order_status or order_status!='closed'):
                                # If no update from websocket, do one last fetch via REST
                                order_update = await exchange.fetch_order(order_id, position.ticker) # type: ignore 
                                order_status = order_update['status']
                                filled_amount = order_update['filled']
                                remaining_amount = order_update['remaining']
                                order_update['multiplier'] = multiplier

                            log(f"Resent market order{order_id} filled. status: {order_status}, filled_amount: {filled_amount}, remaining_amount: {remaining_amount} {json.dumps(order_update, indent=4)}")
                    else:
                        log(f"{position.ticker} {order_id} status (From REST): {json.dumps(order_update, indent=4)}")

                slice.dispatched_price = rounded_limit_price
                slice.dispatched_amount = rounded_slice_amount_in_base_ccy
                position.dispatched_slices.append(slice)

                log(f"Executed slice #{i}", log_level=LogLevel.INFO)
                log(f"{json.dumps(slice.to_dict(), indent=4)}")
                log(f"{position.ticker}, multiplier: {multiplier}, slice_amount_in_base_ccy: {slice_amount_in_base_ccy}, rounded_slice_amount_in_base_ccy, {rounded_slice_amount_in_base_ccy}", log_level=LogLevel.INFO)
                if position.order_type=='limit':
                    log(f"{position.ticker}, limit_price: {limit_price}, rounded_limit_price, {rounded_limit_price}", log_level=LogLevel.INFO)

            except Exception as slice_err:
                log(
                    f"Failed to execute #{i} slice: {slice.to_dict()}. {slice_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}",
                    log_level=LogLevel.ERROR
                    )
                raise slice_err
            finally:
                log(f"{position.ticker} done slice# {i}")
                i += 1
                
        log(f"{position.ticker} patch_executions")
        position.patch_executions()

        log(f"Dispatched slices:")
        for dispatched_slice in position.dispatched_slices:
            log(f"{json.dumps(dispatched_slice.to_dict(), indent=4)}")

        position.filled_amount = position.get_filled_amount()
        position.average_cost = position.get_average_cost()
        position.fees = position.get_fees()

        balances = await exchange.fetch_balance() # type: ignore
        if param['default_type']!='spot':
            updated_position = await exchange.fetch_position(symbol=position.ticker) # type: ignore
            # After position closed, 'updated_position' can be an empty dict. hyperliquid for example.
            amount = (updated_position['contracts'] if updated_position else 0) * position.multiplier # in base ccy
        else:
            base_ccy : str = position.ticker.split("/")[0]
            amount = balances[base_ccy]['total']
        position.pos = amount

        position.done = True

        log(f"Executions:")
        log(f"{json.dumps(position.to_dict(), indent=4)}")

        notification_summary = {
            'ticker' : position.ticker,
            'side' : position.side,
            'num_executions' : len(position.get_executions()),
            'filled_amount' : position.filled_amount,
            'average_cost' : position.average_cost,
            'pos' : position.pos,
            'done' : position.done
        }
        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} execute_one_position done. {position.ticker} {position.side} {position.amount}", message=notification_summary, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

    except Exception as position_execution_err:
        err_msg = f"{position.ticker}  Execution failed: {position_execution_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}"
        log(err_msg)

        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} {position.ticker} execute_one_position failed!!! {position.ticker} {position.side} {position.amount}", message=position.get_executions(), footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger) # type: ignore

        position.done = False
        position.execution_err = err_msg
        
async def work(
    param : Dict,
    exchange : AnyExchange,
    redis_client : StrictRedis,
    notification_params : Dict[str, Any]
):
    incoming_orders_topic_regex : str = param['incoming_orders_topic_regex']
    incoming_orders_topic_regex = incoming_orders_topic_regex.replace("$GATEWAY_ID$", param['gateway_id'])
    incoming_orders_topic_regex_pattern : Pattern = re.compile(incoming_orders_topic_regex)

    executions_publish_topic : str = param['executions_publish_topic'].replace("$GATEWAY_ID$", param['gateway_id'])

    # This is how we avoid reprocess same message twice. We check message hash and cache it.
    processed_hash_queue = deque(maxlen=10)
    
    executions : Dict[str, Dict[str, Any]] = {}
    asyncio.create_task(watch_orders_task(exchange, executions))

    asyncio.create_task(send_heartbeat(exchange))

    while True:
        try:
            keys = redis_client.keys()
            for key in keys:
                try:
                    s_key : str = key.decode("utf-8")
                    if incoming_orders_topic_regex_pattern.match(s_key):
                        orders = None
                        message = redis_client.get(key)
                        if message:
                            message_hash = hashlib.sha256(message).hexdigest()
                            message = message.decode('utf-8')
                            if message_hash not in processed_hash_queue: # Dont process what's been processed before.
                                processed_hash_queue.append(message_hash)

                                orders = json.loads(message)
                                positions : List[DivisiblePosition] = [
                                    DivisiblePosition(
                                        ticker=order['ticker'],
                                        side=order['side'],
                                        amount=order['amount'],
                                        order_type=order['order_type'],
                                        leg_room_bps=order['leg_room_bps'],
                                        reduce_only=order['reduce_only'],
                                        fees_ccy=order['fees_ccy'] if 'fees_ccy' in order else param['default_fees_ccy'],
                                        slices=order['slices'],
                                        wait_fill_threshold_ms=order['wait_fill_threshold_ms'] if order['wait_fill_threshold_ms']>0 else param['wait_fill_threshold_ms']
                                    )
                                    for order in orders
                                ]

                                start = time.time()
                                pending_executions = [ execute_one_position(exchange, position, param, executions, notification_params) for position in positions ]
                                await asyncio.gather(*pending_executions)
                                order_dispatch_elapsed_ms = int((time.time() - start) *1000)

                                i = 0
                                for position in positions:
                                    log(f"{i} {position.ticker}, {position.side} # executions: {len(position.get_executions())}, filled_amount: {position.filled_amount}, average_cost: {position.average_cost}, pos: {position.pos}, done: {position.done}, error: {position.execution_err}, order_dispatch_elapsed_ms: {order_dispatch_elapsed_ms}")
                                    i += 1

                                start = time.time()
                                if redis_client:
                                    redis_client.delete(key)

                                    '''
                                    https://redis.io/commands/set/
                                    '''
                                    expiry_sec : int = 60*15 # additional 15min
                                    _positions = [ position.to_dict() for position in positions ]
                                    redis_client.set(name=executions_publish_topic, value=json.dumps(_positions).encode('utf-8'), ex=60*15)
                                    redis_set_elapsed_ms = int((time.time() - start) *1000)

                                    log(f"positions published back to redis, redis_set_elapsed_ms: {redis_set_elapsed_ms}")

                except Exception as key_error:
                    log(
                        f"Failed to process {key}. Error: {key_error} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}",
                        log_level=LogLevel.ERROR
                        )

        except Exception as loop_error:
            log(f"Error: {loop_error} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")
        finally:
            await asyncio.sleep(param['loop_freq_ms']/1000)

async def main():
    parse_args()

    fh = logging.FileHandler(f"ordergateway_{param['gateway_id']}.log")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)     
    logger.addHandler(fh)

    if not param['apikey']:
        log("Loading credentials from .env")

        load_dotenv()

        encrypt_decrypt_with_aws_kms = os.getenv('ENCRYPT_DECRYPT_WITH_AWS_KMS')
        encrypt_decrypt_with_aws_kms = True if encrypt_decrypt_with_aws_kms=='Y' else False
        
        api_key : str = str(os.getenv('APIKEY'))
        secret : str = str(os.getenv('SECRET'))
        passphrase : str = str(os.getenv('PASSPHRASE'))
    else:
        log("Loading credentials from command line args")

        encrypt_decrypt_with_aws_kms = param['encrypt_decrypt_with_aws_kms']
        api_key : str = param['apikey']
        secret : str = param['secret']
        passphrase : str = param['passphrase']

    notification_params : Dict[str, Any] = param['notification']

    if encrypt_decrypt_with_aws_kms:
        aws_kms_key_id = str(os.getenv('AWS_KMS_KEY_ID'))

        aws_kms = AwsKmsUtil(key_id=aws_kms_key_id, profile_name=None)
        api_key = aws_kms.decrypt(api_key.encode())
        secret = aws_kms.decrypt(secret.encode())
        if passphrase:
            passphrase = aws_kms.decrypt(passphrase.encode())
    
    redis_client : StrictRedis = init_redis_client()

    exchange : Union[AnyExchange, None] = await instantiate_exchange(
        gateway_id=param['gateway_id'],
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        default_type=param['default_type'],
        rate_limit_ms=param['rate_limit_ms']
    )
    if exchange:
        # Once exchange instantiated, try fetch_balance to confirm connectivity and test credentials.
        balances = await exchange.fetch_balance() # type: ignore
        log(f"{param['gateway_id']}: account balances {balances}")
        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} started", message=balances, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

        await work(param=param, exchange=exchange, redis_client=redis_client, notification_params=notification_params)

asyncio.run(main())
