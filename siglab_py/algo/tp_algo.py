import sys
import traceback
import os
import logging
from dotenv import load_dotenv
import argparse
from datetime import datetime, timezone
import time
from typing import List, Dict, Any, Union
import json
import asyncio
from redis import StrictRedis

from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.ordergateway.client import DivisiblePosition, execute_positions
from siglab_py.util.market_data_util import async_instantiate_exchange
from siglab_py.util.trading_util import calc_eff_trailing_sl
from siglab_py.util.notification_util import dispatch_notification
from siglab_py.util.aws_util import AwsKmsUtil

from siglab_py.constants import LogLevel, JSON_SERIALIZABLE_TYPES # type: ignore

current_filename = os.path.basename(__file__)

'''
Error: RuntimeError: aiodns needs a SelectorEventLoop on Windows.
Hack, by far the filthest hack I done in my career: Set SelectorEventLoop on Windows
'''
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

'''
TP algo is an excecution algo. It doesn't decide entries for you. It however, execute the trade with target TP and SL with hard and also gradually tighted trailing stops.

For more on "Gradually Tighted Stops" and 'calc_eff_trailing_sl':
    https://medium.com/@norman-lm-fung/gradually-tightened-trailing-stops-f7854bf1e02b

Now why does TP algo need apikey when gateway.py is sending the orders? TP algo perform position reconciliation (For contracts), and if positions don't match, algo would terminate.
Why would you do that? Imagine if trader want to intervene and close out positions from mobile?

Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python tp_algo.py --gateway_id hyperliquid_01 --default_type linear --rate_limit_ms 100 --encrypt_decrypt_with_aws_kms Y --aws_kms_key_id xxx --apikey xxx --secret xxx --ticker SUSHI/USDC:USDC --side sell --order_type limit --amount_base_ccy 45 --slices 3 --wait_fill_threshold_ms 15000 --leg_room_bps 5 --tp_min_percent 1.5 --tp_max_percent 2.5 --sl_percent_trailing 50 --sl_percent 1 --reversal_num_intervals 2 --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx --load_entry_from_cache N

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
                        "--gateway_id", "hyperliquid_01",
                        "--default_type", "linear",
                        "--rate_limit_ms", "100",
                        "--encrypt_decrypt_with_aws_kms", "Y",
                        "--aws_kms_key_id", "xxx",
                        "--apikey", "xxx",
                        "--secret", "xxx",

                        "--ticker", "SUSHI/USDC:USDC",
                        "--side", "sell",
                        "--order_type", "limit",
                        "--amount_base_ccy", "45",
                        "--slices", "3",
                        "--wait_fill_threshold_ms", "15000",
                        "--leg_room_bps", "5",
                        "--tp_min_percent", "1.5",
                        "--tp_max_percent", "2.5",
                        "--sl_percent_trailing", "50",
                        "--sl_percent", "1",
                        "--reversal_num_intervals", "2",

                        "--load_entry_from_cache", "N",

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
    'trailing_sl_min_percent_linear': 1.0, # This is threshold used for tp_algo to decide if use linear stops tightening, or non-linear. If tp_max_percent far (>100bps), there's more uncertainty if target can be reached: Go with linear.
    'non_linear_pow' : 5, # For non-linear trailing stops tightening. 

    "loop_freq_ms" : 5000, # reduce this if you need trade faster

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

    parser.add_argument("--gateway_id", help="gateway_id: Where are you sending your order?", default=None)

    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)
    parser.add_argument("--encrypt_decrypt_with_aws_kms", help="Y or N. If encrypt_decrypt_with_aws_kms=N, pass in apikey, secret and passphrase unencrypted (Not recommended, for testing only). If Y, they will be decrypted using AMS KMS key.", default='N')
    parser.add_argument("--aws_kms_key_id", help="AWS KMS key ID", default=None)
    parser.add_argument("--apikey", help="Exchange apikey", default=None)
    parser.add_argument("--secret", help="Exchange secret", default=None)
    parser.add_argument("--passphrase", help="Exchange passphrase", default=None)

    parser.add_argument("--ticker", help="Ticker you're trading. Example BTC/USDC:USDC", default=None)
    parser.add_argument("--side", help="order side, for entries. buy or sell", default=None)
    parser.add_argument("--order_type", help="Order type: market or limit", default=None)
    parser.add_argument("--amount_base_ccy", help="Order amount in base ccy (Not # contracts). Always positive, even for sell trades.", default=None)
    parser.add_argument("--leg_room_bps", help="Leg room, for Limit orders only. A more positive leg room is a more aggressive order to get filled. i.e. Buy at higher price, Sell at lower price.", default=5)
    parser.add_argument("--slices", help="Algo can break down larger order into smaller slices. Default: 1", default=1)
    parser.add_argument("--wait_fill_threshold_ms", help="Limit orders will be cancelled if not filled within this time. Remainder will be sent off as market order.", default=15000)

    parser.add_argument("--load_entry_from_cache", help="Y or N (default). This is for algo restart scenario where you don't want make entry again. In this case existing/running position loaded from cache.", default='N')

    parser.add_argument("--tp_min_percent", help="For trailing stops. Min TP in percent, i.e. No TP until pnl at least this much.", default=None)
    parser.add_argument("--tp_max_percent", help="For trailing stops. Max TP in percent, i.e. Price target", default=None)
    parser.add_argument("--sl_percent_trailing", help="For trailing stops. trailing SL in percent, please refer to trading_util.calc_eff_trailing_sl for documentation.", default=None)
    parser.add_argument("--default_effective_tp_trailing_percent", help="Default for sl_percent_trailing when pnl still below tp_min_percent. Default: float('inf'), meaing trailing stop won't be fired.", default=float('inf'))
    parser.add_argument("--sl_percent", help="Hard stop in percent.", default=2)
    parser.add_argument("--reversal_num_intervals", help="How many reversal candles to confirm reversal?", default=3)
    parser.add_argument("--trailing_sl_min_percent_linear", help="This is threshold used for tp_algo to decide if use linear stops tightening, or non-linear. If tp_max_percent far (>200bps for example), there's more uncertainty if target can be reached: Go with linear. Default: 2% (200 bps)", default=2.0)
    parser.add_argument("--non_linear_pow", help="For non-linear trailing stops tightening, have a look at call to 'calc_eff_trailing_sl'. Default: 5", default=5)
    
    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=5000)

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()

    param['gateway_id'] = args.gateway_id
    param['default_type'] = args.default_type
    param['rate_limit_ms'] = int(args.rate_limit_ms)
    
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

    param['ticker'] = args.ticker
    param['side'] = args.side
    param['order_type'] = args.order_type
    param['amount_base_ccy'] = float(args.amount_base_ccy)
    param['leg_room_bps'] = int(args.leg_room_bps)
    param['slices'] = int(args.slices)
    param['wait_fill_threshold_ms'] = int(args.wait_fill_threshold_ms)

    if args.load_entry_from_cache:
        if args.load_entry_from_cache=='Y':
            param['load_entry_from_cache'] = True
        else:
            param['load_entry_from_cache'] = False
    else:
        param['load_entry_from_cache'] = False

    param['tp_min_percent'] = float(args.tp_min_percent)
    param['tp_max_percent'] = float(args.tp_max_percent)
    param['sl_percent_trailing'] = float(args.sl_percent_trailing)
    param['default_effective_tp_trailing_percent'] = float(args.default_effective_tp_trailing_percent)
    param['sl_percent'] = float(args.sl_percent)
    param['reversal_num_intervals'] = int(args.reversal_num_intervals)
    param['trailing_sl_min_percent_linear'] = float(args.trailing_sl_min_percent_linear)
    param['non_linear_pow'] = float(args.non_linear_pow)
    
    param['loop_freq_ms'] = int(args.loop_freq_ms)

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

async def main():
    parse_args()
    redis_client : StrictRedis = init_redis_client()

    gateway_id : str = param['gateway_id']
    ordergateway_pending_orders_topic = 'ordergateway_pending_orders_$GATEWAY_ID$'
    ordergateway_pending_orders_topic = ordergateway_pending_orders_topic.replace("$GATEWAY_ID$", gateway_id)
    
    ordergateway_executions_topic = "ordergateway_executions_$GATEWAY_ID$"
    ordergateway_executions_topic = ordergateway_executions_topic.replace("$GATEWAY_ID$", gateway_id)

    position_cacnle_file : str = f"tp_algo_position_cache_{gateway_id}.csv"

    notification_params : Dict[str, Any] = param['notification']

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

    if encrypt_decrypt_with_aws_kms:
        aws_kms_key_id = str(os.getenv('AWS_KMS_KEY_ID'))

        aws_kms = AwsKmsUtil(key_id=aws_kms_key_id, profile_name=None)
        api_key = aws_kms.decrypt(api_key.encode())
        secret = aws_kms.decrypt(secret.encode())
        if passphrase:
            passphrase = aws_kms.decrypt(passphrase.encode())

    exchange : Union[AnyExchange, None] = await async_instantiate_exchange(
        gateway_id=param['gateway_id'],
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        default_type=param['default_type'],
        rate_limit_ms=param['rate_limit_ms']
    )
    if exchange:
        markets = await exchange.load_markets() # type: ignore
        market = markets[param['ticker']]
        multiplier = market['contractSize'] if 'contractSize' in market and market['contractSize'] else 1

        balances = await exchange.fetch_balance() # type: ignore
        log(f"Balances: {json.dumps(balances, indent=4)}") # type: ignore

        if not param['load_entry_from_cache']:
            # STEP 1. Make entry
            entry_positions : List[DivisiblePosition] = [
                DivisiblePosition(
                    ticker = param['ticker'],
                    side = param['side'],
                    amount = param['amount_base_ccy'],
                    leg_room_bps = param['leg_room_bps'],
                    order_type = param['order_type'],
                    slices = param['slices'],
                    wait_fill_threshold_ms = param['wait_fill_threshold_ms']
                )
            ]
            executed_positions : Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None] = execute_positions(
                                                                                                            redis_client=redis_client,
                                                                                                            positions=entry_positions,
                                                                                                            ordergateway_pending_orders_topic=ordergateway_pending_orders_topic,
                                                                                                            ordergateway_executions_topic=ordergateway_executions_topic
                                                                                                            )
        else:
            with open(position_cacnle_file, 'r') as f:
                executed_position = json.load(f)
                executed_positions = [ executed_position ]
            
        if executed_positions or param['load_entry_from_cache']:
            executed_position = executed_positions[0] # We sent only one DivisiblePosition.
            log(f"{'Entry dispatched.' if not param['load_entry_from_cache'] else 'Entry loaded from cache.'} {json.dumps(executed_position, indent=4)}") # type: ignore
            if executed_position['done']:
                def _reversal(
                    direction : str,  # up or down
                    last_candles
                ) -> bool:
                    if direction == "down" and all([ candle[1]<candle[4] for candle in last_candles ]): # All green?
                        return True
                    elif direction == "up" and all([ candle[1]>candle[4] for candle in last_candles ]): # All red?
                        return True
                    else:
                        return False
                    
                entry_price = executed_position['average_cost']
                amount_filled_usdt = entry_price * abs(executed_position['filled_amount'])
                if param['side'] == 'buy':
                    tp_min_price = entry_price * (1+param['tp_min_percent']/100)
                    tp_max_price = entry_price * (1+param['tp_max_percent']/100)
                    sl_price = entry_price * (1-param['sl_percent']/100)
                else:
                    tp_min_price = entry_price * (1-param['tp_min_percent']/100)
                    tp_max_price = entry_price * (1-param['tp_max_percent']/100)
                    sl_price = entry_price * (1+param['sl_percent']/100)

                if not param['load_entry_from_cache']:
                    executed_position['position'] = {
                        'status' : 'open',
                        'max_unrealized_pnl' : 0,
                        'entry_price' : entry_price,
                        'amount_base_ccy' : executed_position['filled_amount'],
                        'tp_min_price' : tp_min_price,
                        'tp_max_price' : tp_max_price,
                        'sl_price' : sl_price,
                        'multiplier' : multiplier
                    }
                    with open(position_cacnle_file, 'w') as f:
                        json.dump(executed_position, f)

                    dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Execution done, Entry succeeded. {param['ticker']} {param['side']} {param['amount_base_ccy']} (USD amount: {amount_filled_usdt}) @ {entry_price}", message=executed_position['position'], footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                unrealized_pnl : float = 0
                max_unrealized_pnl : float = 0
                unrealized_pnl_percent : float = 0
                max_unrealized_pnl_percent : float = 0
                loss_trailing : float = 0
                effective_tp_trailing_percent : float = param['default_effective_tp_trailing_percent']
                reversal : bool = False
                tp : bool = False
                sl : bool = False
                position_break : bool = False
                while (not tp and not sl and not position_break):
                    try:
                        position_from_exchange = await exchange.fetch_position(param['ticker'])

                        if exchange.options['defaultType']!='spot':
                            if not position_from_exchange and param['load_entry_from_cache']:
                                    position_break = True

                                    err_msg = f"{param['ticker']}: Position break! expected: {executed_position['position']['amount_base_ccy']}, actual: 0"
                                    log(err_msg)
                                    dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Position break! {param['ticker']}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                                
                            if position_from_exchange:
                                position_from_exchange_num_contracts = position_from_exchange['contracts']
                                if position_from_exchange and position_from_exchange['side']=='short':
                                    position_from_exchange_num_contracts = position_from_exchange_num_contracts *-1 if position_from_exchange_num_contracts>0 else position_from_exchange_num_contracts

                                position_from_exchange_base_ccy  = position_from_exchange_num_contracts * multiplier

                                if position_from_exchange_base_ccy!=executed_position['position']['amount_base_ccy']:
                                    position_break = True

                                    err_msg = f"{param['ticker']}: Position break! expected: {executed_position['position']['amount_base_ccy']}, actual: {position_from_exchange_base_ccy}"
                                    log(err_msg)
                                    dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Position break! {param['ticker']}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                        
                        if position_break:
                            log(f"Position break! Exiting execution. Did you manually close the trade?")
                            break
                        
                        else:
                            trailing_candles = await exchange.fetch_ohlcv(symbol=param['ticker'], timeframe='1h', limit=24) # type: ignore
                            trailing_candles = trailing_candles[-param['reversal_num_intervals']:]
                            
                            reversal = _reversal(
                                direction='up' if param['side']=='buy' else 'down',
                                last_candles=trailing_candles
                            )

                            ob = await exchange.fetch_order_book(symbol=param['ticker'], limit=10)
                            best_ask = min([x[0] for x in ob['asks']])
                            best_bid = max([x[0] for x in ob['bids']])
                            mid = (best_ask+best_bid)/2
                            spread_bps = (best_ask/best_bid - 1) * 10000
                            
                            if param['side']=='buy':
                                unrealized_pnl = (mid - entry_price) * param['amount_base_ccy']
                            else:
                                unrealized_pnl = (entry_price - mid) * param['amount_base_ccy']
                            unrealized_pnl_percent = unrealized_pnl / amount_filled_usdt * 100

                            if unrealized_pnl>max_unrealized_pnl:
                                max_unrealized_pnl = unrealized_pnl
                                max_unrealized_pnl_percent = max_unrealized_pnl / amount_filled_usdt * 100
                                executed_position['position']['max_unrealized_pnl'] = max_unrealized_pnl

                                with open(position_cacnle_file, 'w') as f:
                                    json.dump(executed_position, f)

                            if (
                                (unrealized_pnl_percent>=param['tp_min_percent'] and unrealized_pnl<max_unrealized_pnl)
                                or loss_trailing>0 # once your trade pnl crosses tp_min_percent, trailing stops is (and will remain) active.
                            ):
                                loss_trailing = (1 - unrealized_pnl/max_unrealized_pnl) * 100
                            
                            '''
                            Have a look at this for a visual explaination how "Gradually tightened stops" works:
                                https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/trading_util_tests.ipynb
                            '''
                            _effective_tp_trailing_percent = calc_eff_trailing_sl(
                                tp_min_percent = param['tp_min_percent'],
                                tp_max_percent = param['tp_max_percent'],
                                sl_percent_trailing = param['sl_percent_trailing'],
                                pnl_percent_notional = max_unrealized_pnl_percent, # Note: Use [max]_unrealized_pnl_percent, not unrealized_pnl_percent!
                                default_effective_tp_trailing_percent = param['default_effective_tp_trailing_percent'],
                                linear=True if param['tp_max_percent'] >= param['trailing_sl_min_percent_linear'] else False, # If tp_max_percent far (>100bps for example), there's more uncertainty if target can be reached: Go with linear.
                                pow=param['non_linear_pow']
                            )

                            # Once pnl pass tp_min_percent, trailing stops will be activated. Even if pnl fall back below tp_min_percent.
                            effective_tp_trailing_percent = min(effective_tp_trailing_percent, _effective_tp_trailing_percent)

                            log(f"unrealized_pnl: {round(unrealized_pnl,4)}, unrealized_pnl_percent: {round(unrealized_pnl_percent,4)}, max_unrealized_pnl_percent: {round(max_unrealized_pnl_percent,4)}, loss_trailing: {loss_trailing}, effective_tp_trailing_percent: {effective_tp_trailing_percent}, reversal: {reversal}")

                            # STEP 2. Exit
                            if unrealized_pnl>0:
                                if reversal and unrealized_pnl_percent >= param['tp_min_percent']:
                                    tp = True
                                elif loss_trailing>=effective_tp_trailing_percent:
                                    tp = True
                            else:
                                if abs(unrealized_pnl_percent)>=param['sl_percent']:
                                    sl = True

                            if tp or sl:
                                exit_positions : List[DivisiblePosition] = [
                                    DivisiblePosition(
                                        ticker = param['ticker'],
                                        side = 'sell' if param['side']=='buy' else 'buy',
                                        amount = param['amount_base_ccy'],
                                        leg_room_bps = param['leg_room_bps'],
                                        order_type = param['order_type'],
                                        slices = param['slices'],
                                        wait_fill_threshold_ms = param['wait_fill_threshold_ms'],

                                        reduce_only=True
                                    )
                                ]
                                executed_positions : Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None] = execute_positions(
                                                                                                                                redis_client=redis_client,
                                                                                                                                positions=exit_positions,
                                                                                                                            ordergateway_pending_orders_topic=ordergateway_pending_orders_topic,
                                                                                                                            ordergateway_executions_topic=ordergateway_executions_topic
                                                                                                                            )
                                if executed_positions:
                                    executed_position_close = executed_positions[0] # We sent only one DivisiblePosition.
                                    log(f"Position closing. {json.dumps(executed_position, indent=4)}") # type: ignore
                                    if executed_position_close['done']:
                                        executed_position_close['position'] = executed_position['position']

                                        if param['side']=='buy':
                                            closed_pnl = (executed_position_close['average_cost'] - entry_price) * param['amount_base_ccy']
                                        else:
                                            closed_pnl = (entry_price - executed_position_close['average_cost']) * param['amount_base_ccy']
                                        executed_position_close['position']['max_unrealized_pnl'] = max_unrealized_pnl
                                        executed_position_close['position']['closed_pnl'] = closed_pnl
                                        executed_position_close['position']['status'] = 'closed'

                                        with open(position_cacnle_file, 'w') as f:
                                            json.dump(executed_position_close, f)

                                        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Execution done, {'TP' if tp else 'SL'} succeeded. closed_pnl: {closed_pnl}", message=executed_position_close, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                                    else:
                                        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Exit execution failed. {param['ticker']}", message=executed_position_close, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                                    await exchange.close()

                    except Exception as loop_err:
                        log(f"Error: {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}", log_level=LogLevel.ERROR)
                    finally:
                        time.sleep(int(param['loop_freq_ms']/1000))
            
            else:
                dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Entry execution failed. {param['ticker']} {param['side']} {param['amount_base_ccy']}", message=executed_position, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)

asyncio.run(main())