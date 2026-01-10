# type: ignore
import sys
import traceback
import os
import logging
from dotenv import load_dotenv
import argparse
import re
from datetime import datetime, timedelta, timezone
import time
import arrow
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Union, Callable
from io import StringIO
import json
import asyncio
from redis import StrictRedis
import pandas as pd
import numpy as np
import inspect
from tabulate import tabulate

from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.ordergateway.client import DivisiblePosition, execute_positions
from util.datetime_util import parse_trading_window
from util.market_data_util import async_instantiate_exchange, interval_to_ms
from util.trading_util import calc_eff_trailing_sl
from util.notification_util import dispatch_notification
from util.aws_util import AwsKmsUtil

from siglab_py.constants import INVALID, JSON_SERIALIZABLE_TYPES, LogLevel, PositionStatus, OrderSide 


'''
For dry-runs/testing, swap back to StrategyBase, it will not fire an order.
'''
# from strategy_base import StrategyBase as TargetStrategy # Import whatever strategy subclassed from StrategyBase here!
from macdrsi_crosses_15m_tc_strategy import MACDRSICrosses15mTCStrategy as TargetStrategy

current_filename = os.path.basename(__file__)

'''
Error: RuntimeError: aiodns needs a SelectorEventLoop on Windows.
Hack, by far the filthest hack I done in my career: Set SelectorEventLoop on Windows
'''
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

'''
Usage:
    Step 1. Start candles_providers
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python candles_provider.py --provider_id mds_assign_aaa --candle_size 1h --how_many_candles 720 --redis_ttl_ms 3600000
        python candles_provider.py --provider_id mds_assign_bbb --candle_size 15m --how_many_candles 10080 --redis_ttl_ms 3600000

        Note: how_many_candles should be larger than compute_candles_stats.sliding_window_how_many_candles by a few times.
            720 = 24 x 30 days  
            10080 = 60 x 24 x 7 days

    Step 2. Start candles_ta_providers
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python candles_ta_provider.py --candle_size 1h --ma_long_intervals 48 --ma_short_intervals 12 --boillenger_std_multiples 2 --redis_ttl_ms 3600000 --processed_hash_queue_max_size 999 --pypy_compat N
        python candles_ta_provider.py --candle_size 15m --ma_long_intervals 150 --ma_short_intervals 5 --boillenger_std_multiples 2 --redis_ttl_ms 3600000 --processed_hash_queue_max_size 999 --pypy_compat N

        Note, for 15m bars, a sliding window of size 150 means 150 x 15m = 2250 minutes

    Step 3. Start orderbooks_provider
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python orderbooks_provider.py --provider_id mds_assign_ccc --instance_capacity 25 --ts_delta_observation_ms_threshold 150 --ts_delta_consecutive_ms_threshold 150 --redis_ttl_ms 3600000

    Step 4. To trigger candles_providers and orderbooks_provider
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python trigger_provider.py --provider_id mds_assign_aaa --tickers "okx_linear|SOL/USDT:USDT"
        python trigger_provider.py --provider_id mds_assign_bbb --tickers "okx_linear|SOL/USDT:USDT"
        python trigger_provider.py --provider_id mds_assign_ccc --tickers "okx_linear|SOL/USDT:USDT"

    Step 5. Start strategy_executor
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python strategy_executor.py --gateway_id hyperliquid_01 --default_type linear --rate_limit_ms 100 --encrypt_decrypt_with_aws_kms Y --aws_kms_key_id xxx --apikey xxx --secret xxx --ticker SUSHI/USDC:USDC --order_type limit --amount_base_ccy 45 --residual_pos_usdt_threshold 10 --slices 3 --wait_fill_threshold_ms 15000 --leg_room_bps 5 --tp_min_percent 1.5 --tp_max_percent 2.5 --sl_percent_trailing 50 --sl_percent 1 --reversal_num_intervals 3 --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx --load_entry_from_cache Y --economic_calendar_source xxx --block_entry_impacting_events Y --num_intervals_current_ecoevents 96 --trading_window_start Mon_00:00 --trading_window_end Fri_17:00

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

                        "--ticker", "SOL/USDC:USDC",
                        "--order_type", "limit",
                        "--amount_base_ccy", "3",
                        "--slices", "3",
                        "--wait_fill_threshold_ms", "15000",
                        "--leg_room_bps", "5",
                        "--tp_min_percent", "3",
                        "--tp_max_percent", "5",
                        "--sl_percent_trailing", "35",
                        "--sl_percent", "2.5",
                        "--reversal_num_intervals", "3",

                        "--economic_calendar_source", "xxx",
                        "--block_entry_impacting_events","Y",
                        "--num_intervals_current_ecoevents", "96",

                        "--trading_window_start", "Mon_00:00",
                        "--trading_window_end", "Fri_17:00",

                        "--load_entry_from_cache", "Y",

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

    'rolldate_tz' : 'Asia/Hong_Kong', # Roll date based on what timezone?

    # economic_calendar related
    'mapped_regions' : [ 'united_states' ],

    'mapped_event_codes' : [ 
        'core_inflation_rate_mom', 'core_inflation_rate_yoy',
        'inflation_rate_mom', 'inflation_rate_yoy', 
        'fed_interest_rate_decision',
        'fed_chair_speech',
        'core_pce_price_index_mom',
        'core_pce_price_index_yoy',
        'unemployment_rate',
        'non_farm_payrolls',
        'gdp_growth_rate_qoq_adv', 
        'gdp_growth_rate_qoq_final', 
        'gdp_growth_rate_yoy',

        'manual_event'
    ],
    'max_current_economic_calendar_age_sec' : 10,
    'num_intervals_current_ecoevents' : 4* 24, # x4 because lo_interval "15m" per 'lo_candles_w_ta_topic': If you want to convert to 24 hrs ...

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
            "hi_candles_provider_topic" : "mds_assign_aaa",
            "lo_candles_provider_topic" : "mds_assign_bbb",
            "orderbooks_provider_topic" : "mds_assign_ccc",
            "hi_candles_w_ta_topic" : "candles_ta-SOL-USDT-SWAP-okx_linear-1h",
            "lo_candles_w_ta_topic" : "candles_ta-SOL-USDT-SWAP-okx_linear-15m",
            "orderbook_topic" : "orderbooks_SOL/USDT:USDT_okx_linear",
            
            "full_economic_calendars_topic" : "economic_calendars_full_$SOURCE$",
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

POSITION_CACHE_FILE_NAME = f"{TargetStrategy.__name__}_position_cache_$GATEWAY_ID$.csv"
POSITION_CACHE_COLUMNS = [ 
            'exchange', 'ticker',
            'status', 
            'pos', 'pos_usdt', 'multiplier', 'created', 'closed', 
            'pos_entries',
            'spread_bps', 'entry_px', 'close_px', 'last_interval_px',
            'ob_mid', 'ob_best_bid', 'ob_best_ask',
            'unreal', 'unreal_live', 'real',
            'max_unreal_live',
            'max_unreal_live_bps',
            'max_unreal_pessimistic_bps',
            'max_pain',
            'entry_target_price',
            'running_sl_percent_hard',
            'sl_trailing_min_threshold_crossed',
            'sl_percent_trailing',
            'tp_min_target',
        ]

ORDERHIST_CACHE_FILE_NAME = f"{TargetStrategy.__name__}_orderhist_cache_$GATEWAY_ID$.csv"
ORDERHIST_CACHE_COLUMNS = [  'datetime', 'exchange', 'ticker', 'reason', 'side', 'avg_price', 'amount', 'pnl', 'pnl_bps', 'max_pain' ]

def log(message : str, log_level : LogLevel = LogLevel.INFO):
    if log_level.value<LogLevel.WARNING.value:
        logger.info(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(f"{datetime.now()} {message}")

def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument("--gateway_id", help="gateway_id: Where are you sending your order?", default=None)

    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)

    parser.add_argument("--trading_window_start", help="Start of trading window. Set as blank if trading not confined to particular trading window. Format: Fri_00:00", default='')
    parser.add_argument("--trading_window_end", help="End of trading window.", default='')

    parser.add_argument("--encrypt_decrypt_with_aws_kms", help="Y or N. If encrypt_decrypt_with_aws_kms=N, pass in apikey, secret and passphrase unencrypted (Not recommended, for testing only). If Y, they will be decrypted using AMS KMS key.", default='N')
    parser.add_argument("--aws_kms_key_id", help="AWS KMS key ID", default=None)
    parser.add_argument("--apikey", help="Exchange apikey", default=None)
    parser.add_argument("--secret", help="Exchange secret", default=None)
    parser.add_argument("--passphrase", help="Exchange passphrase", default=None)
    parser.add_argument("--verbose", help="logging verbosity, Y or N (default).", default='N')

    parser.add_argument("--ticker", help="Ticker you're trading. Example BTC/USDC:USDC", default=None)
    parser.add_argument("--order_type", help="Order type: market or limit", default=None)
    parser.add_argument("--amount_base_ccy", help="Order amount in base ccy (Not # contracts). Always positive, even for sell trades.", default=None)
    parser.add_argument("--residual_pos_usdt_threshold", help="If pos_usdt<=residual_pos_usdt_threshold (in USD, default $100), PositionStatus will be marked to CLOSED.", default=100)
    parser.add_argument("--leg_room_bps", help="Leg room, for Limit orders only. A more positive leg room is a more aggressive order to get filled. i.e. Buy at higher price, Sell at lower price.", default=5)
    parser.add_argument("--slices", help="Algo can break down larger order into smaller slices. Default: 1", default=1)
    parser.add_argument("--wait_fill_threshold_ms", help="Limit orders will be cancelled if not filled within this time. Remainder will be sent off as market order.", default=15000)

    parser.add_argument("--load_entry_from_cache", help="Y (default) or N. This is for algo restart scenario where you don't want make entry again. In this case existing/running position loaded from cache.", default='Y')

    parser.add_argument("--tp_min_percent", help="For trailing stops. Min TP in percent, i.e. No TP until pnl at least this much.", default=None)
    parser.add_argument("--tp_max_percent", help="For trailing stops. Max TP in percent, i.e. Price target", default=None)
    parser.add_argument("--sl_percent_trailing", help="For trailing stops. trailing SL in percent, please refer to trading_util.calc_eff_trailing_sl for documentation.", default=None)
    parser.add_argument("--default_effective_tp_trailing_percent", help="Default for sl_percent_trailing when pnl still below tp_min_percent. Default: float('inf'), meaing trailing stop won't be fired.", default=float('inf'))
    parser.add_argument("--sl_percent", help="Hard stop in percent.", default=2)
    parser.add_argument("--sl_num_intervals_delay", help="Number of intervals to wait before re-entry allowed after SL. Default 1", default=1)
    parser.add_argument("--reversal_num_intervals", help="How many reversal candles to confirm reversal?", default=3)
    parser.add_argument("--trailing_sl_min_percent_linear", help="This is threshold used for tp_algo to decide if use linear stops tightening, or non-linear. If tp_max_percent far (>200bps for example), there's more uncertainty if target can be reached: Go with linear. Default: 2% (200 bps)", default=2.0)
    parser.add_argument("--non_linear_pow", help="For non-linear trailing stops tightening, have a look at call to 'calc_eff_trailing_sl'. Default: 5", default=5)
    
    parser.add_argument("--economic_calendar_source", help="Source of economic calendar'. Default: None", default=None)
    parser.add_argument("--num_intervals_current_ecoevents", help="Num intervals to block on incoming/outgoing economic events. For 15m bars for example, num_intervals_current_ecoevents=4*24 means 24 hours. Default: 0", default=0)
    parser.add_argument("--block_entry_impacting_events", help="Block entries if any impacting economic events 'impacting_economic_calendars'. Default N", default='N')
    
    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=5000)

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()

    param['gateway_id'] = args.gateway_id
    param['default_type'] = args.default_type
    param['rate_limit_ms'] = int(args.rate_limit_ms)

    param['trading_window_start'] = args.trading_window_start
    param['trading_window_end'] = args.trading_window_end
    
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
    if args.verbose:
        if args.verbose=='Y':
            param['verbose'] = True
        else:
            param['verbose'] = False
    else:
        param['verbose'] = False

    param['ticker'] = args.ticker
    param['order_type'] = args.order_type
    param['amount_base_ccy'] = float(args.amount_base_ccy)
    param['residual_pos_usdt_threshold'] = float(args.residual_pos_usdt_threshold)
    param['leg_room_bps'] = int(args.leg_room_bps)
    param['slices'] = int(args.slices)
    param['wait_fill_threshold_ms'] = int(args.wait_fill_threshold_ms)

    if args.load_entry_from_cache:
        if args.load_entry_from_cache=='Y':
            param['load_entry_from_cache'] = True
        else:
            param['load_entry_from_cache'] = False
    else:
        param['load_entry_from_cache'] = True

    param['tp_min_percent'] = float(args.tp_min_percent)
    param['tp_max_percent'] = float(args.tp_max_percent)
    param['sl_percent_trailing'] = float(args.sl_percent_trailing)
    param['default_effective_tp_trailing_percent'] = float(args.default_effective_tp_trailing_percent)
    param['sl_percent'] = float(args.sl_percent)
    param['sl_num_intervals_delay'] = int(args.sl_num_intervals_delay)
    param['reversal_num_intervals'] = int(args.reversal_num_intervals)
    param['trailing_sl_min_percent_linear'] = float(args.trailing_sl_min_percent_linear)
    param['non_linear_pow'] = float(args.non_linear_pow)

    param['economic_calendar_source'] = args.economic_calendar_source

    if args.block_entry_impacting_events:
        if args.block_entry_impacting_events=='Y':
            param['block_entry_impacting_events'] = True
        else:
            param['block_entry_impacting_events'] = False
    else:
        param['block_entry_impacting_events'] = False
    
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

def fetch_economic_events(redis_client, topic) -> List[Dict]:
    restored = redis_client.get(topic)
    if restored:
        restored = json.loads(restored)
        for economic_calendar in restored:
            '''
            Format: 
                'calendar_id': '1234567'
                'category': 'Building Permits'
                'region': 'united_states'
                'event': 'Building Permits Prel'
                'event_code': 'building_permits_prel'
                'ccy': ''
                'importance': '3'
                'actual': 1.386
                'forecast': 1.45
                'previous': 1.44
                'pos_neg': 'bearish'
                'datetime': datetime.datetime(2024, 6, 20, 20, 30)
                'calendar_item_timestamp_ms': 1718886600000
                'calendar_item_timestamp_sec': 1718886600
                'source': 'xxx'    
            '''
            economic_calendar['datetime'] = arrow.get(economic_calendar['datetime']).datetime
            economic_calendar['datetime'] = economic_calendar['datetime'].replace(tzinfo=None) 
    return restored

async def main(
    order_notional_adj_func : Callable[..., Dict[str, float]],
    allow_entry_initial_func : Callable[..., Dict[str, bool]],
    allow_entry_final_func : Callable[..., Dict[str, Union[bool, float, None]]],
    sl_adj_func : Callable[..., Dict[str, float]],
    trailing_stop_threshold_eval_func : Callable[..., Dict[str, float]],
    tp_eval_func : Callable[..., bool]
):
    parse_args()
    redis_client : StrictRedis = init_redis_client()

    gateway_id : str = param['gateway_id']
    exchange_name : str = gateway_id.split('_')[0]
    ticker : str = param['ticker']
    ordergateway_pending_orders_topic : str = 'ordergateway_pending_orders_$GATEWAY_ID$'
    ordergateway_pending_orders_topic : str = ordergateway_pending_orders_topic.replace("$GATEWAY_ID$", gateway_id)
    
    ordergateway_executions_topic : str = "ordergateway_executions_$GATEWAY_ID$"
    ordergateway_executions_topic : str = ordergateway_executions_topic.replace("$GATEWAY_ID$", gateway_id)

    hi_candles_w_ta_topic : str = param['mds']['topics']['hi_candles_w_ta_topic']
    lo_candles_w_ta_topic : str = param['mds']['topics']['lo_candles_w_ta_topic']
    orderbook_topic : str = param['mds']['topics']['orderbook_topic']

    # economic_calendar_source
    full_economic_calendars_topic : str = param['mds']['topics']['full_economic_calendars_topic']
    full_economic_calendars_topic  = full_economic_calendars_topic.replace('$SOURCE$', param['economic_calendar_source']) if param['economic_calendar_source'] else None

    log(f"hi_candles_w_ta_topic: {hi_candles_w_ta_topic}")
    log(f"lo_candles_w_ta_topic: {lo_candles_w_ta_topic}")
    log(f"orderbook_topic: {orderbook_topic}")
    log(f"ordergateway_pending_orders_topic: {ordergateway_pending_orders_topic}")
    log(f"ordergateway_executions_topic: {ordergateway_executions_topic}")
    log(f"full_economic_calendars_topic: {full_economic_calendars_topic}")

    # aliases
    algo_param = param

    hi_candle_size : str = hi_candles_w_ta_topic.split('-')[-1]
    lo_candle_size : str = lo_candles_w_ta_topic.split('-')[-1]
    hi_interval = hi_candle_size[-1]
    hi_num_intervals : int = int(hi_candle_size.replace(hi_interval,''))
    hi_interval_ms : int = interval_to_ms(hi_interval) * hi_num_intervals
    lo_interval = lo_candle_size[-1]
    lo_num_intervals : int = int(lo_candle_size.replace(lo_interval,''))
    lo_interval_ms : int = interval_to_ms(lo_interval) * lo_num_intervals
    
    num_intervals_current_ecoevents_ms : int = lo_interval_ms * param['num_intervals_current_ecoevents']
    
    strategy_indicators = TargetStrategy.get_strategy_indicators()
    position_cache_columns = POSITION_CACHE_COLUMNS + strategy_indicators
    pd_position_cache = pd.DataFrame(columns=position_cache_columns)

    orderhist_cache = pd.DataFrame(columns=ORDERHIST_CACHE_COLUMNS)

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
        gateway_id=gateway_id,
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        default_type=param['default_type'],
        rate_limit_ms=param['rate_limit_ms'],
        verbose=param['verbose']
    )
    if exchange:
        markets = await exchange.load_markets() 
        market = markets[ticker]
        multiplier = market['contractSize'] if 'contractSize' in market and market['contractSize'] else 1

        balances = await exchange.fetch_balance() 
        log(f"Balances: {json.dumps(balances, indent=4)}") 

        # Lambdas preparation
        order_notional_adj_func_sig = inspect.signature(order_notional_adj_func)
        order_notional_adj_func_params = order_notional_adj_func_sig.parameters.keys()
        allow_entry_initial_func_sig = inspect.signature(allow_entry_initial_func)
        allow_entry_initial_func_params = allow_entry_initial_func_sig.parameters.keys()
        allow_entry_final_func_sig = inspect.signature(allow_entry_final_func)
        allow_entry_final_func_params = allow_entry_final_func_sig.parameters.keys()
        sl_adj_func_sig = inspect.signature(sl_adj_func)
        sl_adj_func_params = sl_adj_func_sig.parameters.keys()
        trailing_stop_threshold_eval_func_sig = inspect.signature(trailing_stop_threshold_eval_func)
        trailing_stop_threshold_eval_func_params = trailing_stop_threshold_eval_func_sig.parameters.keys() 
        tp_eval_func_sig = inspect.signature(tp_eval_func)
        tp_eval_func_params = tp_eval_func_sig.parameters.keys()   

        # Trigger candles providers
        def _trigger_producers(
            redis_client : StrictRedis, 
            exchange_tickers : List, 
            candles_partition_assign_topic : str):
            # https://redis.io/commands/publish/
            redis_client.publish(channel=candles_partition_assign_topic, message=json.dumps(exchange_tickers).encode('utf-8'))
        hi_candles_provider_topic = param['mds']['topics']['hi_candles_provider_topic']
        lo_candles_provider_topic = param['mds']['topics']['lo_candles_provider_topic']
        _trigger_producers(redis_client, [ param['ticker'] ], hi_candles_provider_topic)
        _trigger_producers(redis_client, [ param['ticker'] ], lo_candles_provider_topic)

        # Load cached positions from disk, if any
        if os.path.exists(POSITION_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id)) and os.path.getsize(POSITION_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id))>0:
            pd_position_cache = pd.read_csv(POSITION_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id))
            pd_position_cache.drop(pd_position_cache.columns[pd_position_cache.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            pd_position_cache.replace([np.nan], [None], inplace=True)

            pd_position_cache = pd_position_cache[POSITION_CACHE_COLUMNS]

        if os.path.exists(ORDERHIST_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id)) and os.path.getsize(ORDERHIST_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id))>0:
            orderhist_cache = pd.read_csv(ORDERHIST_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id))
            orderhist_cache.drop(orderhist_cache.columns[orderhist_cache.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            orderhist_cache.replace([np.nan], [None], inplace=True)

            if 'datetime' in orderhist_cache.columns:
                orderhist_cache['datetime'] = pd.to_datetime(orderhist_cache['datetime'])

        block_entries : bool = False
        hi_row, hi_row_tm1 = None, None
        lo_row, lo_row_tm1 = None, None
        pos_unreal_live : float = 0
        unrealized_pnl_live_pessimistic : float = 0 # Evaluated using latest candle's open
        max_unrealized_pnl : float = 0
        pnl_live_bps : float = 0
        pnl_pessimistic_bps : float = 0
        max_unrealized_pnl_percent : float = 0
        loss_trailing : float = 0
        effective_tp_trailing_percent : float = param['default_effective_tp_trailing_percent']
        this_ticker_open_trades : List[Dict] = []
        reversal : bool = False
        tp : bool = False
        sl : bool = False
        executed_position = None
        position_break : bool = False
        while (not tp and not sl and not position_break):
            try:
                dt_now = datetime.now()
                block_entries = False

                dt_targettz = datetime.fromtimestamp(dt_now.timestamp(), tz=ZoneInfo(param['rolldate_tz']))
                today_dayofweek = dt_targettz.weekday()

                delta_hour = int(
                                    (dt_targettz.replace(tzinfo=None) - dt_now).total_seconds()/3600
                                )
                
                log(f"rolldate_tz: {param['rolldate_tz']}, dt_now (local): {dt_now}, dt_targettz: {dt_targettz}, delta_hour: {delta_hour}")
                
                if param['trading_window_start'] and param['trading_window_end']:
                    trading_window : Dict[str, str] = {
                        'start' : param['trading_window_start'],
                        'end' : param['trading_window_end']
                    }
                    parsed_trading_window = parse_trading_window(dt_targettz, trading_window)
                    if not parsed_trading_window['in_window']:
                        block_entries = True

                    log(f"trading_window start: {param['trading_window_start']}, end: {param['trading_window_end']}, in_window: {parsed_trading_window['in_window']}")
                else:
                    log(f"No trading window specified")

                if full_economic_calendars_topic:
                    full_economic_calendars = fetch_economic_events(redis_client, full_economic_calendars_topic)

                    impacting_economic_calendars = None
                    if full_economic_calendars:
                        impacting_economic_calendars = [ x for x in full_economic_calendars 
                                                            if x['event_code'] in param['mapped_event_codes']  
                                                            and x['region'] in param['mapped_regions']
                                                        ]
                        if impacting_economic_calendars:
                            impacting_economic_calendars =  [ x for x in impacting_economic_calendars 
                                                                if(
                                                                    (x['calendar_item_timestamp_ms'] - datetime.now().timestamp()*1000)>0 # Incoming events
                                                                    and (x['calendar_item_timestamp_ms'] - datetime.now().timestamp()*1000) < num_intervals_current_ecoevents_ms
                                                                ) or (
                                                                    (x['calendar_item_timestamp_ms'] - datetime.now().timestamp()*1000)<0 # Passed events
                                                                    and (datetime.now().timestamp()*1000 - x['calendar_item_timestamp_ms']) < num_intervals_current_ecoevents_ms/3 
                                                                )
                                                            ]
                            s_impacting_economic_calendars = " ".join([ x['event_code'] if x['event_code']!='manual_event' else x['event'] for x in impacting_economic_calendars ])

                    log(f"full_economic_calendars #rows: {len(full_economic_calendars) if full_economic_calendars else 0}")
                    log(f"impacting_economic_calendars #rows: {len(impacting_economic_calendars) if impacting_economic_calendars else 0} {s_impacting_economic_calendars}. block_entries: {block_entries}")
                    
                    if param['block_entry_impacting_events'] and impacting_economic_calendars:
                        block_entries = True
                
                log(f"block_entries: {block_entries}")

                position_cache_row = pd_position_cache.loc[(pd_position_cache.exchange==exchange_name) & (pd_position_cache.ticker==ticker)]
                if position_cache_row.shape[0]==0:
                    position_cache_row = {
                        'exchange': exchange_name,
                        'ticker' : ticker,

                        'status' : PositionStatus.UNDEFINED.name,
                        
                        'pos' : None, 
                        'pos_usdt' : None,
                        'multiplier' : multiplier,
                        'created' : None,
                        'closed' : None,

                        'pos_entries' : [],

                        'spread_bps' : None,
                        'entry_px' : None,
                        'close_px' : None,
                        'last_interval_px' : None,

                        'ob_mid' : None,
                        'ob_best_bid' : None,
                        'ob_best_ask' : None,
                        
                        'unreal_live' : 0,
                        'pnl_live_bps' : 0,
                        'pnl_pessimistic_bps' : 0,

                        'max_unreal_live' : 0,
                        'max_unreal_live_bps' : 0,
                        'max_unreal_pessimistic_bps' : 0,
                        'max_pain' : 0,
                        'entry_target_price' : None,
                        'sl_trailing_min_threshold_crossed' : False,
                        'running_sl_percent_hard' : param['sl_percent'],
                        'sl_percent_trailing' : param['sl_percent_trailing'],
                        'tp_min_target' : None,
                    }
                    position_cache_row.update({ind: None for ind in strategy_indicators})
                    pd_position_cache = pd.concat([pd_position_cache, pd.DataFrame([position_cache_row])], axis=0, ignore_index=True)
                    position_cache_row = pd_position_cache.loc[(pd_position_cache.exchange==exchange_name) & (pd_position_cache.ticker==ticker)]
            
                position_cache_row = position_cache_row.iloc[0]

                # Note: arrow.get will populate tzinfo
                pos = position_cache_row['pos'] if position_cache_row['pos'] else 0
                pos_usdt = position_cache_row['pos_usdt'] if position_cache_row['pos_usdt'] else 0
                pos_status = position_cache_row['status']
                if (pos==0 or pos_usdt<=param['residual_pos_usdt_threshold']) and pos_status==PositionStatus.OPEN.name:
                    pos_status = PositionStatus.CLOSED.name
                    pd_position_cache.loc[position_cache_row.name, 'status'] = pos_status 
                if pos_status!=PositionStatus.OPEN.name and (pos and pos!=0):
                    pos_status = PositionStatus.OPEN.name
                    pd_position_cache.loc[position_cache_row.name, 'status'] = pos_status 
                    
                pos_created = position_cache_row['created']
                pos_created = arrow.get(pos_created).datetime if pos_created and isinstance(pos_created, str) else pos_created
                total_sec_since_pos_created = INVALID
                if pos_created:
                    pos_created = pos_created.replace(tzinfo=None)
                    total_sec_since_pos_created = (dt_now - pos_created).total_seconds()
                pos_closed = position_cache_row['closed']
                pos_closed = arrow.get(pos_closed).datetime if pos_closed and isinstance(pos_closed, str) else pos_closed
                if pos_closed:
                    pos_closed = pos_closed.replace(tzinfo=None)
                pos_side = OrderSide.UNDEFINED
                if pos_status!=PositionStatus.UNDEFINED.name:
                    pos_side = OrderSide.BUY if pos and pos>0 else OrderSide.SELL
                pos_entry_px = position_cache_row['entry_px']
                max_unreal_live = position_cache_row['max_unreal_live']
                if not max_unreal_live or pd.isna(max_unreal_live):
                    max_unreal_live = 0

                max_unreal_live_bps = max_unreal_live / abs(pos_usdt) * 10000 if pos_usdt!=0 else 0
                max_unreal_pessimistic_bps = position_cache_row['max_unreal_pessimistic_bps']
                max_pain = position_cache_row['max_pain'] if position_cache_row['max_pain'] else 0
                tp_max_price = position_cache_row['entry_target_price']
                running_sl_percent_hard = position_cache_row['running_sl_percent_hard']
                sl_trailing_min_threshold_crossed = position_cache_row['sl_trailing_min_threshold_crossed']
                effective_sl_percent_trailing = position_cache_row['sl_percent_trailing']
                tp_min_price = position_cache_row['tp_min_target']

                pos_entries = position_cache_row['pos_entries']
                if isinstance(pos_entries, str):
                    datetime_strings = re.findall(r'datetime\.datetime\(([^)]+)\)', pos_entries)
                    
                    pos_entries = []
                    for dt_str in datetime_strings:
                        dt_parts = [int(part.strip()) for part in dt_str.split(',')]
                        if len(dt_parts) == 7:
                            pos_entries.append(datetime(*dt_parts)) 
                        elif len(dt_parts) == 6:
                            pos_entries.append(datetime(*dt_parts, microsecond=0))
                num_pos_entries = len(pos_entries) if pos_entries else 0

                '''
                'fetch_position' is for perpetual. 
                    If you long, you'd see side = 'long' 
                    If you short, you'd see side = 'short'
                    'contracts' and 'notional' is always positive number.

                Example for a short,
                    'id': None
                    'symbol': 'BTC/USDT:USDT'
                    'timestamp': ...
                    'datetime': '...'
                    'lastUpdateTimestamp': ... (13 digits integer, in ms)
                    'initialMargin': 91.77863881
                    'initialMarginPercentage': 0.1005108176234939
                    'maintenanceMargin': 5.11374881
                    'maintenanceMarginPercentage': 0.005600290881174695
                    'entryPrice': 95312.2
                    'notional': 913.122
                    'leverage': 10.0
                    'unrealizedPnl': 0.86
                    'realizedPnl': None
                    'contracts': 0.01
                    'contractSize': 1.0
                    'marginRatio': None
                    'liquidationPrice': None
                    'markPrice': 91226.2
                    'lastPrice': None
                    'collateral': 0.0
                    'marginMode': None
                    'side': 'short'
                    'percentage': None
                    'stopLossPrice': None
                    'takeProfitPrice': None
                    'hedged': False

                For spots/margin trading, you should use 'fetch_balance' instsead. If you short you'd see:
                    BTC: { free: -5.2, total: -5.2 })
                '''
                position_from_exchange = await exchange.fetch_position(param['ticker']) 

                if exchange.options['defaultType']!='spot': 
                    if not position_from_exchange and param['load_entry_from_cache']:
                            position_break = True

                            err_msg = f"{param['ticker']}: Position break! expected: {executed_position['position']['amount_base_ccy']}, actual: 0" 
                            log(err_msg)
                            dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Position break! {param['ticker']}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                        
                    if executed_position and position_from_exchange:
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

                    hi_candles_valid, lo_candles_valid, orderbook_valid = False, False, False
                    trailing_candles = []

                    keys = [item.decode('utf-8') for item in redis_client.keys()] 
                    if hi_candles_w_ta_topic in keys:
                        message = redis_client.get(hi_candles_w_ta_topic)
                        if message:
                            message = message.decode('utf-8')
                        hi_candles_w_ta = json.loads(message) if message else None
                        pd_hi_candles_w_ta = pd.read_json(StringIO(hi_candles_w_ta))
                        pd_hi_candles_w_ta['timestamp_ms'] = pd_hi_candles_w_ta['timestamp_ms'].astype('int64') // 1_000_000
                        hi_row = pd_hi_candles_w_ta.iloc[-1]
                        hi_row_tm1 = pd_hi_candles_w_ta.iloc[-2]
                        candles_age = dt_now.timestamp() *1000 - hi_row['timestamp_ms']
                        if candles_age < hi_interval_ms:
                            hi_candles_valid = True
                        else:
                            hi_candles_valid = False
                            err_msg = {
                                'current_ts_ms' : dt_now.timestamp(),
                                'hi_row_timestamp_ms' : hi_row['timestamp_ms'],
                                'candles_age' : candles_age,
                                'hi_interval_ms' : hi_interval_ms
                            }
                            log(err_msg, LogLevel.CRITICAL)
                            dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid hi_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                    else:
                        hi_candles_valid = False
                        err_msg = f"hi candles missing, topic: {hi_candles_w_ta_topic}"
                        log(err_msg, LogLevel.CRITICAL)
                        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid hi_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                        
                    if lo_candles_w_ta_topic in keys:
                        message = redis_client.get(lo_candles_w_ta_topic)
                        if message:
                            message = message.decode('utf-8')
                        lo_candles_w_ta = json.loads(message) if message else None
                        pd_lo_candles_w_ta = pd.read_json(StringIO(lo_candles_w_ta))
                        pd_lo_candles_w_ta['timestamp_ms'] = pd_lo_candles_w_ta['timestamp_ms'].astype('int64') // 1_000_000
                        lo_row = pd_lo_candles_w_ta.iloc[-1]
                        lo_row_tm1 = pd_lo_candles_w_ta.iloc[-2]
                        candles_age = dt_now.timestamp() *1000 - lo_row['timestamp_ms']
                        if candles_age < lo_interval_ms:
                            lo_candles_valid = True
                            trailing_candles = pd_lo_candles_w_ta[['datetime', 'open', 'high', 'low', 'close']].tail(param['reversal_num_intervals']).values.tolist()

                        else:
                            lo_candles_valid = False
                            err_msg = {
                                'current_ts_ms' : dt_now.timestamp(),
                                'lo_row_timestamp_ms' : lo_row['timestamp_ms'],
                                'candles_age' : candles_age,
                                'lo_interval_ms' : lo_interval_ms
                            }
                            log(err_msg, LogLevel.CRITICAL)
                            dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid lo_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                    else:
                        lo_candles_valid = False
                        err_msg = f"lo candles missing, topic: {lo_candles_w_ta_topic}"
                        log(err_msg, LogLevel.CRITICAL)
                        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid hi_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                    if orderbook_topic in keys:
                        message = redis_client.get(orderbook_topic)
                        if message:
                            message = message.decode('utf-8')
                        ob = json.loads(message) if message else None
                        orderbook_valid = ob['is_valid']

                    else:
                        orderbook_valid = False
                        
                    if not orderbook_valid:
                        ob = await exchange.fetch_order_book(symbol=param['ticker'], limit=10) 
                        err_msg = f"orderbook missing, topic: {orderbook_topic}, fetch from REST instead"
                        log(err_msg, LogLevel.WARNING)
                        
                    if hi_candles_valid and lo_candles_valid: # On turn of interval, candles_provider may need a little time to publish latest candles
                        for indicator in strategy_indicators:
                            indicator_source : str = indicator.split(":")[0]
                            indicator_name = indicator.split(":")[-1]
                            if indicator_source=="lo_row":
                                indicator_value = lo_row[indicator_name]
                            elif indicator_source=="lo_row_tm1":
                                indicator_value = lo_row_tm1[indicator_name]
                            elif indicator_source=="hi_row":
                                indicator_value = hi_row[indicator_name]
                            elif indicator_source=="hi_row_tm1":
                                indicator_value = hi_row_tm1[indicator_name]
                            pd_position_cache.loc[position_cache_row.name, indicator] = indicator_value

                        best_ask = min([x[0] for x in ob['asks']])
                        best_bid = max([x[0] for x in ob['bids']])
                        mid = (best_ask+best_bid)/2
                        spread_bps = (best_ask/best_bid - 1) * 10000

                        last_candles=trailing_candles # alias

                        if pos_status!=PositionStatus.UNDEFINED.name:
                            if pos_side == OrderSide.BUY:
                                pos_unreal_live = (mid - pos_entry_px) * param['amount_base_ccy']
                                unrealized_pnl_live_pessimistic = pos_unreal_live
                                if total_sec_since_pos_created > (lo_interval_ms/1000):
                                    '''
                                    "pessimistic": To align with backtests, motivation is to avoid spikes and trigger trailing stops too early.
                                    But we need be careful with tall candles immediately upon entries.
                                       trailing_candles[-1] is latest candle
                                       trailing_candles[-1][1] is 'open' from latest candle
                                    Example long BTC, a mean reversion trade
                                        pos_entry_px    $97,000
                                        open            $99,000 (This is trailing_candles[-1][1], so it's big red candle)
                                        mid             $97,200 (Seconds after entry)

                                        pos_unreal_live                 $200 per BTC
                                        unrealized_pnl_live_pessimistic $2000 per BTC (This is very misleading! This would cause algo to TP prematurely!)
                                    Thus for new entries, 
                                        unrealized_pnl_live_pessimistic = pos_unreal_live
                                    '''
                                    unrealized_pnl_live_pessimistic = (trailing_candles[-1][1] - pos_entry_px) * param['amount_base_ccy']
                            elif pos_side == OrderSide.SELL:
                                pos_unreal_live = (pos_entry_px - mid) * param['amount_base_ccy']
                                unrealized_pnl_live_pessimistic = pos_unreal_live
                                if total_sec_since_pos_created > lo_interval_ms/1000:
                                    unrealized_pnl_live_pessimistic = (pos_entry_px - trailing_candles[-1][1]) * param['amount_base_ccy']
                        pnl_live_bps = pos_unreal_live / abs(pos_usdt) * 10000 if pos_usdt else 0
                        pnl_pessimistic_bps = unrealized_pnl_live_pessimistic / abs(pos_usdt) * 10000 if pos_usdt else 0

                        pd_position_cache.loc[position_cache_row.name, 'unreal_live'] = pos_unreal_live
                        pd_position_cache.loc[position_cache_row.name, 'pnl_live_bps'] = pnl_live_bps
                        pd_position_cache.loc[position_cache_row.name, 'pnl_pessimistic_bps'] = pnl_pessimistic_bps

                        kwargs = {k: v for k, v in locals().items() if k in trailing_stop_threshold_eval_func_params}
                        trailing_stop_threshold_eval_func_result = trailing_stop_threshold_eval_func(**kwargs)
                        tp_min_percent = trailing_stop_threshold_eval_func_result['tp_min_percent']
                        tp_max_percent = trailing_stop_threshold_eval_func_result['tp_max_percent']

                        pd_position_cache.loc[position_cache_row.name, 'spread_bps'] = spread_bps
                        pd_position_cache.loc[position_cache_row.name, 'ob_mid'] = mid
                        pd_position_cache.loc[position_cache_row.name, 'ob_best_bid'] = best_bid
                        pd_position_cache.loc[position_cache_row.name, 'ob_best_ask'] = best_ask

                        kwargs = {k: v for k, v in locals().items() if k in allow_entry_initial_func_params}
                        allow_entry_func_initial_result = allow_entry_initial_func(**kwargs)
                        allow_entry_long = allow_entry_func_initial_result['long']
                        allow_entry_short = allow_entry_func_initial_result['short']

                        allow_entry = allow_entry_long or allow_entry_short
                        allow_entry = allow_entry and pos_status!=PositionStatus.OPEN.name
                        if allow_entry and not block_entries:
                            kwargs = {k: v for k, v in locals().items() if k in allow_entry_final_func_params}
                            allow_entry_func_final_result = allow_entry_final_func(**kwargs)
                            allow_entry_final_long = allow_entry_func_final_result['long']
                            allow_entry_final_short = allow_entry_func_final_result['short']
                            target_price_long = allow_entry_func_final_result['target_price_long']
                            target_price_short = allow_entry_func_final_result['target_price_short']
                            
                            if allow_entry_final_long or allow_entry_final_short:
                                if allow_entry_final_long:
                                    side = 'buy'
                                    pnl_potential_bps = (target_price_long/mid - 1) *10000 if target_price_long else None
                                elif allow_entry_final_short:
                                    side = 'sell'
                                    pnl_potential_bps = (mid/target_price_short - 1) *10000 if target_price_short else None
                                else:
                                    raise ValueError("Either allow_long or allow_short!")

                                # tp_min_percent adj: Strategies where target_price not based on tp_max_percent, but variable
                                if pnl_potential_bps<tp_max_percent:
                                    tp_minmax_ratio = tp_min_percent/tp_max_percent
                                    tp_max_percent = pnl_potential_bps
                                    tp_min_percent = tp_minmax_ratio * tp_max_percent

                                kwargs = {k: v for k, v in locals().items() if k in order_notional_adj_func_params}
                                order_notional_adj_func_result = order_notional_adj_func(**kwargs)
                                target_order_notional = order_notional_adj_func_result['target_order_notional']
                                
                                entry_positions : List[DivisiblePosition] = [
                                    DivisiblePosition(
                                        ticker = param['ticker'],
                                        side = side,
                                        amount = target_order_notional,
                                        leg_room_bps = param['leg_room_bps'],
                                        order_type = param['order_type'],
                                        slices = param['slices'],
                                        wait_fill_threshold_ms = param['wait_fill_threshold_ms']
                                    )
                                ]
                                log(f"dispatching {side} orders to {gateway_id}")
                                executed_positions : Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None] = execute_positions(
                                                                                                                redis_client=redis_client,
                                                                                                                positions=entry_positions,
                                                                                                                ordergateway_pending_orders_topic=ordergateway_pending_orders_topic,
                                                                                                                ordergateway_executions_topic=ordergateway_executions_topic
                                                                                                                )
                                for executed_position in executed_positions:
                                    if not executed_position['done']:
                                        err_msg = executed_position['execution_err']
                                        log(err_msg, log_level=LogLevel.ERROR)
                                        dispatch_notification(title=f"singlelegta error from order gateway {gateway_id}!!", message=err_msg, log_level=logging.ERROR)
                                        raise ValueError(err_msg)
                                executed_position = executed_positions[0] # We sent only one DivisiblePosition.

                                new_pos_from_exchange =executed_position['filled_amount']
                                amount_filled_usdt = mid * new_pos_from_exchange
                                pos_entry_px = executed_position['average_cost']
                                new_pos_usdt_from_exchange = new_pos_from_exchange * executed_position['average_cost']
                                fees = executed_position['fees']

                                if side=='buy':
                                    tp_max_price = mid * (1 + pnl_potential_bps/10000)
                                    tp_min_price = mid * (1 + tp_min_percent/10000)
                                    sl_price = mid * (1 - param['sl_percent']/100)

                                elif side=='sell':
                                    tp_max_price = mid * (1 - pnl_potential_bps/10000)
                                    tp_min_price = mid * (1 - tp_min_percent/10000)
                                    sl_price = mid * (1 + param['sl_percent']/100)

                                executed_position['position'] = {
                                        'status' : 'open',
                                        'max_unrealized_pnl' : 0,
                                        'pos_entry_px' : pos_entry_px,
                                        'mid' : mid,
                                        'amount_base_ccy' : executed_position['filled_amount'],
                                        'tp_min_price' : tp_min_price,
                                        'tp_max_price' : tp_max_price,
                                        'sl_price' : sl_price,
                                        'multiplier' : multiplier
                                    }

                                pd_position_cache.loc[position_cache_row.name, 'pos'] = pos + new_pos_from_exchange
                                pd_position_cache.loc[position_cache_row.name, 'pos_usdt'] = pos_usdt + new_pos_usdt_from_exchange
                                pd_position_cache.loc[position_cache_row.name, 'status'] = PositionStatus.OPEN.name
                                pos_created = datetime.fromtimestamp(time.time())
                                pd_position_cache.loc[position_cache_row.name, 'created'] = pos_created
                                pd_position_cache.loc[position_cache_row.name, 'closed'] = None
                                pd_position_cache.loc[position_cache_row.name, 'entry_px'] = pos_entry_px
                                pd_position_cache.loc[position_cache_row.name, 'close_px'] = None
                                pd_position_cache.loc[position_cache_row.name, 'unreal_live'] = None
                                pd_position_cache.loc[position_cache_row.name, 'pnl_live_bps'] = None
                                pd_position_cache.loc[position_cache_row.name, 'pnl_pessimistic_bps'] = None
                                pd_position_cache.loc[position_cache_row.name, 'max_unreal_live'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_unreal_live_bps'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_unreal_pessimistic_bps'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_pain'] = None
                                pd_position_cache.loc[position_cache_row.name, 'entry_target_price'] = tp_max_price
                                pd_position_cache.loc[position_cache_row.name, 'running_sl_percent_hard'] = param['sl_percent']
                                pd_position_cache.loc[position_cache_row.name, 'sl_trailing_min_threshold_crossed'] = False
                                pd_position_cache.loc[position_cache_row.name, 'sl_percent_trailing'] = float('inf')
                                pd_position_cache.loc[position_cache_row.name, 'tp_min_target'] = tp_min_price

                                pos_entries.append(pos_created)
                                pd_position_cache.at[position_cache_row.name, 'pos_entries'] = pos_entries

                                # This is for tp_eval_func
                                this_ticker_open_trades.append(
                                    {
                                        'ticker' : param['ticker'],
                                        'side' : side,
                                        'amount' : target_order_notional,
                                        'tp_max_price' : tp_max_price,
                                        'target_price' : tp_max_price # This is the only field needed by backtest_core generic_tp_eval
                                    }
                                )

                                orderhist_cache_row = {
                                                        'datetime' : dt_now,
                                                        'exchange' : exchange_name,
                                                        'ticker' : ticker,
                                                        'reason' : 'entry',
                                                        'side' : side,
                                                        'avg_price' : new_pos_usdt_from_exchange/new_pos_from_exchange,
                                                        'amount': abs(new_pos_usdt_from_exchange),
                                                        'pnl' : 0,
                                                        'pnl_bps' : 0,
                                                        'max_pain' : 0
                                                    }
                                orderhist_cache = pd.concat([orderhist_cache, pd.DataFrame([orderhist_cache_row])], axis=0, ignore_index=True)

                                dispatch_notification(title=f"{param['current_filename']} {gateway_id} Entry succeeded. {param['ticker']} {side} {param['amount_base_ccy']} (USD amount: {amount_filled_usdt}) @ {pos_entry_px}", message=executed_position['position'], footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                        if (
                            ((pnl_live_bps/100)>=tp_min_percent and pos_unreal_live<max_unrealized_pnl) # @todo: unrealized_pnl_percent: from live thus will include the spikes. Do we want to use close from tm1 interval instead?
                            or loss_trailing>0 # once your trade pnl crosses tp_min_percent, trailing stops is (and will remain) active.
                        ):
                            loss_trailing = (1 - pos_unreal_live/max_unrealized_pnl) * 100
                        
                        '''
                        Have a look at this for a visual explaination how "Gradually tightened stops" works:
                            https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/trading_util_tests.ipynb
                        '''
                        if pnl_pessimistic_bps >= tp_min_percent:
                            if not sl_trailing_min_threshold_crossed:
                                sl_trailing_min_threshold_crossed = True
                                pd_position_cache.loc[position_cache_row.name, 'sl_trailing_min_threshold_crossed'] = sl_trailing_min_threshold_crossed

                                msg = {
                                    'side' : pos_side.name,
                                    'mid' : mid,
                                    'pos_entry_px' : pos_entry_px,
                                    'pnl_pessimistic_bps' : pnl_pessimistic_bps,
                                    'sl_trailing_min_threshold_bps' : tp_min_percent
                                }
                                log(msg, LogLevel.CRITICAL)
                                dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} sl_trailing_min_threshold_crossed: True!", message=msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)


                            _effective_tp_trailing_percent = calc_eff_trailing_sl(
                                tp_min_percent = tp_min_percent,
                                tp_max_percent = tp_max_percent,
                                sl_percent_trailing = tp_max_percent,
                                pnl_percent_notional = max_unrealized_pnl_percent, # Note: Use [max]_unrealized_pnl_percent, not unrealized_pnl_percent!
                                default_effective_tp_trailing_percent = param['default_effective_tp_trailing_percent'],
                                linear=True if tp_max_percent >= param['trailing_sl_min_percent_linear'] else False, # If tp_max_percent far (>100bps for example), there's more uncertainty if target can be reached: Go with linear.
                                pow=param['non_linear_pow']
                            )

                            # Once pnl pass tp_min_percent, trailing stops will be activated. Even if pnl fall back below tp_min_percent.
                            effective_tp_trailing_percent = min(effective_tp_trailing_percent, _effective_tp_trailing_percent)

                        log(f"pos_unreal_live: {round(pos_unreal_live,4)}, pnl_live_bps: {round(pnl_live_bps,4)}, unrealized_pnl_live_pessimistic: {unrealized_pnl_live_pessimistic}, pnl_pessimistic_bps: {pnl_pessimistic_bps}, max_unrealized_pnl_percent: {round(max_unrealized_pnl_percent,4)}, loss_trailing: {loss_trailing}, effective_tp_trailing_percent: {effective_tp_trailing_percent}, reversal: {reversal}")

                        # STEP 2. Unwind position
                        if pos_unreal_live>0:
                            kwargs = {k: v for k, v in locals().items() if k in tp_eval_func_params}
                            tp = tp_eval_func(**kwargs)
                            
                        else:
                            if abs(pnl_live_bps/100)>=param['sl_percent']:
                                sl = True

                        if tp or sl:
                            exit_positions : List[DivisiblePosition] = [
                                DivisiblePosition(
                                    ticker = param['ticker'],
                                    side = 'sell' if pos_side==OrderSide.BUY else 'buy',
                                    amount = param['amount_base_ccy'],
                                    leg_room_bps = param['leg_room_bps'],
                                    order_type = param['order_type'],
                                    slices = param['slices'],
                                    wait_fill_threshold_ms = param['wait_fill_threshold_ms'],

                                    reduce_only=True
                                )
                            ]
                            log(f"Closing position. {ticker}, pos: {pos}, pos_usdt: {pos_usdt}") 
                            executed_positions : Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None] = execute_positions(
                                                                                                                            redis_client=redis_client,
                                                                                                                            positions=exit_positions,
                                                                                                                            ordergateway_pending_orders_topic=ordergateway_pending_orders_topic,
                                                                                                                            ordergateway_executions_topic=ordergateway_executions_topic
                                                                                                                        )
                            if executed_positions:
                                executed_position_close = executed_positions[0] # We sent only one DivisiblePosition.
                                if executed_position_close['done']:
                                    if pos_side==OrderSide.BUY:
                                        closed_pnl = (executed_position_close['average_cost'] - pos_entry_px) * param['amount_base_ccy']
                                    else:
                                        closed_pnl = (pos_entry_px - executed_position_close['average_cost']) * param['amount_base_ccy']
                                    
                                    new_pos_from_exchange = abs(pos) + executed_position_close['filled_amount']
                                    new_pos_usdt_from_exchange = new_pos_from_exchange * executed_position_close['average_cost']
                                    fees = executed_position_close['fees']

                                    executed_position_close['position'] = {
                                        'status' : 'TP' if tp else 'SL',
                                        'max_unrealized_pnl' : max_unrealized_pnl,
                                        'pos_entry_px' : pos_entry_px,
                                        'mid' : mid,
                                        'amount_base_ccy' : executed_position_close['filled_amount'],
                                        'closed_pnl' : closed_pnl,
                                    }

                                    new_status = PositionStatus.SL.name if closed_pnl<=0 else PositionStatus.CLOSED.name
                                    pd_position_cache.loc[position_cache_row.name, 'pos'] = new_pos_from_exchange
                                    pd_position_cache.loc[position_cache_row.name, 'pos_usdt'] = new_pos_usdt_from_exchange
                                    pd_position_cache.loc[position_cache_row.name, 'status'] = new_status
                                    pd_position_cache.loc[position_cache_row.name, 'closed'] = dt_now
                                    pd_position_cache.loc[position_cache_row.name, 'unreal_live'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'pnl_live_bps'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'pnl_pessimistic_bps'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'max_unreal_live'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_unreal_live_bps'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_unreal_pessimistic_bps'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_pain'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'close_px'] = mid # mid is approx of actual fill price!
                                    pd_position_cache.at[position_cache_row.name, 'pos_entries'] = []
                                    pd_position_cache.loc[position_cache_row.name, 'running_sl_percent_hard'] = param['sl_percent']
                                    pd_position_cache.loc[position_cache_row.name, 'sl_trailing_min_threshold_crossed'] = False
                                    pd_position_cache.loc[position_cache_row.name, 'sl_percent_trailing'] = param['sl_percent_trailing']
                                    pd_position_cache.loc[position_cache_row.name, 'tp_min_target'] = None

                                    # This is for tp_eval_func
                                    this_ticker_open_trades.clear()

                                    orderhist_cache_row = {
                                                'datetime' : dt_now,
                                                'exchange' : exchange_name,
                                                'ticker' : ticker,
                                                'reason' : new_status,
                                                'side' : 'sell' if pos_side==OrderSide.BUY else 'buy',
                                                'avg_price' : mid, # mid is actually not avg_price!
                                                'amount': abs(new_pos_usdt_from_exchange),
                                                'pos_unreal_live' : pos_unreal_live,
                                                'pnl_live_bps' : pnl_live_bps,
                                                'max_pain' : max_pain
                                            }
                                    orderhist_cache = pd.concat([orderhist_cache, pd.DataFrame([orderhist_cache_row])], axis=0, ignore_index=True)

                                    dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} {'TP' if tp else 'SL'} succeeded. closed_pnl: {closed_pnl}", message=executed_position_close, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                                else:
                                    dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Exit execution failed. {param['ticker']}", message=executed_position_close, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                        log(f"[{gateway_id}", log_level=LogLevel.INFO)
                        log(f"{tabulate(pd_position_cache, headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)

                        pd_position_cache.to_csv(POSITION_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id))
                        orderhist_cache.to_csv(ORDERHIST_CACHE_FILE_NAME.replace("$GATEWAY_ID$", gateway_id))
                        
            except Exception as loop_err:
                log(f"Error: {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}", log_level=LogLevel.ERROR)
            finally:
                time.sleep(int(param['loop_freq_ms']/1000))

asyncio.run(
    main(
        order_notional_adj_func=TargetStrategy.order_notional_adj,
        allow_entry_initial_func=TargetStrategy.allow_entry_initial,
        allow_entry_final_func=TargetStrategy.allow_entry_final,
        sl_adj_func=TargetStrategy.sl_adj,
        trailing_stop_threshold_eval_func=TargetStrategy.trailing_stop_threshold_eval,
        tp_eval_func=TargetStrategy.tp_eval
    )
)