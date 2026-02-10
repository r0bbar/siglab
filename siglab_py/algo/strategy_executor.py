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
from siglab_py.util.datetime_util import parse_trading_window
from siglab_py.util.simple_math import compute_adjacent_levels
from siglab_py.util.market_data_util import async_instantiate_exchange, interval_to_ms, get_old_ticker, get_ticker_map
from siglab_py.util.trading_util import calc_eff_trailing_sl
from siglab_py.util.notification_util import dispatch_notification
from siglab_py.util.aws_util import AwsKmsUtil
from siglab_py.util.module_util import load_module_class
from siglab_py.util.io_util import purge_old_file

from siglab_py.constants import INVALID, JSON_SERIALIZABLE_TYPES, LogLevel, PositionStatus, OrderSide 

current_filename = os.path.basename(__file__)
current_dir : str = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

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
        python candles_provider.py --provider_id aaa --candle_size 1h --how_many_candles 720 --redis_ttl_ms 3600000
        python candles_provider.py --provider_id bbb --candle_size 15m --how_many_candles 672 --redis_ttl_ms 3600000

        Note,
        a. 'ticker_change_map.json' is for ticker change handling.
        b. retart on ticker change?
            candles_provider NO
            gateway.py NO (If will load_market before process each request)
            orderbooks_provider YES (resubscribe on new ticker)

        Note: how_many_candles should be larger than compute_candles_stats.sliding_window_how_many_candles by a few times.
            720 = 24 x 30 days  
            672 = 4 x 24 x 7 days (Each hour has four 15m candles. 672 candles means 672 15m candles)

    Step 2. Start candles_ta_providers
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python candles_ta_provider.py --candle_size 1h --ma_long_intervals 48 --ma_short_intervals 12 --boillenger_std_multiples 2 --redis_ttl_ms 3600000 --processed_hash_queue_max_size 999 --pypy_compat N
        python candles_ta_provider.py --candle_size 15m --ma_long_intervals 150 --ma_short_intervals 75 --boillenger_std_multiples 2 --redis_ttl_ms 3600000 --processed_hash_queue_max_size 999 --pypy_compat N

        Note, 
        a. for 15m bars, a sliding window of size 150 means 150 x 15m = 2250 minutes
        b. sliding window sizes are defined by 'candles_ta_provider', not 'candles_provider'!!!

    Step 3. Start orderbooks_provider
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python orderbooks_provider.py --provider_id ccc --instance_capacity 25 --ts_delta_observation_ms_threshold 150 --ts_delta_consecutive_ms_threshold 150 --redis_ttl_ms 3600000

    Step 4. To trigger candles_providers and orderbooks_provider
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python trigger_provider.py --provider_id aaa --tickers "okx_linear|SOL/USDT:USDT"
        python trigger_provider.py --provider_id bbb --tickers "okx_linear|SOL/USDT:USDT"
        python trigger_provider.py --provider_id ccc --tickers "okx_linear|SOL/USDT:USDT"

    Step 5. Start strategy_executor
        set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
        python strategy_executor.py --target_strategy_name YourStrategyClassName --gateway_id hyperliquid_01 --default_type linear --rate_limit_ms 100 --encrypt_decrypt_with_aws_kms Y --aws_kms_key_id xxx --apikey xxx --secret xxx --ticker SUSHI/USDC:USDC --order_type limit --amount_base_ccy 45 --residual_pos_usdt_threshold 10 --slices 3 --fees_ccy USDT --wait_fill_threshold_ms 15000 --leg_room_bps 5 --tp_min_percent 1.5 --tp_max_percent 2.5 --sl_percent_trailing 50 --sl_hard_percent 1 --reversal_num_intervals 3 --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx --economic_calendar_source xxx --block_entry_impacting_events Y --num_intervals_current_ecoevents 96 --hi_candles_provider_topic mds_assign_aaa --lo_candles_provider_topic mds_assign_bbb --orderbooks_provider_topic mds_assign_ccc --hi_candles_w_ta_topic candles_ta-SOL-USDT-SWAP-hyperliquid-1h --lo_candles_w_ta_topic candles_ta-SOL-USDT-SWAP-hyperliquid-15m --orderbook_topic orderbooks_SOL/USDT:USDT_hyperliquid --trading_window_start Mon_00:00 --trading_window_end Fri_17:00

    Step 6. Start order gateway
        Top of the script for instructions
        https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py

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
                        "--target_strategy_name", "YourStrategyClassName",

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
                        "--fees_ccy", "USDT",
                        "--wait_fill_threshold_ms", "15000",
                        "--leg_room_bps", "5",
                        "--tp_min_percent", "3",
                        "--tp_max_percent", "5",
                        "--sl_percent_trailing", "35",
                        "--sl_hard_percent", "2.5",
                        "--reversal_num_intervals", "3",

                        "--economic_calendar_source", "xxx",
                        "--block_entry_impacting_events","Y",
                        "--num_intervals_current_ecoevents", "96",

                        "hi_candles_provider_topic" : "mds_assign_aaa",
                        "lo_candles_provider_topic" : "mds_assign_bbb",
                        "orderbooks_provider_topic" : "mds_assign_ccc",
                        "hi_candles_w_ta_topic" : "candles_ta-SOL-USDT-SWAP-hyperliquid-1h",
                        "lo_candles_w_ta_topic" : "candles_ta-SOL-USDT-SWAP-hyperliquid-15m",
                        "orderbook_topic" : "orderbooks_SOL/USDT:USDT_hyperliquid",

                        "--trading_window_start", "Mon_00:00",
                        "--trading_window_end", "Fri_17:00",

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
    'trailing_stop_mode': "linear", # linear or parabolic
    'non_linear_pow' : 5, # For non-linear trailing stops tightening. 

    'rolldate_tz' : 'Asia/Hong_Kong', # Roll date based on what timezone?

    'start_timestamp_ms' : None, # You want your algo to start only after what time?

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

    # Ticker change
    'ticker_change_map' : 'ticker_change_map.json',

    "loop_freq_ms" : 1000, # reduce this if you need trade faster

    'current_filename' : current_filename,
    'current_dir' : parent_dir,

    'housekeep_filename_regex_list' : [ 
        "lo_candles_entry_.*\.csv",
        "hi_candles_entry_.*\.csv"
    ],
    'housekeep_max_age_sec' : 60*60*24,

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
            "hi_candles_provider_topic" : None,
            "lo_candles_provider_topic" : None,
            "orderbooks_provider_topic" : None,
            "hi_candles_w_ta_topic" : None,
            "lo_candles_w_ta_topic" : None,
            "orderbook_topic" : None,
            
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

POSITION_CACHE_COLUMNS = [ 
            'exchange', 'ticker',
            'status', 
            'pos', 'pos_usdt', # pos and pos_usdt: For longs, it's positive number. For shorts, it's a negative number. 
            'multiplier', 'created', 'tp_min_crossed', 'closed', 
            'pos_entries',
             
            'entry_px', 
            'tp_max_target',
			'tp_min_target',
            'sl_price',
            'max_pnl_potential_bps',
            'close_px',

            'ob_mid', 'spread_bps', 'ob_best_bid', 'ob_best_ask', 'level_granularity', 'level_below', 'level_above',

            'unreal_live',
            'max_unreal_live',
            'max_pain',
            'max_recovered_pnl',
            'pnl_live_bps',
            'pnl_open_bps',
            'max_unreal_live_bps',
            'max_unreal_open_bps',
			
            'running_sl_percent_hard',
            'sl_trailing_min_threshold_crossed',
            'sl_percent_trailing',
            'effective_tp_trailing_percent',
            'loss_trailing'
        ]


ORDERHIST_CACHE_COLUMNS = [  'datetime', 'timestamp_ms', 'exchange', 'ticker', 'reason', 'reason2', 'side', 'avg_price', 'amount', 'pnl', 'pnl_bps', 'max_pain', 'fees', 'remarks' ]

def log(message : str, log_level : LogLevel = LogLevel.INFO):
    if log_level.value<LogLevel.WARNING.value:
        logger.info(message)

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(message)

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(message)

def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument(
        '--target_strategy_name',
        type=str,
        default=None,
        help='Name of strategy class to be imported as TargetStrategy. If not specified, StrategyBase (dummy strategy which wont fire any order) will be loaded.'
    )

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
    parser.add_argument("--amount_quote_ccy", help="Order amount in quote ccy (USD, USDT or USDC ...etc). Always positive, even for sell trades.", default=None)
    parser.add_argument("--amount_base_ccy", help="Order amount in base ccy (Not # contracts). Always positive, even for sell trades.", default=None)
    parser.add_argument("--max_pos_amount_quote_ccy", help="Allows for sliced entries. Default is set to 'amount_quote_ccy'", default=None)
    parser.add_argument("--max_pos_amount_base_ccy", help="Allows for sliced entries. Default is set to 'amount_base_ccy'", default=None)
    parser.add_argument("--residual_pos_usdt_threshold", help="If pos_usdt<=residual_pos_usdt_threshold (in USD, default $100), PositionStatus will be marked to CLOSED.", default=100)
    parser.add_argument("--leg_room_bps", help="Leg room, for Limit orders only. A more positive leg room is a more aggressive order to get filled. i.e. Buy at higher price, Sell at lower price.", default=5)
    parser.add_argument("--slices", help="Algo can break down larger order into smaller slices. Default: 1", default=1)
    parser.add_argument("--fees_ccy", help="If you're trading crypto, CEX fees USDT, DEX fees USDC in many cases. Default None, in which case gateway won't aggregatge fees from executions for you.", default=None)
    parser.add_argument("--wait_fill_threshold_ms", help="Limit orders will be cancelled if not filled within this time. Remainder will be sent off as market order.", default=15000)
    
    parser.add_argument("--tp_min_percent", help="For trailing stops. Min TP in percent, i.e. No TP until pnl at least this much.", default=None)
    parser.add_argument("--tp_max_percent", help="For trailing stops. Max TP in percent, i.e. Price target", default=None)
    parser.add_argument("--sl_percent_trailing", help="For trailing stops. trailing SL in percent, please refer to trading_util.calc_eff_trailing_sl for documentation.", default=None)
    parser.add_argument("--default_effective_tp_trailing_percent", help="Default for sl_percent_trailing when pnl still below tp_min_percent. Default: float('inf'), meaing trailing stop mechanism will not be activated.", default=float('inf'))
    parser.add_argument("--sl_adj_percent", help="Increment used in SL adj in percent.", default=0)
    parser.add_argument("--sl_hard_percent", help="Hard stop in percent.", default=2)
    parser.add_argument("--sl_num_intervals_delay", help="Number of intervals to wait before re-entry allowed after SL. Default 1", default=1)
    parser.add_argument("--reversal_num_intervals", help="How many reversal candles to confirm reversal?", default=3)
    parser.add_argument("--trailing_stop_mode", help="This is for trailing stops tightening 'calc_eff_trailing_sl': linear or parabolic. Default: linear", default='linear')
    parser.add_argument("--non_linear_pow", help="For non-linear trailing stops tightening, have a look at call to 'calc_eff_trailing_sl'. Default: 5", default=5)
    parser.add_argument("--recover_min_percent", help="This is minimum unreal pnl recovery when your trade is red before trailing stop mechanism will be activated: max_recovered_pnl_percent_notional>=recover_min_percent and abs(max_pain_percent_notional)>=recover_max_pain_percent. Default: float('inf'), meaing trailing stop won't be fired.", default=float('inf'))
    parser.add_argument("--recover_max_pain_percent", help="This is minimum max_pain endured when your trade is red. For trailing stop mechanism will be activated: max_recovered_pnl_percent_notional>=recover_min_percent and abs(max_pain_percent_notional)>=recover_max_pain_percent. Default: float('inf'), meaing trailing stop mechanism will remain inactive.", default=float('inf'))
    
    parser.add_argument("--economic_calendar_source", help="Source of economic calendar'. Default: None", default=None)
    parser.add_argument("--num_intervals_current_ecoevents", help="Num intervals to block on incoming/outgoing economic events. For 15m bars for example, num_intervals_current_ecoevents=4*24 means 24 hours. Default: 0", default=0)
    parser.add_argument("--block_entry_impacting_events", help="Block entries if any impacting economic events 'impacting_economic_calendars'. Default N", default='N')
    
    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=5000)

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    parser.add_argument("--dump_candles", help="This is for trouble shooting only. Y or N (default).", default='N')

    parser.add_argument("--start_timestamp_ms", help="You want your algo to start only after what time? This is timestamp in ms. Default is None (start immediately)", default=None)

    parser.add_argument("--hi_candles_provider_topic", help="hi_candles provider redis topic. Example mds_assign_aaa, this is for strategy_executor to trigger provider. No provider triggering if set to None (default).", default=None)
    parser.add_argument("--lo_candles_provider_topic", help="lo_candles provider redis topic. Example mds_assign_bbb, this is for strategy_executor to trigger provider. No provider triggering if set to None (default).", default=None)
    parser.add_argument("--orderbooks_provider_topic", help="orderbooks provider redis topic. Example mds_assign_ccc, this is for strategy_executor to trigger provider. No provider triggering if set to None (default).", default=None)
    parser.add_argument("--hi_candles_w_ta_topic", help="hi_candles TA redis topic. Example candles_ta-SOL-USDT-SWAP-hyperliquid-1h, this is for strategy_executor source candles TA from redis.", default=None)
    parser.add_argument("--lo_candles_w_ta_topic", help="lo_candles TA redis topic. Example candles_ta-SOL-USDT-SWAP-hyperliquid-15m, this is for strategy_executor source candles TA from redis.", default=None)
    parser.add_argument("--orderbook_topic", help="lo_candles TA redis topic. Example orderbooks_SOL/USDT:USDT_hyperliquid, this is for strategy_executor source orderbooks from redis.", default=None)

    args = parser.parse_args()

    param['target_strategy_name'] = args.target_strategy_name

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
    param['amount_quote_ccy'] = float(args.amount_quote_ccy) if args.amount_quote_ccy else None
    param['max_pos_amount_quote_ccy'] = param['amount_quote_ccy']
    if param['amount_quote_ccy'] and args.max_pos_amount_quote_ccy:
        param['max_pos_amount_quote_ccy'] = max(
            float(args.max_pos_amount_quote_ccy), 
            param['amount_quote_ccy']
        )
    param['amount_base_ccy'] = float(args.amount_base_ccy) if args.amount_base_ccy else None
    param['max_pos_amount_base_ccy'] = param['amount_base_ccy']
    if param['amount_base_ccy'] and args.max_pos_amount_base_ccy:
        param['max_pos_amount_base_ccy'] = max(
            float(args.max_pos_amount_base_ccy), 
            param['amount_base_ccy']
        )
    param['residual_pos_usdt_threshold'] = float(args.residual_pos_usdt_threshold)
    param['leg_room_bps'] = int(args.leg_room_bps)
    param['slices'] = int(args.slices)
    param['fees_ccy'] = args.fees_ccy
    param['wait_fill_threshold_ms'] = int(args.wait_fill_threshold_ms)

    param['tp_min_percent'] = float(args.tp_min_percent)
    param['tp_max_percent'] = float(args.tp_max_percent)
    param['sl_percent_trailing'] = float(args.sl_percent_trailing)
    param['default_effective_tp_trailing_percent'] = float(args.default_effective_tp_trailing_percent)
    param['sl_adj_percent'] = float(args.sl_adj_percent)
    param['sl_hard_percent'] = float(args.sl_hard_percent)
    param['sl_num_intervals_delay'] = int(args.sl_num_intervals_delay)
    param['reversal_num_intervals'] = int(args.reversal_num_intervals)
    param['trailing_stop_mode'] = args.trailing_stop_mode
    param['non_linear_pow'] = float(args.non_linear_pow)
    param['recover_min_percent'] = float(args.recover_min_percent)
    param['recover_max_pain_percent'] = float(args.recover_max_pain_percent)

    param['economic_calendar_source'] = args.economic_calendar_source

    if args.block_entry_impacting_events:
        if args.block_entry_impacting_events=='Y':
            param['block_entry_impacting_events'] = True
        else:
            param['block_entry_impacting_events'] = False
    else:
        param['block_entry_impacting_events'] = False
    
    param['mds']['topics']['hi_candles_provider_topic'] = args.hi_candles_provider_topic.strip() if args.hi_candles_provider_topic else None
    param['mds']['topics']['lo_candles_provider_topic'] = args.lo_candles_provider_topic.strip() if args.lo_candles_provider_topic else None
    param['mds']['topics']['orderbooks_provider_topic'] = args.orderbooks_provider_topic.strip() if args.orderbooks_provider_topic else None

    param['mds']['topics']['hi_candles_w_ta_topic'] = args.hi_candles_w_ta_topic.strip() if args.hi_candles_w_ta_topic else None
    param['mds']['topics']['lo_candles_w_ta_topic'] = args.lo_candles_w_ta_topic.strip() if args.lo_candles_w_ta_topic else None
    param['mds']['topics']['orderbook_topic'] = args.orderbook_topic.strip() if args.orderbook_topic else None

    param['loop_freq_ms'] = int(args.loop_freq_ms)

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

    param['notification']['footer'] = f"From {param['current_filename']} {param['gateway_id']}"

    if args.dump_candles:
        if args.dump_candles=='Y':
            param['dump_candles'] = True
        else:
            param['dump_candles'] = False
    else:
        param['dump_candles'] = False

    param['start_timestamp_ms'] = int(args.start_timestamp_ms) if args.start_timestamp_ms else None

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

async def main():
    parse_args()

    print(f"target_strategy_name: {param['target_strategy_name']}")
    strategy_cls = load_module_class(param["target_strategy_name"])
    if strategy_cls is None:
        from siglab_py.algo.strategy_base import StrategyBase
        TargetStrategy = StrategyBase
    else:
        TargetStrategy = strategy_cls

    orderhist_cache_file_name = f"$STRATEGY_CLASS_NAME$_orderhist_cache_$GATEWAY_ID$.csv"
    orderhist_cache_file_name = orderhist_cache_file_name.replace("$STRATEGY_CLASS_NAME$", TargetStrategy.__name__)
    position_cache_file_name = f"$STRATEGY_CLASS_NAME$_position_cache_$GATEWAY_ID$.csv"
    position_cache_file_name = position_cache_file_name.replace("$STRATEGY_CLASS_NAME$", TargetStrategy.__name__)

    order_notional_adj_func : Callable[..., Dict[str, float]] = TargetStrategy.order_notional_adj
    allow_entry_initial_func : Callable[..., Dict[str, bool]] = TargetStrategy.allow_entry_initial
    allow_entry_final_func : Callable[..., Dict[str, Union[bool, float, None]]] = TargetStrategy.allow_entry_final
    sl_adj_func : Callable[..., Dict[str, float]] = TargetStrategy.sl_adj
    trailing_stop_threshold_eval_func : Callable[..., Dict[str, float]] = TargetStrategy.trailing_stop_threshold_eval
    tp_eval_func : Callable[..., bool] = TargetStrategy.tp_eval
    
    redis_client : StrictRedis = init_redis_client()

    gateway_id : str = param['gateway_id']

    fh = logging.FileHandler(f"strategy_executor_{param['gateway_id']}_{TargetStrategy.__name__}.log")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)     
    logger.addHandler(fh)

    '''
    DO NOT LOG 'param' after secrets (apikey, secret, passphrase) decrypted!!!
    Just to be safe, we do this more explicitly below.
    '''
    _param = param.copy()
    del _param['secret']
    del _param['passphrase']
    log(f"Strategy startup parameters: {json.dumps(_param, indent=4)}")
    del _param

    log(f"orderhist_cache_file_name: {orderhist_cache_file_name}")
    log(f"position_cache_file_name: {position_cache_file_name}")
    
    exchange_name : str = gateway_id.split('_')[0]

    ticker_change_map = None
    with open(param['ticker_change_map'], 'r', encoding='utf-8') as f:
        ticker_change_map : List[Dict[str, Union[str, int]]] = json.load(f)
        log(f"ticker_change_map loaded from {param['ticker_change_map']}")
        log(json.dumps(ticker_change_map))

    ticker : str = param['ticker']
    _ticker = ticker # In case ticker changes ...
    if ticker_change_map:
        old_ticker= get_old_ticker(_ticker, ticker_change_map)
        if old_ticker:
            ticker_change_mapping = get_ticker_map(reference_ticker, ticker_change_map)
            ticker_change_cutoff_sec = int(ticker_change_mapping['cutoff_ms']) / 1000
            if datetime.now().timestamp()<ticker_change_cutoff_sec:
                _ticker = old_ticker

    ordergateway_pending_orders_topic : str = 'ordergateway_pending_orders_$GATEWAY_ID$'
    ordergateway_pending_orders_topic : str = ordergateway_pending_orders_topic.replace("$GATEWAY_ID$", gateway_id)
    
    ordergateway_executions_topic : str = "ordergateway_executions_$GATEWAY_ID$"
    ordergateway_executions_topic : str = ordergateway_executions_topic.replace("$GATEWAY_ID$", gateway_id)

    hi_candles_w_ta_topic : str = param['mds']['topics']['hi_candles_w_ta_topic']
    lo_candles_w_ta_topic : str = param['mds']['topics']['lo_candles_w_ta_topic']
    orderbook_topic : str = param['mds']['topics']['orderbook_topic']

    hi_candles_provider_topic : str = param['mds']['topics']['hi_candles_provider_topic']
    lo_candles_provider_topic : str = param['mds']['topics']['lo_candles_provider_topic']
    orderbooks_provider_topic : str = param['mds']['topics']['orderbooks_provider_topic']

    # economic_calendar_source
    full_economic_calendars_topic : str = param['mds']['topics']['full_economic_calendars_topic']
    full_economic_calendars_topic  = full_economic_calendars_topic.replace('$SOURCE$', param['economic_calendar_source']) if param['economic_calendar_source'] else None

    log(f"hi_candles_w_ta_topic: {hi_candles_w_ta_topic}")
    log(f"lo_candles_w_ta_topic: {lo_candles_w_ta_topic}")
    log(f"hi_candles_provider_topic: {hi_candles_provider_topic}")
    log(f"lo_candles_provider_topic: {lo_candles_provider_topic}")
    log(f"orderbook_topic: {orderbook_topic}")
    log(f"ordergateway_pending_orders_topic: {ordergateway_pending_orders_topic}")
    log(f"ordergateway_executions_topic: {ordergateway_executions_topic}")
    log(f"full_economic_calendars_topic: {full_economic_calendars_topic}")

    # aliases
    algo_param = param
    strategic_specific_algo_param = TargetStrategy.get_strategy_algo_params()
    for entry in strategic_specific_algo_param:
        algo_param[entry['key']] = entry['val']

    hi_candle_size : str = hi_candles_w_ta_topic.split('-')[-1]
    lo_candle_size : str = lo_candles_w_ta_topic.split('-')[-1]
    hi_interval = hi_candle_size[-1]
    hi_num_intervals : int = int(hi_candle_size.replace(hi_interval,''))
    hi_interval_ms : int = interval_to_ms(hi_interval) * hi_num_intervals
    lo_interval = lo_candle_size[-1]
    lo_num_intervals : int = int(lo_candle_size.replace(lo_interval,''))
    lo_interval_ms : int = interval_to_ms(lo_interval) * lo_num_intervals

    min_sl_age_ms : int = lo_interval_ms * param['sl_num_intervals_delay']
    num_intervals_current_ecoevents_ms : int = lo_interval_ms * param['num_intervals_current_ecoevents']

    log(f"hi_candle_size: {hi_candle_size}, hi_interval_ms: {hi_interval_ms:,}")
    log(f"lo_candle_size: {lo_candle_size}, lo_interval_ms: {lo_interval_ms:,}")
    log(f"num_intervals_current_ecoevents: {param['num_intervals_current_ecoevents']}, num_intervals_current_ecoevents_ms: {num_intervals_current_ecoevents_ms:,} (~{num_intervals_current_ecoevents_ms/1000/60/60} hrs)")
    
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
        market = markets[_ticker]
        multiplier = market['contractSize'] if 'contractSize' in market and market['contractSize'] else 1

        balances = await exchange.fetch_balance() 
        log(f"Balances: {json.dumps(balances, indent=4)}") 
        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} strategy {TargetStrategy.__name__} starting", message=balances['total'], footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

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
        
        if hi_candles_provider_topic:
            _trigger_producers(redis_client, [ f"{exchange_name}|{_ticker}" ], hi_candles_provider_topic)
        else:
            log(f"hi_candles_provider_topic not specified, no hi_candles_provider triggering.")
        
        if lo_candles_provider_topic:
            _trigger_producers(redis_client, [ f"{exchange_name}|{_ticker}" ], lo_candles_provider_topic)
        else:
            log(f"lo_candles_provider_topic not specified, no lo_candles_provider triggering.")

        if orderbooks_provider_topic:
            _trigger_producers(redis_client, [ f"{exchange_name}|{_ticker}" ], orderbooks_provider_topic)
        else:
            log(f"orderbooks_provider_topic not specified, no orderbooks_provider triggering.")

        if param['start_timestamp_ms']:
            while datetime.now().timestamp() < int(param['start_timestamp_ms']/1000):
                log(f"Waiting to start at {datetime.fromtimestamp(int(param['start_timestamp_ms']/1000))}")
                time.sleep(10)
        else:
            log(f"Strategy starting immediately.")

        # Load cached positions from disk, if any
        if os.path.exists(position_cache_file_name.replace("$GATEWAY_ID$", gateway_id)) and os.path.getsize(position_cache_file_name.replace("$GATEWAY_ID$", gateway_id))>0:
            pd_position_cache = pd.read_csv(position_cache_file_name.replace("$GATEWAY_ID$", gateway_id))
            pd_position_cache.drop(pd_position_cache.columns[pd_position_cache.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            pd_position_cache.replace([np.nan], [None], inplace=True)

            pd_position_cache = pd_position_cache[POSITION_CACHE_COLUMNS]

        if os.path.exists(orderhist_cache_file_name.replace("$GATEWAY_ID$", gateway_id)) and os.path.getsize(orderhist_cache_file_name.replace("$GATEWAY_ID$", gateway_id))>0:
            orderhist_cache = pd.read_csv(orderhist_cache_file_name.replace("$GATEWAY_ID$", gateway_id))
            orderhist_cache.drop(orderhist_cache.columns[orderhist_cache.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            orderhist_cache.replace([np.nan], [None], inplace=True)

            if 'datetime' in orderhist_cache.columns:
                orderhist_cache['datetime'] = pd.to_datetime(orderhist_cache['datetime'])

        block_entries : bool = False
        any_entry : bool = False
        any_exit : bool = False
        hi_row, hi_row_tm1 = None, None
        lo_row, lo_row_tm1 = None, None
        effective_tp_trailing_percent : float = param['default_effective_tp_trailing_percent']
        this_ticker_open_trades : List[Dict] = []

        loop_counter : int = 0
        loop_start : datetime = datetime.now()
        loop_elapsed_sec : int = 0
        reversal : bool = False
        tp : bool = False
        sl : bool = False
        tp_max_percent : float  = param['tp_max_percent']
        tp_min_percent : float  = param['tp_min_percent']
        executed_position = None
        position_break : bool = False
        lo_row_timestamp_ms : int = 0
        lo_candles_interval_rolled : bool = True
        while (not position_break):
            try:
                if loop_counter>0:
                    loop_elapsed_sec = round((datetime.now() - loop_start).total_seconds(), 2)
                loop_start = datetime.now()

                any_entry, any_exit = False, False

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
                        log(f"Block entries: Outside trading window")

                    log(f"trading_window start: {param['trading_window_start']}, end: {param['trading_window_end']}, in_window: {parsed_trading_window['in_window']}")
                else:
                    log(f"No trading window specified")

                if full_economic_calendars_topic:
                    full_economic_calendars = fetch_economic_events(redis_client, full_economic_calendars_topic)

                    impacting_economic_calendars = None
                    s_impacting_economic_calendars = None
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
                    log(f"impacting_economic_calendars #rows: {len(impacting_economic_calendars) if impacting_economic_calendars else 0} {s_impacting_economic_calendars}.")
                    
                    if param['block_entry_impacting_events'] and impacting_economic_calendars:
                        block_entries = True
                        log(f"Block entries: Incoming events")
                
                if ticker_change_map:
                    old_ticker= get_old_ticker(_ticker, ticker_change_map)
                    if old_ticker:
                        ticker_change_mapping = get_ticker_map(reference_ticker, ticker_change_map)
                        ticker_change_cutoff_sec = int(ticker_change_mapping['cutoff_ms']) / 1000
                        if datetime.now().timestamp()<ticker_change_cutoff_sec:
                            _ticker = old_ticker

                # Ticker changes, delisting handling
                markets = await exchange.load_markets() 
                market = markets[_ticker]
                multiplier = market['contractSize'] if 'contractSize' in market and market['contractSize'] else 1

                position_cache_row = pd_position_cache.loc[(pd_position_cache.exchange==exchange_name) & (pd_position_cache.ticker==_ticker)]
                if position_cache_row.shape[0]==0:
                    position_cache_row = {
                        'exchange': exchange_name,
                        'ticker' : _ticker,

                        'status' : PositionStatus.UNDEFINED.name,
                        
                        'pos' : None, 
                        'pos_usdt' : None,
                        'multiplier' : multiplier,

                        'created' : None,
                        'tp_min_crossed' : None,
                        'closed' : None,

                        'pos_entries' : [],
                        
                        'entry_px' : None,
                        'tp_max_target' : None,
						'tp_min_target' : None,
                        'sl_price' : None,
                        'max_pnl_potential_bps' : None,
                        'close_px' : None,

                        'ob_mid' : None,
                        'spread_bps' : None,
                        'ob_best_bid' : None,
                        'ob_best_ask' : None,
                        
                        'unreal_live' : 0,
						'max_unreal_live' : 0,
                        'max_pain' : 0,
                        'max_recovered_pnl' : 0,
                        'pnl_live_bps' : 0,
                        'pnl_open_bps' : 0,
						'max_unreal_live_bps' : 0,
                        'max_unreal_open_bps' : 0,
                        
                        'running_sl_percent_hard' : param['sl_hard_percent'],
                        'sl_trailing_min_threshold_crossed' : False,
                        'sl_percent_trailing' : param['sl_percent_trailing'],
                        'effective_tp_trailing_percent' : param['default_effective_tp_trailing_percent'],
                        'loss_trailing' : 0
                    }
                    position_cache_row.update({ind: None for ind in strategy_indicators})
                    pd_position_cache = pd.concat([pd_position_cache, pd.DataFrame([position_cache_row])], axis=0, ignore_index=True)
                    position_cache_row = pd_position_cache.loc[(pd_position_cache.exchange==exchange_name) & (pd_position_cache.ticker==_ticker)]
            
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
                
                pos_tp_min_crossed = position_cache_row['tp_min_crossed']
                pos_tp_min_crossed = arrow.get(pos_tp_min_crossed).datetime if pos_tp_min_crossed and isinstance(pos_tp_min_crossed, str) else pos_tp_min_crossed
                if pos_tp_min_crossed:
                    pos_tp_min_crossed = pos_tp_min_crossed.replace(tzinfo=None)

                pos_closed = position_cache_row['closed']
                pos_closed = arrow.get(pos_closed).datetime if pos_closed and isinstance(pos_closed, str) else pos_closed
                if pos_closed:
                    pos_closed = pos_closed.replace(tzinfo=None)
                pos_side = OrderSide.UNDEFINED
                if pos_status!=PositionStatus.UNDEFINED.name:
                    pos_side = OrderSide.BUY if pos and pos>0 else OrderSide.SELL

                '''
                This condition is to block re-entry on same candle.
                lo_row_timestamp_ms is timestamp_ms of latest lo row, which is initialized to zero on start.
                '''
                if pos_status==PositionStatus.CLOSED.name and lo_row_timestamp_ms!=0:
                    total_ms_elapsed_since_lo_interval_rolled = int((datetime.now().timestamp()*1000 - lo_row_timestamp_ms))
                    log(f"total_ms_elapsed_since_lo_interval_rolled: {total_ms_elapsed_since_lo_interval_rolled:,} (~{int(total_ms_elapsed_since_lo_interval_rolled/1000/60):,} min)")
                    if (total_ms_elapsed_since_lo_interval_rolled < lo_interval_ms) and (pos_closed.timestamp()*1000)>lo_row_timestamp_ms:
                        block_entries = True
                        log(f"Block entries: recent TP, block re-entry within same lo candle")

                if pos_status==PositionStatus.SL.name:
                    total_ms_elapsed_since_sl = int((datetime.now() - pos_closed).total_seconds() *1000)
                    log(f"sl_num_intervals_delay: {param['sl_num_intervals_delay']}, min_sl_age_ms: {min_sl_age_ms:,}, total_ms_elapsed_since_sl: {total_ms_elapsed_since_sl:,} (~{int(total_ms_elapsed_since_sl/1000/60):,} min)")
                    if total_ms_elapsed_since_sl < min_sl_age_ms:
                        block_entries = True
                        log(f"Block entries: recent SL")

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

                entry_px = position_cache_row['entry_px']
                tp_max_target = position_cache_row['tp_max_target']
                tp_min_target = position_cache_row['tp_min_target']
                sl_price = position_cache_row['sl_price']
                max_pnl_potential_bps = position_cache_row['max_pnl_potential_bps']
                close_px = position_cache_row['close_px']

                unreal_live = position_cache_row['unreal_live']
                max_unreal_live = position_cache_row['max_unreal_live']
                max_pain = position_cache_row['max_pain']
                max_recovered_pnl = position_cache_row['max_recovered_pnl']
                pnl_live_bps = position_cache_row['pnl_live_bps']
                pnl_open_bps = position_cache_row['pnl_open_bps']
                max_unreal_live_bps = position_cache_row['max_unreal_live_bps']
                max_unreal_open_bps = position_cache_row['max_unreal_open_bps']

                running_sl_percent_hard = position_cache_row['running_sl_percent_hard']
                sl_trailing_min_threshold_crossed = position_cache_row['sl_trailing_min_threshold_crossed']
                # No need read 'sl_percent_trailing' off position_cache_row, it's a static param.
                effective_tp_trailing_percent = position_cache_row['effective_tp_trailing_percent']
                loss_trailing = position_cache_row['loss_trailing']

                pnl_percent_notional = pnl_open_bps/100
                max_pain_percent_notional = max_pain / pos_usdt * 100 if pos_usdt!=0 else 0
                max_recovered_pnl_percent_notional = max_recovered_pnl / pos_usdt * 100 if pos_usdt!=0 else 0

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
                position_from_exchange = await exchange.fetch_position(_ticker) 

                if exchange.options['defaultType']!='spot': 
                    if executed_position and position_from_exchange:
                        position_from_exchange_num_contracts = position_from_exchange['contracts']
                        if position_from_exchange and position_from_exchange['side']=='short':
                            position_from_exchange_num_contracts = position_from_exchange_num_contracts *-1 if position_from_exchange_num_contracts>0 else position_from_exchange_num_contracts

                        position_from_exchange_base_ccy  = position_from_exchange_num_contracts * multiplier

                        if position_from_exchange_base_ccy!=pos: 
                            position_break = True

                            err_msg = f"{_ticker}: Position break! expected: {executed_position['position']['amount_base_ccy']}, actual: {position_from_exchange_base_ccy}" 
                            log(err_msg)
                            dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Position break! {_ticker}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                
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
                                'current_ts_ms' : int(dt_now.timestamp() *1000),
                                'hi_row_timestamp_ms' : int(hi_row['timestamp_ms']),
                                'candles_age' : int(candles_age),
                                'hi_interval_ms' : int(hi_interval_ms)
                            }
                            log(err_msg, LogLevel.CRITICAL)
                            # dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid hi_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
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

                            lo_candles_interval_rolled = False
                            if lo_row['timestamp_ms']!=lo_row_timestamp_ms:
                                lo_row_timestamp_ms = lo_row['timestamp_ms']
                                lo_candles_interval_rolled = True

                            trailing_candles = pd_lo_candles_w_ta \
                                    .tail(param['reversal_num_intervals']) \
                                    .values.tolist()
                                    
                            trailing_candles = [dict(zip(pd_lo_candles_w_ta.columns, row)) for row in trailing_candles]
                            
                        else:
                            lo_candles_valid = False
                            err_msg = {
                                'current_ts_ms' : int(dt_now.timestamp() *1000),
                                'lo_row_timestamp_ms' : int(lo_row['timestamp_ms']),
                                'candles_age' : int(candles_age),
                                'lo_interval_ms' : int(lo_interval_ms)
                            }
                            log(err_msg, LogLevel.CRITICAL)
                            # dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid lo_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                    else:
                        lo_candles_valid = False
                        err_msg = f"lo candles missing, topic: {lo_candles_w_ta_topic}"
                        log(err_msg, LogLevel.CRITICAL)
                        dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Invalid lo_candles", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                    if orderbook_topic in keys:
                        message = redis_client.get(orderbook_topic)
                        if message:
                            message = message.decode('utf-8')
                        ob = json.loads(message) if message else None
                        orderbook_valid = ob['is_valid']
                        err_msg = f"Invalid orderbook, topic: {orderbook_topic}, fetch from REST instead"
                        log(err_msg, LogLevel.WARNING)

                    else:
                        orderbook_valid = False
                        err_msg = f"orderbook missing, topic: {orderbook_topic}, fetch from REST instead"
                        log(err_msg, LogLevel.WARNING)
                        
                    if not orderbook_valid:
                        ob = await exchange.fetch_order_book(symbol=_ticker, limit=10) 

                    best_ask = min([x[0] for x in ob['asks']])
                    best_bid = max([x[0] for x in ob['bids']])
                    mid = (best_ask+best_bid)/2
                    spread_bps = (best_ask/best_bid - 1) * 10000

                    level_granularity = param['level_granularity']
                    adjacent_levels = compute_adjacent_levels(num=mid, level_granularity=level_granularity, num_levels_per_side=2)
                    level_below = adjacent_levels[1]
                    level_above = adjacent_levels[3]

                    # amount_quote_ccy takes precedence over amount_base_ccy
                    if param['amount_quote_ccy']:
                        param['amount_base_ccy'] = param['amount_quote_ccy']/mid
                        param['max_pos_amount_base_ccy'] = param['max_pos_amount_quote_ccy']/mid
                    log(f"Sizing info, mid: {mid}, amount_quote_ccy: {param['amount_quote_ccy']}, amount_base_ccy: {param['amount_base_ccy']}, max_pos_amount_quote_ccy: {param['max_pos_amount_quote_ccy']}, max_pos_amount_base_ccy: {param['max_pos_amount_base_ccy']}")

                    if pos!=0:
                        pos_usdt = mid * pos
                        pd_position_cache.loc[position_cache_row.name, 'pos_usdt'] = pos_usdt

                        unrealized_pnl_optimistic, unrealized_pnl_pessimistic = mid, mid

                        if pos_side == OrderSide.BUY:
                            unreal_live = round((mid - entry_px) * param['amount_base_ccy'], 3)
                            if lo_candles_valid:
                                unrealized_pnl_optimistic = round((trailing_candles[-1]['high'] - entry_px) * param['amount_base_ccy'], 3)
                                unrealized_pnl_pessimistic = round((trailing_candles[-1]['low'] - entry_px) * param['amount_base_ccy'], 3)
                            unrealized_pnl_open = unreal_live
                            if total_sec_since_pos_created > (lo_interval_ms/1000) and lo_candles_valid:
                                '''
                                "unrealized_pnl_open": To align with backtests, motivation is to avoid spikes and trigger trailing stops too early.
                                But we need be careful with tall candles immediately upon entries.
                                    trailing_candles[-1] is latest candle
                                Example long BTC, a mean reversion trade
                                    entry_px        $97,000
                                    open            $99,000 (This is trailing_candles[-1][1], so it's big red candle)
                                    mid             $97,200 (Seconds after entry)

                                    unreal_live                 $200 per BTC
                                    unrealized_pnl_open $2000 per BTC (This is very misleading! This would cause algo to TP prematurely!)
                                Thus for new entries, 
                                    unrealized_pnl_open = unreal_live
                                '''
                                unrealized_pnl_open = (trailing_candles[-1]['open'] - entry_px) * param['amount_base_ccy']
                        elif pos_side == OrderSide.SELL:
                            unreal_live = round((entry_px - mid) * param['amount_base_ccy'], 3)
                            if lo_candles_valid:
                                unrealized_pnl_optimistic = round((trailing_candles[-1]['low'] - entry_px) * param['amount_base_ccy'], 3)
                                unrealized_pnl_pessimistic = round((trailing_candles[-1]['high'] - entry_px) * param['amount_base_ccy'], 3)
                            unrealized_pnl_open = unreal_live
                            if total_sec_since_pos_created > lo_interval_ms/1000 and lo_candles_valid:
                                unrealized_pnl_open = (entry_px - trailing_candles[-1]['open']) * param['amount_base_ccy']

                        if lo_candles_valid:
                            # lamda's may reference candles
                            kwargs = {k: v for k, v in locals().items() if k in trailing_stop_threshold_eval_func_params}
                            trailing_stop_threshold_eval_func_result = trailing_stop_threshold_eval_func(**kwargs)
                            tp_min_percent = trailing_stop_threshold_eval_func_result['tp_min_percent']
                            tp_max_percent = trailing_stop_threshold_eval_func_result['tp_max_percent']

                            log(f"trailing_stop_threshold_eval tp_min_percent: {tp_min_percent}, tp_max_percent: {tp_max_percent}")

                            '''
                            tp_min_percent adj: Strategies where target_price not based on tp_max_percent, but variable
                            Also be careful, not all strategies set target price so max_pnl_potential_bps can be null!
                            '''
                            if max_pnl_potential_bps and (max_pnl_potential_bps/100)<tp_max_percent:
                                tp_minmax_ratio = tp_min_percent/tp_max_percent
                                tp_max_percent = max_pnl_potential_bps/100
                                tp_min_percent = tp_minmax_ratio * tp_max_percent
                            # log(f"param tp_max_percent: {param['tp_max_percent']}, param tp_min_percent: {param['tp_min_percent']}, tp_minmax_ratio: {tp_minmax_ratio}, max_pnl_potential_bps: {max_pnl_potential_bps}, effective tp_max_percent: {tp_max_percent}, effective tp_min_percent: {tp_min_percent}")

                            kwargs = {k: v for k, v in locals().items() if k in sl_adj_func_params}
                            sl_adj_func_result = sl_adj_func(**kwargs)
                            running_sl_percent_hard = sl_adj_func_result['running_sl_percent_hard']

                        else:
                            # Fallback
                            # For 'running_sl_percent_hard', simply skip updating until you have candles again.

                            # sl_adj_func may use candles, so fall back mechanism is simply not further update 'running_sl_percent_hard'
                            # Given sl_adj_func generally only tighten stops as your trade goes greener greener, cap it by param['sl_hard_percent'] just in case.
                            running_sl_percent_hard = min(running_sl_percent_hard, param['sl_hard_percent']) 

                        pnl_live_bps = round(unreal_live / abs(pos_usdt) * 10000, 2) if pos_usdt else 0
                        pnl_open_bps = round(unrealized_pnl_open / abs(pos_usdt)  * 10000, 2) if pos_usdt else 0
                        pnl_percent_notional = pnl_open_bps/100

                        if unreal_live>max_unreal_live:
                            max_unreal_live = unreal_live

                        if pnl_live_bps>max_unreal_live_bps:
                            max_unreal_live_bps = pnl_live_bps                                

                        if pnl_open_bps>max_unreal_open_bps:
                            max_unreal_open_bps = pnl_open_bps

                        if unrealized_pnl_pessimistic < max_pain:
                            max_pain = unrealized_pnl_pessimistic

                        if unrealized_pnl_optimistic < 0 and unrealized_pnl_optimistic>max_pain:
                            recovered_pnl = unrealized_pnl_optimistic - max_pain
                            if recovered_pnl > max_recovered_pnl:
                                max_recovered_pnl = recovered_pnl

                        max_pain_percent_notional = round(max_pain / pos_usdt * 100, 2)
                        max_recovered_pnl_percent_notional = round(max_recovered_pnl / pos_usdt * 100, 2)

                        loss_trailing = round((1 - pnl_live_bps/max_unreal_live_bps) * 100, 2) if pnl_live_bps>0 else 0

                        pd_position_cache.loc[position_cache_row.name, 'unreal_live'] = unreal_live
                        pd_position_cache.loc[position_cache_row.name, 'max_unreal_live'] = max_unreal_live
                        pd_position_cache.loc[position_cache_row.name, 'max_pain'] = max_pain
                        pd_position_cache.loc[position_cache_row.name, 'max_recovered_pnl'] = max_recovered_pnl
                        pd_position_cache.loc[position_cache_row.name, 'pnl_live_bps'] = pnl_live_bps
                        pd_position_cache.loc[position_cache_row.name, 'pnl_open_bps'] = pnl_open_bps
                        pd_position_cache.loc[position_cache_row.name, 'max_unreal_live_bps'] = max_unreal_live_bps
                        pd_position_cache.loc[position_cache_row.name, 'max_unreal_open_bps'] = max_unreal_open_bps

                        pd_position_cache.loc[position_cache_row.name, 'loss_trailing'] = loss_trailing

                        # This is for tp_eval_func
                        this_ticker_open_trades.append(
                            {
                                'ticker' : _ticker,
                                'side' : pos_side.name.lower(), # backtests uses lower case
                                'amount' : pos_usdt,
                                'entry_price' : entry_px,
                                'target_price' : tp_max_target # This is the only field needed by backtest_core generic_tp_eval
                            }
                        )

                        log(f"pnl eval block tp_min_percent: {tp_min_percent}, tp_max_percent: {tp_max_percent}, sl_percent_trailing: {param['sl_percent_trailing']}, max_unreal_open_bps: {max_unreal_open_bps}, effective_tp_trailing_percent: {effective_tp_trailing_percent}, loss_trailing: {loss_trailing}")
                        
                    '''
                    On turn of interval, candles_provider may need a little time to publish latest candles.
                    It's by design strategy_executor will make entries only if you have valid candles.
                    However, TP/SL may be triggered regardless - mid price is coming from orderbook, not historical candles.
                    '''
                    if hi_candles_valid and lo_candles_valid:
                        if param['dump_candles']:
                            pd_hi_candles_w_ta.to_csv(f"hi_candles_{_ticker.replace(':','').replace('/','')}.csv")
                            pd_lo_candles_w_ta.to_csv(f"lo_candles_{_ticker.replace(':','').replace('/','')}.csv")
                            
                        # Strategies uses different indicators, thus: TargetStrategy.get_strategy_indicators()
                        _all_indicators = {}

                        pd_position_cache.loc[position_cache_row.name, "lo_row:datetime"] = lo_row['datetime']
                        pd_position_cache.loc[position_cache_row.name, "hi_row:datetime"] = hi_row['datetime']
                        pd_position_cache.loc[position_cache_row.name, "lo_row:timestamp_ms"] = str(lo_row['timestamp_ms']) # For display purpose, cast to str so won't print scientific notation
                        pd_position_cache.loc[position_cache_row.name, "hi_row:timestamp_ms"] = str(hi_row['timestamp_ms'])
                        pd_position_cache.loc[position_cache_row.name, "lo_row_tm1:id"] = lo_row_tm1.name
                        pd_position_cache.loc[position_cache_row.name, "hi_row_tm1:id"] = hi_row_tm1.name

                        _all_indicators["lo_row:datetime"] = lo_row['datetime'].strftime("%Y-%m-%d %H:%M")
                        _all_indicators["hi_row:datetime"] = hi_row['datetime'].strftime("%Y-%m-%d %H:%M")
                        _all_indicators["lo_row:timestamp_ms"] = int(lo_row['timestamp_ms'])
                        _all_indicators["hi_row:timestamp_ms"] = int(hi_row['timestamp_ms'])
                        _all_indicators["lo_row_tm1:id"] = lo_row_tm1.name
                        _all_indicators["hi_row_tm1:id"] = hi_row_tm1.name
                        _all_indicators['level_below'] = level_below
                        _all_indicators['level_above'] = level_above

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
                            _all_indicators[indicator] = indicator_value
                        
                        last_candles=trailing_candles # alias

                        pd_position_cache.loc[position_cache_row.name, 'ob_mid'] = mid
                        pd_position_cache.loc[position_cache_row.name, 'spread_bps'] = spread_bps
                        pd_position_cache.loc[position_cache_row.name, 'ob_best_bid'] = best_bid
                        pd_position_cache.loc[position_cache_row.name, 'ob_best_ask'] = best_ask
                        pd_position_cache.loc[position_cache_row.name, 'level_granularity'] = param['level_granularity']
                        pd_position_cache.loc[position_cache_row.name, 'level_below'] = level_below
                        pd_position_cache.loc[position_cache_row.name, 'level_above'] = level_above

                        if pos==0: # @todo: align with backtest_core, allow multi-slices entries
                            kwargs = {k: v for k, v in locals().items() if k in allow_entry_initial_func_params}
                            allow_entry_func_initial_result = allow_entry_initial_func(**kwargs)
                            allow_entry_initial_long = allow_entry_func_initial_result['long']
                            allow_entry_initial_short = allow_entry_func_initial_result['short']

                            log(f"block_entries: {block_entries}, allow_entry_initial_long: {allow_entry_initial_long}, allow_entry_initial_short: {allow_entry_initial_short}")

                            assert(not(allow_entry_initial_long and allow_entry_initial_short))
                            
                            allow_entry = allow_entry_initial_long or allow_entry_initial_short
                            allow_entry = allow_entry and pos_status!=PositionStatus.OPEN.name
                            if allow_entry and not block_entries:
                                kwargs = {k: v for k, v in locals().items() if k in allow_entry_final_func_params}
                                allow_entry_func_final_result = allow_entry_final_func(**kwargs)
                                allow_entry_final_long = allow_entry_func_final_result['long']
                                allow_entry_final_short = allow_entry_func_final_result['short']
                                target_price_long = allow_entry_func_final_result['target_price_long']
                                target_price_short = allow_entry_func_final_result['target_price_short']

                                log(f"allow_entry_final_long: {allow_entry_final_long}, allow_entry_final_short: {allow_entry_final_short}")

                                allow_entry_final_long = allow_entry_initial_long and allow_entry_final_long and (
                                    (pos==0 and pos_status in [ PositionStatus.UNDEFINED.name, PositionStatus.CLOSED.name, PositionStatus.SL.name ])
                                    or (pos>0 and pos + param["amount_base_ccy"] <= param['max_pos_amount_base_ccy'])
                                )
                                allow_entry_final_short = allow_entry_initial_short and allow_entry_final_short and (
                                    (pos==0 and pos_status in [ PositionStatus.UNDEFINED.name, PositionStatus.CLOSED.name, PositionStatus.SL.name ])
                                    or (pos<0 and abs(pos) + param["amount_base_ccy"] <= param['max_pos_amount_base_ccy'])
                                )
                                
                                assert(not(allow_entry_final_long and allow_entry_final_short))

                                pnl_potential_bps : Union[float, None] = None
                                if allow_entry_final_long or allow_entry_final_short:
                                    if allow_entry_final_long and target_price_long:
                                        side = 'buy'
                                        pnl_potential_bps = (target_price_long/mid - 1) *10000 if target_price_long else None
                                    elif allow_entry_final_short and target_price_short:
                                        side = 'sell'
                                        pnl_potential_bps = (mid/target_price_short - 1) *10000 if target_price_short else None
                                    else:
                                        raise ValueError("Either allow_long or allow_short!")

                                    '''
                                    tp_min_percent adj: Strategies where target_price not based on tp_max_percent, but variable
                                    Also be careful, not all strategies set target price so max_pnl_potential_bps can be null!
                                    '''
                                    tp_max_percent : float  = param['tp_max_percent']
                                    tp_min_percent : float  = param['tp_min_percent'] # adjusted by trailing_stop_threshold_eval_func
                                    tp_minmax_ratio = tp_min_percent/tp_max_percent
                                    if pnl_potential_bps and pnl_potential_bps<tp_max_percent:
                                        tp_max_percent = pnl_potential_bps/100
                                        tp_min_percent = tp_minmax_ratio * tp_max_percent
                                    # log(f"param tp_max_percent: {param['tp_max_percent']}, param tp_min_percent: {param['tp_min_percent']}, tp_minmax_ratio: {tp_minmax_ratio}, pnl_potential_bps: {pnl_potential_bps}, effective tp_max_percent: {tp_max_percent}, effective tp_min_percent: {tp_min_percent}")
                                    
                                    kwargs = {k: v for k, v in locals().items() if k in order_notional_adj_func_params}
                                    order_notional_adj_func_result = order_notional_adj_func(**kwargs)
                                    target_order_notional = order_notional_adj_func_result['target_order_notional']
                                    
                                    log(f"******** ENTRY (loop# {loop_counter}) ********")
                                    entry_positions : List[DivisiblePosition] = [
                                        DivisiblePosition(
                                            ticker = _ticker,
                                            side = side,
                                            amount = target_order_notional,
                                            leg_room_bps = param['leg_room_bps'],
                                            order_type = param['order_type'],
                                            slices = param['slices'],
                                            wait_fill_threshold_ms = param['wait_fill_threshold_ms'],
                                            fees_ccy=param['fees_ccy']
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
                                            dispatch_notification(
                                                title=f"singlelegta error from order gateway {gateway_id}!!", 
                                                message=err_msg, 
                                                footer=param['notification']['footer'], 
                                                params=notification_params, 
                                                log_level=logging.ERROR)
                                            raise ValueError(err_msg)
                                    executed_position = executed_positions[0] # We sent only one DivisiblePosition.

                                    new_pos_from_exchange = executed_position['filled_amount']
                                    amount_filled_usdt = mid * new_pos_from_exchange
                                    entry_px = executed_position['average_cost']
                                    new_pos_usdt_from_exchange = new_pos_from_exchange * executed_position['average_cost']
                                    fees = executed_position['fees']

                                    if side=='buy':
                                        tp_max_price = entry_px * (1 + tp_max_percent/100)
                                        tp_min_price = entry_px * (1 + tp_min_percent/100)
                                        sl_price = entry_px * (1 - running_sl_percent_hard/100)

                                    elif side=='sell':
                                        tp_max_price = entry_px * (1 - tp_max_percent/100)
                                        tp_min_price = entry_px * (1 - tp_min_percent/100)
                                        sl_price = entry_px * (1 + running_sl_percent_hard/100)

                                    executed_position['position'] = {
                                            'loop_counter' : loop_counter,
                                            'status' : 'open',
                                            'entry_px' : entry_px,
                                            'mid' : mid,
                                            'amount_base_ccy' : executed_position['filled_amount'],
                                            'tp_min_price' : tp_min_price,
                                            'tp_max_price' : tp_max_price,
                                            'sl_price' : sl_price,
                                            'tp_max_percent' : tp_max_percent,
                                            'tp_min_percent' : tp_min_percent,
                                            'running_sl_percent_hard' : running_sl_percent_hard,
                                            'filled_amount' : new_pos_from_exchange,
                                            'pos' : pos + new_pos_from_exchange,
                                            'pos_usdt' : pos_usdt + new_pos_usdt_from_exchange,
                                            'fees' : fees,
                                            'multiplier' : multiplier,
                                            'indicators' : {}
                                        }
                                    for indicator in _all_indicators.keys():
                                        executed_position['position']['indicators'][indicator] = _all_indicators[indicator]

                                    pd_position_cache.loc[position_cache_row.name, 'pos'] = pos + new_pos_from_exchange
                                    pd_position_cache.loc[position_cache_row.name, 'pos_usdt'] = pos_usdt + new_pos_usdt_from_exchange
                                    pd_position_cache.loc[position_cache_row.name, 'status'] = PositionStatus.OPEN.name
                                    pos_created = datetime.fromtimestamp(time.time())
                                    pd_position_cache.loc[position_cache_row.name, 'created'] = pos_created
                                    pd_position_cache.loc[position_cache_row.name, 'tp_min_crossed'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'closed'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'entry_px'] = entry_px
                                    pd_position_cache.loc[position_cache_row.name, 'tp_max_target'] = tp_max_price
                                    pd_position_cache.loc[position_cache_row.name, 'tp_min_target'] = tp_min_price
                                    pd_position_cache.loc[position_cache_row.name, 'sl_price'] = sl_price
                                    pd_position_cache.loc[position_cache_row.name, 'max_pnl_potential_bps'] = pnl_potential_bps
                                    pd_position_cache.loc[position_cache_row.name, 'close_px'] = None
                                    pd_position_cache.loc[position_cache_row.name, 'unreal_live'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_unreal_live'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_pain'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_recovered_pnl'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'pnl_live_bps'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'pnl_open_bps'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_unreal_live_bps'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'max_unreal_open_bps'] = 0
                                    pd_position_cache.loc[position_cache_row.name, 'running_sl_percent_hard'] = param['sl_hard_percent']
                                    pd_position_cache.loc[position_cache_row.name, 'sl_trailing_min_threshold_crossed'] = False
                                    pd_position_cache.loc[position_cache_row.name, 'effective_tp_trailing_percent'] = param['default_effective_tp_trailing_percent']
                                    pd_position_cache.loc[position_cache_row.name, 'loss_trailing'] = 0

                                    pos_entries.append(pos_created)
                                    pd_position_cache.at[position_cache_row.name, 'pos_entries'] = pos_entries

                                    orderhist_cache_row = {
                                                            'datetime' : dt_now,
                                                            'timestamp_ms' : int(dt_now.timestamp() * 1000),
                                                            'exchange' : exchange_name,
                                                            'ticker' : _ticker,
                                                            'reason' : 'entry',
                                                            'reason2' : None,
                                                            'side' : side,
                                                            'avg_price' : new_pos_usdt_from_exchange/new_pos_from_exchange,
                                                            'amount': abs(new_pos_usdt_from_exchange),
                                                            'pnl' : 0,
                                                            'pnl_bps' : 0,
                                                            'max_pain' : 0,
                                                            'fees' : fees,
                                                            'remarks' : None
                                                        }
                                    orderhist_cache = pd.concat([orderhist_cache, pd.DataFrame([orderhist_cache_row])], axis=0, ignore_index=True)

                                    log(executed_position)
                                    dispatch_notification(title=f"{param['current_filename']} {gateway_id} Entry succeeded. {_ticker} {side} {param['amount_base_ccy']} (USD amount: {amount_filled_usdt}) @ {entry_px}", message=executed_position['position'], footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                                    if param['dump_candles']:
                                        pd_hi_candles_w_ta.to_csv(f"hi_candles_entry_{gateway_id}_{_ticker.replace(':','').replace('/','')}_{loop_counter}_{int(dt_now.timestamp())}.csv")
                                        pd_lo_candles_w_ta.to_csv(f"lo_candles_entry_{gateway_id}_{_ticker.replace(':','').replace('/','')}_{loop_counter}_{int(dt_now.timestamp())}.csv")
                                        
                                    any_entry = True
                            
                    '''
                    Have a look at this for a visual explaination how "Gradually tightened stops" works:
                        https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/trading_util_tests.ipynb
                    '''
                    if (
                        (pnl_percent_notional>0 and pnl_percent_notional>=tp_min_percent) # pnl_percent_notional is evaluated using pnl_open_bps to avoid spikes
                        or (
                            pnl_percent_notional<0 
                            and max_recovered_pnl_percent_notional>=param['recover_min_percent']
                            and abs(max_pain_percent_notional)>=param['recover_max_pain_percent']
                        ) # Taking 'abs': Trailing stop can fire if trade moves in either direction - if your trade is losing trade.
                    ):
                        _effective_tp_trailing_percent = calc_eff_trailing_sl(
                            tp_min_percent = tp_min_percent,
                            tp_max_percent = tp_max_percent,
                            sl_percent_trailing = param['sl_percent_trailing'],
                            pnl_percent_notional = max_unreal_open_bps/100, # Note: Use [max]_unrealized_pnl_percent, not unrealized_pnl_percent!
                            default_effective_tp_trailing_percent = param['default_effective_tp_trailing_percent'],
                            linear=param['trailing_stop_mode'],
                            pow=param['non_linear_pow']
                        )

                        # Once pnl pass tp_min_percent, trailing stops will be activated. Even if pnl fall back below tp_min_percent.
                        effective_tp_trailing_percent = min(effective_tp_trailing_percent, round(_effective_tp_trailing_percent, 2))

                        if not sl_trailing_min_threshold_crossed:
                            pos_tp_min_crossed = dt_now
                            sl_trailing_min_threshold_crossed = True
                            pd_position_cache.loc[position_cache_row.name, 'tp_min_crossed'] = pos_tp_min_crossed
                            pd_position_cache.loc[position_cache_row.name, 'sl_trailing_min_threshold_crossed'] = sl_trailing_min_threshold_crossed

                            msg = {
                                'side' : pos_side.name,
                                'mid' : mid,
                                'entry_px' : entry_px,
                                'pnl_open_bps' : pnl_open_bps,
                                'tp_min_percent' : tp_min_percent,
                                'tp_max_percent' : tp_max_percent,
                                'sl_percent_trailing' : param['sl_percent_trailing'],
                                'effective_tp_trailing_percent' : effective_tp_trailing_percent
                            }
                            log(msg, LogLevel.CRITICAL)
                            dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} sl_trailing_min_threshold_crossed: True!", message=msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                        pd_position_cache.loc[position_cache_row.name, 'effective_tp_trailing_percent'] = effective_tp_trailing_percent

                        log(f"calc_eff_trailing_sl tp_min_percent: {tp_min_percent}, tp_max_percent: {tp_max_percent}, sl_percent_trailing: {param['sl_percent_trailing']}, max_unreal_open_bps: {max_unreal_open_bps}, effective_tp_trailing_percent: {effective_tp_trailing_percent}")


                    # STEP 2. Unwind position
                    tp = False
                    sl = False
                    if pos!=0:
                        reason : str = None
                        if unreal_live>0:
                            if lo_candles_valid:
                                kwargs = {k: v for k, v in locals().items() if k in tp_eval_func_params}
                                tp_final = tp_eval_func(**kwargs)

                            else:
                                # tp_eval_func may use candles, so below a fall back mechanism
                                if pos_side==OrderSide.BUY:
                                    tp_final = True if mid>=tp_max_target else False
                                elif pos_side==OrderSide.SELL:
                                    tp_final = True if mid<=tp_max_target else False

                            if effective_tp_trailing_percent==0:
                                tp_final = True
                            tp_trailing_stop = True if loss_trailing>=effective_tp_trailing_percent else False

                            # Potentially let the order to take deeper TP: exclude tp_final
                            # tp = tp_final or tp_trailing_stop
                            tp = tp_trailing_stop

                            if tp_final:
                                reason = f"tp_max_target {tp_max_target} reached."
                            elif tp_trailing_stop:
                                reason = f"Trailing stop fired. tp_min_target: {tp_min_target}, loss_trailing: {loss_trailing}, effective_tp_trailing_percent: {effective_tp_trailing_percent}"

                        else:
                            if abs(pnl_live_bps/100)>=running_sl_percent_hard:
                                sl = True
                                reason = "Hard stop"

                    if tp or sl:
                        log(f"******** EXIT (loop# {loop_counter}) ********")
                        exit_positions : List[DivisiblePosition] = [
                            DivisiblePosition(
                                ticker = _ticker,
                                side = 'sell' if pos_side==OrderSide.BUY else 'buy',
                                amount = pos,
                                leg_room_bps = param['leg_room_bps'],
                                order_type = param['order_type'],
                                slices = param['slices'],
                                wait_fill_threshold_ms = param['wait_fill_threshold_ms'],
                                fees_ccy=param['fees_ccy'],

                                reduce_only=True
                            )
                        ]
                        
                        log(f"Closing position. {_ticker}, pos: {pos}, pos_usdt: {pos_usdt}") 
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
                                    closed_pnl = (executed_position_close['average_cost'] - entry_px) * param['amount_base_ccy']
                                else:
                                    closed_pnl = (entry_px - executed_position_close['average_cost']) * param['amount_base_ccy']
                                closed_pnl = round(closed_pnl, 3)
                                
                                exit_px = executed_position_close['average_cost']
                                new_pos_from_exchange = pos + executed_position_close['filled_amount']
                                new_pos_usdt_from_exchange = new_pos_from_exchange * exit_px
                                fees = executed_position_close['fees']

                                executed_position_close['position'] = {
                                    'loop_counter' : loop_counter,
                                    'status' : 'TP' if tp else 'SL',
                                    'entry_px' : entry_px,
                                    'exit_px' : exit_px,
                                    'mid' : mid,
                                    'amount_base_ccy' : executed_position_close['filled_amount'],
                                    'pnl' : closed_pnl,
                                    'pnl_bps' : closed_pnl/abs(pos_usdt) *10000 if pos_usdt!=0 else 0,
                                    'fees' : fees,
                                    'max_pain' : max_pain,
                                    'running_sl_percent_hard' : running_sl_percent_hard,
                                    'loss_trailing' : loss_trailing,
                                    'effective_tp_trailing_percent' : effective_tp_trailing_percent,
                                    'created' : pos_created.strftime("%Y%m%d %H:%M:%S") if pos_created else None,
                                    'tp_min_crossed' : pos_tp_min_crossed.strftime("%Y%m%d %H:%M:%S") if pos_tp_min_crossed else None,
                                    'closed' : dt_now.strftime("%Y%m%d %H:%M:%S"),
                                    'reason' : reason
                                }

                                new_status = PositionStatus.SL.name if closed_pnl<=0 else PositionStatus.CLOSED.name
                                pd_position_cache.loc[position_cache_row.name, 'pos'] = new_pos_from_exchange
                                pd_position_cache.loc[position_cache_row.name, 'pos_usdt'] = new_pos_usdt_from_exchange
                                pd_position_cache.loc[position_cache_row.name, 'status'] = new_status
                                pd_position_cache.loc[position_cache_row.name, 'closed'] = dt_now
                                pd_position_cache.loc[position_cache_row.name, 'close_px'] = exit_px
                                pd_position_cache.loc[position_cache_row.name, 'unreal_live'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_unreal_live'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_pain'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_recovered_pnl'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'pnl_live_bps'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'pnl_open_bps'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_unreal_live_bps'] = 0
                                pd_position_cache.loc[position_cache_row.name, 'max_unreal_open_bps'] = 0
                                
                                pd_position_cache.at[position_cache_row.name, 'pos_entries'] = []
                                pd_position_cache.loc[position_cache_row.name, 'running_sl_percent_hard'] = param['sl_hard_percent']
                                pd_position_cache.loc[position_cache_row.name, 'sl_trailing_min_threshold_crossed'] = False
                                pd_position_cache.loc[position_cache_row.name, 'effective_tp_trailing_percent'] = param['default_effective_tp_trailing_percent']
                                
                                pd_position_cache.loc[position_cache_row.name, 'loss_trailing'] = 0
                                
                                tp_max_percent  = param['tp_max_percent']
                                tp_min_percent  = param['tp_min_percent']

                                # This is for tp_eval_func
                                this_ticker_open_trades.clear()

                                orderhist_cache_row = {
                                            'datetime' : dt_now,
                                            'timestamp_ms' : int(dt_now.timestamp() * 1000),
                                            'exchange' : exchange_name,
                                            'ticker' : _ticker,
                                            'reason' : new_status,
                                            'reason2' : reason,
                                            'side' : 'sell' if pos_side==OrderSide.BUY else 'buy',
                                            'avg_price' : exit_px,
                                            'amount': abs(new_pos_usdt_from_exchange),
                                            'pnl' : closed_pnl,
                                            'pnl_bps' : closed_pnl/abs(pos_usdt) *10000 if pos_usdt!=0 else 0,
                                            'max_pain' : max_pain,
                                            'fees' : fees,
                                            'remarks' : None
                                        }
                                orderhist_cache = pd.concat([orderhist_cache, pd.DataFrame([orderhist_cache_row])], axis=0, ignore_index=True)

                                log(executed_position_close)
                                dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} {'TP' if tp else 'SL'} on {_ticker} {'long' if pos_side==OrderSide.BUY else 'short'} succeeded. closed_pnl: {closed_pnl}", message=executed_position_close['position'], footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                                any_exit = True

                            else:
                                dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} Exit execution failed. {_ticker} {'long' if pos_side==OrderSide.BUY else 'short'}", message=executed_position_close, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)

                    log(f"loop# {loop_counter} ({loop_elapsed_sec} sec) [{gateway_id}]", log_level=LogLevel.INFO)
                    log(f"{tabulate(pd_position_cache.loc[:, 'exchange':'pos_entries'], headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)
                    log(f"{tabulate(pd_position_cache.loc[:, 'entry_px':'close_px'], headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)
                    log(f"{tabulate(pd_position_cache.loc[:, 'ob_mid':'level_above'], headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)
                    log(f"{tabulate(pd_position_cache.loc[:, 'unreal_live':'max_unreal_open_bps'], headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)
                    log(f"{tabulate(pd_position_cache.loc[:, 'running_sl_percent_hard':'loss_trailing'], headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)

                    if (
                        lo_candles_interval_rolled
                        and lo_candles_valid # Sometimes when you started strategy_executor just when lo_candles rolled
                    ):
                        log(f"candles_provider lo_candles interval rolled! Latest: {lo_row['datetime']} lo_row_timestamp_ms: {lo_row_timestamp_ms}")
                        log(f"{tabulate(pd_position_cache.loc[:, 'lo_row:datetime':'hi_row_tm1:id'], headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)
                        log(f"{tabulate(pd_position_cache.loc[:, strategy_indicators[0]:].transpose(), headers='keys', tablefmt='psql')}", log_level=LogLevel.INFO)

                    def _safe_update_cache(
                        file_name : str,
                        df
                    ):
                        bak_file_name : str = file_name + ".bak" 
                        if os.path.exists(bak_file_name):
                            os.remove(bak_file_name)
                        if os.path.exists(file_name):
                            os.rename(file_name, bak_file_name)
                        df.to_csv(file_name)
                    if (loop_counter%100==0) or (any_entry or any_exit):
                        _safe_update_cache(
                            file_name  = position_cache_file_name.replace("$GATEWAY_ID$", gateway_id),
                            df = pd_position_cache
                        )

                    if any_entry or any_exit:
                        _safe_update_cache(
                            file_name  = orderhist_cache_file_name.replace("$GATEWAY_ID$", gateway_id),
                            df = orderhist_cache
                        )
                        
            except Exception as loop_err:
                err_msg = f"Error: {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}"
                log(err_msg, log_level=LogLevel.ERROR)
                dispatch_notification(title=f"{param['current_filename']} {param['gateway_id']} error. {_ticker}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)
                
            finally:
                time.sleep(int(param['loop_freq_ms']/1000))
                
                if loop_counter==0 or loop_counter%1000==0:
                    try:
                        files_purged : List[str] = purge_old_file(
                            dir = param['current_dir'],
                            filename_regex_list = param['housekeep_filename_regex_list'],
                            max_age_sec = param['housekeep_max_age_sec']
                        )
                        for file_purged in files_purged:
                            logger.info(f"Purged: {file_purged}")
                    except Exception as housekeep_err:
                        log(f"Error while purging old files... {housekeep_err}", log_level=LogLevel.ERROR)

                loop_counter += 1

asyncio.run(
    main()
)