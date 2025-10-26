'''
Command line:
    python macdrsi_h_tc_crypto.py --white_list_tickers BTC/USDT:USDT,ETH/USDT:USDT,BNB/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT --reference_ticker BTC/USDT:USDT  --force_reload Y --block_entries_on_impacting_ecoevents N
    
Debug from vscode, Launch.json:
        {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "justMyCode": true,
                    "args" : [
                                "--white_list_tickers", "BTC/USDT:USDT,ETH/USDT:USDT,BNB/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT",
                                "--reference_ticker", "BTC/USDT:USDT",
                                "--force_reload", "Y",
                                "--block_entries_on_impacting_ecoevents", "N"
                            ]
                }
            ]
        }
'''
import os
import sys
import argparse
import json
from datetime import datetime, timedelta, timezone
import time
from typing import Dict, List, Tuple, Any, Callable
import pandas as pd

from ccxt.base.exchange import Exchange
from ccxt.bybit import bybit

from backtest_core import parseargs, get_logger, spawn_parameters, generic_pnl_eval, generic_tp_eval, generic_sort_filter_universe, run_all_scenario, dump_trades_to_disk

PYPY_COMPAT : bool = True

sys.path.append('../gizmo')
# from market_data_gizmo import fetch_historical_price, fetch_candles, fix_column_types, compute_candles_stats, partition_sliding_window, estimate_fib_retracement
base_dir : str = f"{os.path.dirname(sys.path[0])}\\single_leg_ta"

REPORT_NAME : str = "backtest_macdrsi_crosses_strategy_15m_tc_crypto"
CACHE_CANDLES : str = f"{os.path.dirname(sys.path[0])}\\cache\\candles"

white_list_tickers : List[str] = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "DOT/USDT:USDT",
    "TON/USDT:USDT",
    "MATIC/USDT:USDT",
    "SHIB/USDT:USDT",
    "LTC/USDT:USDT",
    "BCH/USDT:USDT",
    "UNI/USDT:USDT",
    "NEAR/USDT:USDT",
    "ICP/USDT:USDT",
    "APT/USDT:USDT"
]

white_list_tickers : List[str] = [ "SOL/USDT:USDT" ]

force_reload : bool = False

num_candles_limit = 100 # Depends on exchange but generally 100 ok!
param = {
        'apiKey' : None,
        'secret' : None,
        'password' : None,    # Other exchanges dont require this! This is saved in exchange.password!
        'subaccount' : None,
        'rateLimit' : 100,    # In ms
        'options' : {
            'defaultType': 'linear',
            'leg_room_bps' : 5,
            'trade_fee_bps' : 10,

            'list_ts_field' : 'listTime' # list_ts_field: Response field in exchange.markets[symbol] to indiate timestamp of symbol's listing date in ms. For bybit, markets['launchTime'] is list date. For okx, it's markets['listTime'].
        }
    }

exchanges = [ 
    bybit(param), 
]

exchanges[0].name='bybit_linear'

commission_bps : float = 5

'''
******** STRATEGY_SPECIFIC parameters ********
'''
additional_trade_fields : List[str] = [
     # Add fields you want to include in trade extract
]


'''
******** GENERIC parameters ********
'''
strategy_mode_values : List[str]= [ 'long_short'] # 'long_only', 'short_only', 'long_short'

'''
For example, Monday's are weird. Entries, SL adjustments ...etc may have STRATEGY_SPECIFIC logic around this.
'''
CAUTIOUS_DAYOFWEEK : List[int] = [ 0 ]
how_many_last_candles : int = 3
last_candles_timeframe : str = 'lo' # Either hi or lo (default)
enable_wait_entry : bool = True
enable_sliced_entry : bool = False
enable_athatl_logic : bool = False # If you have special logic in 'allow_entry_initial' or 'allow_entry_final'.

'''
Economic events comes from 'economic_calanedar.csv' in same folder.

Block entries if pending economic event in next x-intervals (applied on lo timeframe)
Set to -1 to disable this.
'''
adj_sl_on_ecoevents = False
block_entries_on_impacting_ecoevents = True
num_intervals_block_pending_ecoevents = 3
ECOEVENTS_MAPPED_REGIONS = [ 'united_states' ]

mapped_event_codes = [ 
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
        'gdp_growth_rate_yoy'
    ]

num_intervals_current_ecoevents = 8

sl_num_intervals_delay_values : List[float] = [ 15*4*8 ]
sl_hard_percent_values : List[float] = [ 2.5 ]
sl_percent_trailing_values : List[float] = [ 35 ]
use_gradual_tightened_trailing_stops : bool = True
trailing_stop_mode : str = "linear" # linear or parabolic

'''
This is for trailing stops slope calc.
Say if your trade's max profit potential is tp_max_percent=3%=300bps.
tp_min_percent = 0.3 means you will NOT TP until at least pnl > 0.3% or 30bps.
'''
tp_min_percent = 3
tp_max_percent = 5

POST_MOVE_NUM_INTERVALS : int = 24*3
POST_MOVE_PERCENT_THRESHOLD : int = 3

enable_hi_timeframe_confirm : bool = True

start_dates : List[datetime] = [ 
    datetime(2024, 4, 1)
]

hi_how_many_candles_values : List[Tuple[str, int, int]] = [ 
    ('1h', 24*3, 24*572)
]

lo_how_many_candles_values : List[Tuple[str, int, int]] = [ 
    ('15m', 15 *10, 15*4*24 *572)
]

hi_ma_short_vs_long_interval_values : List[Tuple[int, int]] = [ (12, 30) ]
lo_ma_short_vs_long_interval_values : List[Tuple[int, int]] = [ (5, 10) ]

rsi_sliding_window_how_many_candles : int = 14 # For RSI, 14 is standard.  If you want see spikes >70 and <30, use this config.
rsi_trend_sliding_window_how_many_candles : int = 30 # This is for purpose of RSI trend identification (Locating local peaks/troughs in RSI). This should typically be multiples of 'rsi_sliding_window_how_many_candles'.
rsi_upper_threshold_values : List[float] = [ 60 ]
rsi_lower_threshold_values : List[float] = [ 40 ]
rsi_midrangeonly : bool = False

target_fib_level : float = 0.618
boillenger_std_multiples_values : List[float] = [ 2 ]
allow_entry_sit_bb : bool = True
hurst_exp_window_how_many_candles : int = 125 # For hurst, at least 125.


# 'strategy_mode' decides if strategy can long_only, short_only, long_short at get go of back test. If long_above_btc_ema_short_below==True, strategy can long at bottom only if BTC (General market) stands above say 90d EMA. Or short only if BTC below 90d EMA for the given point in time.
ref_ema_num_days_fast : int = 5
ref_ema_num_days_slow : int = 90
long_above_ref_ema_short_below : bool = True
ref_price_vs_ema_percent_threshold : float = 2
ath_atl_close_gap_threshold_percent : float = 3

ema_short_slope_threshold_values : List[float] = [ 999 ] # 999 essentially turn it off

initial_cash_values : List[float] = [ 100000 ]

entry_percent_initial_cash_values : List[float] = [ 70 ]
target_position_size_percent_total_equity_values : List[float] = [ 100 ]
min_volume_usdt_threshold_values : List[float] = [ 100000 ]
clip_order_notional_to_best_volumes : bool = False
constant_order_notional : bool = True if min(start_dates) <= datetime(2024,1,1) else False # This is avoid snowball effect in long dated back tests

dayofweek_adj_map_order_notional : Dict = {
        0 : 1,
        1 : 1,
        2 : 1,
        3 : 1,
        4 : 1,
        5 : 1,
        6 : 1
}

dayofweek_sl_adj_map : Dict = {
    0 : 1,
    1 : 1,
    2 : 1,
    3 : 1,
    4 : 1,
    5 : 1,
    6 : 0.5
}

# Segmentation related parameters https://norman-lm-fung.medium.com/time-series-slicer-and-price-pattern-extractions-81f9dd1108fd
sliding_window_ratio : float = 16
smoothing_window_size_ratio : int = 3
linregress_stderr_threshold : float = 10
max_recur_depth : int = 2
min_segment_size_how_many_candles : int = 15
segment_consolidate_slope_ratio_threshold : float = 2
sideway_price_condition_threshold : float = 0.05 # i.e. Price if stay within 5% between start and close it's considered 'Sideway' market.

ECONOMIC_CALENDARS_FILE : str = "economic_calanedar_archive.csv"

default_level_granularity : float = 0.001

args = parseargs()
force_reload = args['force_reload']
white_list_tickers : List[str] = args['white_list_tickers']
reference_ticker : str = args['reference_ticker']
block_entries_on_impacting_ecoevents = args['block_entries_on_impacting_ecoevents']
enable_sliced_entry = args['enable_sliced_entry']
asymmetric_tp_bps : int = args['asymmetric_tp_bps']

full_report_name = f"{REPORT_NAME}_{start_dates[0].strftime('%Y%m%d')}"
trade_extract_filename : str = f"{full_report_name}_{white_list_tickers[0].replace(':','').replace('/','')}_trades.csv"

logger = get_logger(full_report_name)

import inspect
import builtins
def is_external(obj):
    if inspect.ismodule(obj):
        return True
    module = getattr(obj, '__module__', None)
    return module and not module.startswith('__')  # Exclude built-in/dunder modules

local_vars = {
    k: v 
    for k, v in locals().items() 
    if not (k.startswith('__') and k.endswith('__'))  # Exclude dunders
    and not is_external(v)  # Exclude anything from external modules
}

algo_params : List[Dict] = spawn_parameters(local_vars)

logger.info(f"#algo_params: {len(algo_params)}")


'''
******** STRATEGY_SPECIFIC Logic here ********
a. order_notional_adj
    Specific logic to adjust order sizes based on market condition(s) for example.
b. entry (initial + final)
    'allow_entry_initial' is first pass entry conditions determination.
    If 'allow_entry_initial' allow entry, 'allow_entry_final' will perform the second pass entry condition determinations.
    'allow_entry_final' is generally for more expensive operations, keep 'allow_entry_initial' fast and nimble.
c. 'pnl_eval' (You may wish to use specific prices to mark your TPs)
d. 'tp_eval' (Logic to fire TP)
e. 'sl_adj'
    Adjustment to sl_percent_hard
f. 'trailing_stop_threshold_eval'
g. 'sort_filter_universe' (optional, if 'white_list_tickers' only has one ticker for example, then you don't need bother)
h. 'additional_trade_fields' to be included in the trade extract file
'''
def order_notional_adj(
    algo_param : Dict,
):
    initial_cash : float = algo_param['initial_cash']
    entry_percent_initial_cash : float = algo_param['entry_percent_initial_cash']
    target_order_notional = initial_cash * entry_percent_initial_cash/100
    return {
         'target_order_notional' : target_order_notional
    }
     
def allow_entry_initial(
    lo_row_tm1,
    hi_row_tm1,
	last_candles
) -> Dict[str, bool]:
    return {
        'long' : _allow_entry_initial('long', lo_row_tm1, hi_row_tm1, last_candles),
        'short' : _allow_entry_initial('short', lo_row_tm1, hi_row_tm1, last_candles)
    }
def _allow_entry_initial(
	long_or_short : str,  # long or short
    lo_row_tm1,
    hi_row_tm1,
	last_candles
) -> Dict[str, bool]:
	if long_or_short == "long":
		if (
                lo_row_tm1['macd_cross'] == 'bullish'
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bullish_cross_last_id']
                      and 
                      (lo_row_tm1.name - lo_row_tm1['macd_bullish_cross_last_id']) < 5
                )
                and lo_row_tm1['rsi_trend']=="up"
                and lo_row_tm1['close']>hi_row_tm1['ema_close']
        ):
			return True
		else:
			return False
	elif long_or_short == "short":
		if (
                lo_row_tm1['macd_cross'] == 'bearish'
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bearish_cross_last_id']
                      and 
                      (lo_row_tm1.name - lo_row_tm1['macd_bearish_cross_last_id']) < 5
                )
                and lo_row_tm1['rsi_trend']=="down"
                and lo_row_tm1['close']<hi_row_tm1['ema_close']
        ):
			return True
		else:
			return False

def allow_entry_final(
    lo_row,
    algo_param : Dict
    
) -> bool:
    reference_ticker = algo_param['reference_ticker']
    timestamp_ms : int = lo_row['timestamp_ms']
    open : float = lo_row['open']

    entry_price_long, entry_price_short = open, open
    allow_long, allow_short = True, True
    reference_price = None
    
    pnl_potential_bps = algo_param['tp_max_percent']*100

    target_price_long = entry_price_long * (1 + pnl_potential_bps/10000)
    target_price_short = entry_price_short * (1 - pnl_potential_bps/10000)

    return {
            'long' : allow_long,
            'short' : allow_short,

            # In additional to allow or not, allow_entry_final also calculate a few things which you may need to mark the entry trades.
            'entry_price_long' : entry_price_long,
            'entry_price_short' : entry_price_short,
            'target_price_long' : target_price_long,
            'target_price_short' : target_price_short,
            'reference_price' : reference_price
        }

allow_slice_entry = allow_entry_initial

def sl_adj(
    max_unrealized_pnl_live : float,
    current_position_usdt : float,
    algo_param : Dict
):
    tp_min_percent = algo_param['tp_min_percent']
    max_pnl_percent_notional = max_unrealized_pnl_live / current_position_usdt * 100
    running_sl_percent_hard = algo_param['sl_hard_percent']
    return {
         'running_sl_percent_hard' : running_sl_percent_hard
    }

def trailing_stop_threshold_eval(
        algo_param : Dict
    ) -> Dict[str, float]:
        tp_min_percent = algo_param['tp_min_percent']
        tp_max_percent = algo_param['tp_max_percent']
        return {
            'tp_min_percent' : tp_min_percent,
            'tp_max_percent' : tp_max_percent
        }

def pnl_eval (
        this_candle,
        lo_row_tm1,
        running_sl_percent_hard : float,
        this_ticker_open_trades : List[Dict],
        algo_param : Dict
) -> Dict[str, float]:
    return generic_pnl_eval(
        this_candle,
        running_sl_percent_hard,
        this_ticker_open_trades,
        algo_param,
        long_tp_indicator_name=None,
        short_tp_indicator_name=None
    )

def tp_eval (
        this_ticker_open_positions_side : str,
        lo_row,
        this_ticker_open_trades : List[Dict],
        algo_param : Dict
) -> bool:
    '''
    Be very careful, backtest_core 'generic_pnl_eval' may use a) some indicator (tp_indicator_name), or b) target_price to evaluate 'unrealized_pnl_tp'.
    'tp_eval' only return True or False but it needs be congruent with backtest_core 'generic_pnl_eval', otherwise incorrect rosy pnl may be reported.
    '''
    return generic_tp_eval(lo_row, this_ticker_open_trades)

def sort_filter_universe(
    tickers : List[str],
    exchange : Exchange,

    # Use "i" (row index) to find current/last interval's market data or TAs from "all_exchange_candles"
    i,
    all_exchange_candles : Dict[str, Dict[str, Dict[str, pd.DataFrame]]],

    max_num_tickers : int = 10
) -> List[str]:
    return generic_sort_filter_universe(
         tickers=tickers,
         exchange=exchange,
         i=i,
         all_exchange_candles=all_exchange_candles,
         max_num_tickers=max_num_tickers
    )

algo_results : List[Dict] = run_all_scenario(
    algo_params=algo_params,
    exchanges=exchanges,
    order_notional_adj_func=order_notional_adj,
    allow_entry_initial_func=allow_entry_initial,
    allow_entry_final_func=allow_entry_final,
    allow_slice_entry_func=allow_slice_entry,
    sl_adj_func=sl_adj,
    trailing_stop_threshold_eval_func=trailing_stop_threshold_eval,
    pnl_eval_func=pnl_eval,
    tp_eval_func=tp_eval,
    sort_filter_universe_func=sort_filter_universe,

    logger=logger
)

dump_trades_to_disk(
     algo_results,
     trade_extract_filename,
     logger
     )
