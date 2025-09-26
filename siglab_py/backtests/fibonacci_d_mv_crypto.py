'''
Command line:
    python fibonacci_d_mv_crypto.py --white_list_tickers BTC/USDT:USDT --reference_ticker BTC/USDT:USDT
    
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
                                "--white_list_tickers", "BTC/USDT:USDT",
                                "--reference_ticker", "BTC/USDT:USDT"
                            ]
                }
            ]
        }
'''
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Union
import pandas as pd

from ccxt.base.exchange import Exchange
from ccxt.binance import binance # Use any crypto exchange that support the pair you want to trade

from backtest_core import get_logger, spawn_parameters, generic_pnl_eval, generic_tp_eval, run_all_scenario, dump_trades_to_disk

PYPY_COMPAT : bool = False

sys.path.append('../gizmo')
# from market_data_gizmo import fetch_historical_price, fetch_candles, fix_column_types, compute_candles_stats, partition_sliding_window, estimate_fib_retracement
base_dir : str = f"{os.path.dirname(sys.path[0])}\\single_leg_ta"

REPORT_NAME : str = "backtest_fibonacci_strategy_d_mv_crypto"
CACHE_CANDLES : str = f"{os.path.dirname(sys.path[0])}\\cache\\candles"

white_list_tickers : List[str] = [ 'BTC/USDT' ]

force_reload : bool = True

'''
To reuse previously pulled candles or candles from other utility, specify file name here.
    https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/ccxt_candles_ta_to_csv.py
    https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/futu_candles_ta_to_csv.py
'''
reference_candles_file : Union[str, None] = None
hi_candles_file : Union[str, None] = None
lo_candles_file : Union[str, None] = None

num_candles_limit = 100 # Depends on exchange but generally 100 ok!
param = {
        'apiKey' : None,
        'secret' : None,
        'password' : None,    # Other exchanges dont require this! This is saved in exchange.password!
        'subaccount' : None,
        'rateLimit' : 100,    # In ms
        'options' : {
            'defaultType': 'spot',
            'leg_room_bps' : 5,
            'trade_fee_bps' : 10,

            'list_ts_field' : 'listTime' # list_ts_field: Response field in exchange.markets[symbol] to indiate timestamp of symbol's listing date in ms. For bybit, markets['launchTime'] is list date. For okx, it's markets['listTime'].
        }
    }

exchanges : List[Exchange] = [ 
    binance(param),  # type: ignore
]

exchanges[0].name='binance_linear' # type: ignore

commission_bps : float = 5

'''
******** STRATEGY_SPECIFIC parameters ********
'''
target_fib_level : float = 0.618

additional_trade_fields : List[str] = [ # These are fields I wish to include in trade extract file for examination
    'lo_tm1_close',

    'hi_tm1_normalized_ema_long_slope',

    'hi_tm1_min_short_periods',
    'hi_tm1_idmin_short_periods',
    'hi_tm1_idmin_dt_short_periods',
    'hi_tm1_max_short_periods',
    'hi_tm1_idmax_short_periods',
    'hi_tm1_idmax_dt_short_periods',

    'hi_tm1_min_long_periods',
    'hi_tm1_idmin_long_periods',
    'hi_tm1_idmin_dt_long_periods',
    'hi_tm1_max_long_periods',
    'hi_tm1_idmax_long_periods',
    'hi_tm1_idmax_dt_long_periods',
]


'''
******** GENERIC parameters ********
'''
strategy_mode_values : List[str]= [ 'long_short'] # 'long_only', 'short_only', 'long_short'


start_dates : List[datetime] = [ 
    # datetime(2024, 4, 1),
    # datetime(2023, 1,1)
    datetime(2021, 3, 1),
]

# 'hi' refers to 'higher timeframe'
hi_how_many_candles_values : List[Tuple[str, int, int]] = [ 
    # ('1d', 30, 478),
    # ('1d', 30, 887),
    ('1d', 30, 1595),
]

'''
'lo' refers to 'lower timeframe': Tuple (interval, sliding window size, total num candles to fetch)
- sliding window size: backtest_core. will parse into 'lo_stats_computed_over_how_many_candles', which will be passed as 'sliding_window_how_many_candles' to 'compute_candles_stats'. Things like EMAs depends on this.
- total num candles to fetch: It's just your test duration from start to end, how many candles are there?
'''
lo_how_many_candles_values : List[Tuple[str, int, int]] = [ 
    # ('1h', 24, 24*478),
    # ('1h', 24, 24*887),
    ('1h', 24, 24*1595),
]

# For 'lower timeframe' as well as 'higher timeframe', EMA's are evaluated with 'long periods' and 'short periods'. In example below, 'long' is 24 hours, 'short' is 8 hours.
hi_ma_short_vs_long_interval_values : List[Tuple[int, int]] = [ (15, 30) ]
lo_ma_short_vs_long_interval_values : List[Tuple[int, int]] = [ (8, 24) ]

# 'strategy_mode' decides if strategy can long_only, short_only, long_short at get go of back test. If long_above_btc_ema_short_below==True, strategy can long at bottom only if BTC (General market) stands above say 90d EMA. Or short only if BTC below 90d EMA for the given point in time.
ref_ema_num_days_fast : int = 30
ref_ema_num_days_slow : int = 90
long_above_ref_ema_short_below : bool = False

'''
For example, Monday's are weird. Entries, SL adjustments ...etc may have STRATEGY_SPECIFIC logic around day of week, or not.
'''
CAUTIOUS_DAYOFWEEK : List[int] = [0,1,2,3,4]
how_many_last_candles : int = 3 # How many candles to be included in Reversal Check? Note the last is always 'current' candle.
last_candles_timeframe : str = 'hi' # Either hi or lo (default)
enable_sliced_entry : bool = True # This relates to param 'entry_percent_initial_cash_values' as well: If you entry notional say 33% of total equity. If enable_sliced_entry=True, your entries can be done in slices.

initial_cash_values : List[float] = [ 100000 ]
entry_percent_initial_cash_values : List[float] = [ 70 ]
target_position_size_percent_total_equity_values : List[float] = [ 100 ]
min_volume_usdt_threshold_values : List[float] = [ 100000 ]
clip_order_notional_to_best_volumes : bool = False
constant_order_notional : bool = True if min(start_dates) <= datetime(2024,1,1) else False # This is avoid snowball effect in long dated back tests

'''
Economic events comes from 'economic_calanedar_archive.csv' in same folder.

Block entries if pending economic event in next x-intervals (applied on lo timeframe)
Set to -1 to disable this.
'''
adj_sl_on_ecoevents = False
block_entries_on_impacting_ecoevents = False
num_intervals_block_pending_ecoevents = 24
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

sl_num_intervals_delay_values : List[float] = [ 0 ] # Delay before re-entry after SL in num intervals (lo timeframe)
sl_hard_percent_values : List[float] = [ 3 ] # Hard SL (Evaluated 'pessimistically': For example if you long and candle spike down -3%, even close -1%, SL will be triggered)
sl_percent_trailing_values : List[float] = [ 50 ] # How much give back to street for trailing stops to fire?
'''
References: 'calc_eff_trailing_sl' in backtest_core
    https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/trading_util_tests.ipynb
    https://medium.com/@norman-lm-fung/gradually-tightened-trailing-stops-f7854bf1e02b
'''
use_gradual_tightened_trailing_stops : bool = True
trailing_stop_mode : str = "linear" # linear or parabolic
tp_min_percent = 0.5
tp_max_percent = 5

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
    6 : 1
}

POST_MOVE_NUM_INTERVALS : int = 24*2
POST_MOVE_PERCENT_THRESHOLD : int = 3

enable_hi_timeframe_confirm : bool = True

boillenger_std_multiples_values : List[float] = [ 2 ] # Even if your strategy don't use it, compute_candles_stats takes this as argument.
rsi_upper_threshold_values : List[float] = [ 70 ]
rsi_lower_threshold_values : List[float] = [ 30 ]
rsi_sliding_window_how_many_candles : int = 14 # For RSI, 14 is standard.  If you want see spikes >70 and <30, use this config.
rsi_trend_sliding_window_how_many_candles : int = 24*7 # This is for purpose of RSI trend identification (Locating local peaks/troughs in RSI). This should typically be multiples of 'rsi_sliding_window_how_many_candles'.
hurst_exp_window_how_many_candles : int = 125 # For hurst, at least 125.

# Segmentation related parameters https://norman-lm-fung.medium.com/time-series-slicer-and-price-pattern-extractions-81f9dd1108fd
sliding_window_ratio : float = 16
smoothing_window_size_ratio : int = 3
linregress_stderr_threshold : float = 10
max_recur_depth : int = 2
min_segment_size_how_many_candles : int = 15
segment_consolidate_slope_ratio_threshold : float = 2
sideway_price_condition_threshold : float = 0.05 # i.e. Price if stay within 5% between start and close it's considered 'Sideway' market.

ECONOMIC_CALENDARS_FILE : str = "economic_calanedar_archive.csv"

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--white_list_tickers", help="Comma seperated list, example: BTC/USDT:USDT,ETH/USDT:USDT,XRP/USDT:USDT ", default="BTC/USDT:USDT")
    parser.add_argument("--reference_ticker", help="This is ticker for bull / bear determination.", default="BTC/USDT:USDT")
    parser.add_argument("--asymmetric_tp_bps", help="A positive asymmetric_tp_bps means you are taking deeper TPs. A negative asymmetric_tp_bps means shallower", default=0)
    args = parser.parse_args()
    if args.white_list_tickers:
        white_list_tickers = args.white_list_tickers.split(',')
        
    reference_ticker = args.reference_ticker if args.reference_ticker else white_list_tickers[0] # type: ignore
    asymmetric_tp_bps = int(args.asymmetric_tp_bps)

    return {
        'white_list_tickers' : white_list_tickers, # type: ignore
        'reference_ticker' : reference_ticker,
        'asymmetric_tp_bps' : asymmetric_tp_bps
    }
args = parseargs()
white_list_tickers : List[str] = args['white_list_tickers']
reference_ticker : str = args['reference_ticker']
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
g. 'additional_trade_fields' to be included in trade extract file (Entries only. Not applicable to exit trades.)
'''
def order_notional_adj(
    algo_param : Dict,
):
    initial_cash : float = algo_param['initial_cash']
    entry_percent_initial_cash : float = algo_param['entry_percent_initial_cash']
    '''
    Slicing: If 100% target_order_notional on first entry, essentially you'd have disabled slicing. Alternative is adjust entry cash check.
    '''
    target_order_notional = initial_cash * entry_percent_initial_cash/100
            
    return {
         'target_order_notional' : target_order_notional
    }
     
def allow_entry_initial(
	key : str,
    lo_row_tm1,
    hi_row_tm1,
    hi_fib_eval_result
) -> Dict[str, bool]:
    return {
        'long' : _allow_entry_initial(key, 'long', lo_row_tm1, hi_row_tm1, hi_fib_eval_result),
        'short' : _allow_entry_initial(key, 'short', lo_row_tm1, hi_row_tm1, hi_fib_eval_result)
    }
def _allow_entry_initial(
	key : str,
	long_or_short : str,  # long or short
    lo_row_tm1,
    hi_row_tm1,
    hi_fib_eval_result,
) -> bool:
	if (
          hi_row_tm1 is None
          or (not hi_row_tm1['normalized_ema_long_slope'])
          or not hi_fib_eval_result
    ):
		return False
    
	if long_or_short == "long":
		if (
                lo_row_tm1['close']<hi_row_tm1['min_long_periods']
                and hi_row_tm1['normalized_ema_long_slope']>0
                and hi_row_tm1['close']>hi_row_tm1['open']
        ):
			return True
		else:
			return False
	elif long_or_short == "short":
		if (
                lo_row_tm1['close']>hi_row_tm1['max_long_periods']
                and hi_row_tm1['normalized_ema_long_slope']<0
                and hi_row_tm1['close']<hi_row_tm1['open']
        ):
			return True
		else:
			return False
	else:
		raise ValueError("Long or Short?")

def allow_entry_final(
    key : str,
    exchange : Exchange,
    lo_row,
    hi_row_tm1,
    hi_fib_eval_result,
    reversal_camp_cache,
    fetch_historical_price_func : Callable[..., float],
    pd_reference_price_cache : pd.DataFrame,
    algo_param : Dict
) -> Dict[str, Union[bool, float, None]]:
    reference_ticker = algo_param['reference_ticker']
    timestamp_ms : int = lo_row['timestamp_ms']

    entry_price_long = lo_row['open']
    entry_price_short = lo_row['open']

    reference_price = None
    if algo_param['long_above_ref_ema_short_below']:
        reference_price = fetch_historical_price_func(
                                    exchange=exchange, 
                                    normalized_symbol=reference_ticker, 
                                    pd_reference_price_cache=pd_reference_price_cache,
                                    timestamp_ms=timestamp_ms, 
                                    ref_timeframe='1m')
        
    allow_long = True if (
                hi_row_tm1['normalized_ema_long_slope']>0
                and entry_price_long<hi_fib_eval_result['long_periods']['fib_target']
        ) else False

    allow_short = True if (
                hi_row_tm1['normalized_ema_long_slope']<0
                and entry_price_short>hi_fib_eval_result['long_periods']['fib_target']
        ) else False

    return {
            'long' : allow_long,
            'short' : allow_short,

            # In additional to allow or not, allow_entry_final also calculate a few things which you may need to mark the entry trades.
            'entry_price_long' : entry_price_long,
            'entry_price_short' : entry_price_short,
            'target_price_long' : hi_fib_eval_result['long_periods']['fib_target'],
            'target_price_short' : hi_fib_eval_result['long_periods']['fib_target'],

            'reference_price' : reference_price
        }

def allow_slice_entry(
    lo_row,
    last_candles,
    algo_param : Dict,
    pnl_percent_notional : float
):
    enable_sliced_entry = algo_param['enable_sliced_entry'] 

    allow_slice_entry_long = enable_sliced_entry and (
                                (pnl_percent_notional>0)
                                or (last_candles[0]['is_green'] and last_candles[-2]['is_green'])
                            )
    allow_slice_entry_short = enable_sliced_entry and (
                                (pnl_percent_notional>0)
                                or (not last_candles[0]['is_green'] and not last_candles[-2]['is_green'])
                            )

    return {
         'long' : allow_slice_entry_long,
         'short' : allow_slice_entry_short
    }

def sl_adj(
    lo_row,
    hi_row,
    post_move_candles,
    pos_side,
    avg_entry_price,
    this_ticker_open_trades : List[Dict],
    algo_param : Dict
):
    running_sl_percent_hard = algo_param['sl_hard_percent']

    if pos_side=='buy':
        target_price = max([ trade['target_price'] for trade in this_ticker_open_trades]) # Most aggressive target
        pnl_potential_bps = (target_price/avg_entry_price -1) *10000
    else:
        target_price = min([ trade['target_price'] for trade in this_ticker_open_trades]) # Most aggressive target
        pnl_potential_bps = (avg_entry_price/target_price -1) *10000

    running_sl_percent_hard = min(
            running_sl_percent_hard,
            (pnl_potential_bps/100) * 1.5
        )

    return {
         'running_sl_percent_hard' : running_sl_percent_hard
    }


def trailing_stop_threshold_eval(
        lo_row,
		pos_side : str,
		avg_entry_price : float,
		this_ticker_open_trades : List[Dict],
        algo_param : Dict
    ) -> Dict[str, float]:
        tp_min_percent = algo_param['tp_min_percent']
        tp_max_percent = algo_param['tp_max_percent']
		
        if pos_side=='buy':
            target_price = max([ trade['target_price'] for trade in this_ticker_open_trades]) # Most aggressive target
            pnl_potential_bps = (target_price/avg_entry_price -1) *10000
			
        else:
            target_price = min([ trade['target_price'] for trade in this_ticker_open_trades]) # Most aggressive target
            pnl_potential_bps = (avg_entry_price/target_price -1) *10000
		
        tp_max_percent = min((pnl_potential_bps/100), tp_max_percent)
        tp_min_percent = max(tp_max_percent/3, tp_min_percent)
		
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
        long_tp_indicator_name=None, # type: ignore
        short_tp_indicator_name=None # type: ignore
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

algo_results : List[Dict] = run_all_scenario(
    algo_params=algo_params,
    exchanges=exchanges,
    order_notional_adj_func=order_notional_adj, # type: ignore
    allow_entry_initial_func=allow_entry_initial,  # type: ignore
    allow_entry_final_func=allow_entry_final, # type: ignore
    allow_slice_entry_func=allow_slice_entry, # type: ignore
    sl_adj_func=sl_adj,
    trailing_stop_threshold_eval_func=trailing_stop_threshold_eval,
    pnl_eval_func=pnl_eval,
    tp_eval_func=tp_eval,

    logger=logger
)

dump_trades_to_disk(
     algo_results,
     trade_extract_filename,
     logger
     )
