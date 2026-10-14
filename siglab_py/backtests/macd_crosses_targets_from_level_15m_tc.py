'''
Command line:
    python macd_crosses_targets_from_level_15m_tc.py --exchange_name binance --white_list_tickers ETH/USDT:USDT --reference_ticker ETH/USDT:USDT --force_reload N --recompute_ta N --block_entries_on_impacting_ecoevents N --start_date 2025-01-01 --num_days 365 --some_multiples 3 --tp_min_percent -1.00 --tp_max_percent -1.00 --sl_hard_percent 1.00
    
    --tp_min_percent and --tp_max_percent: -1?          -- TP targets are ATR based. tp_min_percent = 1 ATR, tp_max_percent = 3 ATR. Instead of fixed TP/SL targets, it's mutliple of ATR. This makes it flexible to adapt strategy to other tickers with different vols.
    --some_multiples                                    -- It's mapped to target_atr_multiplier. Have a look at these lambdas: allow_entry_final, trailing_stop_threshold_eval, sl_adj   
    --sl_hard_percent: 1                                -- SL also ATR based, but 1% is a cap.
     

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
                                "--white_list_tickers", "ETH/USDT:USDT",
                                "--reference_ticker", "ETH/USDT:USDT",
                                "--start_date", "2025-01-01",
                                "--num_days", "365",
                                "--exchange_name", "binance",
                                "--some_multiples", "3",
                                "--tp_min_percent", "-1",
                                "--tp_max_percent", "-1",
                                "--sl_hard_percent", "1"
                                "--force_reload", "N",
                                "--recompute_ta", "N",
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
from typing import Dict, List, Tuple, Any, Callable, Union
import math
from pprint import pformat
import pandas as pd
import numpy as np

import ccxt
from ccxt.base.exchange import Exchange

from backtest_core import parseargs, get_logger, spawn_parameters, generic_pnl_eval, generic_tp_eval, generic_sort_filter_universe, run_all_scenario, dump_trades_to_disk
from siglab_py.util.simple_math import compute_adjacent_levels
from siglab_py.util.datetime_util import is_us_cash_open_effective
from siglab_py.util.analytic_util import evaluate_trading_context
from siglab_py.util.market_data_util import PolygonMarketDataProvider

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PYPY_COMPAT : bool = True

sys.path.append('../gizmo')

REPORT_NAME : str = "backtest_macd_crosses_strategy_15m_tc"
CACHE_CANDLES : str = f"{os.path.dirname(sys.path[0])}\\cache\\candles"

white_list_tickers : List[str] = [ "SOL/USDT:USDT" ]

force_reload : bool = False
recompute_ta : bool = False

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
commission_bps : float = 5

'''
******** STRATEGY_SPECIFIC parameters ********
'''
target_atr_multiplier : float = 3
min_adx_threshold : float = 30

slippage_cost_bps : float = 7

trading_context_how_many_candles : int = 24*14
trading_context_vp_1_how_many_candles : int = 24*7 # 7d worth of 1h bars
trading_context_vp_2_how_many_candles : int = 24*30 # one month worth of 1h bars
trading_context_vp_3_how_many_candles : int = 24*30*3 # three months worth of 1h bars
sliding_window_how_many_candles_atr : int = 24*14 # one week 1h bars
sliding_window_how_many_candles_adx : int = 24*14 # two weeks 1h bars
level_granularity = 0.01 # target_price determination involves min(swing_hi, level_above) max(swing_lo, level_below). If set to 0, you get rounded levels like $91, 92, 93, but pnl much worse!

additional_trade_fields : List[str] = [
     # Add fields you want to include in trade extract
]


'''
******** GENERIC parameters ********
'''
build_trading_context : bool = False

strategy_mode_values : List[str]= [ 'long_short'] # 'long_only', 'short_only', 'long_short'

'''
For example, Monday's are weird. Entries, SL adjustments ...etc may have STRATEGY_SPECIFIC logic around this.
'''
CAUTIOUS_DAYOFWEEK : List[int] = [ 0 ]
allow_entry_dayofweek : List[int] = [0,1,2,3,4]
how_many_last_candles : int = 2
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

tp_num_intervals_delay : int = 0
sl_num_intervals_delay_values : List[float] = [ 4 ]
sl_hard_percent_values : List[float] = [ 1.00 ]
sl_percent_trailing_values : List[float] = [ 10 ]
use_gradual_tightened_trailing_stops : bool = True
trailing_stop_mode : str = "linear" # linear or parabolic

'''
This is for trailing stops slope calc.
Say if your trade's max profit potential is tp_max_percent=3%=300bps.
tp_min_percent = 0.3 means you will NOT TP until at least pnl > 0.3% or 30bps.
'''
tp_min_percent = 1.00
tp_max_percent = 3.00
sl_adj_percent : float = 0.25

POST_MOVE_NUM_INTERVALS : int = 24*3
POST_MOVE_PERCENT_THRESHOLD : int = 3

enable_hi_timeframe_confirm : bool = True

# min_allow_entry_timestamp_ms : int = 1769448600000

start_dates : List[datetime] = [ 
    datetime(2026, 1, 1),
    # datetime(2025, 1, 1),
    # datetime(2024, 1, 1),
]

hi_how_many_candles_values : List[Tuple[str, int, int]] = [ 
    ('1h', 6, 24*238),
    # ('1h', 6, 24*365*1)
]

lo_how_many_candles_values : List[Tuple[str, int, int]] = [ 
    ('15m', 10, 4*24 *238),
    # ('15m', 10, 4*24 *365*1)
]

hi_ma_short_vs_long_interval_values : List[Tuple[int, int]] = [ (3, 6) ]
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
constant_order_notional : bool = True

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
exchange_name = args['exchange_name']
api_key = args['api_key']
force_reload = args['force_reload']
recompute_ta = args['recompute_ta']
white_list_tickers : List[str] = args['white_list_tickers']
reference_ticker : str = args['reference_ticker']
block_entries_on_impacting_ecoevents = args['block_entries_on_impacting_ecoevents']
enable_sliced_entry = args['enable_sliced_entry']
asymmetric_tp_bps : int = args['asymmetric_tp_bps']
some_multiples : float = args['some_multiples']

ccxt_exchange_map = {
    'binance': ccxt.binance,
    'bybit': ccxt.bybit,
    'okx' : ccxt.okx,
    'hyperliquid' : ccxt.hyperliquid,
    'lighter' : ccxt.lighter
}

if exchange_name in ccxt_exchange_map:
    exchange = ccxt_exchange_map[exchange_name](param)
    exchange.name=exchange_name

elif exchange_name=='polygonio':
    exchange = PolygonMarketDataProvider(api_key=api_key, rate_limit_ms=10)

exchanges = [ 
    exchange
]

start_date = args['start_date']
num_days = args['num_days']
tp_min_percent = args['tp_min_percent']
tp_max_percent = args['tp_max_percent']
sl_hard_percent = args['sl_hard_percent']
sl_hard_percent_values = [ sl_hard_percent ]
target_atr_multiplier = some_multiples

start_dates = [ start_date ]

_hi_how_many_candles_values = []
_lo_how_many_candles_values = []
for x in hi_how_many_candles_values:
    _hi_how_many_candles_values.append((x[0], x[1], 24*num_days))
hi_how_many_candles_values = _hi_how_many_candles_values
for x in lo_how_many_candles_values:
    _lo_how_many_candles_values.append((x[0], x[1], 4*24*num_days))
lo_how_many_candles_values = _lo_how_many_candles_values

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

# If you need it? Just an example here.
def stage_strat_specific_preentry_data(
        lo_row,
        pd_hi_candles,
        algo_param : Dict,
        strategy_specific_data_cache : Dict[str, Any],
    ) -> Dict[str, str]:
        trading_context_how_many_candles = algo_param['trading_context_how_many_candles'] if 'trading_context_how_many_candles' in algo_param else 24*7
        trading_context_vp_1_how_many_candles = algo_param['trading_context_vp_1_how_many_candles'] if 'trading_context_vp_1_how_many_candles' in algo_param else 24*7
        trading_context_vp_2_how_many_candles = algo_param['trading_context_vp_2_how_many_candles'] if 'trading_context_vp_2_how_many_candles' in algo_param else 24*30
        trading_context_vp_3_how_many_candles = algo_param['trading_context_vp_3_how_many_candles'] if 'trading_context_vp_3_how_many_candles' in algo_param else 24*30*3

        sliding_window_how_many_candles_atr = algo_param['sliding_window_how_many_candles_atr'] if 'sliding_window_how_many_candles_atr' in algo_param else 24*7
        sliding_window_how_many_candles_adx = algo_param['sliding_window_how_many_candles_adx'] if 'sliding_window_how_many_candles_adx' in algo_param else 24*7

        _pd_hi_candles = pd_hi_candles[
             (pd_hi_candles.timestamp_ms<lo_row['timestamp_ms'])
             & (~pd_hi_candles['dayofweek'].isin([5,6]))
        ]

        if _pd_hi_candles.shape[0]<trading_context_how_many_candles:
            return
        
        _pd_hi_candles = _pd_hi_candles.iloc[-trading_context_how_many_candles:].copy()

        if _pd_hi_candles.empty:
            return
        
        if algo_param['build_trading_context']:
            if (
                'trading_context' not in strategy_specific_data_cache
                or lo_row['hour']==1 # update daily
            ):
                trading_context = strategy_specific_data_cache['trading_context'] if 'trading_context' in strategy_specific_data_cache else None

                evaluation_timestamp_ms = trading_context['evaluation_timestamp_ms'] if trading_context else 0
                if not trading_context or (lo_row['timestamp_ms'] - evaluation_timestamp_ms) > 1000*60*60*24: # Update only if prev evaluation older than 24 hours
                    # Evaluate certain TA (ADX, atr_bps ...) using different sliding window
                    def ta_overrides(
                        pd_candles : pd.DataFrame,
                        sliding_window_how_many_candles_atr : int,
                        sliding_window_how_many_candles_adx : int
                    ):
                        pd_candles.loc[:,'h_l'] = pd_candles['high'] - pd_candles['low']
                        pd_candles.loc[:,'h_pc'] = abs(pd_candles['high'] - pd_candles['close'].shift(1))
                        pd_candles.loc[:,'l_pc'] = abs(pd_candles['low'] - pd_candles['close'].shift(1))
                        pd_candles.loc[:,'tr'] = pd_candles[['h_l', 'h_pc', 'l_pc']].max(axis=1)
                        pd_candles.loc[:,'atr'] = pd_candles['tr'].rolling(window=sliding_window_how_many_candles_atr).mean()
                        pd_candles.loc[:,'atr_bps'] = pd_candles['atr']/pd_candles['ema_close'] *10000

                        def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
                            if period <= 0:
                                raise ValueError("period must be > 0")
                            n = len(series)
                            if n < period:
                                # Not enough data; return NaNs
                                return pd.Series(np.nan, index=series.index)
                            
                            alpha = 1.0 / period
                            ewm = series.ewm(alpha=alpha, adjust=False).mean()
                            pivot = period - 1
                            init_avg = series.iloc[:period].mean()
                            correction = init_avg - ewm.iloc[pivot]
                            result = ewm.copy()
                            indices = np.arange(pivot, n)
                            decay = (1 - alpha) ** (indices - pivot)
                            result.iloc[pivot:] = ewm.iloc[pivot:] + correction * decay
                            result.iloc[:pivot] = np.nan
                            return result
                        high, low = pd_candles['high'], pd_candles['low']
                        up_move = high - high.shift(1)
                        down_move = low.shift(1) - low

                        plus_DM = pd.Series(
                            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                            index=pd_candles.index
                        )
                        minus_DM = pd.Series(
                            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                            index=pd_candles.index
                        )
                        smoothed_plus_DM = wilder_smooth(plus_DM, sliding_window_how_many_candles_adx)
                        smoothed_minus_DM = wilder_smooth(minus_DM, sliding_window_how_many_candles_adx)
                        smoothed_atr = wilder_smooth(pd_candles['tr'], sliding_window_how_many_candles_adx)
                        plus_DI = 100 * smoothed_plus_DM / smoothed_atr
                        minus_DI = 100 * smoothed_minus_DM / smoothed_atr
                        di_sum = plus_DI + minus_DI
                        di_diff = (plus_DI - minus_DI).abs()
                        dx = 100 * di_diff / di_sum
                        pd_candles['adx'] = wilder_smooth(dx, sliding_window_how_many_candles_adx)

                    trading_context = evaluate_trading_context(
                        pd_candles=_pd_hi_candles,

                        sliding_window_how_many_candles=trading_context_how_many_candles,
                        volume_profile_1_num_intervals=trading_context_vp_1_how_many_candles,
                        volume_profile_2_num_intervals=trading_context_vp_2_how_many_candles,
                        volume_profile_3_num_intervals=trading_context_vp_3_how_many_candles,
                        value_area_pct=0.7
                    )

                    trading_context['evaluation_timestamp_ms'] = lo_row['timestamp_ms']

                    ta_overrides(pd_candles=_pd_hi_candles, sliding_window_how_many_candles_atr=sliding_window_how_many_candles_atr, sliding_window_how_many_candles_adx=sliding_window_how_many_candles_adx)

                    last_row =  _pd_hi_candles.iloc[-1]
                    adx = round(last_row['adx'], 2)
                    atr_bps = round(last_row['atr_bps'], 2)

                    trading_context['adx'] = adx
                    trading_context['atr_bps'] = atr_bps

                    prettyprint = pformat(trading_context, indent=2, width=100)
                    logger.info("#trading_context")
                    logger.info(prettyprint)

                    strategy_specific_data_cache['trading_context'] = trading_context
        
        return {}

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
    lo_row_tm1,
    algo_param : Dict,
) -> Dict[str, float]:
    initial_cash : float = algo_param['initial_cash']
    entry_percent_initial_cash : float = algo_param['entry_percent_initial_cash']
    target_order_notional = initial_cash * entry_percent_initial_cash/100
    return {
         'target_order_notional' : target_order_notional
    }
     
def allow_entry_initial(
    lo_row,
    lo_row_tm1,
    hi_row_tm1,
	last_candles,
    pd_lo_candles : pd.DataFrame,
    strategy_specific_data_cache : Dict[str, Any],
	algo_param : Dict,
	this_ticker_open_trades : List
) -> Dict[str, bool]:
    return {
        'long' : _allow_entry_initial('long', lo_row, lo_row_tm1, hi_row_tm1, last_candles, pd_lo_candles, strategy_specific_data_cache, algo_param, this_ticker_open_trades),
        'short' : _allow_entry_initial('short', lo_row, lo_row_tm1, hi_row_tm1, last_candles, pd_lo_candles, strategy_specific_data_cache, algo_param, this_ticker_open_trades)
    }
def _allow_entry_initial(
	long_or_short : str,  # long or short
	lo_row, # In live trading/strategy_executor, last_candle[-1] is the current row.
    lo_row_tm1, # This corresponds to lo_row_tm1 in strategy_executor
    hi_row_tm1,
	last_candles,
    pd_lo_candles : pd.DataFrame,
    strategy_specific_data_cache : Dict[str, Any],
	algo_param : Dict,
	this_ticker_open_trades : List
) -> bool:
	allow_entry_dayofweek = algo_param['allow_entry_dayofweek']
	if lo_row['dayofweek'] not in allow_entry_dayofweek:
		return False
    
	utc_now = datetime.fromtimestamp(int(lo_row['timestamp_ms']/1000), tz=timezone.utc)
	us_cash_open_effective : bool = is_us_cash_open_effective(utc_now=utc_now)

	if long_or_short == "long":
		if (
                lo_row_tm1['macd_cross_last'] == 'bullish'
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bullish_cross_last_id']
                      and lo_row_tm1['nums_intervals_since_last_cross'] < 3
                )
                and (lo_row_tm1['macd_divergence'] not in ['normal_bearish_divergence', 'hidden_bearish_divergence'])
                and (
                    ('min_adx_threshold' not in algo_param or not algo_param['min_adx_threshold']) 
                    or (lo_row_tm1['adx']>algo_param['min_adx_threshold'])                     
                )
                and not us_cash_open_effective
        ):
			return True
		else:
			return False
          
	elif long_or_short == "short":
		if (
                lo_row_tm1['macd_cross_last'] == 'bearish'
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bearish_cross_last_id']
                      and lo_row_tm1['nums_intervals_since_last_cross'] < 3
                )
                and (lo_row_tm1['macd_divergence'] not in ['normal_bullish_divergence', 'hidden_bullish_divergence'])
                and (
                    ('min_adx_threshold' not in algo_param or not algo_param['min_adx_threshold']) 
                    or (lo_row_tm1['adx']>algo_param['min_adx_threshold'])                     
                )
                and not us_cash_open_effective
        ):
			return True
		else:
			return False

def allow_entry_final(
    exchange : Exchange,
    lo_row,
    lo_row_tm1,
    algo_param : Dict
) -> Dict[str, Union[bool, float, None]]:   
    tp_min_percent = lo_row_tm1['atr_bps']/100 *algo_param['target_atr_multiplier']
    tp_max_percent = tp_min_percent*3
    entry_price_long = lo_row['open'] * (1 + 0.5 * algo_param['slippage_cost_bps']/10000)
    entry_price_short = lo_row['open'] * (1 - 0.5 * algo_param['slippage_cost_bps']/10000)
    tp_long_target : float = entry_price_long * (1 + tp_max_percent/100)
    tp_short_target : float = entry_price_short * (1 - tp_max_percent/100)

    return {
            'long' : True,
            'short' : True,

            # In additional to allow or not, allow_entry_final also calculate a few things which you may need to mark the entry trades.
            'entry_price_long' : entry_price_long,
            'entry_price_short' : entry_price_short,
            'target_price_long' : tp_long_target,
            'target_price_short' : tp_short_target,
            'reference_price' : None
        }

allow_slice_entry = allow_entry_initial

def sl_adj(
    lo_row_tm1 : datetime,
    algo_param : Dict
):
    sl_hard_percent = algo_param['sl_hard_percent']
    running_sl_percent_hard = min(lo_row_tm1['atr_bps']/100 *algo_param['target_atr_multiplier'], sl_hard_percent)
    return {
         'running_sl_percent_hard' : running_sl_percent_hard
    }
	

def trailing_stop_threshold_eval(
        lo_row,
        lo_row_tm1,
        pos_side,
        algo_param : Dict
    ) -> Dict[str, float]:
        tp_min_percent = lo_row_tm1['atr_bps']/100 *algo_param['target_atr_multiplier']
        tp_max_percent = tp_min_percent*3
        return {
            'tp_min_percent' : tp_min_percent,
            'tp_max_percent' : tp_max_percent
        }

def pnl_eval (
        this_candle,
        lo_row_tm1,
        tp_min_percent : float,
        tp_max_percent : float,
        running_sl_percent_hard : float,
        this_ticker_open_trades : List[Dict],
        algo_param : Dict
) -> Dict[str, float]:
    return generic_pnl_eval(
        this_candle,
        lo_row_tm1,
        tp_min_percent,
        tp_max_percent,
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

    stage_strat_specific_preentry_data_func=stage_strat_specific_preentry_data,

    logger=logger
)

dump_trades_to_disk(
     algo_results,
     trade_extract_filename,
     logger
     )
