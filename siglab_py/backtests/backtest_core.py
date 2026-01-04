# type: ignore Sorry sorry
import os
import logging
import argparse
import arrow
from datetime import datetime, timedelta, timezone
import time
from typing import List, Dict, Any, Union, Callable
import uuid
import math
import json
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ccxt.base.exchange import Exchange

from siglab_py.util.retry_util import retry
from siglab_py.util.market_data_util import fetch_candles, fix_column_types, fetch_historical_price, timestamp_to_week_of_month
from siglab_py.util.trading_util import calc_eff_trailing_sl
from siglab_py.util.analytic_util import compute_candles_stats, lookup_fib_target, partition_sliding_window
from siglab_py.util.simple_math import bucket_series, bucketize_val

def get_logger(report_name : str):
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger(report_name)
    log_level = logging.INFO # DEBUG --> INFO --> WARNING --> ERROR
    logger.setLevel(log_level)
    format_str = '%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str)

    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(f"{report_name}.log", mode='w')
    fh.setLevel(log_level)
    fh.setFormatter(formatter)     
    logger.addHandler(fh)

    return logger

def spawn_parameters(
    flattened_parameters
) -> List[Dict[str, Any]]:
    algo_params : List[Dict[str, Any]] = []
    for key in flattened_parameters:
        _key = key.lower()
        if _key in [ 'exchanges' ]:
            continue

        val = flattened_parameters[key]
        if not algo_params:
            assert(_key=="pypy_compat")
            param_dict = {_key : val}
            algo_params.append(param_dict)

        else:
            cloned_algo_params = None

            for existing_algo_param in algo_params:
                if type(val) not in [list, List]:
                    existing_algo_param[_key] = val
                else:
                    if _key == 'hi_how_many_candles_values':
                        for x in val:
                            existing_algo_param['hi_stats_computed_over_how_many_candles'] = x[1]
                            existing_algo_param['hi_candle_size'] = x[0]
                            existing_algo_param['hi_how_many_candles'] = x[2]
                    elif _key == 'hi_ma_short_vs_long_interval_values':
                        for x in val:
                            existing_algo_param['hi_ma_short_interval'] = x[0]
                            existing_algo_param['hi_ma_long_interval'] = x[1]

                    elif _key == 'lo_how_many_candles_values':
                        for x in val:
                            existing_algo_param['lo_stats_computed_over_how_many_candles'] = x[1]
                            existing_algo_param['lo_candle_size'] = x[0]
                            existing_algo_param['lo_how_many_candles'] = x[2]

                    elif _key == 'lo_ma_short_vs_long_interval_values':
                        for x in val:
                            existing_algo_param['lo_ma_short_interval'] = x[0]
                            existing_algo_param['lo_ma_long_interval'] = x[1]

                    elif _key  in [ 'white_list_tickers', 'additional_trade_fields', 'cautious_dayofweek', 'allow_entry_dayofweek', 'mapped_event_codes', 'ecoevents_mapped_regions' ]:
                        existing_algo_param[_key] = val

                    else:
                        if len(val)>1:
                            cloned_algo_params = []

                        if _key not in [ 'start_dates']:
                            _key = _key.replace("_values","")
                        elif _key == 'start_dates':
                            _key = 'start_date'
                            
                        i = 0
                        for x in val:
                            
                            if i==0:
                                 existing_algo_param[_key] = x
                            else:
                                cloned_algo_param = existing_algo_param.copy()
                                cloned_algo_param[_key] = x
                                cloned_algo_params.append(cloned_algo_param)
                            i+=1
                        
                        if cloned_algo_params:
                            algo_params = algo_params + cloned_algo_params
                            cloned_algo_params.clear()
                            cloned_algo_params = None

    param_id : int = 0
    for algo_param in algo_params:
        start_date = algo_param.pop('start_date')
        name_exclude_start_date = ""
        for key in algo_param:
            name_exclude_start_date += f"{key}: {algo_param[key]}|"
        name = "start_date: {start_date}|" + name_exclude_start_date
        algo_param['param_id'] = param_id
        algo_param['start_date'] = start_date
        algo_param['name'] = name
        algo_param['name_exclude_start_date'] = name_exclude_start_date

        # Purpose is to avoid snowball effect in equity curves in long dated back tests.
        if 'constant_order_notional' not in algo_param:
            algo_param['constant_order_notional'] = True
        algo_param['target_order_notional'] = None
        if algo_param['constant_order_notional']:
            algo_param['target_order_notional'] = algo_param['initial_cash'] * algo_param['entry_percent_initial_cash']/100
        
        param_id+=1
    return algo_params

def create_plot_canvas(key : str, pd_hi_candles : pd.DataFrame, pd_lo_candles : pd.DataFrame):
        SMALL_SIZE = 7
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

        plt.rc('figure', figsize=(25, 25))
        plt.ion()
        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=SMALL_SIZE)
        plt.rc('axes', labelsize=SMALL_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=SMALL_SIZE)

        fig, axes = plt.subplots(5, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})

        date_numbers_hi = mdates.date2num(pd_hi_candles['datetime'])
        date_numbers_lo = mdates.date2num(pd_lo_candles['datetime'])
        major_locator = mdates.AutoDateLocator(minticks=3, maxticks=7)

        for ax in axes:
            ax.set_xticklabels([])
            ax.minorticks_on()
            ax.grid()
            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            ax.tick_params(axis="x", which='major', labelbottom=True, rotation=45)

        time_series_canvas = axes[0]
        time_series_canvas.minorticks_on()
        time_series_canvas.grid()
        time_series_canvas.set_ylabel(f'Close px and boillenger band {key}')
        time_series_canvas.tick_params(axis="x", which='major', labelbottom=True, rotation=45)
        time_series_canvas.plot(date_numbers_lo, pd_lo_candles['close'], color='darkblue', linewidth=2, label=f"close")
        time_series_canvas.plot(date_numbers_hi, pd_hi_candles['boillenger_upper'], color='lightblue', linestyle='--', linewidth=0.5, label=f"upper boillenger (hi)")
        time_series_canvas.plot(date_numbers_hi, pd_hi_candles['boillenger_lower'], color='lightblue', linestyle='--', linewidth=0.5, label=f"lower boillenger (hi)")
        time_series_canvas.plot(date_numbers_lo, pd_lo_candles['boillenger_upper'], color='gray', linestyle='-', linewidth=1, label=f"upper boillenger (lo)")
        time_series_canvas.plot(date_numbers_lo, pd_lo_candles['boillenger_lower'], color='gray', linestyle='-', linewidth=1, label=f"lower boillenger (lo)")
        time_series_canvas.legend()

        boillenger_channel_height_canvas = axes[1]
        boillenger_channel_height_canvas.minorticks_on()
        boillenger_channel_height_canvas.grid()
        boillenger_channel_height_canvas.set_ylabel(f'boillenger channel height vs ATR band {key}')
        boillenger_channel_height_canvas.tick_params(axis="x", which='major', labelbottom=True, rotation=45)
        boillenger_channel_height_canvas.plot(date_numbers_hi, pd_hi_candles['boillenger_channel_height'], color='lightblue', linewidth=0.5, label=f"boillenger channel height (hi)")
        boillenger_channel_height_canvas.plot(date_numbers_hi, pd_hi_candles['atr'], color='lightblue', linestyle='dashed', linewidth=0.5, label=f"ATR (hi)")
        boillenger_channel_height_canvas.plot(date_numbers_lo, pd_lo_candles['boillenger_channel_height'], color='gray', linewidth=0.5, label=f"boillenger channel height (lo)")
        boillenger_channel_height_canvas.plot(date_numbers_lo, pd_lo_candles['atr'], color='gray', linestyle='dashed', linewidth=0.5, label=f"ATR (lo)")
        boillenger_channel_height_canvas.legend()

        rsi_canvas = axes[2]
        rsi_canvas.minorticks_on()
        rsi_canvas.grid()
        rsi_canvas.set_ylabel(f'RSI {key}')
        rsi_canvas.tick_params(axis="x", which='major', labelbottom=True, rotation=45)
        rsi_canvas.plot(date_numbers_hi, pd_hi_candles['rsi'], color='lightblue', linewidth=2, label=f"RSI (hi)")
        rsi_canvas.plot(date_numbers_lo, pd_lo_candles['rsi'], color='gray', linestyle='dashed', linewidth=2, label=f"RSI (lo)")

        macd_canvas_hi = axes[3]
        macd_canvas_hi.minorticks_on()
        macd_canvas_hi.grid()
        macd_canvas_hi.set_ylabel(f'MACD hi {key}')
        macd_canvas_hi.tick_params(axis="x", which='major', labelbottom=True, rotation=45)
        macd_canvas_hi.plot(date_numbers_hi, pd_hi_candles['macd'], color='lightblue', linewidth=0.5, label=f"MACD (hi)")
        macd_canvas_hi.plot(date_numbers_hi, pd_hi_candles['signal'], color='lightblue', linewidth=0.5, label=f"signal (hi)")
        bar_colors = ['red' if value < 0 else 'green' for value in pd_hi_candles['macd_minus_signal']]
        macd_canvas_hi.bar(date_numbers_hi, pd_hi_candles['macd_minus_signal'], width=0.005, color=bar_colors, label="MACD Histogram (hi)")

        macd_canvas_lo = axes[4]
        macd_canvas_lo.minorticks_on()
        macd_canvas_lo.grid()
        macd_canvas_lo.set_ylabel(f'MACD lo {key}')
        macd_canvas_lo.tick_params(axis="x", which='major', labelbottom=True, rotation=45)
        macd_canvas_lo.plot(date_numbers_lo, pd_lo_candles['macd'], color='gray', linewidth=0.5, label=f"MACD (lo)")
        macd_canvas_lo.plot(date_numbers_lo, pd_lo_candles['signal'], color='gray', linewidth=0.5, label=f"signal (lo)")
        bar_colors_lo = ['red' if value < 0 else 'green' for value in pd_lo_candles['macd_minus_signal']]
        macd_canvas_lo.bar(date_numbers_lo, pd_lo_candles['macd_minus_signal'], width=0.005, color=bar_colors_lo, label="MACD Histogram (lo)")

        return {
            'plt' : plt,
            'time_series_canvas' : time_series_canvas
        }

def plot_segments(
        pd_candles : pd.DataFrame,
        ts_partitions : Dict,
        jpg_filename : str = None
        ):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    minima = ts_partitions['minima']
    maxima = ts_partitions['maxima']
    segments = ts_partitions['segments']

    fig = plt.figure(figsize=(15, 8), facecolor='black')
    gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 

    # Price Chart
    ax0 = plt.subplot(gs[0])
    ax0.plot(pd_candles['datetime'], pd_candles['close'], label='Close', color='dodgerblue')
    ax0.plot(pd_candles['datetime'], pd_candles['smoothed_close'], label='Smoothed Close', color='yellow')
    ax0.plot(pd_candles['datetime'], pd_candles['ema_close'], label='3m EMA', linestyle='--', color='orange')
    ax0.fill_between(pd_candles['datetime'], pd_candles['close'], pd_candles['ema_close'], where=(pd_candles['close'] > pd_candles['ema_close']), interpolate=True, color='dodgerblue', alpha=0.3, label='Bull Market')
    ax0.fill_between(pd_candles['datetime'], pd_candles['close'], pd_candles['ema_close'], where=(pd_candles['close'] <= pd_candles['ema_close']), interpolate=True, color='red', alpha=0.3, label='Bear Market')

    ax0.set_title('Close vs EMA', color='white')
    ax0.set_xlabel('Date', color='white')
    ax0.set_ylabel('Price', color='white')
    legend = ax0.legend()
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')

    # @CRITICAL close vs smoothed_close and merge_distance
    for maxima_index in maxima:
        ax0.plot(pd_candles['datetime'][maxima_index], pd_candles['close'][maxima_index], marker='+', markersize=8, color='yellow', label='maxima')
    for minima_index in minima:
        ax0.plot(pd_candles['datetime'][minima_index], pd_candles['close'][minima_index], marker='o', markersize=5, color='yellow', label='minima')

    for segment in segments:
        
        ax0.axvline(x=pd_candles['datetime'][segment['end']], color='gray', linewidth=2, linestyle='--')

        if 'maxima_idx_boillenger' in segment and segment['maxima_linregress_boillenger'] is not None:
            '''
            We don't need to compute y_series like this:
                slope_maxima = segment['maxima_linregress_boillenger'].slope
                intercept_maxima = segment['maxima_linregress_boillenger'].intercept
                segment_maxima_dates = pd_candles['datetime'][segment['maxima_idx_boillenger']]
                y_series = [ slope_maxima * idx + intercept_maxima for idx in segment['maxima_idx_boillenger'] ]
            But, syntax is just for reference.
            '''
            x_series = [pd_candles.loc[idx]['datetime'] for idx in segment['maxima_idx_boillenger'] if idx in pd_candles.index] # x = dates
            y_series = [segment['maxima_close_boillenger'][i] for i, idx in enumerate(segment['maxima_idx_boillenger']) if idx in pd_candles.index] # y = boillenger upper
            ax0.plot(
                    x_series,
                    y_series,
                color='green', linestyle='--', label='Maxima Linear Regression')
        
        if 'minima_idx_boillenger' in segment and segment['minima_linregress_boillenger'] is not None:
            x_series = [pd_candles.loc[idx]['datetime'] for idx in segment['minima_idx_boillenger'] if idx in pd_candles.index] # x = dates
            y_series = [segment['minima_close_boillenger'][i] for i, idx in enumerate(segment['minima_idx_boillenger']) if idx in pd_candles.index] # y = boillenger lower
            ax0.plot(
                    x_series,
                    y_series,
                color='red', linestyle='--', label='Minima Linear Regression')

    ax0.set_facecolor('black')
    
    ax0.tick_params(axis='x', colors='white')
    ax0.tick_params(axis='y', colors='white')
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()

    if jpg_filename:
        plt.savefig(jpg_filename, format='jpg', dpi=300)

def segments_to_df(segments : List[Dict]) -> pd.DataFrame:
    segments = [ 
                            {
                                'start' : segment['start'],
                                'end' : segment['end'],
                                'start_datetime' : segment['start_datetime'] if not type(segment['start_datetime']) is str else arrow.get(segment['start_datetime']).datetime.replace(tzinfo=None),
                                'end_datetime' : segment['end_datetime'] if not type(segment['end_datetime']) is str else arrow.get(segment['end_datetime']).datetime.replace(tzinfo=None),
                                'start_close' : segment['start_close'],
                                'end_close' : segment['end_close'],
                                'window_size_num_intervals' : segment['window_size_num_intervals'],
                                'cur_recur_depth' : segment['cur_recur_depth'],
                                'up_or_down' : segment['up_or_down'],
                                'class' : segment['class'],
                                'maxima_linregress_slope' : segment['maxima_linregress_full'].slope,
                                'maxima_linregress_intercept' : segment['maxima_linregress_full'].intercept,
                                'maxima_linregress_std_err' : segment['maxima_linregress_full'].stderr,
                                'minima_linregress_slope' : segment['minima_linregress_full'].slope,
                                'minima_linregress_intercept' : segment['minima_linregress_full'].intercept,
                                'minima_linregress_std_err' : segment['minima_linregress_full'].stderr

                            }
                            for segment in segments ] 
    for segment in segments:
        segment['start_ts'] = int(segment['start_datetime'].timestamp())
        segment['end_ts'] = int(segment['end_datetime'].timestamp())
    pd_segments = pd.DataFrame(segments)
    return pd_segments

def generic_check_signal_thresholds(
    signal_thresholds : List[Dict[str, Any]],
    this_candle : Dict[str, Any],
    adj_bps : float = 0
) -> bool:
    '''
    WARNING!!! Do not put any strategy specific logic here!!!
    Thanks.
    '''
    return all([
        this_candle[signal['lhs']] > (this_candle[signal['rhs']] + adj_bps/10000)
        if signal['op'] == '>' 
        else this_candle[signal['lhs']] < (this_candle[signal['rhs']] + adj_bps/10000)
        for signal in signal_thresholds
    ])

def generic_pnl_eval (
            this_candle,
            running_sl_percent_hard : float,
            this_ticker_open_trades : List[Dict],
            algo_param : Dict,
            long_tp_indicator_name : str = None,
            short_tp_indicator_name : str = None
    ) -> Dict[str, float]:
        '''
        WARNING!!! Do not put any strategy specific logic here!!!
        Thanks.
        '''
        unrealized_pnl_interval, unrealized_pnl_open, unrealized_pnl_live_optimistic, unrealized_pnl_live_pessimistic, unrealized_pnl_tp, unrealized_pnl_sl, unrealized_pnl_close_approx = 0, 0, 0, 0, 0, 0, 0
        assert(len(set([ trade['side'] for trade in this_ticker_open_trades]))==1) # open trades should be in same direction
        this_ticker_open_positions_side = this_ticker_open_trades[-1]['side']

        lo_dayofweek = this_candle['dayofweek']
        cautious_dayofweek : List[int] = algo_param['cautious_dayofweek']

        lo_close = this_candle['close']
        lo_open = this_candle['open']
        lo_high = this_candle['high']
        lo_low = this_candle['low']

        # ATR, Fib618, bollengers are price levels. RSI/MFI..etc are not prices. Be careful.
        long_tp_price = this_candle[long_tp_indicator_name] if long_tp_indicator_name else None
        short_tp_price = this_candle[short_tp_indicator_name] if short_tp_indicator_name else None

        _asymmetric_tp_bps = algo_param['asymmetric_tp_bps'] if lo_dayofweek in cautious_dayofweek else 0

        for trade in this_ticker_open_trades:
            target_price = trade['target_price'] if 'target_price' in trade else None
            if not long_tp_indicator_name and not short_tp_indicator_name:
                assert(target_price)

            if this_ticker_open_positions_side=='buy':
                unrealized_pnl_interval += (lo_close - trade['entry_price']) * trade['size']
                unrealized_pnl_open += (lo_open - trade['entry_price']) * trade['size']
                unrealized_pnl_live_optimistic += (lo_high - trade['entry_price']) * trade['size']
                unrealized_pnl_live_pessimistic += (lo_low - trade['entry_price']) * trade['size']
                unrealized_pnl_close_approx += (min(lo_close*(1+_asymmetric_tp_bps/10000), lo_high) - trade['entry_price']) * trade['size'] # Less accurate to use close price
                if (
                    long_tp_indicator_name 
                    and not target_price # If entry trades are tagged target_price, it should take precedence over indicator
                ):
                    unrealized_pnl_tp += (min(long_tp_price*(1+_asymmetric_tp_bps/10000), lo_high) - trade['entry_price']) * trade['size']
                else:
                    if target_price:
                        if (lo_high>target_price and lo_low<target_price):
                            unrealized_pnl_tp += (target_price - trade['entry_price']) * trade['size']
                        else:
                            unrealized_pnl_tp += unrealized_pnl_close_approx # This is worst, try not to estimate pnl with close price!
                    else:
                        unrealized_pnl_tp += unrealized_pnl_close_approx # This is worst, try not to estimate pnl with close price!
                unrealized_pnl_sl += -1 * (trade['entry_price'] * trade['size'] * (running_sl_percent_hard/100)) 

            else:
                unrealized_pnl_interval += (trade['entry_price'] - lo_close) * trade['size']
                unrealized_pnl_open += (trade['entry_price'] - lo_open) * trade['size']
                unrealized_pnl_live_optimistic += (trade['entry_price'] - lo_low) * trade['size']
                unrealized_pnl_live_pessimistic += (trade['entry_price'] - lo_high) * trade['size']
                unrealized_pnl_close_approx += (trade['entry_price'] - max(lo_close*(1-_asymmetric_tp_bps/10000), lo_low)) * trade['size']
                if (
                    short_tp_indicator_name 
                    and not target_price # If entry trades are tagged target_price, it should take precedence over indicator
                ):
                    unrealized_pnl_tp += (trade['entry_price'] - max(short_tp_price*(1-_asymmetric_tp_bps/10000), lo_low)) * trade['size']
                else:
                    if target_price:
                        if (lo_high>target_price and lo_low<target_price):
                            unrealized_pnl_tp += (trade['entry_price'] - target_price) * trade['size']
                        else:
                            unrealized_pnl_tp += unrealized_pnl_close_approx # This is worst, try not to estimate pnl with close price!
                    else:
                        unrealized_pnl_tp += unrealized_pnl_close_approx # This is worst, try not to estimate pnl with close price!

                unrealized_pnl_sl += -1 * (trade['entry_price'] * trade['size'] * (running_sl_percent_hard/100))

        return {
            'unrealized_pnl_interval' : unrealized_pnl_interval,
            'unrealized_pnl_open' : unrealized_pnl_open,
            'unrealized_pnl_live_optimistic' : unrealized_pnl_live_optimistic,
            'unrealized_pnl_live_pessimistic' : unrealized_pnl_live_pessimistic,
            'unrealized_pnl_tp' : unrealized_pnl_tp,
            'unrealized_pnl_sl' : unrealized_pnl_sl
        }

def generic_tp_eval (
        lo_row,
        this_ticker_open_trades : List[Dict]
) -> bool:
    low : float = lo_row['low']
    high : float = lo_row['high']
    
    for trade in this_ticker_open_trades:
        if trade['target_price']<=high and trade['target_price']>=low:
             return True
    return False

def generic_sort_filter_universe(
    tickers : List[str],
    exchange : Exchange,

    # Use "i" (row index) to find current/last interval's market data or TAs from "all_exchange_candles"
    i,
    all_exchange_candles : Dict[str, Dict[str, Dict[str, pd.DataFrame]]],

    max_num_tickers : int = 10
) -> List[str]:
    if not tickers:
        return None
    
    sorted_filtered_tickers : List[str] = tickers.copy()

    # Custom strategy specific sort logic here. Sort first before you filter!
    sorted_filtered_tickers.sort()

    # Custom filtering logic
    if len(sorted_filtered_tickers)>max_num_tickers:
         sorted_filtered_tickers = sorted_filtered_tickers[:max_num_tickers]

    return sorted_filtered_tickers

@retry(num_attempts=3)
def fetch_price(
            exchange,
            normalized_symbol : str,
            pd_reference_price_cache : pd.DataFrame,
            timestamp_ms : int,
            ref_timeframe : str = '1m'
    ) -> float:
    cached_row = pd_reference_price_cache[pd_reference_price_cache.timestamp_ms==timestamp_ms]
    if cached_row.shape[0]>0:
        reference_price = cached_row.iloc[-1]['price']
    else:
        reference_price = fetch_historical_price(
                                        exchange=exchange, 
                                        normalized_symbol=normalized_symbol, 
                                        timestamp_ms=timestamp_ms, 
                                        ref_timeframe=ref_timeframe)
        cached_row = {
            'exchange' : exchange,
            'ticker' : normalized_symbol,
            'datetime' : datetime.fromtimestamp(int(timestamp_ms/1000)),
            'datetime_utc' : datetime.fromtimestamp(int(timestamp_ms/1000), tz=timezone.utc),
            'timestamp_ms' : timestamp_ms,
            'price' : reference_price
        }
        # pd_reference_price_cache = pd.concat([pd_reference_price_cache, pd.DataFrame([cached_row])], axis=0, ignore_index=True)
        pd_reference_price_cache.loc[len(pd_reference_price_cache)] = cached_row
    return reference_price

def fetch_cycle_ath_atl(
        exchange, 
        symbol, 
        timeframe, 
        start_date : datetime, 
        end_date : datetime
    ):
    ath = float('-inf')
    atl = float('inf')
    all_ohlcv = []

    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    while start_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_ts, limit=100)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            start_ts = ohlcv[-1][0] + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"fetch_cycle_ath_atl Oops: {e}")
    
    for candle in all_ohlcv:
        high = candle[2]
        low = candle[3]
        ath = max(ath, high)
        atl = min(atl, low)
    
    return {
        'ath' : ath,
        'atl' : atl
    }

'''
******** THE_LOOP ********

This is the loop which replay candles to back tests. No STRATEGY_SPECIFIC logic should be here!!!
'''
def run_scenario(
        algo_param : Dict,
        exchanges : List[Exchange],
        all_exchange_candles : Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        pd_ref_candles_fast : pd.DataFrame,
        pd_ref_candles_slow : pd.DataFrame,
        tickers : List[str],
        ref_candles_partitions : Dict,
        pd_hi_candles_partitions : pd.DataFrame,
        pd_lo_candles_partitions : pd.DataFrame,
        economic_calendars_loaded : bool,
        pd_economic_calendars : pd.DataFrame,

        order_notional_adj_func : Callable[..., float],
        allow_entry_initial_func : Callable[..., bool],
        allow_entry_final_func : Callable[..., bool],
        allow_slice_entry_func : Callable[..., bool],
        sl_adj_func : Callable[..., Dict[str, float]],
        trailing_stop_threshold_eval_func : Callable[..., Dict[str, float]],
        pnl_eval_func : Callable[..., Dict[str, float]],
        tp_eval_func : Callable[..., bool],
        sort_filter_universe_func : Callable[..., List[str]],

        logger,

        pypy_compat : bool = False,
        plot_timeseries : bool = True,
    ):
    exceptions : Dict = {}

    if not pypy_compat:
        pd_ref_candles_segments = segments_to_df(ref_candles_partitions['segments'])
        pd_hi_candles_segments = segments_to_df(pd_hi_candles_partitions['segments'])
        pd_lo_candles_segments = segments_to_df(pd_lo_candles_partitions['segments'])

    min_sl_age_ms : int = 0
    if algo_param['lo_candle_size'][-1]=="m":
        one_interval_ms = 60*1000
        min_sl_age_ms = algo_param['sl_num_intervals_delay'] * one_interval_ms
        num_intervals_block_pending_ecoevents_ms = one_interval_ms*algo_param['num_intervals_block_pending_ecoevents']
    elif algo_param['lo_candle_size'][-1]=="h":
        one_interval_ms = 60*60*1000
        min_sl_age_ms = algo_param['sl_num_intervals_delay'] * one_interval_ms
        num_intervals_block_pending_ecoevents_ms = one_interval_ms*algo_param['num_intervals_block_pending_ecoevents']
    elif algo_param['lo_candle_size'][-1]=="d":
        one_interval_ms = 60*60*24*1000
        min_sl_age_ms = algo_param['sl_num_intervals_delay'] * one_interval_ms
        num_intervals_block_pending_ecoevents_ms = one_interval_ms*algo_param['num_intervals_block_pending_ecoevents']

    commission_bps = algo_param['commission_bps']

    initial_cash : float = algo_param['initial_cash']
    entry_percent_initial_cash : float = algo_param['entry_percent_initial_cash']
    target_position_size_percent_total_equity : float = algo_param['target_position_size_percent_total_equity']

    class GlobalState:
        def __init__(self, initial_cash) -> None:
            self.cash = initial_cash
            self.total_equity = self.cash
            self.total_commission = 0

    gloabl_state = GlobalState(initial_cash=initial_cash) # This cash position is shared across all tickers in universe
    current_position_usdt = 0
    
    all_trades : List = []

    compiled_candles_by_exchange_pairs : List[Dict[str, pd.DataFrame]]= {}
    hi_num_intervals, lo_num_intervals  = 99999999, 99999999

    for exchange in exchanges:
        for ticker in tickers:
            key : str = f"{exchange.name}-{ticker}"
            pd_hi_candles : pd.DataFrame  = all_exchange_candles[exchange.name][ticker]['hi_candles']
            pd_lo_candles : pd.DataFrame = all_exchange_candles[exchange.name][ticker]['lo_candles']

            # market_data_gizmo sometimes insert dummy row(s) between start_date and actual first candle fetched
            if pd_hi_candles[~pd_hi_candles.close.notna()].shape[0]>0:
                pd_hi_candles.drop(pd_hi_candles[~pd_hi_candles.close.notna()].index[0], inplace=True)

            hi_num_intervals = min(hi_num_intervals, pd_hi_candles.shape[0])
            lo_num_intervals = min(lo_num_intervals, pd_lo_candles.shape[0])
            
            compiled_candles_by_exchange_pairs[key] = {}
            compiled_candles_by_exchange_pairs[key]['hi_candles'] = pd_hi_candles
            compiled_candles_by_exchange_pairs[key]['lo_candles'] = pd_lo_candles

    all_canvas = {}
    if plot_timeseries:
        for exchange in exchanges:
            for ticker in tickers:
                key = f"{exchange.name}-{ticker}"
                pd_hi_candles = compiled_candles_by_exchange_pairs[key]['hi_candles']
                pd_lo_candles = compiled_candles_by_exchange_pairs[key]['lo_candles']

                canvas = create_plot_canvas(key, pd_hi_candles, pd_lo_candles)
                all_canvas[f"{key}-param_id{algo_param['param_id']}"] = canvas

    order_notional_adj_func_sig = inspect.signature(order_notional_adj_func)
    order_notional_adj_func_params = order_notional_adj_func_sig.parameters.keys()
    allow_entry_initial_func_sig = inspect.signature(allow_entry_initial_func)
    allow_entry_initial_func_params = allow_entry_initial_func_sig.parameters.keys()
    allow_entry_final_func_sig = inspect.signature(allow_entry_final_func)
    allow_entry_final_func_params = allow_entry_final_func_sig.parameters.keys()
    allow_slice_entry_func_sig = inspect.signature(allow_slice_entry_func)
    allow_slice_entry_func_params = allow_slice_entry_func_sig.parameters.keys()
    sl_adj_func_sig = inspect.signature(sl_adj_func)
    sl_adj_func_params = sl_adj_func_sig.parameters.keys()
    trailing_stop_threshold_eval_func_sig = inspect.signature(trailing_stop_threshold_eval_func)
    trailing_stop_threshold_eval_func_params = trailing_stop_threshold_eval_func_sig.parameters.keys()
    tp_eval_func_sig = inspect.signature(tp_eval_func)
    tp_eval_func_params = tp_eval_func_sig.parameters.keys()
    sort_filter_universe_func_sig = inspect.signature(sort_filter_universe_func)
    sort_filter_universe_func_params = sort_filter_universe_func_sig.parameters.keys()
    
    BUCKETS_m100_100 = bucket_series(
						values=list([i for i in range(-100,100)]), 
						outlier_threshold_percent=10, 
						level_granularity=algo_param['default_level_granularity'] if 'default_level_granularity' in algo_param else 0.01
					)
    
    REFERENCE_PRICE_CACHE_COLUMNS = [ 
            'exchange', 'ticker', 'datetime', 'datetime_utc', 'timestamp_ms', 'price'
    ]
    reference_price_cache = {}

    def _max_camp(
            camp1 : bool,
            camp2 : bool,
            camp3 : bool
    ) -> int:
        camp : int = 1 if camp1 else 0
        if camp2:
            camp = 2
        if camp3:
            camp =3
        return camp
    REVERSAL_CAMP_ITEM = {
        'camp1' : False,
        'camp2' : False,
        'camp3' : False,
        'camp1_price' : None,
        'camp2_price' : None,
        'camp3_price' : None,

        'datetime' : None # Last update
    }
    reversal_camp_cache = {}
    lo_boillenger_lower_breached_cache = {}
    lo_boillenger_upper_breached_cache = {}
    ath, atl = None, None
    target_order_notional = 0
    for i in range(algo_param['how_many_last_candles'], lo_num_intervals):
        for exchange in exchanges:
            
            kwargs = {k: v for k, v in locals().items() if k in sort_filter_universe_func_params}
            sorted_filtered_tickers = sort_filter_universe_func(**kwargs)

            for ticker in sorted_filtered_tickers:
                key = f"{exchange.name}-{ticker}"
                if key not in reversal_camp_cache:
                    reversal_camp_cache[key] = REVERSAL_CAMP_ITEM.copy()

                pd_reference_price_cache : pd.DataFrame = None
                reference_price_cache_file : str = f"refpx_{ticker.replace('/','').replace(':','')}.csv"
                if reference_price_cache_file not in reference_price_cache:
                    if os.path.isfile(reference_price_cache_file):
                        pd_reference_price_cache = pd.read_csv(reference_price_cache_file)
                        pd_reference_price_cache.drop(pd_reference_price_cache.columns[pd_reference_price_cache.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
                        reference_price_cache[reference_price_cache_file] = pd_reference_price_cache
                else:
                    pd_reference_price_cache = reference_price_cache[reference_price_cache_file]
                if reference_price_cache_file not in reference_price_cache:
                    pd_reference_price_cache = pd.DataFrame(columns=REFERENCE_PRICE_CACHE_COLUMNS)
                reference_price_cache[reference_price_cache_file] = pd_reference_price_cache

                pd_candles = compiled_candles_by_exchange_pairs[key]
                pd_hi_candles = pd_candles['hi_candles']
                pd_lo_candles = pd_candles['lo_candles']
                
                lo_row = pd_lo_candles.iloc[i]
                lo_row_tm1 = pd_lo_candles.iloc[i-1]

                lo_datetime = lo_row['datetime']
                tm1 = lo_row_tm1['datetime']
                
                lo_year = lo_row['year']
                lo_month = lo_row['month']
                lo_day = lo_row['day']
                lo_hour = lo_row['hour']
                lo_minute = lo_row['minute']
                lo_timestamp_ms = lo_row['timestamp_ms']
                lo_dayofweek = lo_row['dayofweek']
                lo_open = lo_row['open']
                lo_high = lo_row['high']
                lo_low = lo_row['low']
                lo_mid = (lo_high + lo_low)/2
                lo_close = lo_row['close']
                lo_candle_open_close = lo_open - lo_close
                lo_candle_hi_lo = lo_high - lo_low
                lo_volume = lo_row['volume']
                lo_atr = lo_row['atr']
                lo_rsi = lo_row['rsi']
                lo_rsi_bucket = lo_row['rsi_bucket']
                lo_rsi_trend = lo_row['rsi_trend']
                lo_mfi = lo_row['mfi']
                lo_mfi_bucket = lo_row['mfi_bucket']
                lo_macd_minus_signal = lo_row['macd_minus_signal']
                lo_boillenger_upper = lo_row['boillenger_upper']
                lo_boillenger_lower = lo_row['boillenger_lower']
                lo_boillenger_mid = (lo_boillenger_upper + lo_boillenger_lower) / 2
                lo_boillenger_height = lo_boillenger_upper - lo_boillenger_lower 
                lo_boillenger_channel_height = lo_row['boillenger_channel_height']
                lo_aggressive_up = lo_row['aggressive_up']
                lo_aggressive_down = lo_row['aggressive_down']
                lo_fvg_high = lo_row['fvg_high']
                lo_fvg_low = lo_row['fvg_low']
                lo_hurst_exp = lo_row['hurst_exp']
                lo_ema_volume_short_periods = lo_row['ema_volume_short_periods']
                lo_ema_short_slope = lo_row['ema_short_slope'] if 'ema_short_slope' in pd_lo_candles.columns else 0
                lo_normalized_ema_short_slope = lo_row['normalized_ema_short_slope'] if 'normalized_ema_short_slope' in pd_lo_candles.columns else 0
                lo_ema_long_slope = lo_row['ema_long_slope'] if 'ema_long_slope' in pd_lo_candles.columns else 0
                lo_normalized_ema_long_slope = lo_row['normalized_ema_long_slope'] if 'normalized_ema_long_slope' in pd_lo_candles.columns else 0
                lo_tm1_normalized_ema_long_slope = lo_row_tm1['normalized_ema_long_slope'] if 'normalized_ema_long_slope' in pd_lo_candles.columns else 0

                lo_tm1_close = lo_row_tm1['close']
                lo_tm1_rsi = lo_row_tm1['rsi']
                lo_tm1_rsi_bucket = lo_row_tm1['rsi_bucket']
                lo_tm1_rsi_trend = lo_row_tm1['rsi_trend']

                lo_max_short_periods = lo_row['max_short_periods']
                lo_idmax_short_periods = int(lo_row['idmax_short_periods']) if not math.isnan(lo_row['idmax_short_periods']) else None
                lo_idmax_dt_short_periods = pd_lo_candles.at[lo_idmax_short_periods, 'datetime'] if not (lo_idmax_short_periods is None or pd.isna(lo_idmax_short_periods)) else None
                lo_max_long_periods = lo_row['max_long_periods']
                lo_idmax_long_periods = int(lo_row['idmax_long_periods']) if not math.isnan(lo_row['idmax_long_periods']) else None
                lo_idmax_dt_long_periods = pd_lo_candles.at[lo_idmax_long_periods, 'datetime'] if not (lo_idmax_long_periods is None or pd.isna(lo_idmax_long_periods)) else None

                lo_tm1_max_short_periods = lo_row_tm1['max_short_periods']
                lo_tm1_idmax_short_periods = int(lo_row_tm1['idmax_short_periods']) if not math.isnan(lo_row_tm1['idmax_short_periods']) else None
                lo_tm1_idmax_dt_short_periods = pd_lo_candles.at[lo_tm1_idmax_short_periods, 'datetime'] if not (lo_tm1_idmax_short_periods is None or pd.isna(lo_tm1_idmax_short_periods)) else None
                lo_tm1_max_long_periods = lo_row_tm1['max_long_periods']
                lo_tm1_idmax_long_periods = int(lo_row_tm1['idmax_long_periods']) if not math.isnan(lo_row_tm1['idmax_long_periods']) else None
                lo_tm1_idmax_dt_long_periods = pd_lo_candles.at[lo_tm1_idmax_long_periods, 'datetime'] if not (lo_tm1_idmax_long_periods is None or pd.isna(lo_tm1_idmax_long_periods)) else None

                lo_min_short_periods = lo_row['min_short_periods']
                lo_idmin_short_periods = int(lo_row['idmin_short_periods']) if not math.isnan(lo_row['idmin_short_periods']) else None
                lo_idmin_dt_short_periods = pd_lo_candles.at[lo_idmin_short_periods,'datetime'] if not (lo_idmin_short_periods is None or pd.isna(lo_idmin_short_periods)) else None
                lo_min_long_periods = lo_row['min_long_periods']
                lo_idmin_long_periods = int(lo_row['idmin_long_periods']) if not math.isnan(lo_row['idmin_long_periods']) else None
                lo_idmin_dt_long_periods = pd_lo_candles.at[lo_idmin_long_periods,'datetime'] if not (lo_idmin_long_periods is None or pd.isna(lo_idmin_long_periods)) else None
                
                lo_tm1_min_short_periods = lo_row_tm1['min_short_periods']
                lo_tm1_idmin_short_periods = int(lo_row_tm1['idmin_short_periods']) if not math.isnan(lo_row_tm1['idmin_short_periods']) else None
                lo_tm1_idmin_dt_short_periods = pd_lo_candles.at[lo_tm1_idmin_short_periods,'datetime'] if not (lo_tm1_idmin_short_periods is None or pd.isna(lo_tm1_idmin_short_periods)) else None
                lo_tm1_min_long_periods = lo_row_tm1['min_long_periods']
                lo_tm1_idmin_long_periods = int(lo_row_tm1['idmin_long_periods']) if not math.isnan(lo_row_tm1['idmin_long_periods']) else None
                lo_tm1_idmin_dt_long_periods = pd_lo_candles.at[lo_tm1_idmin_long_periods,'datetime'] if not (lo_tm1_idmin_long_periods is None or pd.isna(lo_tm1_idmin_long_periods)) else None

                if not ath or not atl:
                    ath_atl = fetch_cycle_ath_atl(exchange=exchange, symbol=ticker, timeframe='1d', start_date=(algo_param['start_date'] - timedelta(days=365*4)), end_date=algo_param['start_date'])
                    ath = ath_atl['ath']
                    atl = ath_atl['atl']

                if lo_close>ath:
                    ath = lo_close
                if lo_close<atl:
                    atl = lo_close

                # Incoming economic calendars? num_incoming_economic_calendars is used to Block entries if incoming events (total_num_ecoevents==0 to make entries).
                num_impacting_economic_calendars : int = 0
                num_bullish_ecoevents, num_bearish_ecoevents, total_num_ecoevents = 0, 0, 0
                if economic_calendars_loaded and algo_param['block_entries_on_impacting_ecoevents']:
                    pd_impacting_economic_calendars = pd_economic_calendars[pd_economic_calendars.event_code.isin(algo_param['mapped_event_codes'])]
                    pd_impacting_economic_calendars = pd_impacting_economic_calendars[
                                ( 
                                    (
                                        pd_impacting_economic_calendars.calendar_item_timestamp_ms>=lo_timestamp_ms) # Incoming
                                        & (lo_timestamp_ms>=(pd_impacting_economic_calendars.calendar_item_timestamp_ms - num_intervals_block_pending_ecoevents_ms)
                                    ) 
                                )
                                |
                                ( 
                                    (
                                        pd_impacting_economic_calendars.calendar_item_timestamp_ms<lo_timestamp_ms) # Passed
                                        & (lo_timestamp_ms<=(pd_impacting_economic_calendars.calendar_item_timestamp_ms + num_intervals_block_pending_ecoevents_ms/3)
                                    ) 
                                )
                            ]
                    num_impacting_economic_calendars = pd_impacting_economic_calendars.shape[0]

                    if num_impacting_economic_calendars>0:
                        pd_passed_economic_calendars = pd_impacting_economic_calendars[pd_impacting_economic_calendars.calendar_item_timestamp_ms>(lo_timestamp_ms+one_interval_ms)] # Careful with look ahead bias
                        num_bullish_ecoevents = pd_passed_economic_calendars[pd_passed_economic_calendars.pos_neg=='bullish'].shape[0]
                        num_bearish_ecoevents = pd_passed_economic_calendars[pd_passed_economic_calendars.pos_neg=='bearish'].shape[0]
                        num_neutral_ecoevents = pd_passed_economic_calendars[pd_passed_economic_calendars.pos_neg=='neutral'].shape[0]

                        # If adj_sl_on_ecoevents==True, total_num_ecoevents is used to set sl_percent_adj
                        total_num_ecoevents = num_bullish_ecoevents + num_bearish_ecoevents + num_neutral_ecoevents
                    
                lo_fib_eval_result = lookup_fib_target(lo_row_tm1, pd_lo_candles)
                lo_fib_short_periods_fib_target, lo_fib_short_periods_price_swing, lo_fib_long_periods_fib_target, lo_fib_long_periods_price_swing = None, None, None, None
                if lo_fib_eval_result:
                    lo_fib_short_periods_fib_target = lo_fib_eval_result['short_periods']['fib_target']
                    lo_fib_long_periods_fib_target = lo_fib_eval_result['long_periods']['fib_target']

                current_ref_candles_segment_index, last_ref_candles_segmment_index = -1, -1
                current_ref_candles_segment, last_ref_candles_segment = None, None
                current_ref_candles_segment_class, last_ref_candles_segment_class = None, None
                if not pypy_compat:
                    if pd_ref_candles_segments[(pd_ref_candles_segments.start_ts<=lo_datetime.timestamp()) & (pd_ref_candles_segments.end_ts>lo_datetime.timestamp()) ].shape[0]>0:
                        current_ref_candles_segment_index = pd_ref_candles_segments[(pd_ref_candles_segments.start_ts<=lo_datetime.timestamp()) & (pd_ref_candles_segments.end_ts>lo_datetime.timestamp()) ].index.to_list()[0] # Take first
                        current_ref_candles_segment = pd_ref_candles_segments.iloc[current_ref_candles_segment_index]
                        if current_ref_candles_segment is not None and not current_ref_candles_segment.empty:
                            current_ref_candles_segment_class = current_ref_candles_segment['class']
                        last_ref_candles_segmment_index = current_ref_candles_segment_index
                        last_ref_candles_segment = current_ref_candles_segment
                        if current_ref_candles_segment_index>0:
                            last_ref_candles_segmment_index = current_ref_candles_segment_index-1
                            last_ref_candles_segment = pd_ref_candles_segments.iloc[current_ref_candles_segment_index]
                            if last_ref_candles_segment is not None and not last_ref_candles_segment.empty:
                                last_ref_candles_segment_class = last_ref_candles_segment['class']
            
                current_hi_candles_segment_index, last_hi_candles_segmment_index = -1, -1
                current_hi_candles_segment, last_hi_candles_segment = None, None
                current_hi_candles_segment_class, last_hi_candles_segment_class = None, None
                if not pypy_compat:
                    if pd_hi_candles_segments[(pd_hi_candles_segments.start_ts<=lo_datetime.timestamp()) & (pd_hi_candles_segments.end_ts>lo_datetime.timestamp()) ].shape[0]>0:
                        current_hi_candles_segment_index = pd_hi_candles_segments[(pd_hi_candles_segments.start_ts<=lo_datetime.timestamp()) & (pd_hi_candles_segments.end_ts>lo_datetime.timestamp()) ].index.to_list()[0] # Take first
                        current_hi_candles_segment = pd_hi_candles_segments.iloc[current_hi_candles_segment_index]
                        if current_hi_candles_segment is not None and not current_hi_candles_segment.empty:
                            current_hi_candles_segment_class = current_hi_candles_segment['class']
                        last_hi_candles_segmment_index = current_hi_candles_segment_index
                        last_hi_candles_segment = current_hi_candles_segment
                        if current_hi_candles_segment_index>0:
                            last_hi_candles_segmment_index = current_hi_candles_segment_index-1
                            last_hi_candles_segment = pd_hi_candles_segments.iloc[current_hi_candles_segment_index]
                            if last_hi_candles_segment is not None and not last_hi_candles_segment.empty:
                                last_hi_candles_segment_class = last_hi_candles_segment['class']

                current_lo_candles_segment_index, last_lo_candles_segmment_index = -1, -1
                current_lo_candles_segment, last_lo_candles_segment = None, None
                current_lo_candles_segment_class, last_lo_candles_segment_class = None, None
                if not pypy_compat:
                    if pd_lo_candles_segments[(pd_lo_candles_segments.start_ts<=lo_datetime.timestamp()) & (pd_lo_candles_segments.end_ts>lo_datetime.timestamp()) ].shape[0]>0:
                        current_lo_candles_segment_index = pd_lo_candles_segments[(pd_lo_candles_segments.start_ts<=lo_datetime.timestamp()) & (pd_lo_candles_segments.end_ts>lo_datetime.timestamp()) ].index.to_list()[0] # Take first
                        current_lo_candles_segment = pd_lo_candles_segments.iloc[current_lo_candles_segment_index]
                        if current_lo_candles_segment is not None and not current_lo_candles_segment.empty:
                            current_lo_candles_segment_class = current_lo_candles_segment['class']
                        last_lo_candles_segmment_index = current_lo_candles_segment_index
                        last_lo_candles_segment = current_lo_candles_segment
                        if current_lo_candles_segment_index>0:
                            last_lo_candles_segmment_index = current_lo_candles_segment_index-1
                            last_lo_candles_segment = pd_lo_candles_segments.iloc[current_lo_candles_segment_index]
                            if last_lo_candles_segment is not None and not last_lo_candles_segment.empty: 
                                last_lo_candles_segment_class = last_lo_candles_segment['class']

                # Find corresponding row in pd_hi_candles
                def _find_ref_row(lo_year, lo_month, lo_day, pd_ref_candles):
                    ref_row = None
                    ref_matching_rows = pd_ref_candles[(pd_ref_candles.year==lo_year) & (pd_ref_candles.month==lo_month) & (pd_ref_candles.day==lo_day)]
                    if not ref_matching_rows.empty:
                        ref_row = ref_matching_rows.iloc[0]
                        ref_row.has_inflection_point = False

                        recent_rows = pd_ref_candles[(pd_ref_candles['datetime'] <= lo_datetime)].tail(3)
                        if not recent_rows['close_above_or_below_ema'].isna().all():
                            ref_row.has_inflection_point = True

                    else:
                        logger.warning(f"{key} ref_row not found for year: {lo_year}, month: {lo_month}, day: {lo_day}")

                    return ref_row
                
                def _search_hi_tm1(hi_row, lo_row, pd_hi_candles):
                    row_index = hi_row.name -1
                    hi_row_tm1 = pd_hi_candles.iloc[row_index] if hi_row is not None else None
                    hi_row_tm1 = hi_row_tm1 if hi_row_tm1['timestamp_ms'] < lo_row['timestamp_ms'] else None
                    if row_index>1:
                        while hi_row_tm1['timestamp_ms'] >= lo_row['timestamp_ms']:
                            row_index = row_index -1
                            hi_row_tm1 = pd_hi_candles.iloc[row_index]
                    return hi_row_tm1

                hi_row, hi_row_tm1 = None, None
                if lo_datetime>=algo_param['start_date']:
                    if algo_param['lo_candle_size'][-1]=="m":
                        matching_rows = pd_hi_candles[(pd_hi_candles.year==lo_year) & (pd_hi_candles.month==lo_month) & (pd_hi_candles.day==lo_day) & (pd_hi_candles.hour==lo_hour)]
                        if not matching_rows.empty:
                            hi_row = matching_rows.iloc[0]
                            
                        else:
                            logger.warning(f"{key} hi_row not found for year: {lo_year}, month: {lo_month}, day: {lo_day}, hour: {lo_hour}")
                            continue

                        hi_row_tm1 = _search_hi_tm1(hi_row, lo_row, pd_hi_candles)
                        if hi_row_tm1 is not None:
                            assert(hi_row_tm1['timestamp_ms'] < lo_row['timestamp_ms']) # No look ahead bias!!!
                        else:
                            continue

                        # Be careful with look ahead bias!!!
                        target_ref_candle_date = lo_datetime + timedelta(days=-1)
                        ref_row_fast = _find_ref_row(
                            target_ref_candle_date.year, 
                            target_ref_candle_date.month, 
                            target_ref_candle_date.day, 
                            pd_ref_candles_fast)
                        ref_row_slow = _find_ref_row(
                            target_ref_candle_date.year, 
                            target_ref_candle_date.month, 
                            target_ref_candle_date.day, 
                            pd_ref_candles_slow)

                    elif algo_param['lo_candle_size'][-1]=="h":
                        matching_rows = pd_hi_candles[(pd_hi_candles.year==lo_year) & (pd_hi_candles.month==lo_month) & (pd_hi_candles.day==lo_day)]
                        if not matching_rows.empty:
                            hi_row = matching_rows.iloc[0]
                            
                        else:
                            logger.warning(f"{key} hi_row not found for year: {lo_year}, month: {lo_month}, day: {lo_day}")
                            continue

                        hi_row_tm1 = _search_hi_tm1(hi_row, lo_row, pd_hi_candles)
                        if hi_row_tm1 is not None:
                            assert(hi_row_tm1['timestamp_ms'] < lo_row['timestamp_ms']) # No look ahead bias!!!
                        else:
                            continue
                        
                        # Be careful with look ahead bias!!!
                        target_ref_candle_date = lo_datetime + timedelta(days=-1)
                        ref_row_fast = _find_ref_row(
                            target_ref_candle_date.year, 
                            target_ref_candle_date.month, 
                            target_ref_candle_date.day, 
                            pd_ref_candles_fast)
                        ref_row_slow = _find_ref_row(
                            target_ref_candle_date.year, 
                            target_ref_candle_date.month, 
                            target_ref_candle_date.day, 
                            pd_ref_candles_slow)

                    elif algo_param['lo_candle_size'][-1]=="d":
                        # Not supported atm
                        hi_row, hi_row_tm1 = None, None

                    hi_datetime, hi_year, hi_month, hi_day, hi_hour, hi_minute, hi_timestamp_ms = None, None, None, None, None, None, None
                    hi_open, hi_high, hi_low, hi_close, hi_volume = None, None, None, None, None
                    hi_atr, hi_rsi, hi_rsi_bucket, hi_rsi_trend, hi_mfi, hi_mfi_bucket, hi_macd_minus_signal, hi_boillenger_upper, hi_boillenger_lower, hi_boillenger_channel_height = None, None, None, None, None, None, None, None, None, None
                    hi_hurst_exp, hi_ema_volume_long_periods = None, None
                    hi_tm1_rsi, hi_tm1_rsi_bucket, hi_tm1_rsi_trend = None, None, None
                    hi_fib_eval_result = None
                    if hi_row is not None:
                        hi_datetime = hi_row['datetime']
                        hi_year = hi_row['year']
                        hi_month = hi_row['month']
                        hi_day = hi_row['day']
                        hi_hour = hi_row['hour']
                        hi_minute = hi_row['minute']
                        hi_timestamp_ms = hi_row['timestamp_ms']
                        hi_open = hi_row['open']
                        hi_high = hi_row['high']
                        hi_low = hi_row['low']
                        hi_close = hi_row['close']
                        hi_volume = hi_row['volume']
                        hi_atr = hi_row['atr']
                        hi_rsi = hi_row['rsi']
                        hi_rsi_bucket = hi_row['rsi_bucket']
                        hi_rsi_trend = hi_row['rsi_trend']
                        hi_mfi = hi_row['mfi']
                        hi_mfi_bucket = hi_row['mfi_bucket']
                        hi_macd_minus_signal = hi_row['macd_minus_signal']
                        hi_boillenger_upper = hi_row['boillenger_upper']
                        hi_boillenger_lower = hi_row['boillenger_lower']
                        hi_boillenger_channel_height = hi_row['boillenger_channel_height']
                        hi_hurst_exp = hi_row['hurst_exp']

                        hi_tm1_rsi = hi_row_tm1['rsi']
                        hi_tm1_rsi_bucket = hi_row_tm1['rsi_bucket']
                        hi_tm1_rsi_trend = hi_row_tm1['rsi_trend']

                        hi_ema_volume_long_periods = hi_row['ema_volume_long_periods']
                        hi_ema_short_slope = hi_row['ema_short_slope'] if 'ema_short_slope' in pd_hi_candles.columns else 0
                        hi_normalized_ema_short_slope = hi_row['normalized_ema_short_slope'] if 'normalized_ema_short_slope' in pd_hi_candles.columns else 0
                        hi_ema_long_slope = hi_row['ema_long_slope'] if 'ema_long_slope' in pd_hi_candles.columns else 0
                        hi_normalized_ema_long_slope = hi_row['normalized_ema_long_slope'] if 'normalized_ema_long_slope' in pd_hi_candles.columns else 0
                        hi_tm1_normalized_ema_long_slope = hi_row_tm1['normalized_ema_long_slope'] if 'normalized_ema_long_slope' in pd_hi_candles.columns else 0

                        hi_max_short_periods = hi_row['max_short_periods']
                        hi_idmax_short_periods = int(hi_row['idmax_short_periods']) if not math.isnan(hi_row['idmax_short_periods']) else None
                        hi_idmax_dt_short_periods = pd_hi_candles.at[hi_idmax_short_periods,'datetime'] if not(hi_idmax_short_periods is None or pd.isna(hi_idmax_short_periods)) else None
                        hi_max_long_periods = hi_row['max_long_periods']
                        hi_idmax_long_periods = int(hi_row['idmax_long_periods']) if not math.isnan(hi_row['idmax_long_periods']) else None
                        hi_idmax_dt_long_periods = pd_hi_candles.at[hi_idmax_long_periods,'datetime'] if not(hi_idmax_long_periods is None or pd.isna(hi_idmax_long_periods)) else None

                        hi_tm1_max_short_periods = hi_row_tm1['max_short_periods']
                        hi_tm1_idmax_short_periods = int(hi_row_tm1['idmax_short_periods']) if not math.isnan(hi_row_tm1['idmax_short_periods']) else None
                        hi_tm1_idmax_dt_short_periods = pd_hi_candles.at[hi_tm1_idmax_short_periods,'datetime'] if not(hi_tm1_idmax_short_periods is None or pd.isna(hi_tm1_idmax_short_periods)) else None
                        hi_tm1_max_long_periods = hi_row_tm1['max_long_periods']
                        hi_tm1_idmax_long_periods = int(hi_row_tm1['idmax_long_periods']) if not math.isnan(hi_row_tm1['idmax_long_periods']) else None
                        hi_tm1_idmax_dt_long_periods = pd_hi_candles.at[hi_tm1_idmax_long_periods,'datetime'] if not(hi_tm1_idmax_long_periods is None or pd.isna(hi_tm1_idmax_long_periods)) else None

                        hi_min_short_periods = hi_row['min_short_periods']
                        hi_idmin_short_periods = int(hi_row['idmin_short_periods']) if not math.isnan(hi_row['idmin_short_periods']) else None
                        hi_idmin_dt_short_periods = pd_hi_candles.at[hi_idmin_short_periods,'datetime'] if not (hi_idmin_short_periods is None or pd.isna(hi_idmin_short_periods)) else None
                        hi_min_long_periods = hi_row['min_long_periods']
                        hi_idmin_long_periods = int(hi_row['idmin_long_periods']) if not math.isnan(hi_row['idmin_long_periods']) else None
                        hi_idmin_dt_long_periods = pd_hi_candles.at[hi_idmin_long_periods,'datetime'] if not (hi_idmin_long_periods is None or pd.isna(hi_idmin_long_periods)) else None

                        hi_tm1_min_short_periods = hi_row_tm1['min_short_periods']
                        hi_tm1_idmin_short_periods = int(hi_row_tm1['idmin_short_periods']) if not math.isnan(hi_row_tm1['idmin_short_periods']) else None
                        hi_tm1_idmin_dt_short_periods = pd_hi_candles.at[hi_tm1_idmin_short_periods,'datetime'] if not (hi_tm1_idmin_short_periods is None or pd.isna(hi_tm1_idmin_short_periods)) else None
                        hi_tm1_min_long_periods = hi_row_tm1['min_long_periods']
                        hi_tm1_idmin_long_periods = int(hi_row_tm1['idmin_long_periods']) if not math.isnan(hi_row_tm1['idmin_long_periods']) else None
                        hi_tm1_idmin_dt_long_periods = pd_hi_candles.at[hi_tm1_idmin_long_periods,'datetime'] if not (hi_tm1_idmin_long_periods is None or pd.isna(hi_tm1_idmin_long_periods)) else None

                        hi_fib_eval_result = lookup_fib_target(hi_row_tm1, pd_hi_candles)
                        hi_fib_short_periods_fib_target, hi_fib_short_periods_price_swing, hi_fib_long_periods_fib_target, hi_fib_long_periods_price_swing = None, None, None, None
                        if hi_fib_eval_result:
                            hi_fib_short_periods_fib_target = hi_fib_eval_result['short_periods']['fib_target']
                            hi_fib_long_periods_fib_target = hi_fib_eval_result['long_periods']['fib_target']

                    last_candles, post_move_candles, post_move_price_change, post_move_price_change_percent = None, None, None, None
                    if algo_param['last_candles_timeframe']=='lo':
                        last_candles = pd_lo_candles[pd_lo_candles['timestamp_ms']<=lo_timestamp_ms].tail(algo_param['how_many_last_candles']).to_dict('records')
                        assert(all([ candle['timestamp_ms']<=lo_timestamp_ms for candle in last_candles ]))
                        post_move_candles = pd_lo_candles[pd_lo_candles['timestamp_ms']<=lo_timestamp_ms].tail(algo_param['post_move_num_intervals']).to_dict('records')
                        
                    elif algo_param['last_candles_timeframe']=='hi' and hi_row is not None:
                        last_candles = pd_hi_candles[pd_hi_candles['timestamp_ms']<=hi_timestamp_ms].tail(algo_param['how_many_last_candles']).to_dict('records')
                        assert(all([ candle['timestamp_ms']<=hi_timestamp_ms for candle in last_candles ]))
                        post_move_candles = pd_hi_candles[pd_hi_candles['timestamp_ms']<=hi_timestamp_ms].tail(algo_param['post_move_num_intervals']).to_dict('records')

                    post_move_price_change, post_move_price_change_percent = 0, 0
                    if post_move_candles and len(post_move_candles)>=2:
                        post_move_price_change = post_move_candles[-1]['close'] - post_move_candles[0]['open']
                        post_move_price_change_percent = 0
                        if post_move_price_change>0:
                            post_move_price_change_percent = (post_move_candles[-1]['close']/post_move_candles[0]['open'] -1) * 100
                        else:
                            post_move_price_change_percent = -(post_move_candles[0]['close']/post_move_candles[-1]['open'] -1) * 100

                    ref_close_fast, ref_ema_close_fast = None, None
                    if ref_row_fast is not None:
                        ref_close_fast = ref_row_fast['close']
                        ref_ema_close_fast = ref_row_fast['ema_close']

                    ref_close_slow, ref_ema_close_slow = None, None
                    if ref_row_slow is not None:
                        ref_close_slow = ref_row_slow['close']
                        ref_ema_close_slow = ref_row_slow['ema_close']

                    # POSITION NOTIONAL MARKING lo_low, lo_high. pessimistic!
                    def _refresh_current_position(timestamp_ms):
                        current_position_usdt_buy = sum([x['size'] * lo_close for x in all_trades if not x['closed'] and x['side']=='buy'])
                        current_position_usdt_sell = sum([x['size'] * lo_close for x in all_trades if not x['closed'] and x['side']=='sell'])
                        current_position_usdt = current_position_usdt_buy + current_position_usdt_sell
                        this_ticker_historical_trades = [ trade for trade in all_trades if trade['symbol']==ticker ]
                        this_ticker_open_trades = [ trade for trade in this_ticker_historical_trades if not trade['closed'] ]
                        this_ticker_current_position_usdt_buy = sum([x['size'] * lo_close for x in this_ticker_open_trades if x['side']=='buy'])
                        this_ticker_current_position_usdt_sell = sum([x['size'] * lo_close for x in this_ticker_open_trades if x['side']=='sell'])

                        entries_since_sl : Union[int, None] = -1

                        avg_entry_price = None
                        pos_side = '---'
                        max_trade_age_ms = timestamp_ms
                        if this_ticker_open_trades:
                            max_trade_age_ms = timestamp_ms - max([trade['timestamp_ms'] for trade in this_ticker_open_trades ])

                            avg_entry_price = sum([ trade['entry_price']*trade['size'] for trade in this_ticker_open_trades]) / sum([ trade['size'] for trade in this_ticker_open_trades])

                            sides = [ x['side'] for x in this_ticker_open_trades ]
                            if len(set(sides))==1:
                                if sides[0]=='buy':
                                    pos_side = 'buy'
                                else:
                                    pos_side = 'sell'

                        max_sl_trade_age_ms = None
                        this_ticker_sl_trades = [ trade for trade in this_ticker_historical_trades if trade['reason']=='SL' ]
                        if this_ticker_sl_trades:
                            last_sl_timestamp_ms = max([trade['timestamp_ms'] for trade in this_ticker_sl_trades ])
                            max_sl_trade_age_ms = timestamp_ms - last_sl_timestamp_ms
                            entries_since_sl = len([trade for trade in this_ticker_historical_trades if trade['timestamp_ms']>last_sl_timestamp_ms and trade['reason']=='entry' and trade['closed']] )

                        # In single legged trading, we either long or short for a particular ticker at any given moment
                        assert(
                            (this_ticker_current_position_usdt_buy>=0 and this_ticker_current_position_usdt_sell==0) 
                            or (this_ticker_current_position_usdt_buy==0 and this_ticker_current_position_usdt_sell>=0))
                        
                        if this_ticker_current_position_usdt_buy>0:
                            this_ticker_open_positions_side = 'buy'
                            this_ticker_current_position_usdt = this_ticker_current_position_usdt_buy
                        elif this_ticker_current_position_usdt_sell>0:
                            this_ticker_open_positions_side = 'sell'
                            this_ticker_current_position_usdt = this_ticker_current_position_usdt_sell
                        else:
                            this_ticker_open_positions_side = 'flat'
                            this_ticker_current_position_usdt = 0

                        return {
                            'avg_entry_price' : avg_entry_price,
                            'side' : pos_side,
                            'current_position_usdt_buy' : current_position_usdt_buy,
                            'current_position_usdt_sell' : current_position_usdt_sell,
                            'current_position_usdt' : current_position_usdt,
                            'this_ticker_open_trades' : this_ticker_open_trades,
                            'this_ticker_current_position_usdt_buy' : this_ticker_current_position_usdt_buy,
                            'this_ticker_current_position_usdt_sell' : this_ticker_current_position_usdt_sell,
                            'this_ticker_open_positions_side' : this_ticker_open_positions_side,
                            'this_ticker_current_position_usdt' : this_ticker_current_position_usdt,
                            'max_trade_age_ms' : max_trade_age_ms,
                            'max_sl_trade_age_ms' : max_sl_trade_age_ms,
                            'entries_since_sl' : entries_since_sl
                        }
                    
                    current_positions_info = _refresh_current_position(lo_timestamp_ms)
                    avg_entry_price = current_positions_info['avg_entry_price']
                    pos_side = current_positions_info['side']
                    current_position_usdt_buy = current_positions_info['current_position_usdt_buy']
                    current_position_usdt_sell = current_positions_info['current_position_usdt_sell']
                    current_position_usdt = current_positions_info['current_position_usdt']
                    this_ticker_open_trades = current_positions_info['this_ticker_open_trades']
                    this_ticker_current_position_usdt_buy = current_positions_info['this_ticker_current_position_usdt_buy']
                    this_ticker_current_position_usdt_sell = current_positions_info['this_ticker_current_position_usdt_sell']
                    this_ticker_open_positions_side = current_positions_info['this_ticker_open_positions_side']
                    this_ticker_current_position_usdt = current_positions_info['this_ticker_current_position_usdt']
                    max_trade_age_ms = current_positions_info['max_trade_age_ms']
                    max_sl_trade_age_ms = current_positions_info['max_sl_trade_age_ms']
                    entries_since_sl = current_positions_info['entries_since_sl']
                    block_entry_since_last_sl = True if max_sl_trade_age_ms and max_sl_trade_age_ms<=min_sl_age_ms else False

                    def _close_open_positions(
                            key, ticker, 
                            this_ticker_current_position_usdt, 
                            this_ticker_open_positions_side, 
                            current_position_usdt, 
                            trade_pnl,
                            effective_tp_trailing_percent, 
                            row, 
                            reason, 
                            reason2,
                            gloabl_state, all_trades, all_canvas,
                            algo_param,
                            standard_pnl_percent_buckets=BUCKETS_m100_100
                        ):
                        def _gains_losses_to_label(gains_losses_percent):
                            gains_losses_percent_label = bucketize_val(gains_losses_percent, buckets=standard_pnl_percent_buckets)

                            if gains_losses_percent>=0:
                                return f"gain {gains_losses_percent_label}%"
                            else:
                                return f"loss {gains_losses_percent_label}%"
                        
                        def _how_long_before_closed_sec_to_label(how_long_before_closed_sec):
                            how_long_before_closed_sec_label = None
                            how_long_before_closed_hr = how_long_before_closed_sec/(60*60)
                            if how_long_before_closed_hr<=1:
                                how_long_before_closed_sec_label = "<=1hr"
                            elif how_long_before_closed_hr>1 and how_long_before_closed_hr<=8:
                                how_long_before_closed_sec_label = ">1hr <=8hr"
                            elif how_long_before_closed_hr>8 and how_long_before_closed_hr<=24:
                                how_long_before_closed_sec_label = ">8hr <=24hr"
                            elif how_long_before_closed_hr>24 and how_long_before_closed_hr<=24*7:
                                how_long_before_closed_sec_label = ">24hr <=7days"
                            elif how_long_before_closed_hr>24*7 and how_long_before_closed_hr<=24*7*2:
                                how_long_before_closed_sec_label = ">7days <=14days"
                            else:
                                how_long_before_closed_sec_label = ">14days"
                            return how_long_before_closed_sec_label
                        
                        this_ticker_open_trades = [ trade for trade in all_trades if not trade['closed'] and trade['symbol']==ticker]

                        entry_dt = min([ trade['trade_datetime'] for trade in this_ticker_open_trades ])
                        entry_dayofweek = entry_dt.dayofweek
                        entry_hour = entry_dt.hour

                        this_datetime = row['datetime']
                        this_timestamp_ms = row['timestamp_ms']
                        dayofweek = row['dayofweek']
                        high = row['high']
                        low = row['low']
                        close = row['close']
                        ema_short_slope = row['ema_short_slope'] if 'ema_short_slope' in row else None
                        ema_long_slope = row['ema_long_slope'] if 'ema_long_slope' in row else None

                        # Step 1. mark open trades as closed first
                        entry_commission, exit_commission = 0, 0
                        for trade in this_ticker_open_trades:
                            entry_commission += trade['commission']
                            if this_ticker_open_positions_side=='buy':
                                exit_commission += close * trade['size'] * commission_bps / 10000
                                
                            else:
                                exit_commission += close * trade['size'] * commission_bps / 10000
                            trade['trade_pnl'] = 0 # trade_pnl parked under closing trade
                            trade['trade_pnl_bps'] = 0
                            trade['closed'] = True
                        max_pain = min([ trade['max_pain'] for trade in this_ticker_open_trades])
                        max_pain_percent = max_pain/this_ticker_current_position_usdt * 100
                        max_pain_percent_label = _gains_losses_to_label(max_pain_percent)

                        timestamp_ms_from_closed_trades = min([ trade['timestamp_ms'] for trade in this_ticker_open_trades])
                        num_impacting_economic_calendars = min([ trade['num_impacting_economic_calendars'] if 'num_impacting_economic_calendars' in trade else 0 for trade in this_ticker_open_trades])
                        max_camp = max([ trade['max_camp'] for trade in this_ticker_open_trades])
                        entry_post_move_price_change_percent = max([ trade['post_move_price_change_percent'] if 'post_move_price_change_percent' in trade else 0 for trade in this_ticker_open_trades ])
                        
                        # Step 2. Update global_state
                        trade_pnl_less_comm = trade_pnl - (entry_commission + exit_commission)
                        gains_losses_percent = trade_pnl_less_comm/this_ticker_current_position_usdt * 100
                        gains_losses_percent_label = _gains_losses_to_label(gains_losses_percent)
                        how_long_before_closed_sec = (this_timestamp_ms - timestamp_ms_from_closed_trades) / 1000
                        how_long_before_closed_sec_label = _how_long_before_closed_sec_to_label(how_long_before_closed_sec)

                        gloabl_state.total_equity += trade_pnl_less_comm
                        gloabl_state.total_commission += exit_commission
                        cash_before = gloabl_state.cash
                        gloabl_state.cash = gloabl_state.total_equity
                        cash_after = gloabl_state.cash
                        running_total_num_positions : int = len([ 1 for x in all_trades if x['reason']=='entry' and not x['closed']])

                        # Step 3. closing trade
                        # closing_price = low if this_ticker_open_positions_side=='buy' else high # pessimistic!
                        closing_price = close
                        closing_trade = {
                            'trade_datetime' : this_datetime,
                            'timestamp_ms' : this_timestamp_ms,
                            'dayofweek' : dayofweek,
                            'entry_dt' : entry_dt,
                            'entry_dayofweek' : entry_dayofweek,
                            'entry_hour' : entry_hour,
                            'exchange' : exchange.name,
                            'symbol' : ticker,
                            'side' : 'sell' if this_ticker_open_positions_side=='buy' else 'buy',
                            'size' : this_ticker_current_position_usdt / closing_price, # in base ccy
                            'entry_price' : closing_price, # pessimistic!
                            'closed' : True,
                            'reason' : reason,
                            'reason2' : reason2,
                            'total_equity' : gloabl_state.total_equity,
                            'this_ticker_current_position_usdt' : this_ticker_current_position_usdt,
                            'current_position_usdt' : current_position_usdt,
                            'running_total_num_positions' : running_total_num_positions,
                            'cash_before' : cash_before,
                            'cash_after' : cash_after,
                            'order_notional' : this_ticker_current_position_usdt,
                            'trade_pnl' : trade_pnl,
                            'commission' : exit_commission,
                            'max_pain' : max_pain,
                            'trade_pnl_less_comm': trade_pnl_less_comm,
                            'trade_pnl_bps' : (trade_pnl / this_ticker_current_position_usdt) * 100 * 100 if this_ticker_current_position_usdt!=0 else 0,
                            'gains_losses_percent' : gains_losses_percent,
                            'gains_losses_percent_label' : gains_losses_percent_label,
                            'how_long_before_closed_sec' : how_long_before_closed_sec,
                            'how_long_before_closed_sec_label' : how_long_before_closed_sec_label,
                            'max_pain_percent' : max_pain_percent,
                            'max_pain_percent_label' : max_pain_percent_label,
                            'ema_short_slope' : ema_short_slope,
                            'ema_long_slope' : ema_long_slope,
                            'num_impacting_economic_calendars' : num_impacting_economic_calendars,
                            'max_camp' : max_camp,
                            'entry_post_move_price_change_percent' : entry_post_move_price_change_percent
                        }
                        _last_open_trade = this_ticker_open_trades[-1]
                        additional_fields = {field: _last_open_trade[field] if field in _last_open_trade else None for field in algo_param['additional_trade_fields']}
                        closing_trade.update(additional_fields)
                        all_trades.append(closing_trade)

                        if plot_timeseries:
                            '''
                            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html
                            linestyle='-' means solid line. If you don't supply linestyle, the vertical line wont show!!!
                            '''
                            color = 'green' if reason=='TP' or (reason=='HC' and trade_pnl_less_comm>0) else 'red'    
                            all_canvas[f"{key}-param_id{algo_param['param_id']}"]['time_series_canvas'].axvline(x=this_datetime, color=color, linewidth=2, linestyle='--')
                            all_canvas[f"{key}-param_id{algo_param['param_id']}"]['time_series_canvas'].scatter([this_datetime, this_datetime], [low, high], color=color)
                    
                    # UNREAL EVALUATION. We're being pessimistic! We use low/high for estimating unrealized_pnl for buys and sells respectively here.
                    pnl_percent_notional = 0
                    if current_position_usdt>0:    
                        unrealized_pnl, unrealized_pnl_interval, unrealized_pnl_open, unrealized_pnl_live_optimistic, unrealized_pnl_live_pessimistic, unrealized_pnl_live, max_pnl_percent_notional, unrealized_pnl_boillenger, unrealized_pnl_sl, max_unrealized_pnl_live, max_pain, recovered_pnl_optimistic, recovered_pnl_pessimistic, max_recovered_pnl = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 # USDT
                        _asymmetric_tp_bps = algo_param['asymmetric_tp_bps'] if lo_dayofweek in algo_param['cautious_dayofweek'] else 0
                        
                        max_unrealized_pnl_live = max([ trade['max_unrealized_pnl_live'] if 'max_unrealized_pnl_live' in trade else 0 for trade in this_ticker_open_trades ])
                        # 'min' max_pain becaues max_pain is a negative number. It's a loss!
                        max_pain = min([ trade['max_pain'] if 'max_pain' in trade else 0 for trade in this_ticker_open_trades ])
                        max_recovered_pnl = max([ trade['max_recovered_pnl'] if 'max_recovered_pnl' in trade else 0 for trade in this_ticker_open_trades ])
                        trade_datetime = max([ trade['trade_datetime'] if 'trade_datetime' in trade else 0 for trade in this_ticker_open_trades ])
                        entry_post_move_price_change_percent = max([ trade['post_move_price_change_percent'] if 'post_move_price_change_percent' in trade else 0 for trade in this_ticker_open_trades ])
                        max_camp = max([ trade['max_camp'] for trade in this_ticker_open_trades])
                        running_sl_percent_hard = max([ trade['running_sl_percent_hard'] for trade in this_ticker_open_trades])

                        max_pnl_potential_percent = None
                        if any([ trade for trade in this_ticker_open_trades if 'target_price' in trade ]):
                            max_pnl_potential_percent = max([ (trade['target_price']/trade['entry_price'] -1) *100 if trade['side']=='buy' else (trade['entry_price']/trade['target_price'] -1) *100 for trade in this_ticker_open_trades if 'target_price' in trade ])

                        kwargs = {k: v for k, v in locals().items() if k in sl_adj_func_params}
                        sl_adj_func_result = sl_adj_func(**kwargs)
                        running_sl_percent_hard = sl_adj_func_result['running_sl_percent_hard']

                        # this_ticker_open_trades should be updated after SL adj eval
                        for trade in this_ticker_open_trades:
                            trade['running_sl_percent_hard'] = running_sl_percent_hard
                        
                        kwargs = {k: v for k, v in locals().items() if k in trailing_stop_threshold_eval_func_params}
                        trailing_stop_threshold_eval_func_result = trailing_stop_threshold_eval_func(**kwargs)
                        tp_min_percent = trailing_stop_threshold_eval_func_result['tp_min_percent']
                        tp_max_percent = trailing_stop_threshold_eval_func_result['tp_max_percent']
                        recover_min_percent = algo_param['recover_min_percent'] if 'recover_min_percent' in algo_param else None
                        recover_max_pain_percent = algo_param['recover_max_pain_percent'] if 'recover_max_pain_percent' in algo_param else None

                        # tp_min_percent adj: Strategies where target_price not based on tp_max_percent, but variable
                        if max_pnl_potential_percent and max_pnl_potential_percent<tp_max_percent:
                            tp_minmax_ratio = tp_min_percent/tp_max_percent
                            tp_max_percent = max_pnl_potential_percent
                            tp_min_percent = tp_minmax_ratio * tp_max_percent

                        unrealized_pnl_eval_result = pnl_eval_func(lo_row, lo_row_tm1, running_sl_percent_hard, this_ticker_open_trades, algo_param)
                        unrealized_pnl_interval = unrealized_pnl_eval_result['unrealized_pnl_interval']
                        unrealized_pnl_open = unrealized_pnl_eval_result['unrealized_pnl_open']
                        unrealized_pnl_live_optimistic = unrealized_pnl_eval_result['unrealized_pnl_live_optimistic']
                        unrealized_pnl_live_pessimistic = unrealized_pnl_eval_result['unrealized_pnl_live_pessimistic']
                        unrealized_pnl_tp = unrealized_pnl_eval_result['unrealized_pnl_tp']
                        unrealized_pnl_sl = unrealized_pnl_eval_result['unrealized_pnl_sl']
                        unrealized_pnl_live = unrealized_pnl_live_pessimistic

                        if unrealized_pnl_live>0 and unrealized_pnl_live_optimistic>max_unrealized_pnl_live:
                            max_unrealized_pnl_live = unrealized_pnl_live_optimistic
                            for trade in this_ticker_open_trades:
                                trade['max_unrealized_pnl_live'] = max_unrealized_pnl_live # Evaluated optimistically!!!
                        
                        # Do this before max_pain updated
                        if unrealized_pnl_live<0 and unrealized_pnl_live_optimistic>max_pain:
                            recovered_pnl_optimistic = unrealized_pnl_live_optimistic - max_pain
                            recovered_pnl_pessimistic = unrealized_pnl_live_pessimistic - max_pain
                            if recovered_pnl_optimistic>max_recovered_pnl:
                                max_recovered_pnl = recovered_pnl_optimistic
                                for trade in this_ticker_open_trades:
                                    trade['max_recovered_pnl'] = max_recovered_pnl
                                    
                        if unrealized_pnl_live<0:
                            if unrealized_pnl_live<max_pain:
                                max_pain = unrealized_pnl_live
                                for trade in this_ticker_open_trades:
                                    trade['max_pain'] = max_pain # unrealized_pnl_live is set to unrealized_pnl_live_pessimistic!

                        if unrealized_pnl_live<0 and max_unrealized_pnl_live>0:
                            # If out unrealized_pnl_live already fell from positive to negative, reset max_unrealized_pnl_live back to zero
                            max_unrealized_pnl_live = 0
                            for trade in this_ticker_open_trades:
                                trade['max_unrealized_pnl_live'] = max_unrealized_pnl_live

                        unrealized_pnl = unrealized_pnl_live
                        pnl_percent_notional = unrealized_pnl_open / current_position_usdt * 100 # This is evaluated using open (Don't use close, that's forward bias!)
                        max_pnl_percent_notional = max_unrealized_pnl_live / current_position_usdt * 100
                        max_pain_percent_notional = max_pain / current_position_usdt * 100
                        max_recovered_pnl_percent_notional = max_recovered_pnl / current_position_usdt * 100

                        if (
                            (pnl_percent_notional>0 and pnl_percent_notional>=tp_min_percent)
                            or (
                                recover_max_pain_percent
                                and pnl_percent_notional<0 
                                and max_recovered_pnl_percent_notional>=recover_min_percent
                                and abs(max_pain_percent_notional)>=recover_max_pain_percent
                            ) # Taking 'abs': Trailing stop can fire if trade moves in either direction - if your trade is losing trade.
                        ): 
                            '''
                            
                            'effective_tp_trailing_percent' is initialized to float('inf') on entries. Whenever 'pnl_percent_notional' crosses 'tp_min_percent', trailing stop mechanism kicks in.

                            https://norman-lm-fung.medium.com/gradually-tightened-trailing-stops-f7854bf1e02b

                            'effective_tp_trailing_percent' is used to TRIGGER trailing stop.
                            Please be careful if you're marking closing trade with candle close.
                            '''
                            if algo_param['use_gradual_tightened_trailing_stops']:
                                effective_tp_trailing_percent = calc_eff_trailing_sl(
                                        tp_min_percent = tp_min_percent,
                                        tp_max_percent = tp_max_percent,
                                        sl_percent_trailing = algo_param['sl_percent_trailing'],
                                        pnl_percent_notional = max_pnl_percent_notional if pnl_percent_notional>0 else max_recovered_pnl_percent_notional,
                                        default_effective_tp_trailing_percent = float('inf'),
                                        linear=True if algo_param['trailing_stop_mode']=='linear' else False, # trailing_stop_mode: linear vs parabolic
                                        pow=5
                                    )
                            else:
                                effective_tp_trailing_percent = algo_param['sl_percent_trailing']
                        
                        # 1. SL
                        if (
                            unrealized_pnl_live < 0 
                            or (
                                (unrealized_pnl_live>0 and unrealized_pnl_live<max_unrealized_pnl_live)
                                or (
                                    unrealized_pnl_live<0
                                    and recovered_pnl_pessimistic<max_recovered_pnl
                                    and abs(max_recovered_pnl_percent_notional)>=recover_min_percent
                                    and abs(max_pain_percent_notional)>=recover_max_pain_percent
                                )
                            )
                        ):
                            # unrealized_pnl_live is set to unrealized_pnl_live_pessimistic!
                            loss_hard = abs(unrealized_pnl_live)/this_ticker_current_position_usdt * 100 if unrealized_pnl_live<0 else 0

                            if unrealized_pnl_live>0:
                                loss_trailing = (1 - unrealized_pnl_live/max_unrealized_pnl_live) * 100 if unrealized_pnl_live>0 and unrealized_pnl_live<max_unrealized_pnl_live else 0
                            elif unrealized_pnl_live<0:
                                loss_trailing = (1 - recovered_pnl_pessimistic/max_recovered_pnl) * 100 if unrealized_pnl_live<0 and recovered_pnl_pessimistic<max_recovered_pnl else 0

                            if loss_hard>=running_sl_percent_hard:
                                unrealized_pnl = (running_sl_percent_hard/algo_param['sl_hard_percent']) * unrealized_pnl_sl
                                reason2 = "sl_hard_percent"
                            elif (
                                    loss_trailing>=effective_tp_trailing_percent # loss_trailing is evaluated pessimistically.
                                    # and pnl_percent_notional>tp_min_percent
                                    # and unrealized_pnl_live >= sl_trailing_min_threshold_usdt
                                ):
                                ''' 
                                If you're using 'effective_tp_trailing_percent' to approx unrealised pnl, make sure "loss_trailing>=effective_tp_trailing_percent" is the only condition.
                                Don't AND this with other condition. Otherwise use close price to approx unrealised pnl instead!!!
                                '''
                                if unrealized_pnl_live>0:
                                    unrealized_pnl = min(
                                        ((100-effective_tp_trailing_percent)/100) * max_unrealized_pnl_live,
                                        this_ticker_current_position_usdt * algo_param['tp_max_percent']/100
                                    )
                                else:
                                    unrealized_pnl = max_pain + ((100-effective_tp_trailing_percent)/100) * max_recovered_pnl
                                # unrealized_pnl = unrealized_pnl_interval # less accurate
                                reason2 = "sl_percent_trailing"

                            if (
                                (loss_hard>=running_sl_percent_hard) 
                                or (
                                    loss_trailing>=effective_tp_trailing_percent 
                                    # and pnl_percent_notional>tp_min_percent
                                    # and unrealized_pnl_live >= sl_trailing_min_threshold_usdt
                                )
                            ):
                                block_entry_since_last_sl = True
                                reason = 'SL' if unrealized_pnl<0 else 'TP'
                                _close_open_positions(
                                    key, 
                                    ticker, 
                                    this_ticker_current_position_usdt, 
                                    this_ticker_open_positions_side, 
                                    current_position_usdt, 
                                    unrealized_pnl, 
                                    effective_tp_trailing_percent,
                                    lo_row, reason, reason2, gloabl_state, all_trades, all_canvas,
                                    algo_param
                                )
                                current_positions_info = _refresh_current_position(lo_timestamp_ms)
                                avg_entry_price = current_positions_info['avg_entry_price']
                                pos_side = current_positions_info['side']
                                current_position_usdt_buy = current_positions_info['current_position_usdt_buy']
                                current_position_usdt_sell = current_positions_info['current_position_usdt_sell']
                                current_position_usdt = current_positions_info['current_position_usdt']
                                this_ticker_open_trades = current_positions_info['this_ticker_open_trades']
                                this_ticker_current_position_usdt_buy = current_positions_info['this_ticker_current_position_usdt_buy']
                                this_ticker_current_position_usdt_sell = current_positions_info['this_ticker_current_position_usdt_sell']
                                this_ticker_open_positions_side = current_positions_info['this_ticker_open_positions_side']
                                this_ticker_current_position_usdt = current_positions_info['this_ticker_current_position_usdt']
                                max_sl_trade_age_ms = current_positions_info['max_sl_trade_age_ms']

                                # sl_percent_trailing = algo_param['sl_percent_trailing'] # Reset! Remember!
                                this_ticker_open_positions_side='flat' # Reset!
                                reversal_camp_cache[key] = REVERSAL_CAMP_ITEM.copy()

                        # 2. TP: Trigger by unrealized_pnl_live_optimistic, not unrealized_pnl_live (which is unrealized_pnl_live_pessimistic). Pnl estimation from unrealized_pnl_boillenger however!!!
                        if this_ticker_current_position_usdt>0 and unrealized_pnl_live_optimistic>0:
                            kwargs = {k: v for k, v in locals().items() if k in tp_eval_func_params}
                            tp_eval_func_result = tp_eval_func(**kwargs)

                            if tp_eval_func_result:
                                unrealized_pnl_tp = min(
                                    unrealized_pnl_tp,
                                    this_ticker_current_position_usdt * algo_param['tp_max_percent']/100
                                )
                                _close_open_positions(
                                    key, ticker, 
                                    this_ticker_current_position_usdt, 
                                    this_ticker_open_positions_side, 
                                    current_position_usdt, 
                                    unrealized_pnl_tp, 
                                    effective_tp_trailing_percent,
                                    lo_row, 'TP', '', gloabl_state, all_trades, all_canvas,
                                    algo_param
                                )
                                current_position_usdt -= this_ticker_current_position_usdt
                                this_ticker_current_position_usdt = 0
                                this_ticker_open_positions_side='flat' # Reset!
                                reversal_camp_cache[key] = REVERSAL_CAMP_ITEM.copy()

                    def _position_size_and_cash_check(
                            current_position_usdt : float, # All positions added together (include other tickers)
                            this_ticker_current_position_usdt : float, # This position only
                            target_order_notional : float,
                            total_equity : float,
                            target_position_size_percent_total_equity : float,
                            cash : float
                    ) -> bool:
                        return (
                            (current_position_usdt + target_order_notional <= total_equity * (target_position_size_percent_total_equity/100)) 
                            and (this_ticker_current_position_usdt + target_order_notional <= total_equity * (target_position_size_percent_total_equity/100))
                            and cash >= target_order_notional
                        )
                    
                    entry_adj_bps = 0 # Essentially disable this for time being.

                    if(
                        lo_low<=(lo_boillenger_lower*(1-entry_adj_bps/10000))
                    ):
                        lo_boillenger_lower_breached_history = lo_boillenger_lower_breached_cache.get(key, [])
                        lo_boillenger_upper_breached_history = lo_boillenger_upper_breached_cache.get(key, [])
                        lo_boillenger_lower_breached_cache[key] = lo_boillenger_lower_breached_history
                        lo_boillenger_upper_breached_cache[key] = lo_boillenger_upper_breached_history
                        lo_boillenger_upper_breached_history.clear()
                        lo_boillenger_lower_breached_history.append(lo_datetime)
                        reversal_camp_cache[key] = REVERSAL_CAMP_ITEM.copy()
                    elif(
                        lo_high>=(lo_boillenger_upper*(1+entry_adj_bps/10000))
                    ):
                        lo_boillenger_lower_breached_history = lo_boillenger_lower_breached_cache.get(key, [])
                        lo_boillenger_upper_breached_history = lo_boillenger_upper_breached_cache.get(key, [])
                        lo_boillenger_lower_breached_cache[key] = lo_boillenger_lower_breached_history
                        lo_boillenger_upper_breached_cache[key] = lo_boillenger_upper_breached_history
                        lo_boillenger_lower_breached_history.clear()
                        lo_boillenger_upper_breached_history.append(lo_datetime)
                        reversal_camp_cache[key] = REVERSAL_CAMP_ITEM.copy()

                    if algo_param['constant_order_notional']:
                        target_order_notional = algo_param['target_order_notional']
                    else:
                        kwargs = {k: v for k, v in locals().items() if k in order_notional_adj_func_params}
                        order_notional_adj_func_result = order_notional_adj_func(**kwargs)
                        target_order_notional = order_notional_adj_func_result['target_order_notional']
                                    
                    order_notional_long, order_notional_short = target_order_notional, target_order_notional
                    if algo_param['clip_order_notional_to_best_volumes']:
                        order_notional_long = min(lo_volume * lo_low, target_order_notional)
                        order_notional_short = min(lo_volume * lo_high, target_order_notional)
                    
                    kwargs = {k: v for k, v in locals().items() if k in allow_entry_initial_func_params}
                    allow_entry_initial_func_result = allow_entry_initial_func(**kwargs)
                    allow_entry_initial_long = allow_entry_initial_func_result['long']
                    allow_entry_initial_short = allow_entry_initial_func_result['short']
                    allow_entry_final_long = False
                    allow_entry_final_short = False

                    if algo_param['enable_sliced_entry']:
                        kwargs = {k: v for k, v in locals().items() if k in allow_slice_entry_func_params}
                        allow_slice_entry_func_result = allow_slice_entry_func(**kwargs)
                        
                    # 3. Entries
                    if (
                            algo_param['strategy_mode'] in [ 'long_only', 'long_short'] 
                            and order_notional_long>0
                            and (not algo_param['block_entries_on_impacting_ecoevents'] or num_impacting_economic_calendars==0)
                            and not block_entry_since_last_sl
                            and (
                                (
                                    this_ticker_open_positions_side=='flat'
                                    and allow_entry_initial_long
                                    and _position_size_and_cash_check(current_position_usdt, this_ticker_current_position_usdt, order_notional_long, gloabl_state.total_equity, target_position_size_percent_total_equity, gloabl_state.cash)
                                ) or (
                                    this_ticker_open_positions_side=='buy'
                                    and _position_size_and_cash_check(current_position_usdt, this_ticker_current_position_usdt, order_notional_long, gloabl_state.total_equity, target_position_size_percent_total_equity, gloabl_state.cash)
                                    and (algo_param['enable_sliced_entry'] and allow_slice_entry_func_result['long'])
                                )
                            )
                    ):
                        # Long
                        order_notional = order_notional_long

                        if not reversal_camp_cache[key]['camp1'] and not reversal_camp_cache[key]['camp2'] and not reversal_camp_cache[key]['camp3']:
                            reversal_camp_cache[key]['camp1'] = True
                            reversal_camp_cache[key]['camp1_price'] = lo_close
                        elif reversal_camp_cache[key]['camp1'] and not reversal_camp_cache[key]['camp2'] and not reversal_camp_cache[key]['camp3']:
                            reversal_camp_cache[key]['camp2'] = True
                            reversal_camp_cache[key]['camp2_price'] = lo_close
                        elif reversal_camp_cache[key]['camp1'] and reversal_camp_cache[key]['camp2'] and not reversal_camp_cache[key]['camp3']:
                            reversal_camp_cache[key]['camp3'] = True
                            reversal_camp_cache[key]['camp3_price'] = lo_close
                        reversal_camp_cache[key]['datetime'] = lo_datetime

                        fetch_historical_price_func = fetch_price
                        kwargs = {k: v for k, v in locals().items() if k in allow_entry_final_func_params}
                        allow_entry_final_func_result = allow_entry_final_func(**kwargs)

                        allow_entry_final_long = allow_entry_final_func_result['long']
                        if (allow_entry_final_long):
                            order_notional_adj_factor = algo_param['dayofweek_adj_map_order_notional'][lo_dayofweek]
                            if order_notional>0 and order_notional_adj_factor>0:
                                max_camp = _max_camp(reversal_camp_cache[key]['camp1'], reversal_camp_cache[key]['camp2'], reversal_camp_cache[key]['camp3'])
                                target_price = allow_entry_final_func_result['target_price_long']
                                reference_price = allow_entry_final_func_result['reference_price']
                                sitting_on_boillenger_band = allow_entry_final_func_result['sitting_on_boillenger_band'] if 'sitting_on_boillenger_band' in allow_entry_final_func_result else None
                                _additional_trade_fields = {k: v for k, v in locals().items() if k in algo_param['additional_trade_fields']}
                                
                                commission = order_notional_adj_factor*order_notional * commission_bps / 10000
                                gloabl_state.total_commission += commission

                                cash_before = gloabl_state.cash
                                gloabl_state.cash = gloabl_state.cash - order_notional_adj_factor*order_notional - commission
                                cash_after = gloabl_state.cash

                                running_total_num_positions : int = len([ 1 for x in all_trades if x['reason']=='entry' and not x['closed']])

                                entry_price = allow_entry_final_func_result['entry_price_long']
                                reversal_camp_cache[key]['price'] = entry_price

                                pnl_potential_bps = (target_price/entry_price - 1) *10000 if target_price else None

                                new_trade_0 = {
                                                    'trade_datetime' : lo_datetime,
                                                    'timestamp_ms' : lo_timestamp_ms,
                                                    'dayofweek' : lo_dayofweek,
                                                    'exchange' : exchange.name,
                                                    'symbol' : ticker,
                                                    'side' : 'buy',
                                                    'size' : order_notional_adj_factor*order_notional/lo_close, # in base ccy. 
                                                    'entry_price' : entry_price,
                                                    'target_price' : target_price,
                                                    'pnl_potential_bps' : pnl_potential_bps,
                                                    'ref_ema_close_fast' : ref_ema_close_fast,
                                                    'running_sl_percent_hard' : algo_param['sl_hard_percent'],
                                                    'closed' : False,
                                                    'reason' : 'entry',
                                                    'reason2' : '',
                                                    'total_equity' : gloabl_state.total_equity,
                                                    'this_ticker_current_position_usdt' : this_ticker_current_position_usdt,
                                                    'current_position_usdt' : current_position_usdt,
                                                    'running_total_num_positions' : running_total_num_positions,
                                                    'cash_before' : cash_before,
                                                    'cash_after' : cash_after,
                                                    'order_notional' : order_notional,
                                                    'commission' : commission,
                                                    'max_pain' : 0,
                                                    'num_impacting_economic_calendars' : num_impacting_economic_calendars,
                                                    'max_camp': max_camp,
                                                    'post_move_price_change_percent' : post_move_price_change_percent
                                                }
                                all_trades.append(new_trade_0)
                                new_trade_0.update(_additional_trade_fields)

                                # Resets!
                                effective_tp_trailing_percent = float('inf')
                                lo_boillenger_lower_breached_history = lo_boillenger_lower_breached_cache.get(key, [])
                                lo_boillenger_lower_breached_history.clear()

                            if plot_timeseries:
                                '''
                                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html
                                linestyle='-' means solid line. If you don't supply linestyle, the vertical line wont show!!!
                                '''
                                all_canvas[f"{key}-param_id{algo_param['param_id']}"]['time_series_canvas'].axvline(x=lo_datetime, color='gray', linewidth=1, linestyle='-')
                                all_canvas[f"{key}-param_id{algo_param['param_id']}"]['time_series_canvas'].scatter([lo_datetime, lo_datetime], [lo_low, lo_high], color='gray')

                    elif (
                            algo_param['strategy_mode'] in [ 'short_only', 'long_short'] 
                            and order_notional_short>0
                            and (not algo_param['block_entries_on_impacting_ecoevents'] or num_impacting_economic_calendars==0)
                            and not block_entry_since_last_sl
                            and (
                                (
                                    this_ticker_open_positions_side=='flat'
                                    and allow_entry_initial_short
                                    and _position_size_and_cash_check(current_position_usdt, this_ticker_current_position_usdt, order_notional_short, gloabl_state.total_equity, target_position_size_percent_total_equity, gloabl_state.cash)
                                ) or (
                                    this_ticker_open_positions_side=='sell'
                                    and _position_size_and_cash_check(current_position_usdt, this_ticker_current_position_usdt, order_notional_short, gloabl_state.total_equity, target_position_size_percent_total_equity, gloabl_state.cash)
                                    and (algo_param['enable_sliced_entry'] and allow_slice_entry_func_result['short'])
                                )
                            )
                    ):
                        # Short
                        order_notional = order_notional_short

                        if not reversal_camp_cache[key]['camp1'] and not reversal_camp_cache[key]['camp2'] and not reversal_camp_cache[key]['camp3']:
                            reversal_camp_cache[key]['camp1'] = True
                            reversal_camp_cache[key]['camp1_price'] = lo_close
                        elif reversal_camp_cache[key]['camp1'] and not reversal_camp_cache[key]['camp2'] and not reversal_camp_cache[key]['camp3']:
                            reversal_camp_cache[key]['camp2'] = True
                            reversal_camp_cache[key]['camp2_price'] = lo_close
                        elif reversal_camp_cache[key]['camp1'] and reversal_camp_cache[key]['camp2'] and not reversal_camp_cache[key]['camp3']:
                            reversal_camp_cache[key]['camp3'] = True
                            reversal_camp_cache[key]['camp3_price'] = lo_close
                        reversal_camp_cache[key]['datetime'] = lo_datetime

                        fetch_historical_price_func = fetch_price
                        kwargs = {k: v for k, v in locals().items() if k in allow_entry_final_func_params}
                        allow_entry_final_func_result = allow_entry_final_func(**kwargs)

                        allow_entry_final_short = allow_entry_final_func_result['short']
                        if (allow_entry_final_short):
                            order_notional_adj_factor = algo_param['dayofweek_adj_map_order_notional'][lo_dayofweek]
                            if order_notional>0 and order_notional_adj_factor>0:
                                max_camp = _max_camp(reversal_camp_cache[key]['camp1'], reversal_camp_cache[key]['camp2'], reversal_camp_cache[key]['camp3'])
                                target_price = allow_entry_final_func_result['target_price_short']
                                reference_price = allow_entry_final_func_result['reference_price']
                                sitting_on_boillenger_band = allow_entry_final_func_result['sitting_on_boillenger_band'] if 'sitting_on_boillenger_band' in allow_entry_final_func_result else None
                                _additional_trade_fields = {k: v for k, v in locals().items() if k in algo_param['additional_trade_fields']}

                                commission = order_notional_adj_factor*order_notional * commission_bps / 10000
                                gloabl_state.total_commission += commission

                                cash_before = gloabl_state.cash
                                gloabl_state.cash = gloabl_state.cash - order_notional_adj_factor*order_notional - commission
                                cash_after = gloabl_state.cash

                                running_total_num_positions : int = len([ 1 for x in all_trades if x['reason']=='entry' and not x['closed']])

                                entry_price = allow_entry_final_func_result['entry_price_short']
                                reversal_camp_cache[key]['price'] = entry_price

                                pnl_potential_bps = (entry_price/target_price - 1) *10000 if target_price else None

                                new_trade_0 = {
                                                    'trade_datetime' : lo_datetime,
                                                    'timestamp_ms' : lo_timestamp_ms,
                                                    'dayofweek' : lo_dayofweek,
                                                    'exchange' : exchange.name,
                                                    'symbol' : ticker,
                                                    'side' : 'sell',
                                                    'size' : order_notional_adj_factor*order_notional/lo_close, # in base ccy
                                                    'entry_price' : entry_price,
                                                    'target_price' : target_price,
                                                    'pnl_potential_bps' : pnl_potential_bps,
                                                    'ref_ema_close_fast' : ref_ema_close_fast,
                                                    'running_sl_percent_hard' : algo_param['sl_hard_percent'],
                                                    'closed' : False,
                                                    'reason' : 'entry',
                                                    'reason2' : '',
                                                    'total_equity' : gloabl_state.total_equity,
                                                    'this_ticker_current_position_usdt' : this_ticker_current_position_usdt,
                                                    'current_position_usdt' : current_position_usdt,
                                                    'running_total_num_positions' : running_total_num_positions,
                                                    'cash_before' : cash_before,
                                                    'cash_after' : cash_after,
                                                    'order_notional' : order_notional,
                                                    'commission' : commission,
                                                    'max_pain' : 0,
                                                    'num_impacting_economic_calendars' : num_impacting_economic_calendars,
                                                    'max_camp': max_camp,
                                                    'post_move_price_change_percent' : post_move_price_change_percent
                                                }
                                all_trades.append(new_trade_0)
                                new_trade_0.update(_additional_trade_fields)

                                # Resets!
                                effective_tp_trailing_percent = float('inf')
                                lo_boillenger_upper_breached_history = lo_boillenger_upper_breached_cache.get(key, [])
                                lo_boillenger_upper_breached_history.clear()

                            if plot_timeseries:
                                '''
                                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html
                                linestyle='-' means solid line. If you don't supply linestyle, the vertical line wont show!!!
                                '''
                                all_canvas[f"{key}-param_id{algo_param['param_id']}"]['time_series_canvas'].axvline(x=lo_datetime, color='gray', linewidth=1, linestyle='-')
                                all_canvas[f"{key}-param_id{algo_param['param_id']}"]['time_series_canvas'].scatter([lo_datetime, lo_datetime], [lo_low, lo_high], color='gray')
                    
                    iter_info = f"param_id: {algo_param['param_id']}, {key} i: {i} {lo_datetime}, # trades: {len(all_trades)}, equity: {round(gloabl_state.total_equity,2)}"
                    if i%100==0 and i%1000!=0:
                        print(iter_info)
                    elif i%1000==0:
                        logger.info(iter_info)
                    
                if i==pd_lo_candles.shape[0]-1:
                    # HC
                    if this_ticker_current_position_usdt>0:
                        _close_open_positions(key, ticker, this_ticker_current_position_usdt, this_ticker_open_positions_side, current_position_usdt, unrealized_pnl, None, lo_row, 'HC', '', gloabl_state, all_trades, all_canvas, algo_param)
                    
            sorted_filtered_tickers.clear()
            sorted_filtered_tickers = None

        if gloabl_state.total_equity<target_order_notional:
            logger.warning(f"total_equity {gloabl_state.total_equity} < target_order_notional {target_order_notional} exiting prematurely on {lo_datetime}!!!")
            break

    if plot_timeseries:
        for exchange in exchanges:
            for ticker in tickers:
                key = f"{exchange.name}-{ticker}-param_id{algo_param['param_id']}"
                canvas = all_canvas[key]
                canvas['plt'].savefig(f"ts_{key.replace('/','').replace(':','')}.jpg", format='jpg', dpi=300)

    for reference_price_cache_file in reference_price_cache:
        reference_price_cache[reference_price_cache_file].sort_values("timestamp_ms", inplace=True)
        reference_price_cache[reference_price_cache_file].to_csv(reference_price_cache_file)

    num_tp = len([ x for x in all_trades if x['reason']=='TP'])
    num_sl = len([ x for x in all_trades if x['reason']=='SL'])
    num_hc_tp = len([ x for x in all_trades if x['reason']=='HC' and x['trade_pnl']>0 ] )
    num_hc_sl = len([ x for x in all_trades if x['reason']=='HC' and x['trade_pnl']<=0 ] )
    num_hc = num_hc_tp + num_hc_sl
                 
    return {
        'realized_pnl' : sum([x['trade_pnl'] for x in all_trades if 'trade_pnl' in x]) - gloabl_state.total_commission,
        'total_commission' : gloabl_state.total_commission,
        'hit_ratio' : (num_tp + num_hc_tp) / (num_tp + num_sl + num_hc),
        'num_tp' : num_tp,
        'num_sl' : num_sl,
        'num_hc' : num_hc,
        'num_entry' : num_tp + num_sl + num_hc,
        'trades' : all_trades,
        'exceptions' : exceptions
    }

def run_all_scenario(
    algo_params : List[Dict[str, Any]],
    exchanges : List[Exchange],

    order_notional_adj_func : Callable[..., float],
    allow_entry_initial_func : Callable[..., bool],
    allow_entry_final_func : Callable[..., bool],
    allow_slice_entry_func : Callable[..., bool],
    sl_adj_func : Callable[..., Dict[str, float]],
    trailing_stop_threshold_eval_func : Callable[..., Dict[str, float]],
    pnl_eval_func : Callable[..., Dict[str, float]],
    tp_eval_func : Callable[..., bool],
    sort_filter_universe_func : Callable[..., List[str]],

    logger,

    reference_start_dt : datetime = datetime(2021,1,1, tzinfo=timezone.utc),
) -> List[Dict]:
    all_exceptions = []

    start = datetime.now()
    max_test_end_date = start

    economic_calendars_file = algo_params[0]['economic_calendars_file']
    ecoevents_mapped_regions = algo_params[0]['ecoevents_mapped_regions']
    pd_economic_calendars = None
    economic_calendars_loaded : bool = False
    if os.path.isfile(economic_calendars_file):
        pd_economic_calendars  = pd.read_csv(economic_calendars_file)
        pd_economic_calendars = pd_economic_calendars[pd_economic_calendars.region.isin(ecoevents_mapped_regions)]
        economic_calendars_loaded = True if pd_economic_calendars.shape[0]>0 else False

    i : int = 1
    algo_results : List[Dict] = []
    best_realized_pnl, best_algo_result = 0, None
    for algo_param in algo_params:

        algo_result : Dict = {
            'param' : algo_param
        }
        algo_results.append(algo_result)

        # We calc test_end_date with 'lo' (not with 'hi'), we assume it'd be the same.
        test_start_date = algo_param['start_date']
        test_fetch_start_date = test_start_date
        lo_candle_size = algo_param['lo_candle_size']
        lo_num_intervals = int(lo_candle_size[0])
        lo_interval = lo_candle_size[-1]
        lo_how_many_candles = algo_param['lo_how_many_candles']
        if lo_interval=="m":
            test_end_date = test_start_date + timedelta(minutes=lo_num_intervals*lo_how_many_candles) 
            test_fetch_start_date = test_fetch_start_date - timedelta(minutes=algo_param['lo_stats_computed_over_how_many_candles']*2) 
            test_end_date_ref = test_end_date + timedelta(minutes=algo_param['lo_stats_computed_over_how_many_candles']*4) 
        elif lo_interval=="h":
            test_end_date = test_start_date + timedelta(hours=lo_num_intervals*lo_how_many_candles) 
            test_fetch_start_date = test_fetch_start_date - timedelta(hours=algo_param['lo_stats_computed_over_how_many_candles']*2) 
            test_end_date_ref = test_end_date + timedelta(hours=algo_param['lo_stats_computed_over_how_many_candles']*4) 
        elif lo_interval=="d":
            test_end_date = test_start_date + timedelta(days=lo_num_intervals*lo_how_many_candles) 
            test_fetch_start_date = test_fetch_start_date - timedelta(days=algo_param['lo_stats_computed_over_how_many_candles']*2) 
            test_end_date_ref = test_end_date + timedelta(days=algo_param['lo_stats_computed_over_how_many_candles']*4) 
        test_end_date = test_end_date if test_end_date < max_test_end_date else max_test_end_date
        test_end_date_ref = test_end_date_ref if test_end_date_ref < max_test_end_date else max_test_end_date
        cutoff_ts = int(test_fetch_start_date.timestamp()) # in seconds
        

        ####################################### STEP 1. Fetch candles (Because each test may have diff test_end_date, you need re-fetch candles for each algo_param) #######################################
        '''
        cutoff_ts in seconds, example '1668135382'

        exchanges[0].fetch_ohlcv('ETHUSDT', "1m", cutoff_ts)

        Candles format, first field is timestamp in ms:
            [
                [1502942400000, 301.13, 301.13, 301.13, 301.13, 0.42643],
                [1502942460000, 301.13, 301.13, 301.13, 301.13, 2.75787],
                [1502942520000, 300.0, 300.0, 300.0, 300.0, 0.0993],
                [1502942580000, 300.0, 300.0, 300.0, 300.0, 0.31389],
                ...
            ]
        '''
        delisted : List[str] = []

        data_fetch_start : float = time.time()
        
        # Fetch BTC
        reference_ticker : str = algo_param['reference_ticker']
        target_candle_file_name_fast : str = f'{reference_ticker.replace("^","").replace("/","").replace(":","")}_fast_candles_{datetime(2021,1,1, tzinfo=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date_ref.strftime("%Y-%m-%d-%H-%M-%S")}_1d.csv'
        target_candle_file_name_slow : str = f'{reference_ticker.replace("^","").replace("/","").replace(":","")}_slow_candles_{datetime(2021,1,1, tzinfo=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date_ref.strftime("%Y-%m-%d-%H-%M-%S")}_1d.csv'
        logger.info(f"reference_ticker: {reference_ticker}, target_candle_file_name_fast: {target_candle_file_name_fast}, target_candle_file_name_slow: {target_candle_file_name_slow}, reference_candles_file: {algo_param['reference_candles_file'] if 'reference_candles_file' in algo_param else '---'}")
        if algo_param['force_reload'] or not os.path.isfile(target_candle_file_name_fast):
            if algo_param['force_reload'] and 'reference_candles_file' in algo_param and algo_param['reference_candles_file'] and os.path.isfile(algo_param['reference_candles_file']):
                pd_ref_candles_fast = pd.read_csv(algo_param['reference_candles_file'])
                pd_ref_candles_slow : pd.DataFrame = pd_ref_candles_fast.copy(deep=True)
                logger.info(f"reference candles loaded from {algo_param['reference_candles_file']}")

            else:
                ref_candles : Dict[str, pd.DataFrame] = fetch_candles(
                                                                                        start_ts=int(reference_start_dt.timestamp()), 
                                                                                        end_ts=int(test_end_date_ref.timestamp()), 
                                                                                        exchange=exchanges[0], 
                                                                                        normalized_symbols=[reference_ticker], 
                                                                                        candle_size = '1d', 
                                                                                        num_candles_limit=algo_param['num_candles_limit'],
                                                                                        logger=logger,
                                                                                        cache_dir=algo_param['cache_candles'],
                                                                                        list_ts_field=exchanges[0].options['list_ts_field'] if 'list_ts_field' in exchanges[0].options else None
                                                                                        )
                logger.info(f"Reference candles fetched: {reference_ticker}, start: {reference_start_dt}, end: {test_end_date_ref}")
                pd_ref_candles_fast : pd.DataFrame = ref_candles[reference_ticker]
                pd_ref_candles_slow : pd.DataFrame = pd_ref_candles_fast.copy(deep=True)

            compute_candles_stats(pd_candles=pd_ref_candles_fast, boillenger_std_multiples=2, sliding_window_how_many_candles=algo_param['ref_ema_num_days_fast'], slow_fast_interval_ratio=int(algo_param['ref_ema_num_days_fast']/2), rsi_sliding_window_how_many_candles=algo_param['rsi_sliding_window_how_many_candles'], rsi_trend_sliding_window_how_many_candles=algo_param['rsi_trend_sliding_window_how_many_candles'], hurst_exp_window_how_many_candles=algo_param['hurst_exp_window_how_many_candles'], target_fib_level=algo_param['target_fib_level'], pypy_compat=algo_param['pypy_compat'])
            compute_candles_stats(pd_candles=pd_ref_candles_slow, boillenger_std_multiples=2, sliding_window_how_many_candles=algo_param['ref_ema_num_days_slow'], slow_fast_interval_ratio=int(algo_param['ref_ema_num_days_slow']/2), rsi_sliding_window_how_many_candles=algo_param['rsi_sliding_window_how_many_candles'], rsi_trend_sliding_window_how_many_candles=algo_param['rsi_trend_sliding_window_how_many_candles'], hurst_exp_window_how_many_candles=algo_param['hurst_exp_window_how_many_candles'], target_fib_level=algo_param['target_fib_level'], pypy_compat=algo_param['pypy_compat'])
            logger.info(f"Reference candles {reference_ticker} compute_candles_stats done.")
            
            pd_ref_candles_fast.to_csv(target_candle_file_name_fast)
            pd_ref_candles_slow.to_csv(target_candle_file_name_slow)

        else:
            pd_ref_candles_fast : pd.DataFrame = pd.read_csv(target_candle_file_name_fast)
            pd_ref_candles_slow : pd.DataFrame = pd.read_csv(target_candle_file_name_slow)
            fix_column_types(pd_ref_candles_fast)
            fix_column_types(pd_ref_candles_slow)
            logger.info(f"Reference candles {reference_ticker} loaded from target_candle_file_name_fast: {target_candle_file_name_fast}, target_candle_file_name_slow: {target_candle_file_name_slow}")

        total_seconds = (test_end_date_ref - test_start_date).total_seconds()
        total_hours = total_seconds / 3600
        total_days = total_hours / 24
        sliding_window_how_many_candles : int = 0
        sliding_window_how_many_candles = int(total_days / algo_param['sliding_window_ratio'])

        ref_candles_partitions, pd_hi_candles_partitions, pd_lo_candles_partitions = None, None, None
        if not algo_param['pypy_compat']:
            ref_candles_partitions = partition_sliding_window(
                    pd_candles = pd_ref_candles_fast, 
                    sliding_window_how_many_candles = sliding_window_how_many_candles, 
                    smoothing_window_size_ratio = algo_param['smoothing_window_size_ratio'], 
                    linregress_stderr_threshold = algo_param['linregress_stderr_threshold'],
                    max_recur_depth = algo_param['max_recur_depth'], 
                    min_segment_size_how_many_candles = algo_param['min_segment_size_how_many_candles'], 
                    segment_consolidate_slope_ratio_threshold = algo_param['segment_consolidate_slope_ratio_threshold'],
                    sideway_price_condition_threshold = algo_param['sideway_price_condition_threshold']
                )
            candle_segments_jpg_file_name : str = f'{reference_ticker.replace("^","").replace("/","").replace(":","")}_refcandles_w_segments_{datetime(2021,1,1, tzinfo=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date_ref.strftime("%Y-%m-%d-%H-%M-%S")}_1d.jpg'
            plot_segments(pd_ref_candles_fast, ref_candles_partitions, candle_segments_jpg_file_name)

            candle_segments_file_name : str = f'{reference_ticker.replace("^","").replace("/","").replace(":","")}_refcandles_w_segments_{datetime(2021,1,1, tzinfo=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date_ref.strftime("%Y-%m-%d-%H-%M-%S")}_1d.csv'
            pd_ref_candles_segments = segments_to_df(ref_candles_partitions['segments'])
            pd_ref_candles_segments.to_csv(candle_segments_file_name)
        
        all_exchange_candles : Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
        for exchange in exchanges:
            markets = exchange.load_markets()
            if exchange.name not in all_exchange_candles:
                all_exchange_candles[exchange.name]  = {} 

                if algo_param['white_list_tickers']:
                    tickers = algo_param['white_list_tickers']
                else:
                    tickers = list(markets.keys())

                for ticker in tickers:
                    if ticker not in markets:
                        err_msg = f"{ticker}: {'no longer in markets'}"
                        logger.error(err_msg)
                        delisted.append(ticker)
                    else:    
                        all_exchange_candles[exchange.name][ticker] = {}

                        _ticker = ticker.split(":")[0].replace("/","")
                        total_seconds = (test_end_date - test_fetch_start_date).total_seconds()
                        total_hours = total_seconds / 3600
                        total_days = total_hours / 24
                        sliding_window_how_many_candles : int = 0
                        sliding_window_how_many_candles = int(total_days / algo_param['sliding_window_ratio'])
                    
                        pd_hi_candles = None
                        target_candle_file_name : str = f'{_ticker}_candles_{test_fetch_start_date.strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date.strftime("%Y-%m-%d-%H-%M-%S")}_{algo_param["hi_candle_size"]}.csv'
                        if algo_param['force_reload'] or not os.path.isfile(target_candle_file_name):
                            if algo_param['force_reload'] and 'hi_candles_file' in algo_param and algo_param['hi_candles_file'] and os.path.isfile(algo_param['hi_candles_file']):
                                pd_hi_candles : pd.DataFrame = pd.read_csv(algo_param['hi_candles_file'])

                            else:
                                hi_candles : Dict[str, pd.DataFrame] = fetch_candles(
                                                                                                start_ts=cutoff_ts, 
                                                                                                end_ts=int(test_end_date.timestamp()), 
                                                                                                exchange=exchange, normalized_symbols=[ ticker ], 
                                                                                                candle_size = algo_param['hi_candle_size'], 
                                                                                                num_candles_limit=algo_param['num_candles_limit'],
                                                                                                logger=logger,
                                                                                                cache_dir=algo_param['cache_candles'],
                                                                                                list_ts_field=exchange.options['list_ts_field']
                                                                                                )
                                pd_hi_candles : pd.DataFrame = hi_candles[ticker]
                                logger.info(f"pd_hi_candles fetched: {ticker} {pd_hi_candles.shape}, start: {cutoff_ts}, end: {int(test_end_date.timestamp())}")
                            compute_candles_stats(pd_candles=pd_hi_candles, boillenger_std_multiples=algo_param['boillenger_std_multiples'], sliding_window_how_many_candles=algo_param['hi_stats_computed_over_how_many_candles'], slow_fast_interval_ratio=(algo_param['hi_stats_computed_over_how_many_candles']/algo_param['hi_ma_short_interval']), rsi_sliding_window_how_many_candles=algo_param['rsi_sliding_window_how_many_candles'], rsi_trend_sliding_window_how_many_candles=algo_param['rsi_trend_sliding_window_how_many_candles'], hurst_exp_window_how_many_candles=algo_param['hurst_exp_window_how_many_candles'], target_fib_level=algo_param['target_fib_level'], pypy_compat=algo_param['pypy_compat'])
                            logger.info(f"pd_hi_candles {ticker} compute_candles_stats done: {target_candle_file_name}")
                            pd_hi_candles.to_csv(target_candle_file_name)

                            if pd_hi_candles is not None and pd_hi_candles.shape[0]>0:
                                first_candle_datetime = datetime.fromtimestamp(pd_hi_candles.iloc[0]['timestamp_ms']/1000)
                                last_candle_datetime = datetime.fromtimestamp(pd_hi_candles.iloc[-1]['timestamp_ms']/1000)

                                assert(last_candle_datetime>first_candle_datetime)
                            else:
                                err_msg = f"{ticker} no hi candles?"
                                logger.error(err_msg)
                        else:
                            pd_hi_candles : pd.DataFrame = pd.read_csv(target_candle_file_name)
                            fix_column_types(pd_hi_candles)
                            logger.info(f"pd_hi_candles {ticker} {pd_hi_candles.shape} loaded from {target_candle_file_name}")

                        if not algo_param['pypy_compat']:
                            pd_hi_candles_partitions = partition_sliding_window(
                                        pd_candles = pd_hi_candles, 
                                        sliding_window_how_many_candles = sliding_window_how_many_candles, 
                                        smoothing_window_size_ratio = algo_param['smoothing_window_size_ratio'], 
                                        linregress_stderr_threshold = algo_param['linregress_stderr_threshold'],
                                        max_recur_depth = algo_param['max_recur_depth'], 
                                        min_segment_size_how_many_candles = algo_param['min_segment_size_how_many_candles'], 
                                        segment_consolidate_slope_ratio_threshold = algo_param['segment_consolidate_slope_ratio_threshold'],
                                        sideway_price_condition_threshold = algo_param['sideway_price_condition_threshold']
                                    )
                            candle_segments_jpg_file_name : str = f'{_ticker}_hicandles_w_segments_{test_fetch_start_date.strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date.strftime("%Y-%m-%d-%H-%M-%S")}_{algo_param["hi_candle_size"]}.jpg'
                            plot_segments(pd_hi_candles, pd_hi_candles_partitions, candle_segments_jpg_file_name)

                            candle_segments_file_name : str = f'{_ticker}_hicandles_w_segments_{test_fetch_start_date.strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date.strftime("%Y-%m-%d-%H-%M-%S")}_{algo_param["hi_candle_size"]}.csv'
                            pd_hi_candles_segments = segments_to_df(pd_hi_candles_partitions['segments'])
                            pd_hi_candles_segments.to_csv(candle_segments_file_name)

                        pd_lo_candles = None
                        _ticker = ticker.split(":")[0].replace("/","")
                        target_candle_file_name : str = f'{_ticker}_candles_{test_fetch_start_date.strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date.strftime("%Y-%m-%d-%H-%M-%S")}_{algo_param["lo_candle_size"]}.csv'
                        if algo_param['force_reload'] or not os.path.isfile(target_candle_file_name):
                            if algo_param['force_reload'] and 'lo_candles_file' in algo_param and algo_param['lo_candles_file'] and os.path.isfile(algo_param['lo_candles_file']):
                                pd_lo_candles : pd.DataFrame = pd.read_csv(algo_param['lo_candles_file'])

                            else:
                                lo_candles : Dict[str, pd.DataFrame] = fetch_candles(
                                                                                                start_ts=cutoff_ts, 
                                                                                                end_ts=int(test_end_date.timestamp()), 
                                                                                                exchange=exchange, normalized_symbols=[ ticker ], 
                                                                                                candle_size = algo_param['lo_candle_size'], 
                                                                                                num_candles_limit=algo_param['num_candles_limit'],
                                                                                                logger=logger,
                                                                                                cache_dir=algo_param['cache_candles'],
                                                                                                list_ts_field=exchange.options['list_ts_field']
                                                                                                )
                                pd_lo_candles : pd.DataFrame = lo_candles[ticker]
                                logger.info(f"pd_lo_candles fetched: {ticker} {pd_lo_candles.shape}, start: {cutoff_ts}, end: {int(test_end_date.timestamp())}")
                            compute_candles_stats(pd_candles=pd_lo_candles, boillenger_std_multiples=algo_param['boillenger_std_multiples'], sliding_window_how_many_candles=algo_param['lo_stats_computed_over_how_many_candles'], slow_fast_interval_ratio=(algo_param['lo_stats_computed_over_how_many_candles']/algo_param['lo_ma_short_interval']), rsi_sliding_window_how_many_candles=algo_param['rsi_sliding_window_how_many_candles'], rsi_trend_sliding_window_how_many_candles=algo_param['rsi_trend_sliding_window_how_many_candles'], hurst_exp_window_how_many_candles=algo_param['hurst_exp_window_how_many_candles'], target_fib_level=algo_param['target_fib_level'], pypy_compat=algo_param['pypy_compat'])
                            logger.info(f"pd_lo_candles {ticker} compute_candles_stats done. {target_candle_file_name}")
                            pd_lo_candles.to_csv(target_candle_file_name)
                            
                            if pd_lo_candles is not None and pd_lo_candles.shape[0]>0:
                                first_candle_datetime = datetime.fromtimestamp(pd_lo_candles.iloc[0]['timestamp_ms']/1000)
                                last_candle_datetime = datetime.fromtimestamp(pd_lo_candles.iloc[-1]['timestamp_ms']/1000)
                                
                                assert(last_candle_datetime>first_candle_datetime)
                            else:
                                err_msg = f"{ticker} no lo candles?"
                                logger.error(err_msg)
                        else:
                            pd_lo_candles : pd.DataFrame = pd.read_csv(target_candle_file_name)
                            fix_column_types(pd_lo_candles)
                            logger.info(f"pd_lo_candles {ticker} {pd_lo_candles.shape} loaded from {target_candle_file_name}")

                        if not algo_param['pypy_compat']:
                            pd_lo_candles_partitions = partition_sliding_window(
                                        pd_candles = pd_lo_candles, 
                                        sliding_window_how_many_candles = sliding_window_how_many_candles, 
                                        smoothing_window_size_ratio = algo_param['smoothing_window_size_ratio'], 
                                        linregress_stderr_threshold = algo_param['linregress_stderr_threshold'],
                                        max_recur_depth = algo_param['max_recur_depth'], 
                                        min_segment_size_how_many_candles = algo_param['min_segment_size_how_many_candles'], 
                                        segment_consolidate_slope_ratio_threshold = algo_param['segment_consolidate_slope_ratio_threshold'],
                                        sideway_price_condition_threshold = algo_param['sideway_price_condition_threshold']
                                    )
                            candle_segments_jpg_file_name : str = f'{_ticker}_locandles_w_segments_{test_fetch_start_date.strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date.strftime("%Y-%m-%d-%H-%M-%S")}_{algo_param["lo_candle_size"]}.jpg'
                            plot_segments(pd_lo_candles, pd_lo_candles_partitions, candle_segments_jpg_file_name)

                            candle_segments_file_name : str = f'{_ticker}_locandles_w_segments_{test_fetch_start_date.strftime("%Y-%m-%d-%H-%M-%S")}_{test_end_date.strftime("%Y-%m-%d-%H-%M-%S")}_{algo_param["hi_candle_size"]}.csv'
                            pd_lo_candles_segments = segments_to_df(pd_lo_candles_partitions['segments'])
                            pd_lo_candles_segments.to_csv(candle_segments_file_name)
                    
                        all_exchange_candles[exchange.name][ticker]['hi_candles'] = pd_hi_candles
                        all_exchange_candles[exchange.name][ticker]['lo_candles'] = pd_lo_candles

        data_fetch_finish : float = time.time()

        ####################################### STEP 2. Trade simulation #######################################
        logger.info(f"Start run_scenario")
        scenario_start : float = time.time()
        result = run_scenario(
                algo_param=algo_param, 
                exchanges=exchanges, 
                all_exchange_candles=all_exchange_candles, 
                pd_ref_candles_fast=pd_ref_candles_fast, 
                pd_ref_candles_slow=pd_ref_candles_slow, 
                ref_candles_partitions=ref_candles_partitions,
                pd_hi_candles_partitions=pd_hi_candles_partitions,
                pd_lo_candles_partitions=pd_lo_candles_partitions,
                economic_calendars_loaded=economic_calendars_loaded,
                pd_economic_calendars=pd_economic_calendars,
                tickers=tickers, 

                order_notional_adj_func=order_notional_adj_func,
                allow_entry_initial_func=allow_entry_initial_func,
                allow_entry_final_func=allow_entry_final_func,
                allow_slice_entry_func=allow_slice_entry_func,
                sl_adj_func=sl_adj_func,
                trailing_stop_threshold_eval_func=trailing_stop_threshold_eval_func,
                pnl_eval_func=pnl_eval_func,
                tp_eval_func=tp_eval_func,
                sort_filter_universe_func=sort_filter_universe_func,

                logger=logger,

                pypy_compat=algo_param['pypy_compat'],
                plot_timeseries=True
            )
        scenario_finish = time.time()
        
        data_fetch_elapsed_ms = (data_fetch_finish - data_fetch_start) * 1000
        scenario_elapsed_ms = (scenario_finish - scenario_start) * 1000

        logger.info(f"Done run_scenario. data_fetch_elapsed_ms: {data_fetch_elapsed_ms} ms, scenario_elapsed_ms: {scenario_elapsed_ms}")
        
        algo_result['orders'] = result['trades']
        result.pop('trades')
        algo_result['summary'] = {
            # Key parameters
            'initial_cash' : algo_param['initial_cash'],
            'entry_percent_initial_cash' : algo_param['entry_percent_initial_cash'],
            'strategy_mode' : algo_param['strategy_mode'],
            'ref_ema_num_days_fast' : algo_param['ref_ema_num_days_fast'],
            'ref_ema_num_days_slow' : algo_param['ref_ema_num_days_slow'],
            'long_above_ref_ema_short_below' : algo_param['long_above_ref_ema_short_below'],
            'ref_price_vs_ema_percent_threshold' : algo_param['ref_price_vs_ema_percent_threshold'] if 'ref_price_vs_ema_percent_threshold' in algo_param else None,
            'rsi_upper_threshold' : algo_param['rsi_upper_threshold'],
            'rsi_lower_threshold' : algo_param['rsi_lower_threshold'],
            'boillenger_std_multiples' : algo_param['boillenger_std_multiples'],
            'ema_short_slope_threshold' : algo_param['ema_short_slope_threshold'] if 'ema_short_slope_threshold' in algo_param else None,
            'num_intervals_block_pending_ecoevents' : algo_param['num_intervals_block_pending_ecoevents'],
            'num_intervals_current_ecoevents' : algo_param['num_intervals_current_ecoevents'],
            'sl_hard_percent' : algo_param['sl_hard_percent'],
            'sl_percent_trailing' : algo_param['sl_percent_trailing'],
            'use_gradual_tightened_trailing_stops' : algo_param['use_gradual_tightened_trailing_stops'],
            'sl_num_intervals_delay' : algo_param['sl_num_intervals_delay'],
            'tp_min_percent' : algo_param['tp_min_percent'],
            'tp_max_percent' : algo_param['tp_max_percent'],
            'asymmetric_tp_bps' : algo_param['asymmetric_tp_bps'],

            # Key output
            'realized_pnl' : result['realized_pnl'], # Commission already taken out
            'total_commission' : result['total_commission'],
            'hit_ratio' : result['hit_ratio'],
            'num_tp' : result['num_tp'],
            'num_sl' : result['num_sl'],
            'num_hc' : result['num_hc'],
            'num_entry' : result['num_entry'],
            'data_fetch_elapsed_ms' : data_fetch_elapsed_ms,
            'scenario_elapsed_ms' : scenario_elapsed_ms,
            'num_exceptions' : len(result['exceptions'])
        }

        all_exceptions = all_exceptions + list(result['exceptions'].items())
        logger.error(list(result['exceptions'].items()))

        logger.info(f"Done ({i}/{len(algo_params)}) {algo_param['name_exclude_start_date']}")
        logger.info(json.dumps(algo_result['summary'], indent=4))

        if result['realized_pnl']>best_realized_pnl or not best_algo_result:
            best_algo_result = algo_result['summary']
        
        i = i + 1

    finish = datetime.now()
    elapsed = (finish-start).seconds


    logger.info(f"Backtest done in {elapsed}sec over {len(algo_params)} scenario's with start_date {test_start_date} over {len(exchanges)} exchange(s) and {len(tickers)} tickers.")

    logger.info(f"*** Best result realized_pnl: {best_algo_result['realized_pnl']}")
    logger.info(json.dumps(best_algo_result, indent=4))

    pd_results = pd.DataFrame([ x['summary'] for x in algo_results])
    pd_results.loc['avg', 'realized_pnl'] = pd_results['realized_pnl'].mean(numeric_only=True, axis=0)
    pd_results.loc['avg', 'total_commission'] = pd_results['total_commission'].mean(numeric_only=True, axis=0)
    pd_results.loc['avg', 'hit_ratio'] = pd_results['hit_ratio'].mean(numeric_only=True, axis=0)

    return algo_results

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_reload", help="Reload candles? Both candles and TA previously computed will be loaded from disk. Y or N (default)", default=False)
    parser.add_argument("--white_list_tickers", help="Comma seperated list, example: BTC/USDT:USDT,ETH/USDT:USDT,XRP/USDT:USDT ", default="BTC/USDT:USDT")
    parser.add_argument("--reference_ticker", help="This is ticker for bull / bear determination. The Northstar.", default="BTC/USDT:USDT")
    parser.add_argument("--block_entries_on_impacting_ecoevents", help="Block entries on economic event? Y (default) or N", default=True)
    parser.add_argument("--enable_sliced_entry", help="Block entries on economic event? Y or N (default)", default=False)
    parser.add_argument("--asymmetric_tp_bps", help="A positive asymmetric_tp_bps means you are taking deeper TPs. A negative asymmetric_tp_bps means shallower", default=0)
    args = parser.parse_args()
    
    if args.force_reload:
        if args.force_reload=='Y':
            force_reload = True
        else:
            force_reload = False
    else:
        force_reload = False

    if args.white_list_tickers:
        white_list_tickers = args.white_list_tickers.split(',')
        
    reference_ticker = args.reference_ticker if args.reference_ticker else white_list_tickers[0]

    if args.block_entries_on_impacting_ecoevents:
        if args.block_entries_on_impacting_ecoevents=='Y':
            block_entries_on_impacting_ecoevents = True
        else:
            block_entries_on_impacting_ecoevents = False
    else:
        block_entries_on_impacting_ecoevents = True

    if args.enable_sliced_entry:
        if args.enable_sliced_entry=='Y':
            enable_sliced_entry = True
        else:
            enable_sliced_entry = False
    else:
        enable_sliced_entry = False

    asymmetric_tp_bps = int(args.asymmetric_tp_bps)

    return {
        'force_reload': force_reload,
        'white_list_tickers' : white_list_tickers,
        'reference_ticker' : reference_ticker,
        'block_entries_on_impacting_ecoevents' : block_entries_on_impacting_ecoevents,
        'enable_sliced_entry'  : enable_sliced_entry,
        'asymmetric_tp_bps' : asymmetric_tp_bps
    }

def dump_trades_to_disk(
    algo_results,
    filename,
    logger
):
    flattenned_trades : List[Dict[str, Any]]= []
    for algo_result in algo_results:
        for order in algo_result['orders']:
            try:
                order['name'] = algo_result['param']['name']
                order['name_exclude_start_date'] = algo_result['param']['name_exclude_start_date']
                
                order['initial_cash'] = algo_result['param']['initial_cash']
                order['entry_percent_initial_cash'] = algo_result['param']['entry_percent_initial_cash']
                order['clip_order_notional_to_best_volumes'] = algo_result['param']['clip_order_notional_to_best_volumes']
                order['target_position_size_percent_total_equity'] = algo_result['param']['target_position_size_percent_total_equity']

                order['reference_ticker'] = algo_result['param']['reference_ticker']
                order['strategy_mode'] = algo_result['param']['strategy_mode']
                order['boillenger_std_multiples'] = algo_result['param']['boillenger_std_multiples']
                order['ema_short_slope_threshold'] = algo_result['param']['ema_short_slope_threshold'] if 'ema_short_slope_threshold' in algo_result['param'] else None
                order['how_many_last_candles'] = algo_result['param']['how_many_last_candles']
                order['last_candles_timeframe'] = algo_result['param']['last_candles_timeframe']
                order['enable_wait_entry'] = algo_result['param']['enable_wait_entry'] if 'enable_wait_entry' in algo_result['param'] else None
                order['allow_entry_sit_bb'] = algo_result['param']['allow_entry_sit_bb']  if 'allow_entry_sit_bb' in algo_result['param'] else None
                order['enable_sliced_entry'] = algo_result['param']['enable_sliced_entry']
                order['adj_sl_on_ecoevents'] = algo_result['param']['adj_sl_on_ecoevents']
                order['block_entries_on_impacting_ecoevents'] = algo_result['param']['block_entries_on_impacting_ecoevents']
                order['num_intervals_block_pending_ecoevents'] = algo_result['param']['num_intervals_block_pending_ecoevents']
                order['num_intervals_current_ecoevents'] = algo_result['param']['num_intervals_current_ecoevents']
                order['enable_hi_timeframe_confirm'] = algo_result['param']['enable_hi_timeframe_confirm'] if 'enable_hi_timeframe_confirm' in algo_result['param'] else None
                order['sl_num_intervals_delay'] = algo_result['param']['sl_num_intervals_delay']
                order['sl_hard_percent'] = algo_result['param']['sl_hard_percent']
                order['sl_percent_trailing'] = algo_result['param']['sl_percent_trailing']
                order['use_gradual_tightened_trailing_stops'] = algo_result['param']['use_gradual_tightened_trailing_stops']
                order['tp_min_percent'] = algo_result['param']['tp_min_percent']
                order['tp_max_percent'] = algo_result['param']['tp_max_percent']
                order['asymmetric_tp_bps'] = algo_result['param']['asymmetric_tp_bps']
            
                order['hi_candle_size'] = algo_result['param']['hi_candle_size']
                order['hi_stats_computed_over_how_many_candles'] = algo_result['param']['hi_stats_computed_over_how_many_candles']
                order['hi_how_many_candles'] = algo_result['param']['hi_how_many_candles']
                order['hi_ma_short_interval'] = algo_result['param']['hi_ma_short_interval']
                order['hi_ma_long_interval'] = algo_result['param']['hi_ma_long_interval']

                order['lo_candle_size'] = algo_result['param']['lo_candle_size']
                order['lo_stats_computed_over_how_many_candles'] = algo_result['param']['lo_stats_computed_over_how_many_candles']
                order['lo_how_many_candles'] = algo_result['param']['lo_how_many_candles']
                order['lo_ma_short_interval'] = algo_result['param']['lo_ma_short_interval']
                order['lo_ma_long_interval'] = algo_result['param']['lo_ma_long_interval']

                order['target_fib_level'] = algo_result['param']['target_fib_level']
                order['rsi_sliding_window_how_many_candles'] = algo_result['param']['rsi_sliding_window_how_many_candles']
                order['rsi_trend_sliding_window_how_many_candles'] = algo_result['param']['rsi_trend_sliding_window_how_many_candles']
                order['hurst_exp_window_how_many_candles'] = algo_result['param']['hurst_exp_window_how_many_candles']

                order['ref_ema_num_days_fast'] = algo_result['param']['ref_ema_num_days_fast']
                order['re_ema_num_days_slow'] = algo_result['param']['ref_ema_num_days_slow']
                order['long_above_ref_ema_short_below'] = algo_result['param']['long_above_ref_ema_short_below']
                order['ref_price_vs_ema_percent_threshold'] = algo_result['param']['ref_price_vs_ema_percent_threshold'] if 'ref_price_vs_ema_percent_threshold' in algo_result['param'] else None
                order['rsi_upper_threshold'] = algo_result['param']['rsi_upper_threshold']
                order['rsi_lower_threshold'] = algo_result['param']['rsi_lower_threshold']

                order['id'] = str(uuid.uuid4())
                order['trade_year'] = order['trade_datetime'].year
                order['trade_month'] = order['trade_datetime'].month
                order['trade_day'] = order['trade_datetime'].day
                order['trade_dayofweek'] = order['dayofweek']
                order['trade_week_of_month'] = timestamp_to_week_of_month(
                    int(order['trade_datetime'].timestamp() * 1000)
                )
                
                flattenned_trades.append(order)

            except Exception as error:
                logger.error(f"Error while processing flattenned trades! {error}")

    if len(flattenned_trades)>0:
        pd_flattenned_trades = pd.DataFrame(flattenned_trades)
        pd_flattenned_trades.to_csv(filename)

        logger.info(f"Trade extract: {filename}")