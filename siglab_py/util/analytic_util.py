import tzlocal
from datetime import datetime, timezone
from typing import List, Dict, Union, NoReturn, Any, Tuple
from enum import Enum
from pathlib import Path
import math
import pandas as pd
import numpy as np
from hurst import compute_Hc # compatible with pypy

from ccxt.base.exchange import Exchange as CcxtExchange
from ccxt import deribit

from siglab_py.util.simple_math import bucket_series, bucketize_val
from siglab_py.util.market_data_util import fix_column_types
from siglab_py.constants import TrendDirection

def classify_candle(
    candle : pd.Series,
    min_candle_height_ratio : float = 3
) -> Union[str, None]:
    candle_class : Union[str, None] = None
    open = candle['open']
    high = candle['high']
    low = candle['low']
    close = candle['close']
    candle_full_height = high - low # always positive
    candle_body_height = close - open # can be negative
    candle_height_ratio = candle_full_height / abs(candle_body_height) if candle_body_height!=0 else float('inf')

    if (
        candle_height_ratio>=min_candle_height_ratio
        and close>low
    ):
        candle_class = 'hammer'
    elif (
        candle_height_ratio>=min_candle_height_ratio
        and close<high
    ):
        candle_class = 'shooting_star'
    # Keep add more ...

    return candle_class

# Fibonacci
MAGIC_FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.00, 1.618, 2.618, 3.618, 4.236]

def estimate_fib_retracement(
        swing_low: float, 
        swing_low_idx: int, 
        swing_high: float, 
        swing_high_idx: int,
        target_fib_level: float = 0.618
    ) -> float:
        price_range = swing_high - swing_low

        # https://blog.quantinsti.com/fibonacci-retracement-trading-strategy-python/
        if swing_low_idx < swing_high_idx:
            retracement_price = swing_high - (price_range * target_fib_level)
        else:
            retracement_price = swing_low + (price_range * target_fib_level)
            
        return retracement_price

def calculate_slope(
    pd_data : pd.DataFrame,
    src_col_name : str,
    slope_col_name : str,
    sliding_window_how_many_candles : int
):
    import statsmodels.api as sm # in-compatible with pypy

    X = sm.add_constant(range(len(pd_data[src_col_name])))
    rolling_slope = pd_data[src_col_name].rolling(window=sliding_window_how_many_candles).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_data[slope_col_name] = rolling_slope
    max_abs_slope = pd_data[slope_col_name].abs().rolling(window=sliding_window_how_many_candles).max()
    pd_data[f"normalized_{slope_col_name}"] = pd_data[slope_col_name] / max_abs_slope
    normalized_slope_rolling = pd_data[f"normalized_{slope_col_name}"].rolling(window=sliding_window_how_many_candles)
    pd_data[f"normalized_{slope_col_name}_min"] = normalized_slope_rolling.min()
    pd_data[f"normalized_{slope_col_name}_max"] = normalized_slope_rolling.max()
    pd_data[f"normalized_{slope_col_name}_idmin"] = normalized_slope_rolling.apply(lambda x : x.idxmin())
    pd_data[f"normalized_{slope_col_name}_idmax"] = normalized_slope_rolling.apply(lambda x : x.idxmax())

def trend_from_highs(series: np.ndarray) -> float:
    valid_series = series[~np.isnan(series)]
    unique_maxima = valid_series[np.concatenate(([True], np.diff(valid_series) != 0))]
    if len(unique_maxima) < 2:
        return TrendDirection.UNDEFINED.value
    first, last = unique_maxima[0], unique_maxima[-1]
    if first > last:
        return TrendDirection.LOWER_HIGHS.value
    elif first < last:
        return TrendDirection.HIGHER_HIGHS.value
    else:
        return TrendDirection.SIDEWAYS.value

def trend_from_lows(series: np.ndarray) -> float:
    valid_series = series[~np.isnan(series)]
    unique_minima = valid_series[np.concatenate(([True], np.diff(valid_series) != 0))]
    if len(unique_minima) < 2:
        return TrendDirection.UNDEFINED.value
    first, last = unique_minima[0], unique_minima[-1]
    if first > last:
        return TrendDirection.LOWER_LOWS.value
    elif first < last:
        return TrendDirection.HIGHER_LOWS.value
    else:
        return TrendDirection.SIDEWAYS.value
    
    
'''
compute_candles_stats will calculate typical/basic technical indicators using in many trading strategies:
    a. Basic SMA/EMAs (And slopes)
    b. EMA crosses
    c. ATR
    d. Boillenger bands (Yes incorrect spelling sorry)
    e. FVG
    f. Hurst Exponent
    g. RSI, MFI
    h. MACD
    i. Fibonacci
    j. Inflections points: where 'close' crosses EMA from above or below.

Parameters:
    a. boillenger_std_multiples: For boillenger upper and lower calc
    b. sliding_window_how_many_candles: Moving averages calculation
    c. rsi_ema: RSI calculated using EMA or SMA?
    d. boillenger_ema: Boillenger calculated using SMA or EMA?
    e. slow_fast_interval_ratios
        MACD calculated using two moving averages. 
        Slow line using 'sliding_window_how_many_candles' intervals. 
        Fast line using 'sliding_window_how_many_candles/slow_fast_interval_ratios' intervals. 
        Example, 
            if Slow line is calculated using 24 candles and short_long_interval_ratios = 3, 
                then Fast line is calculated using 24/3 = 8 candles.
'''
def compute_candles_stats(
        pd_candles : pd.DataFrame,
        boillenger_std_multiples : float,
        sliding_window_how_many_candles : int,
        rsi_ema : bool = True,
        boillenger_ema : bool = False,
        slow_fast_interval_ratio : float = 3,
        rsi_sliding_window_how_many_candles : int = 14, # RSI standard 14
        rsi_trend_sliding_window_how_many_candles : int = 24*7, # This is for purpose of RSI trend identification (Locating local peaks/troughs in RSI). This should typically be multiples of 'rsi_sliding_window_how_many_candles'.
        hurst_exp_window_how_many_candles : Union[int, None] = None, # Hurst exp standard 100-200
        boillenger_std_multiples_for_aggressive_moves_detect : int = 3, # Aggressive moves if candle low/high breaches boillenger bands from 3 standard deviations.
        target_fib_level : float = 0.618,
        pypy_compat : bool = True
        ):
    BUCKETS_m0_100 = bucket_series(
						values=list([i for i in range(0,100)]), 
						outlier_threshold_percent=10, 
						level_granularity=0.1
					)
    
    pd_candles['candle_height'] = pd_candles['high'] - pd_candles['low']
    pd_candles['candle_body_height'] = pd_candles['close'] - pd_candles['open']

    '''
    market_data_gizmo inserted dummy lines --> Need exclude those or "TypeError: unorderable types for comparison": pd_btc_candles = pd_btc_candles[pd_btc_candles.close.notnull()]

    pd_btc_candles.loc[
        (pd_btc_candles['close_above_or_below_ema'] != pd_btc_candles['close_above_or_below_ema'].shift(1)) & 
        (abs(pd_btc_candles['gap_close_vs_ema']) > avg_gap_close_vs_ema), 
        'close_vs_ema_inflection'
    ] = np.sign(pd_btc_candles['close'] - pd_btc_candles['ema_long_periods']) <-- TypeError: unorderable types for comparison
    '''
    # pd_candles = pd_candles[pd_candles.close.notnull()] # Don't make a copy. Drop in-place
    
    fix_column_types(pd_candles) # Do this AFTER filtering. Or you'd mess up index, introduce error around idmax, idmin. fix_column_types will drop all 'unnamed' columns and reset_index.

    pd_candles['is_green'] =  pd_candles['close'] >= pd_candles['open']

    pd_candles['candle_class'] = pd_candles.apply(lambda row: classify_candle(row), axis=1) # type: ignore

    close_short_periods_rolling = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio))
    close_long_periods_rolling = pd_candles['close'].rolling(window=sliding_window_how_many_candles)
    close_short_periods_ewm = pd_candles['close'].ewm(span=int(sliding_window_how_many_candles/slow_fast_interval_ratio), adjust=False)
    close_long_periods_ewm = pd_candles['close'].ewm(span=sliding_window_how_many_candles, adjust=False)

    pd_candles['pct_change_close'] = pd_candles['close'].pct_change() * 100
    pd_candles['sma_short_periods'] = close_short_periods_rolling.mean()
    pd_candles['sma_long_periods'] = close_long_periods_rolling.mean()
    pd_candles['ema_short_periods'] = close_short_periods_ewm.mean()
    pd_candles['ema_long_periods'] = close_long_periods_ewm.mean()
    pd_candles['ema_close'] = pd_candles['ema_long_periods'] # Alias, shorter name
    pd_candles['std'] = close_long_periods_rolling.std()
    pd_candles['std_percent'] = pd_candles['std'] / pd_candles['ema_close'] * 100

    pd_candles['vwap_short_periods'] = (pd_candles['close'] * pd_candles['volume']).rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).sum() / pd_candles['volume'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).sum()
    pd_candles['vwap_long_periods'] = (pd_candles['close'] * pd_candles['volume']).rolling(window=sliding_window_how_many_candles).sum() / pd_candles['volume'].rolling(window=sliding_window_how_many_candles).sum()

    pd_candles['candle_height_percent'] = pd_candles['candle_height'] / pd_candles['ema_close'] * 100
    pd_candles['candle_height_percent_rounded'] = pd_candles['candle_height_percent'].round().astype('Int64')

    pd_candles['candle_body_height_percent'] = pd_candles['candle_body_height'] / pd_candles['ema_close'] * 100
    pd_candles['candle_body_height_percent_rounded'] = pd_candles['candle_body_height_percent'].round().astype('Int64')

    '''
    To annualize volatility:
        if candle_interval == '1m':
            annualization_factor = np.sqrt(365 * 24 * 60)  # 1-minute candles
        elif candle_interval == '1h':
            annualization_factor = np.sqrt(365 * 24)       # 1-hour candles
        elif candle_interval == '1d':
            annualization_factor = np.sqrt(365)            # 1-day candles
        pd_candles['annualized_volatility'] = (
            pd_candles['interval_historical_volatility'] * annualization_factor
        )

    Why log return? Trading Dude https://python.plainenglish.io/stop-using-percentage-returns-logarithmic-returns-explained-with-code-64a4634b883a
    '''
    pd_candles['log_return'] = np.log(pd_candles['close'] / pd_candles['close'].shift(1))
    pd_candles['interval_hist_vol'] = pd_candles['log_return'].rolling(window=sliding_window_how_many_candles).std()
    
    time_gap_sec = int(pd_candles['timestamp_ms'].iloc[1] - pd_candles['timestamp_ms'].iloc[0])/1000
    seconds_in_year = 365 * 24 * 60 * 60
    candles_per_year = seconds_in_year / time_gap_sec
    annualization_factor = np.sqrt(candles_per_year)
    pd_candles['annualized_hist_vol'] = pd_candles['interval_hist_vol'] * annualization_factor

    pd_candles['chop_against_ema'] = (
        (~pd_candles['is_green'] & (pd_candles['close'] > pd_candles['ema_close'])) |  # Case 1: Green candle and close > EMA
        (pd_candles['is_green'] & (pd_candles['close'] < pd_candles['ema_close']))   # Case 2: Red candle and close < EMA
    )

    pd_candles['ema_volume_short_periods'] = pd_candles['volume'].ewm(span=sliding_window_how_many_candles/slow_fast_interval_ratio, adjust=False).mean()
    pd_candles['ema_volume_long_periods'] = pd_candles['volume'].ewm(span=sliding_window_how_many_candles, adjust=False).mean()

    pd_candles['ema_cross'] = None
    pd_candles['ema_cross_last'] = None
    pd_candles['ema_bullish_cross_last_id'] = None
    pd_candles['ema_bearish_cross_last_id'] = None
    ema_short_periods_prev = pd_candles['ema_short_periods'].shift(1)
    ema_long_periods_prev = pd_candles['ema_long_periods'].shift(1)
    ema_short_periods_curr = pd_candles['ema_short_periods']
    ema_long_periods_curr = pd_candles['ema_long_periods']
    bullish_ema_crosses = (ema_short_periods_prev <= ema_long_periods_prev) & (ema_short_periods_curr > ema_long_periods_curr)
    bearish_ema_crosses = (ema_short_periods_prev >= ema_long_periods_prev) & (ema_short_periods_curr < ema_long_periods_curr)
    pd_candles.loc[bullish_ema_crosses, 'ema_cross'] = 1
    pd_candles.loc[bearish_ema_crosses, 'ema_cross'] = -1
    bullish_indices = pd.Series(pd_candles.index.where(pd_candles['ema_cross'] == 1), index=pd_candles.index).astype('Int64')
    bearish_indices = pd.Series(pd_candles.index.where(pd_candles['ema_cross'] == -1), index=pd_candles.index).astype('Int64')
    pd_candles['ema_bullish_cross_last_id'] = bullish_indices.rolling(window=pd_candles.shape[0], min_periods=1).max().astype('Int64')
    pd_candles['ema_bearish_cross_last_id'] = bearish_indices.rolling(window=pd_candles.shape[0], min_periods=1).max().astype('Int64')
    conditions = [
        (pd_candles['ema_bullish_cross_last_id'].notna() & 
        pd_candles['ema_bearish_cross_last_id'].notna() &
        (pd_candles['ema_bullish_cross_last_id'] > pd_candles['ema_bearish_cross_last_id'])),
        
        (pd_candles['ema_bullish_cross_last_id'].notna() & 
        pd_candles['ema_bearish_cross_last_id'].notna() &
        (pd_candles['ema_bearish_cross_last_id'] > pd_candles['ema_bullish_cross_last_id'])),
        
        (pd_candles['ema_bullish_cross_last_id'].notna() & 
        pd_candles['ema_bearish_cross_last_id'].isna()),
        
        (pd_candles['ema_bearish_cross_last_id'].notna() & 
        pd_candles['ema_bullish_cross_last_id'].isna())
    ]
    choices = ['bullish', 'bearish', 'bullish', 'bearish']
    pd_candles['ema_cross_last'] = np.select(conditions, choices, default=None) # type: ignore
    pd_candles.loc[bullish_ema_crosses, 'ema_cross'] = 'bullish'
    pd_candles.loc[bearish_ema_crosses, 'ema_cross'] = 'bearish'

    pd_candles['max_short_periods'] = close_short_periods_rolling.max()
    pd_candles['max_long_periods'] = close_long_periods_rolling.max()
    pd_candles['idmax_short_periods'] = close_short_periods_rolling.apply(lambda x : x.idxmax())
    pd_candles['idmax_long_periods'] = close_long_periods_rolling.apply(lambda x : x.idxmax())

    pd_candles['min_short_periods'] = close_short_periods_rolling.min()
    pd_candles['min_long_periods'] = close_long_periods_rolling.min()
    pd_candles['idmin_short_periods'] = close_short_periods_rolling.apply(lambda x : x.idxmin())
    pd_candles['idmin_long_periods'] = close_long_periods_rolling.apply(lambda x : x.idxmin())

    pd_candles['max_candle_body_height_percent_long_periods'] = pd_candles['candle_body_height_percent'].rolling(window=sliding_window_how_many_candles).max()
    pd_candles['idmax_candle_body_height_percent_long_periods'] = pd_candles['candle_body_height_percent'].rolling(window=sliding_window_how_many_candles).apply(lambda x : x.idxmax())
    pd_candles['min_candle_body_height_percent_long_periods'] = pd_candles['candle_body_height_percent'].rolling(window=sliding_window_how_many_candles).min()
    pd_candles['idmin_candle_body_height_percent_long_periods'] = pd_candles['candle_body_height_percent'].rolling(window=sliding_window_how_many_candles).apply(lambda x : x.idxmin())

    pd_candles['price_swing_short_periods'] = np.where(
        pd_candles['idmax_short_periods'] > pd_candles['idmin_short_periods'],
        pd_candles['max_short_periods'] - pd_candles['min_short_periods'],  # Up swing
        pd_candles['min_short_periods'] - pd_candles['max_short_periods']   # Down swing (negative)
    )

    pd_candles['price_swing_long_periods'] = np.where(
        pd_candles['idmax_long_periods'] > pd_candles['idmin_long_periods'],
        pd_candles['max_long_periods'] - pd_candles['min_long_periods'],  # Up swing
        pd_candles['min_long_periods'] - pd_candles['max_long_periods']   # Down swing (negative)
    )

    pd_candles['trend_from_highs_long_periods'] = np.where(
												pd.isna(pd_candles['max_long_periods']),
                                                None, # type: ignore
												pd_candles['max_long_periods'].rolling(window=sliding_window_how_many_candles).apply(trend_from_highs, raw=True)
												)
    pd_candles['trend_from_lows_long_periods'] = np.where(
												pd.isna(pd_candles['min_long_periods']),
                                                None, # type: ignore
												pd_candles['min_long_periods'].rolling(window=sliding_window_how_many_candles).apply(trend_from_lows, raw=True)
												)
    pd_candles['trend_from_highs_short_periods'] = np.where(
												pd.isna(pd_candles['max_short_periods']),
                                                None, # type: ignore
												pd_candles['max_short_periods'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(trend_from_highs, raw=True)
												)
    pd_candles['trend_from_lows_short_periods'] = np.where(
												pd.isna(pd_candles['min_short_periods']),
                                                None, # type: ignore
												pd_candles['min_short_periods'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(trend_from_lows, raw=True)
												)

    # ATR https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58 
    pd_candles.loc[:,'h_l'] = pd_candles['high'] - pd_candles['low']
    pd_candles.loc[:,'h_pc'] = abs(pd_candles['high'] - pd_candles['close'].shift(1))
    pd_candles.loc[:,'l_pc'] = abs(pd_candles['low'] - pd_candles['close'].shift(1))
    pd_candles.loc[:,'tr'] = pd_candles[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    pd_candles.loc[:,'atr'] = pd_candles['tr'].rolling(window=sliding_window_how_many_candles).mean()
    pd_candles.loc[:,'atr_avg_short_periods'] = pd_candles['atr'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).mean()
    pd_candles.loc[:,'atr_avg_long_periods'] = pd_candles['atr'].rolling(window=sliding_window_how_many_candles).mean()
    

    '''
    @hardcode @todo
    Hurst https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
    Smaller Windows (e.g., 50–100)
    Larger Windows (e.g., 200+)

    Sometimes you may encounter "Exception has occurred: FloatingPointError invalid value encountered in scalar divide"
    And for example adjusting window size from 120 to 125 will resolve the issue.
    '''
    if not hurst_exp_window_how_many_candles:
        hurst_exp_window_how_many_candles = (sliding_window_how_many_candles if sliding_window_how_many_candles>=125 else 125)
    pd_candles['hurst_exp'] = pd_candles['close'].rolling(
        window=hurst_exp_window_how_many_candles
        ).apply(lambda x: compute_Hc(x, kind='price', simplified=True)[0])


    # Boillenger https://www.quantifiedstrategies.com/python-bollinger-band-trading-strategy/
    pd_candles.loc[:,'boillenger_upper'] = (pd_candles['sma_long_periods'] if not boillenger_ema else pd_candles['ema_long_periods']) + pd_candles['std'] * boillenger_std_multiples
    pd_candles.loc[:,'boillenger_lower'] = (pd_candles['sma_long_periods'] if not boillenger_ema else pd_candles['ema_long_periods']) - pd_candles['std'] * boillenger_std_multiples
    pd_candles.loc[:,'boillenger_channel_height'] = pd_candles['boillenger_upper'] - pd_candles['boillenger_lower']

    pd_candles.loc[:,'boillenger_upper_agg'] = (pd_candles['sma_long_periods'] if not boillenger_ema else pd_candles['ema_long_periods']) + pd_candles['std'] * boillenger_std_multiples_for_aggressive_moves_detect
    pd_candles.loc[:,'boillenger_lower_agg'] = (pd_candles['sma_long_periods'] if not boillenger_ema else pd_candles['ema_long_periods']) - pd_candles['std'] * boillenger_std_multiples_for_aggressive_moves_detect
    pd_candles.loc[:,'boillenger_channel_height_agg'] = pd_candles['boillenger_upper_agg'] - pd_candles['boillenger_lower_agg']

    def detect_aggressive_movement(
        index: int,
        pd_candles: pd.DataFrame,
        sliding_window_how_many_candles: int,
        up_or_down: bool = True
    ):
        window_start = max(0, index - sliding_window_how_many_candles + 1)
        window = pd_candles.iloc[window_start:index + 1]
        first_breach_index = None
        candle_high, candle_low, candle_height = None, None, None
        
        if up_or_down:
            aggressive_mask = window['close'] >= window['boillenger_upper_agg']
            if aggressive_mask.any():
                first_breach_index = aggressive_mask.idxmax()
                candle_high = pd_candles.at[first_breach_index, 'high'] 
                candle_low = pd_candles.at[first_breach_index, 'low']
                candle_height = candle_high - candle_low  # type: ignore
        else:
            aggressive_mask = window['close'] <= window['boillenger_lower_agg']
            if aggressive_mask.any():
                first_breach_index = aggressive_mask.idxmax()
                candle_high = pd_candles.at[first_breach_index, 'high'] 
                candle_low = pd_candles.at[first_breach_index, 'low']
                candle_height = candle_high - candle_low  # type: ignore

        return {
            'aggressive_move': aggressive_mask.any(),
            'first_breach_index': first_breach_index,
            'candle_high' : candle_high,
            'candle_low' : candle_low,
            'candle_height' : candle_height
        }
    
    pd_candles['aggressive_up'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=True
        )['aggressive_move']
    )
    pd_candles['aggressive_up_index'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=True
        )['first_breach_index']
    )
    pd_candles['aggressive_up_candle_height'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=True
        )['candle_height']
    )
    pd_candles['aggressive_up_candle_high'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=True
        )['candle_high']
    )
    pd_candles['aggressive_up_candle_low'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=True
        )['candle_low']
    )
    pd_candles['aggressive_down'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=False
        )['aggressive_move']
    )
    pd_candles['aggressive_down_index'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=False
        )['first_breach_index']
    )
    pd_candles['aggressive_down_candle_height'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=False
        )['candle_height']
    )
    pd_candles['aggressive_down_candle_high'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=False
        )['candle_high']
    )
    pd_candles['aggressive_down_candle_low'] = pd_candles.index.to_series().apply(
        lambda idx: detect_aggressive_movement(
            idx, pd_candles, sliding_window_how_many_candles, up_or_down=False
        )['candle_low']
    )


    # FVG - Fair Value Gap https://atas.net/technical-analysis/fvg-trading-what-is-fair-value-gap-meaning-strategy/
    def compute_fvg(row, pd_candles):
        fvg_low = None
        fvg_high = None

        if row['aggressive_up_index'] is not None and not math.isnan(row['aggressive_up_index']):
            idx = row['aggressive_up_index']
            last_high = pd_candles.at[idx - 1, 'high']
            if idx + 1 < len(pd_candles):
                next_low = pd_candles.at[idx + 1, 'low']
            else:
                next_low = None
            
            fvg_low = next_low
            fvg_high = last_high

        elif row['aggressive_down_index'] is not None and not math.isnan(row['aggressive_down_index']):
            idx = row['aggressive_down_index']
            last_low = pd_candles.at[idx - 1, 'low']
            if idx + 1 < len(pd_candles):
                next_high = pd_candles.at[idx + 1, 'high']
            else:
                next_high = None

            fvg_low = last_low
            fvg_high = next_high

        return pd.Series({'fvg_low': fvg_low, 'fvg_high': fvg_high})

    fvg_result = pd_candles.apply(lambda row: compute_fvg(row, pd_candles), axis=1)
    pd_candles[['fvg_low', 'fvg_high']] = fvg_result
    pd_candles['fvg_gap'] = pd_candles['fvg_high'] - pd_candles['fvg_low']

    def compute_fvg_mitigated(row, pd_candles):
        mitigated = False
        if row['aggressive_down_index'] is not None and not math.isnan(row['aggressive_down_index']):
            idx = int(row['aggressive_down_index'])
            mitigated = pd_candles.iloc[idx + 1:row.name]['close'].gt(row['fvg_low']).any()
        elif row['aggressive_up_index'] is not None and not math.isnan(row['aggressive_up_index']):
            idx = int(row['aggressive_up_index'])
            mitigated = pd_candles.iloc[idx + 1:row.name]['close'].lt(row['fvg_high']).any()
        return mitigated

    pd_candles['fvg_mitigated'] = pd_candles.apply(lambda row: compute_fvg_mitigated(row, pd_candles), axis=1) # type: ignore

    '''
        RSI
        Divergences from Bybit Learn https://www.youtube.com/watch?v=G9oUTi-PI18&t=809s 
        RSI Reversals from BK Traders https://www.youtube.com/watch?v=MvkbrHjiQlI
    '''
    pd_candles.loc[:,'close_delta'] = pd_candles['close'].diff()
    pd_candles.loc[:,'close_delta_percent'] = pd_candles['close'].pct_change()
    lo_up = pd_candles['close_delta'].clip(lower=0)
    lo_down = -1 * pd_candles['close_delta'].clip(upper=0)
    pd_candles.loc[:,'up'] = lo_up
    pd_candles.loc[:,'down'] = lo_down

    if rsi_ema == True:
        # Use exponential moving average
        lo_ma_up = lo_up.ewm(
            com = rsi_sliding_window_how_many_candles -1, 
            adjust=True, 
            min_periods = rsi_sliding_window_how_many_candles).mean()
        lo_ma_down = lo_down.ewm(
            com = (rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles) - 1, 
            adjust=True, 
            min_periods = rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).mean()

    else:
        # Use simple moving average
        lo_ma_up = lo_up.rolling(window = rsi_sliding_window_how_many_candles).mean()
        lo_ma_down = lo_down.rolling(window = rsi_sliding_window_how_many_candles).mean()
        
    lo_rs = lo_ma_up / lo_ma_down
    pd_candles.loc[:,'rsi'] = 100 - (100/(1 + lo_rs))
    pd_candles['rsi_bucket'] = pd_candles['rsi'].apply(lambda x: bucketize_val(x, buckets=BUCKETS_m0_100))
    pd_candles['ema_rsi'] = pd_candles['rsi'].ewm(
        span=rsi_sliding_window_how_many_candles, 
        adjust=False).mean()

    rsi_rolling = pd_candles['rsi'].rolling(window=int(rsi_trend_sliding_window_how_many_candles))
    pd_candles['rsi_max'] = rsi_rolling.max()
    pd_candles['rsi_idmax'] = rsi_rolling.apply(lambda x : x.idxmax())
    pd_candles['rsi_min'] = rsi_rolling.min()
    pd_candles['rsi_idmin'] = rsi_rolling.apply(lambda x : x.idxmin())

    def rsi_trend(
            row,
            rsi_upper_threshold : float = 70,
            rsi_lower_threshold : float = 30):
        if pd.isna(row['rsi_idmax']) or pd.isna(row['rsi_idmin']):
            return np.nan
        if row['rsi_idmax'] > row['rsi_idmin']:
            return 'down' if row.name > row['rsi_idmax'] and row['rsi'] <= rsi_upper_threshold else 'up'
        else:
            return 'up' if row.name > row['rsi_idmin'] and row['rsi'] >= rsi_lower_threshold else 'down'

    pd_candles['rsi_trend'] = pd_candles.apply(lambda row: rsi_trend(row), axis=1)

    pd_candles['rsi_trend_from_highs'] = np.where(
												pd.isna(pd_candles['rsi_max']),
                                                None, # type: ignore
												pd_candles['rsi_max'].rolling(window=rsi_trend_sliding_window_how_many_candles).apply(trend_from_highs, raw=True)
												)
    pd_candles['rsi_trend_from_lows'] = np.where(
												pd.isna(pd_candles['rsi_min']),
                                                None, # type: ignore
												pd_candles['rsi_min'].rolling(window=rsi_trend_sliding_window_how_many_candles).apply(trend_from_lows, raw=True)
												)

    def _rsi_divergence(row):
        trend_from_highs_long_periods = TrendDirection(row['trend_from_highs_long_periods']) if row['trend_from_highs_long_periods'] is not None and not pd.isna(row['trend_from_highs_long_periods']) else None  # type: ignore
        rsi_trend_from_highs = TrendDirection(row['rsi_trend_from_highs']) if row['rsi_trend_from_highs'] is not None and not pd.isna(row['rsi_trend_from_highs']) else None # type: ignore
        
        if trend_from_highs_long_periods and rsi_trend_from_highs and trend_from_highs_long_periods == TrendDirection.LOWER_HIGHS and rsi_trend_from_highs == TrendDirection.HIGHER_HIGHS:
            return 'bullish_divergence'
        elif trend_from_highs_long_periods and rsi_trend_from_highs and trend_from_highs_long_periods == TrendDirection.HIGHER_HIGHS and rsi_trend_from_highs == TrendDirection.LOWER_HIGHS:
            return 'bearish_divergence'
        return 'no_divergence'
    pd_candles['rsi_divergence'] = pd_candles.apply(_rsi_divergence, axis=1)
    

    # MFI (Money Flow Index) https://randerson112358.medium.com/algorithmic-trading-strategy-using-money-flow-index-mfi-python-aa46461a5ea5 
    pd_candles['typical_price'] = (pd_candles['high'] + pd_candles['low'] + pd_candles['close']) / 3
    pd_candles['money_flow'] = pd_candles['typical_price'] * pd_candles['volume']
    pd_candles['money_flow_positive'] = pd_candles['money_flow'].where(
        pd_candles['typical_price'] > pd_candles['typical_price'].shift(1), 0
    )
    pd_candles['money_flow_negative'] = pd_candles['money_flow'].where(
        pd_candles['typical_price'] < pd_candles['typical_price'].shift(1), 0
    )
    pd_candles['positive_flow_sum'] = pd_candles['money_flow_positive'].rolling(
        rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).sum()
    pd_candles['negative_flow_sum'] = pd_candles['money_flow_negative'].rolling(
        rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).sum()
    pd_candles['money_flow_ratio'] = pd_candles['positive_flow_sum'] / pd_candles['negative_flow_sum']
    pd_candles['mfi'] = 100 - (100 / (1 + pd_candles['money_flow_ratio']))
    pd_candles['mfi_bucket'] = pd_candles['mfi'].apply(lambda x: bucketize_val(x, buckets=BUCKETS_m0_100))
    

    # MACD https://www.investopedia.com/terms/m/macd.asp
    # https://www.youtube.com/watch?v=jmPCL3l08ss
    pd_candles['macd'] = pd_candles['ema_short_periods'] - pd_candles['ema_long_periods']
    pd_candles['signal'] = pd_candles['macd'].ewm(span=int(sliding_window_how_many_candles/slow_fast_interval_ratio), adjust=False).mean()
    pd_candles['macd_minus_signal'] = pd_candles['macd'] - pd_candles['signal'] # MACD histogram
    macd_cur = pd_candles['macd_minus_signal']
    macd_prev = pd_candles['macd_minus_signal'].shift(1)
    bullish_macd_crosses = (macd_prev < 0) & (macd_cur > 0)
    bearish_macd_crosses = (macd_prev > 0) & (macd_cur < 0)
    pd_candles.loc[bullish_macd_crosses, 'macd_cross'] = 1
    pd_candles.loc[bearish_macd_crosses, 'macd_cross'] = -1
    bullish_indices = pd.Series(pd_candles.index.where(pd_candles['macd_cross'] == 1), index=pd_candles.index).astype('Int64')
    bearish_indices = pd.Series(pd_candles.index.where(pd_candles['macd_cross'] == -1), index=pd_candles.index).astype('Int64')
    pd_candles['macd_bullish_cross_last_id'] = bullish_indices.rolling(window=pd_candles.shape[0], min_periods=1).max().astype('Int64')
    pd_candles['macd_bearish_cross_last_id'] = bearish_indices.rolling(window=pd_candles.shape[0], min_periods=1).max().astype('Int64')
    conditions = [
        (pd_candles['macd_bullish_cross_last_id'].notna() & 
        pd_candles['macd_bearish_cross_last_id'].notna() &
        (pd_candles['macd_bullish_cross_last_id'] > pd_candles['macd_bearish_cross_last_id'])),
        
        (pd_candles['macd_bullish_cross_last_id'].notna() & 
        pd_candles['macd_bearish_cross_last_id'].notna() &
        (pd_candles['macd_bearish_cross_last_id'] > pd_candles['macd_bullish_cross_last_id'])),
        
        (pd_candles['macd_bullish_cross_last_id'].notna() & 
        pd_candles['macd_bearish_cross_last_id'].isna()),
        
        (pd_candles['macd_bearish_cross_last_id'].notna() & 
        pd_candles['macd_bullish_cross_last_id'].isna())
    ]
    choices = ['bullish', 'bearish', 'bullish', 'bearish']
    pd_candles['macd_cross_last'] = np.select(conditions, choices, default=None) # type: ignore
    pd_candles.loc[bullish_macd_crosses, 'macd_cross'] = 'bullish'
    pd_candles.loc[bearish_macd_crosses, 'macd_cross'] = 'bearish'

    if not pypy_compat:
        calculate_slope(
            pd_data=pd_candles,
            src_col_name='close',
            slope_col_name='close_short_slope',
            sliding_window_how_many_candles=int(sliding_window_how_many_candles/slow_fast_interval_ratio)
        )

        calculate_slope(
            pd_data=pd_candles,
            src_col_name='close',
            slope_col_name='close_long_slope',
            sliding_window_how_many_candles=int(sliding_window_how_many_candles)
        )
        
        calculate_slope(
            pd_data=pd_candles,
            src_col_name='ema_short_periods',
            slope_col_name='ema_short_slope',
            sliding_window_how_many_candles=int(sliding_window_how_many_candles/slow_fast_interval_ratio)
        )

        calculate_slope(
            pd_data=pd_candles,
            src_col_name='ema_long_periods',
            slope_col_name='ema_long_slope',
            sliding_window_how_many_candles=int(sliding_window_how_many_candles)
        )

        calculate_slope(
            pd_data=pd_candles,
            src_col_name='boillenger_upper',
            slope_col_name='boillenger_upper_slope',
            sliding_window_how_many_candles=int(sliding_window_how_many_candles)
        )

        calculate_slope(
            pd_data=pd_candles,
            src_col_name='boillenger_lower',
            slope_col_name='boillenger_lower_slope',
            sliding_window_how_many_candles=int(sliding_window_how_many_candles)
        )

        calculate_slope(
            pd_data=pd_candles,
            src_col_name='ema_rsi',
            slope_col_name='ema_rsi_slope',
            sliding_window_how_many_candles=int(rsi_trend_sliding_window_how_many_candles)
        )

        pd_candles['regular_divergence'] = (
            (pd_candles['ema_long_slope'] > 0) & (pd_candles['ema_rsi_slope'] < 0) |
            (pd_candles['ema_long_slope'] < 0) & (pd_candles['ema_rsi_slope'] > 0)
        )

        calculate_slope(
            pd_data=pd_candles,
            src_col_name='hurst_exp',
            slope_col_name='hurst_exp_slope',
            sliding_window_how_many_candles=hurst_exp_window_how_many_candles
        )
        

    # Fibonacci
    pd_candles[f'fib_{target_fib_level}_short_periods'] = pd_candles.apply(lambda rw : estimate_fib_retracement(rw['min_short_periods'], rw['idmin_short_periods'], rw['max_short_periods'], rw['idmax_short_periods'], target_fib_level), axis=1)
    pd_candles[f'fib_{target_fib_level}_long_periods'] = pd_candles.apply(lambda rw : estimate_fib_retracement(rw['min_long_periods'], rw['idmin_long_periods'], rw['max_long_periods'], rw['idmax_long_periods'], target_fib_level), axis=1)


    # Inflection points
    pd_candles['gap_close_vs_ema'] = pd_candles['close'] - pd_candles['ema_long_periods']
    pd_candles['gap_close_vs_ema_percent'] = pd_candles['gap_close_vs_ema']/pd_candles['close'] *100

    pd_candles['close_above_or_below_ema'] = None
    pd_candles.loc[pd_candles['gap_close_vs_ema'] > 0, 'close_above_or_below_ema'] = 'above'
    pd_candles.loc[pd_candles['gap_close_vs_ema'] < 0, 'close_above_or_below_ema'] = 'below'

    pd_candles.loc[
        (pd_candles['close_above_or_below_ema'] != pd_candles['close_above_or_below_ema'].shift(-1)),
        'close_vs_ema_inflection'
    ] = np.sign(pd_candles['close'] - pd_candles['ema_long_periods'])

def lookup_fib_target(
            row,
            pd_candles,
            target_fib_level : float = 0.618
        ) -> Union[Dict, None]:
            if row is None:
                return None
            
            fib_target_short_periods = None
            fib_target_long_periods = None

            max_short_periods = row['max_short_periods']
            idmax_short_periods = int(row['idmax_short_periods']) if not math.isnan(row['idmax_short_periods']) else None
            max_long_periods = row['max_long_periods']
            idmax_long_periods = int(row['idmax_long_periods']) if not math.isnan(row['idmax_long_periods']) else None

            min_short_periods = row['min_short_periods']
            idmin_short_periods = int(row['idmin_short_periods']) if not math.isnan(row['idmin_short_periods']) else None
            min_long_periods = row['min_long_periods']
            idmin_long_periods = int(row['idmin_long_periods']) if not math.isnan(row['idmin_long_periods']) else None
                        
            if idmax_short_periods and idmin_short_periods and idmax_short_periods>0 and idmin_short_periods>0:
                if idmax_short_periods>idmin_short_periods and idmax_short_periods < len(pd_candles):
                    # Falling from prev peak
                    last_peak = pd_candles.iloc[idmax_short_periods]
                    fib_target_short_periods = last_peak[f'fib_{target_fib_level}_short_periods'] if not math.isnan(last_peak[f'fib_{target_fib_level}_short_periods']) else None
                    
                else:
                    # Bouncing from prev bottom
                    if idmin_short_periods < len(pd_candles):
                        last_bottom = pd_candles.iloc[idmin_short_periods]
                        fib_target_short_periods = last_bottom[f'fib_{target_fib_level}_short_periods'] if not math.isnan(last_bottom[f'fib_{target_fib_level}_short_periods']) else None
                        
            if idmax_long_periods and idmin_long_periods and idmax_long_periods>0 and idmin_long_periods>0:
                if idmax_long_periods>idmin_long_periods and idmax_long_periods < len(pd_candles):
                    # Falling from prev peak
                    last_peak = pd_candles.iloc[idmax_long_periods]
                    fib_target_long_periods = last_peak[f'fib_{target_fib_level}_long_periods'] if not math.isnan(last_peak[f'fib_{target_fib_level}_long_periods']) else None
                    
                else:
                    # Bouncing from prev bottom
                    if idmin_long_periods < len(pd_candles):
                        last_bottom = pd_candles.iloc[idmin_long_periods]
                        fib_target_long_periods = last_bottom[f'fib_{target_fib_level}_long_periods'] if not math.isnan(last_bottom[f'fib_{target_fib_level}_long_periods']) else None
            
            return {
                'short_periods' : {
                    'idmin' : idmin_short_periods,
                    'idmax' : idmax_short_periods,
                    'min' : min_short_periods,
                    'max' : max_short_periods,
                    'fib_target' : fib_target_short_periods,
                },
                'long_periods' : {
                    'idmin' : idmin_long_periods,
                    'idmax' : idmax_long_periods,
                    'min' : min_long_periods,
                    'max' : max_long_periods,
                    'fib_target' : fib_target_long_periods
                }
            }

''' 
The implementation from Geeksforgeeks https://www.geeksforgeeks.org/find-indices-of-all-local-maxima-and-local-minima-in-an-array/ is wrong. 
If you have consecutive-duplicates, things will gall apart!
    Example 1: values = [ 1, 2, 3, 7, 11, 15, 13, 12, 11, 6, 5, 7, 11, 8]
        The default implementation correctly identify "15" as a peak.

    Example 2: values = [ 1, 2, 3, 7, 11, 15, 15, 13, 12, 11, 6, 5, 7, 11, 8]
        The default implementation will mark "11" as local maxima because there are two consecutive 15's.

Fix: https://stackoverflow.com/questions/75013708/python-finding-local-minima-and-maxima?noredirect=1#comment132376733_75013708
'''
def find_local_max_min(values: List[float], merge_distance: int = 5) -> Union[Dict[str, List], None]:
    mx = []
    mn = []

    n = len(values)
    if n < 2:
        return None

    if values[0] > values[1]:
        mn.append(0)
    elif values[0] < values[1]:
        mx.append(0)

    for i in range(1, n-1):
        if all(values[i] >= values[j] for j in range(i-10, i+11) if 0 <= j < n):
            mx.append(i)
        elif all(values[i] <= values[j] for j in range(i-10, i+11) if 0 <= j < n):
            mn.append(i)

    if values[-1] > values[-2]:
        mx.append(n-1)
    elif values[-1] < values[-2]:
        mn.append(n-1)

    # Merge nearby maxima and minima
    mx_merged = []
    mn_merged = []

    def merge_nearby_points(points):
        merged = []
        start = points[0]
        for i in range(1, len(points)):
            if points[i] - start > merge_distance:
                merged.append(start + (points[i-1] - start) // 2)  # Take the middle point
                start = points[i]
        merged.append(start + (points[-1] - start) // 2)  # Take the middle point for the last segment
        return merged

    mx_merged = merge_nearby_points(mx)
    mn_merged = merge_nearby_points(mn)

    return {
        'local_max': mx_merged,
        'local_min': mn_merged
    }

def partition_sliding_window(
        pd_candles : pd.DataFrame,
        sliding_window_how_many_candles : int,
        smoothing_window_size_ratio : int,
        linregress_stderr_threshold : float,
        max_recur_depth : int,
        min_segment_size_how_many_candles : int,
        segment_consolidate_slope_ratio_threshold : float,
        sideway_price_condition_threshold : float
        ) -> Dict[str, Any]:
    
    window_size = int(sliding_window_how_many_candles/smoothing_window_size_ratio)
    # window_size = 8 # @hack
    smoothed_colse = pd.Series(pd_candles['close']).rolling(window=window_size, min_periods=window_size).mean()
    pd_candles['smoothed_close'] = smoothed_colse

    pd_candles['maxima'] = False
    pd_candles['minima'] = False
    maxima = []
    minima = []
    maxima_minima = find_local_max_min(values = pd_candles['close'].to_list(), merge_distance=1) # @CRITICAL close vs smoothed_close and merge_distance
    if maxima_minima:
        maxima = maxima_minima['local_max']
        minima = maxima_minima['local_min']
        maxima = [x for x in maxima if x>=pd_candles.index.min()]
        minima = [x for x in minima if x>=pd_candles.index.min()]    
        pd_candles.loc[maxima, 'maxima'] = True
        pd_candles.loc[minima, 'minima'] = True

    inflection_points = pd_candles[(pd_candles.close_vs_ema_inflection == 1) | (pd_candles.close_vs_ema_inflection == -1)].index.tolist()
    inflection_points = [ index-1 for index in inflection_points ]
    if (pd_candles.shape[0]-1) not in inflection_points:
        inflection_points.append(pd_candles.shape[0]-1)
    
    last_point = inflection_points[0]
    sparse_inflection_points = [ last_point ]
    for point in inflection_points:
        if (point not in sparse_inflection_points) and ((point-last_point)>min_segment_size_how_many_candles):
            sparse_inflection_points.append(point)
            last_point = point
    inflection_points = sparse_inflection_points

    def _compute_new_segment(
            pd_candles : pd.DataFrame,
            start_index : int,
            end_index : int,
            cur_recur_depth : int,
            linregress_stderr_threshold : float = 50,
            max_recur_depth : int = 2,
            min_segment_size_how_many_candles : int = 15
    ) -> Union[List[Dict], None]:
        new_segments : Union[List[Dict], None] = None

        if end_index>pd_candles.shape[0]-1:
            end_index = pd_candles.shape[0]-1

        if start_index==end_index:
            return []
                    
        start_upper = pd_candles.iloc[start_index]['boillenger_upper']
        end_upper = pd_candles.iloc[end_index]['boillenger_upper']
        start_lower = pd_candles.iloc[start_index]['boillenger_lower']
        end_lower = pd_candles.iloc[end_index]['boillenger_lower']

        start_datetime = pd_candles.iloc[start_index]['datetime']
        end_datetime = pd_candles.iloc[end_index]['datetime']
        start_timestamp_ms = pd_candles.iloc[start_index]['timestamp_ms'] 
        end_timestamp_ms = pd_candles.iloc[end_index]['timestamp_ms'] 
        start_close = pd_candles.iloc[start_index]['close']
        end_close = pd_candles.iloc[end_index]['close']
        
        # Using Boillenger upper and lower only
        maxima_idx_boillenger = [ start_index, end_index ]
        maxima_close_boillenger = [ start_upper, end_upper ]
        minima_idx_boillenger = [ start_index, end_index ]
        minima_close_boillenger = [ start_lower, end_lower ]

        from scipy.stats import linregress # in-compatible with pypy

        maxima_linregress_boillenger = linregress(maxima_idx_boillenger, maxima_close_boillenger)
        minima_linregress_boillenger = linregress(minima_idx_boillenger, minima_close_boillenger)

        # Using Boillenger upper and lower AND Local maxima/minima
        maxima_idx_full = [start_index] + [ x for x in maxima if x>=start_index+1 and x<end_index ] + [end_index]
        maxima_close_full = [ start_upper if not math.isnan(start_upper) else start_close ] + [ pd_candles.loc[x]['close'] for x in maxima if x>start_index and x<end_index ] + [ end_upper ]
        minima_idx_full = [start_index] + [ x for x in minima if x>=start_index+1 and x<end_index ] + [end_index]
        minima_close_full = [ start_lower if not math.isnan(start_lower) else start_close ] + [ pd_candles.loc[x]['close'] for x in minima if x>start_index and x<end_index ] + [ end_lower ]
        
        maxima_linregress_full = linregress(maxima_idx_full, maxima_close_full)
        minima_linregress_full = linregress(minima_idx_full, minima_close_full)
        
        largest_candle_index : int = int(pd_candles.iloc[start_index:end_index,:]['candle_height'].idxmax())
        if (
            (abs(maxima_linregress_full.stderr) < linregress_stderr_threshold and abs(minima_linregress_full.stderr) < linregress_stderr_threshold) # type: ignore Otherwise, Error: Cannot access attribute "stderr" for class "_"
            or cur_recur_depth>=max_recur_depth
            or (start_index==largest_candle_index or end_index==largest_candle_index+1)
            or (
                (end_index-largest_candle_index < min_segment_size_how_many_candles)
                or (largest_candle_index - start_index < min_segment_size_how_many_candles)
            )
        ):
            new_segment = {
                'start' : start_index,
                'end' : end_index,
                'start_datetime' : start_datetime,
                'end_datetime' : end_datetime,
                'start_timestamp_ms' : start_timestamp_ms,
                'end_timestamp_ms' : end_timestamp_ms,
                'start_close' : start_close,
                'end_close' : end_close,
                'window_size_num_intervals' : end_index - start_index,
                'cur_recur_depth' : cur_recur_depth,
                'up_or_down' : 'up' if end_close>=start_close else 'down',

                'volume' : pd_candles[start_index:end_index]['volume'].sum(), # in base_ccy

                'maxima_idx_boillenger' : maxima_idx_boillenger,
                'maxima_close_boillenger' : maxima_close_boillenger,
                'minima_idx_boillenger' : minima_idx_boillenger,
                'minima_close_boillenger' : minima_close_boillenger,
                
                'maxima_linregress_boillenger' : maxima_linregress_boillenger,
                'minima_linregress_boillenger' : minima_linregress_boillenger,
                'maxima_linregress_full' : maxima_linregress_full,
                'minima_linregress_full' : minima_linregress_full,
            }
            new_segments = [ new_segment ]
        else:
            
            new_segments1 = _compute_new_segment(pd_candles, start_index, largest_candle_index, cur_recur_depth+1)
            new_segments2 = _compute_new_segment(pd_candles, largest_candle_index+1, end_index, cur_recur_depth+1)
            new_segments = (new_segments1 or []) + (new_segments2 or [])

        return new_segments
        
    segments = []
    for end_index in inflection_points:
        if not segments:
            start_index = 0
            
            inscope_maxima = [ x for x in maxima if x>=0 and x<end_index ] 
            inscope_minima = [ x for x in minima if x>=0 and x<end_index ]

            if inscope_maxima and inscope_minima:
                if sliding_window_how_many_candles<end_index:
                    start_index = sliding_window_how_many_candles
                new_segments = _compute_new_segment(pd_candles, start_index, end_index, 0, linregress_stderr_threshold, max_recur_depth, min_segment_size_how_many_candles)
                segments = (segments or []) + (new_segments or [])
            
        else:
            start_index = segments[-1]['end']
            if start_index!=end_index:
                new_segments = _compute_new_segment(pd_candles, start_index, end_index, 0, linregress_stderr_threshold, max_recur_depth, min_segment_size_how_many_candles)
                if new_segments:
                    segments = segments + new_segments


    '''
    You have five kinds of wedges:
    a. Rising parallel
    b. Rising converging
    c. Side way
    d. Falling parallel
    e. Falling converging

    Here, we're merging 'parallel' segments based on slope of 'maxima_linregress_boillenger' and 'minima_linregress_boillenger' from adjacent segments.
    '''
    consolidated_segements = [ segments[0] ]
    for segment in segments:
        if segment not in consolidated_segements:
            last_segment = consolidated_segements[-1]
            last_segment_maxima_slope = last_segment['maxima_linregress_boillenger'].slope
            last_segment_minima_slope = last_segment['minima_linregress_boillenger'].slope
            this_segment_maxima_slope = segment['maxima_linregress_boillenger'].slope
            this_segment_minima_slope = segment['minima_linregress_boillenger'].slope
            if math.isnan(last_segment_maxima_slope) or math.isnan(last_segment_minima_slope):
                consolidated_segements.append(segment)
            else:
                if (
                    abs(last_segment_maxima_slope/this_segment_maxima_slope-1)<segment_consolidate_slope_ratio_threshold 
                    and abs(last_segment_minima_slope/this_segment_minima_slope-1)<segment_consolidate_slope_ratio_threshold
                ):
                    consolidated_segements.pop()
                    
                    start_index = last_segment['maxima_idx_boillenger'][0]
                    end_index = segment['maxima_idx_boillenger'][-1]
                    maxima_idx_boillenger = [ start_index, end_index ]
                    maxima_close_boillenger = [ last_segment['maxima_close_boillenger'][0], segment['maxima_close_boillenger'][-1] ]
                    minima_idx_boillenger = maxima_idx_boillenger
                    minima_close_boillenger = [ last_segment['minima_close_boillenger'][0], segment['minima_close_boillenger'][-1] ]

                    from scipy.stats import linregress # in-compatible with pypy

                    maxima_linregress_boillenger = linregress(maxima_idx_boillenger, maxima_close_boillenger)
                    minima_linregress_boillenger = linregress(minima_idx_boillenger, minima_close_boillenger)

                    # Using Boillenger upper and lower AND Local maxima/minima
                    start_upper = pd_candles.iloc[start_index]['boillenger_upper']
                    end_upper = pd_candles.iloc[end_index]['boillenger_upper']
                    start_lower = pd_candles.iloc[start_index]['boillenger_lower']
                    end_lower = pd_candles.iloc[end_index]['boillenger_lower']
                    maxima_idx_full = [last_segment['start']] + [ x for x in maxima if x>=start_index+1 and x<end_index ] + [segment['end']]
                    maxima_close_full = [ start_upper ] + [ pd_candles.loc[x]['close'] for x in maxima if x>start_index and x<end_index ] + [ end_upper ]
                    minima_idx_full = [last_segment['start']] + [ x for x in minima if x>=start_index+1 and x<end_index ] + [segment['end']]
                    minima_close_full = [ start_lower ] + [ pd_candles.loc[x]['close'] for x in minima if x>start_index and x<end_index ] + [ end_lower ]

                    maxima_linregress_full = linregress(maxima_idx_full, maxima_close_full)
                    minima_linregress_full = linregress(minima_idx_full, minima_close_full)
                    
                    new_segment = {
                        'start' : last_segment['start'],
                        'end' : segment['end'],
                        'start_datetime' : last_segment['start_datetime'],
                        'end_datetime' : segment['end_datetime'],
                        'start_timestamp_ms' : last_segment['start_timestamp_ms'],
                        'end_timestamp_ms' : segment['end_timestamp_ms'],
                        'start_close' : last_segment['start_close'],
                        'end_close' : segment['end_close'],
                        'window_size_num_intervals' : end_index - start_index,
                        'cur_recur_depth' : max(last_segment['cur_recur_depth'], segment['cur_recur_depth']),
                        'up_or_down' : 'up' if segment['end_close']>=last_segment['start_close'] else 'down',

                        'volume' : pd_candles[start_index:end_index]['volume'].sum(), # in base_ccy

                        'maxima_idx_boillenger' : maxima_idx_boillenger,
                        'maxima_close_boillenger' : maxima_close_boillenger,
                        'minima_idx_boillenger' : minima_idx_boillenger,
                        'minima_close_boillenger' : minima_close_boillenger,
                        
                        'maxima_linregress_boillenger' : maxima_linregress_boillenger,
                        'minima_linregress_boillenger' : minima_linregress_boillenger,
                        'maxima_linregress_full' : maxima_linregress_full,
                        'minima_linregress_full' : minima_linregress_full,
                    }
                    consolidated_segements.append(new_segment)
                else:
                    consolidated_segements.append(segment)

    '''
    Depending on 'sliding_window_how_many_candles', pd_candles['boillenger_upper'] and pd_candles['boillenger_lower'] from 'compute_candles_stats' may be nan in first few segments.
    So here, we're back filling pd_candles['boillenger_upper'] and pd_candles['boillenger_lower'] from subsequent segments.
    '''
    last_segment = consolidated_segements[-1]
    for i in range(len(consolidated_segements)-1, -1, -1):
        segment = consolidated_segements[i]
        if math.isnan(segment['maxima_close_boillenger'][0]) or math.isnan(segment['minima_close_boillenger'][0]):
            start_index = segment['start']
            end_index = segment['end']
            start_close = segment['start_close']

            # Using Boillenger upper and lower only
            maxima_idx_boillenger = segment['maxima_idx_boillenger']
            minima_idx_boillenger = segment['minima_idx_boillenger']
            maxima_close_boillenger = segment['maxima_close_boillenger']
            minima_close_boillenger = segment['minima_close_boillenger']
            if math.isnan(maxima_close_boillenger[-1]) or not math.isnan(minima_close_boillenger[-1]):
                maxima_close_boillenger[-1] = last_segment['maxima_close_boillenger'][0]
                minima_close_boillenger[-1] = last_segment['minima_close_boillenger'][0]
            end_boillenger_height = maxima_close_boillenger[-1] - minima_close_boillenger[-1]
            maxima_close_boillenger[0] = segment['start_close'] + end_boillenger_height/2
            minima_close_boillenger[0] = segment['start_close'] - end_boillenger_height/2

            from scipy.stats import linregress # in-compatible with pypy

            maxima_linregress_boillenger = linregress(maxima_idx_boillenger, maxima_close_boillenger)
            minima_linregress_boillenger = linregress(minima_idx_boillenger, minima_close_boillenger)

            # Using Boillenger upper and lower AND Local maxima/minima
            start_upper = maxima_close_boillenger[0]
            end_upper = maxima_close_boillenger[-1]
            start_lower = minima_close_boillenger[0]
            end_lower = minima_close_boillenger[-1]
            maxima_idx_full = [start_index] + [ x for x in maxima if x>=start_index+1 and x<end_index ] + [end_index]
            maxima_close_full = [ start_upper if not math.isnan(start_upper) else start_close ] + [ pd_candles.loc[x]['close'] for x in maxima if x>start_index and x<end_index ] + [ end_upper ]
            minima_idx_full = [start_index] + [ x for x in minima if x>=start_index+1 and x<end_index ] + [end_index]
            minima_close_full = [ start_lower if not math.isnan(start_lower) else start_close ] + [ pd_candles.loc[x]['close'] for x in minima if x>start_index and x<end_index ] + [ end_lower ]
            
            maxima_linregress_full = linregress(maxima_idx_full, maxima_close_full)
            minima_linregress_full = linregress(minima_idx_full, minima_close_full)
            
            segment['maxima_linregress_boillenger'] = maxima_linregress_boillenger
            segment['minima_linregress_boillenger'] = minima_linregress_boillenger
            segment['maxima_linregress_full'] = maxima_linregress_full
            segment['minima_linregress_full'] = minima_linregress_full

        last_segment = segment
    
    '''
    You have five kinds of wedges:
    a. Rising parallel
    b. Rising converging/diverging
    c. Side way
    d. Falling parallel
    e. Falling converging/diverging
    '''
    def classify_segment(
            segment : Dict,
            segment_consolidate_slope_ratio_threshold : float,
            sideway_price_condition_threshold : float
        ):
        start_close = segment['start_close']
        end_close = segment['end_close']
        maxima_close_boillenger = segment['maxima_close_boillenger']
        minima_close_boillenger = segment['minima_close_boillenger']
        start_height : float = maxima_close_boillenger[0] - minima_close_boillenger[0]
        end_height : float = maxima_close_boillenger[-1] - minima_close_boillenger[-1]
        upper_slope = segment['maxima_linregress_boillenger'].slope
        lower_slope = segment['minima_linregress_boillenger'].slope
        is_parallel : bool = True if abs((upper_slope/lower_slope) -1) > segment_consolidate_slope_ratio_threshold else False
        is_rising : bool = True if end_close>start_close else False
        is_sideway : bool = True if abs((start_close/end_close)-1) < sideway_price_condition_threshold else False

        is_converging : bool = True if start_height>end_height and start_height/end_height>2 else False
        is_diverging : bool = True if end_height>start_height and end_height/start_height>2 else False
        
        if is_sideway:
            segment['class'] = 'sideway'
            if is_converging:
                segment['class'] = 'sideway_converging'
            elif is_diverging:
                segment['class'] = 'sideway_diverging'

        else:
            if is_rising:
                if is_parallel:
                    segment['class'] = 'rising_parallel'
                else:
                    if is_converging:
                        segment['class'] = 'rising_converging'
                    elif is_diverging:
                        segment['class'] = 'rising_diverging'
                    else:
                        segment['class'] = 'rising_parallel'
            else:
                if is_parallel:
                    segment['class'] = 'falling_parallel'
                else:
                    if is_converging:
                        segment['class'] = 'falling_converging'
                    elif is_diverging:
                        segment['class'] = 'falling_diverging'
                    else:
                        segment['class'] = 'falling_parallel'
        
    for segment in consolidated_segements:
        classify_segment(segment, segment_consolidate_slope_ratio_threshold, sideway_price_condition_threshold)
        
    return {
        'minima' : minima,
        'maxima' : maxima,
        'segments' : consolidated_segements
    }

# This relies on statsmodels.api, which is not pypy compatible
def compute_pair_stats(
    pd_candles : pd.DataFrame,
    how_many_candles : int = 24*7
) -> None:
    import statsmodels.api as sm

    def _compute_hedge_ratio(
                prices0 : List[float],
                prices1 : List[float]
        ):
        model = sm.OLS(prices0, prices1).fit()
        hedge_ratio = model.params[0]
        return hedge_ratio
    
    pd_candles['hedge_ratio'] = np.nan
    for j in range(how_many_candles, pd_candles.shape[0]):
        window = pd_candles.iloc[j-how_many_candles:j]
        hedge_ratio = _compute_hedge_ratio(window['close_1'].values, window['close_2'].values) # type: ignore
        pd_candles.loc[j, 'hedge_ratio'] = hedge_ratio

    pd_candles['close_spread'] = pd_candles['close_1'] - (pd_candles['close_2'] * pd_candles['hedge_ratio']) # You're fitting one hedge_ratio over a windows
    mean = pd_candles['close_spread'].rolling(how_many_candles).mean()
    std = pd_candles['close_spread'].rolling(how_many_candles).std()
    pd_candles['close_spread_mean'] = mean
    pd_candles['close_spread_std'] = std
    pd_candles['zscore_close_spread'] = (pd_candles['close_spread'] - mean)/std
    pd_candles['zscore_close_spread_min'] = pd_candles['zscore_close_spread'].rolling(how_many_candles).min()
    pd_candles['zscore_close_spread_max'] = pd_candles['zscore_close_spread'].rolling(how_many_candles).max()

    calculate_slope(
        pd_data=pd_candles,
        src_col_name='zscore_close_spread',
        slope_col_name='zscore_slope',
        sliding_window_how_many_candles=how_many_candles
    )