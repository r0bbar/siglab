import unittest
from typing import List
from pathlib import Path

from util.analytic_util import compute_candles_stats, lookup_fib_target

import pandas as pd

'''
Manual checks against for example Tradingview here: \siglab\siglab_py\tests\manual
'''

# @unittest.skip("Skip all integration tests.")
class AnalyticUtilTests(unittest.TestCase):

    def test_compute_candle_stats(self):
        '''
        Folder structure:
            \ siglab
                \ siglab_py	<-- python project root
                    \ sigab_py
                        __init__.py
                        \ util
                            __init__.py
                            market_data_util.py
                        \ tests
                            \ unit
                                __init__.py
                                analytic_util_tests.py <-- Tests here
                            
                \ siglab_rs <-- Rust project root
                \ data	 <-- Data files here!
        '''
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        csv_path = data_dir / "sample_btc_candles.csv"
        pd_candles : pd.DataFrame = pd.read_csv(csv_path)
        compute_candles_stats(
            pd_candles=pd_candles,
            boillenger_std_multiples=2,
            sliding_window_how_many_candles=20,
            pypy_compat=True # Slopes calculation? Set pypy_compat to False
        )

        expected_columns : List[str] = [
            'exchange', 'symbol', 'timestamp_ms', 
            'open', 'high', 'low', 'close', 'volume', 
            'datetime', 'datetime_utc', 'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 
            'pct_chg_on_close', 'candle_height', 'candle_body_height',
            'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr', 
            'is_green', 'pct_change_close', 
            'sma_short_periods', 'sma_long_periods', 'ema_short_periods', 'ema_long_periods', 'ema_close', 
            'std', 'std_percent', 
            'vwap_short_periods', 'vwap_long_periods',
            'candle_height_percent', 'candle_height_percent_rounded', 'candle_body_height_percent', 'candle_body_height_percent_rounded',
            'log_return', 'interval_hist_vol', 'annualized_hist_vol',
            'gap_ema_close_vs_ema', 'gap_ema_close_vs_ema_percent',
            'chop_against_ema', 
            'ema_volume_short_periods', 'ema_volume_long_periods', 
            'ema_cross', 'ema_cross_last', 'ema_bullish_cross_last_id', 'ema_bearish_cross_last_id',
            'max_short_periods', 'max_long_periods', 'idmax_short_periods', 'idmax_long_periods', 'min_short_periods', 'min_long_periods', 'idmin_short_periods', 'idmin_long_periods', 
            'max_candle_body_height_percent_long_periods', 'idmax_candle_body_height_percent_long_periods',
            'min_candle_body_height_percent_long_periods', 'idmin_candle_body_height_percent_long_periods',
            'price_swing_short_periods', 'price_swing_long_periods',
            'trend_from_highs_long_periods', 'trend_from_lows_long_periods', 'trend_from_highs_short_periods', 'trend_from_lows_short_periods',
            'h_l', 'h_pc', 'l_pc', 'tr', 'atr', 'atr_avg_short_periods', 'atr_avg_long_periods',
            'hurst_exp', 
            'boillenger_upper', 'boillenger_lower', 'boillenger_channel_height', 'boillenger_upper_agg', 'boillenger_lower_agg', 'boillenger_channel_height_agg', 
            'aggressive_up', 'aggressive_up_index', 'aggressive_up_candle_height', 'aggressive_up_candle_high', 'aggressive_up_candle_low', 'aggressive_down', 'aggressive_down_index', 'aggressive_down_candle_height', 'aggressive_down_candle_high', 'aggressive_down_candle_low', 
            'fvg_low', 'fvg_high', 'fvg_gap', 'fvg_mitigated', 
            'close_delta', 'close_delta_percent', 'up', 'down', 
            'rsi', 'rsi_bucket', 'ema_rsi', 'rsi_max', 'rsi_idmax', 'rsi_min', 'rsi_idmin', 'rsi_trend', 'rsi_trend_from_highs', 'rsi_trend_from_lows', 'rsi_divergence',
            'typical_price', 
            'money_flow', 'money_flow_positive', 'money_flow_negative', 'positive_flow_sum', 'negative_flow_sum', 'money_flow_ratio', 'mfi', 'mfi_bucket',
            'macd', 'signal', 'macd_minus_signal', 
            'fib_0.618_short_periods', 'fib_0.618_long_periods', 
            'gap_close_vs_ema', 
            'close_above_or_below_ema', 
            'close_vs_ema_inflection'
        ]

        missing_columns = [ expected for expected in expected_columns if expected not in pd_candles.columns.to_list() ]
        unexpected_columns = [ actual for actual in pd_candles.columns.to_list() if actual not in expected_columns ]

        assert(pd_candles.columns.to_list()==expected_columns)

    def test_lookup_fib_target(self):
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        csv_path = data_dir / "sample_btc_candles.csv"
        pd_candles : pd.DataFrame = pd.read_csv(csv_path)
        target_fib_level : float = 0.618
        compute_candles_stats(
            pd_candles=pd_candles,
            boillenger_std_multiples=2,
            sliding_window_how_many_candles=20,
            target_fib_level=target_fib_level,
            pypy_compat=True # Slopes calculation? Set pypy_compat to False
        )

        last_row = pd_candles.iloc[-1]
        result = lookup_fib_target(
            row=last_row, 
            pd_candles=pd_candles,
            target_fib_level=target_fib_level
            )
        if result:
            assert(result['short_periods']['min']<result['short_periods']['fib_target']<result['short_periods']['max'])
            assert(result['long_periods']['min']<result['long_periods']['fib_target']<result['long_periods']['max'])
        

