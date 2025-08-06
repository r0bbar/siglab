import unittest
from typing import List
from pathlib import Path

from util.analytic_util import compute_candles_stats

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
            pypy_compat=True
        )

        expected_columns : List[str] = [
            'exchange', 'symbol', 'timestamp_ms', 
            'open', 'high', 'low', 'close', 'volume', 
            'datetime', 'datetime_utc', 'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 
            'pct_chg_on_close', 'candle_height', 
            'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr', 
            'is_green', 'pct_change_close', 
            'sma_short_periods', 'sma_long_periods', 'ema_short_periods', 'ema_long_periods', 'ema_close', 
            'std', 'std_percent', 
            'vwap_short_periods', 'vwap_long_periods',
            'candle_height_percent', 'candle_height_percent_rounded', 
            'log_return', 'interval_hist_vol', 'annualized_hist_vol',
            'chop_against_ema', 
            'ema_volume_short_periods', 'ema_volume_long_periods', 
            'max_short_periods', 'max_long_periods', 'idmax_short_periods', 'idmax_long_periods', 'min_short_periods', 'min_long_periods', 'idmin_short_periods', 'idmin_long_periods', 
            'price_swing_short_periods', 'price_swing_long_periods',
            'higher_highs_long_periods', 'lower_lows_long_periods', 'higher_highs_short_periods', 'lower_lows_short_periods',
            'h_l', 'h_pc', 'l_pc', 'tr', 'atr', 'atr_avg_short_periods', 'atr_avg_long_periods',
            'hurst_exp', 
            'boillenger_upper', 'boillenger_lower', 'boillenger_channel_height', 'boillenger_upper_agg', 'boillenger_lower_agg', 'boillenger_channel_height_agg', 
            'aggressive_up', 'aggressive_up_index', 'aggressive_up_candle_height', 'aggressive_up_candle_high', 'aggressive_up_candle_low', 'aggressive_down', 'aggressive_down_index', 'aggressive_down_candle_height', 'aggressive_down_candle_high', 'aggressive_down_candle_low', 
            'fvg_low', 'fvg_high', 'fvg_gap', 'fvg_mitigated', 
            'close_delta', 'close_delta_percent', 'up', 'down', 
            'rsi', 'ema_rsi', 'rsi_max', 'rsi_idmax', 'rsi_min', 'rsi_idmin', 'rsi_trend', 'rsi_higher_highs', 'rsi_lower_lows', 'rsi_divergence',
            'typical_price', 
            'money_flow', 'money_flow_positive', 'money_flow_negative', 'positive_flow_sum', 'negative_flow_sum', 'money_flow_ratio', 'mfi', 
            'macd', 'signal', 'macd_minus_signal', 
            'fib_0.618_short_periods', 'fib_0.618_long_periods', 
            'gap_close_vs_ema', 
            'close_above_or_below_ema', 
            'close_vs_ema_inflection'
        ]

        missing_columns = [ expected for expected in expected_columns if expected not in pd_candles.columns.to_list() ]
        unexpected_columns = [ actual for actual in pd_candles.columns.to_list() if actual not in expected_columns ]

        assert(pd_candles.columns.to_list()==expected_columns)