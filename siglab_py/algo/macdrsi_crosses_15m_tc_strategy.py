from datetime import datetime, timezone
from abc import ABC, abstractmethod
from pprint import pformat
import logging
from typing import List, Dict, Any, Union

from siglab_py.constants import OrderSide 
from siglab_py.algo.strategy_base import StrategyBase
from siglab_py.util.datetime_util import is_us_cash_open_effective

logger = logging.getLogger()

class MACDRSICrosses15mTCStrategy(StrategyBase):
    def __init__(self, *args: object) -> None:
        pass
    
    @staticmethod
    def order_notional_adj(
        algo_param : Dict,
    ) -> Dict[str, float]:
        return StrategyBase.order_notional_adj(algo_param) # type: ignore

    @staticmethod
    def allow_entry_initial(
        lo_row,
        lo_row_tm1,
        algo_param : Dict,
    )  -> Dict[str, bool]:
        allow_long, allow_short = False, False

        allow_entry_dayofweek = algo_param['allow_entry_dayofweek']
        if lo_row['dayofweek'] not in allow_entry_dayofweek:
            return False
        
        utc_now = datetime.fromtimestamp(int(lo_row['timestamp_ms']/1000), tz=timezone.utc)
        us_cash_open_effective : bool = is_us_cash_open_effective(utc_now=utc_now)
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
            allow_long = True
            
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
            allow_short = True

        return {
            'long' : allow_long,
            'short' : allow_short
        }

    @staticmethod
    def allow_entry_final(
        lo_row,
        algo_param : Dict
    ) -> Dict[str, Union[bool, float, None]]:
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
    
    @staticmethod
    def sl_adj(
        lo_row_tm1,
        algo_param : Dict,
        
        strategy_specific_data_cache : Dict[str, Any],
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        target_atr_multiplier : float = algo_param['target_atr_multiplier'] if 'target_atr_multiplier' in algo_param else 1

        sl_hard_percent = algo_param['sl_hard_percent']
        running_sl_percent_hard = min(lo_row_tm1['atr_bps']/100 *target_atr_multiplier, sl_hard_percent)
        assert(running_sl_percent_hard<=algo_param['sl_hard_percent'])
        assert(running_sl_percent_hard>=0)
        return {
            'running_sl_percent_hard' : running_sl_percent_hard
        }
    
    @staticmethod
    def trailing_stop_threshold_eval(
        lo_row_tm1,
        algo_param : Dict,
        strategy_specific_data_cache : Dict[str, Any],
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        target_atr_multiplier : float = algo_param['target_atr_multiplier'] if 'target_atr_multiplier' in algo_param else 1
        tp_min_percent = lo_row_tm1['atr_bps']/100 *target_atr_multiplier
        tp_max_percent = tp_min_percent*3
        return {
            'tp_min_percent' : tp_min_percent,
            'tp_max_percent' : tp_max_percent
        }

    @staticmethod
    def tp_eval (
            mid : float,
            tp_max_target : float,
            pos_side : OrderSide
    ) -> bool:
        tp : bool = False
        if pos_side==OrderSide.BUY:
            tp = True if mid>=tp_max_target else False
        elif pos_side==OrderSide.SELL:
            tp = True if mid<=tp_max_target else False
        return tp

    @staticmethod
    def get_strategy_indicators() -> List[str]:
        return [ 
            'lo_row_tm1:macd_cross_last', 'lo_row_tm1:macd_divergence', 'lo_row_tm1:macd_bullish_cross_last_id', 'lo_row_tm1:macd_bearish_cross_last_id', 'lo_row_tm1:nums_intervals_since_last_cross',
            'lo_row_tm1:close',
            'lo_row_tm1:adx',
        ]

    @staticmethod
    def get_strategy_algo_params() -> List[Dict[str, Any]]:
        return [
            {
                'key' : 'level_granularity',
                'val' : 0.05
            },
        ]