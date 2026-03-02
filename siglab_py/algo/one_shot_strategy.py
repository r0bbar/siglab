import math
from pprint import pformat
from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
import logging

from siglab_py.constants import OrderSide 
from siglab_py.algo.strategy_base import StrategyBase
from siglab_py.util.simple_math import compute_adjacent_levels

logger = logging.getLogger()
        
'''

'''
class OneShotStrategy(StrategyBase):
    def __init__(self, *args: object) -> None:
        pass
    
    @staticmethod
    def order_notional_adj(
        algo_param : Dict,
    ) -> Dict[str, float]:
        return StrategyBase.order_notional_adj(algo_param) # type: ignore

    @staticmethod
    def allow_entry_initial(
        mid,
        lo_row_tm1,
        hi_row_tm1,
        last_candles
    )  -> Dict[str, bool]:
        allow_long : bool = True
        allow_short : bool = True
            
        return {
            'long' : allow_long,
            'short' : allow_short
        }

    @staticmethod
    def allow_entry_final(
        mid, # Different from backtest but more appropriate
        lo_row_tm1,
        algo_param : Dict
    ) -> Dict[str, Union[bool, float, None]]:
        allow_long : bool = True
        allow_short : bool = True
        
        entry_price_long, entry_price_short = mid, mid

        tp_max_bps = algo_param['tp_max_percent']*100

        target_price_long : float = mid * (1 + tp_max_bps/10000)
        target_price_short : float = mid * (1 - tp_max_bps/10000)

        return {
                'long' : allow_long,
                'short' : allow_short,

                # In additional to allow or not, allow_entry_final also calculate a few things which you may need to mark the entry trades.
                'entry_price_long' : entry_price_long,
                'entry_price_short' : entry_price_short,
                'target_price_long' : target_price_long,
                'target_price_short' : target_price_short,
                'reference_price' : None
            }
    
    @staticmethod
    def sl_adj(
        max_unreal_live : float,
        pos_usdt : float,
        this_ticker_open_trades : List[Dict],
        algo_param : Dict,
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        running_sl_percent_hard = algo_param['sl_hard_percent']
        sl_adj_percent = algo_param['sl_adj_percent']

        # sl_hard_percent adj: Strategies where target_price not based on tp_max_percent, but variable
        max_pnl_potential_percent = None
        if any([ trade for trade in this_ticker_open_trades if 'target_price' in trade ]):
            max_pnl_potential_percent = max([ (trade['target_price']/trade['entry_price'] -1) *100 if trade['side']=='buy' else (trade['entry_price']/trade['target_price'] -1) *100 for trade in this_ticker_open_trades if 'target_price' in trade ])

            if max_pnl_potential_percent and max_pnl_potential_percent<algo_param['tp_max_percent']:
                tp_minmax_ratio = algo_param['tp_min_percent']/algo_param['tp_max_percent']
                tp_min_sl_hard_ratio  = algo_param['sl_hard_percent']/algo_param['tp_min_percent']

                tp_max_percent = max_pnl_potential_percent
                tp_min_percent = tp_minmax_ratio * tp_max_percent
                running_sl_percent_hard = tp_min_sl_hard_ratio * tp_min_percent
        
        max_pnl_percent_notional = max_unreal_live / pos_usdt * 100 if pos_usdt else 0

        if max_pnl_percent_notional>=algo_param['sl_hard_percent']:
            running_sl_percent_hard = min(
                    running_sl_percent_hard,
                    max(0, running_sl_percent_hard - max_pnl_percent_notional * sl_adj_percent)
                )

        assert(running_sl_percent_hard<=algo_param['sl_hard_percent'])

        return {
            'running_sl_percent_hard' : running_sl_percent_hard
        }
    
    @staticmethod
    def trailing_stop_threshold_eval(
        algo_param : Dict,
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        result = StrategyBase.trailing_stop_threshold_eval(algo_param)
        tp_min_percent = result['tp_min_percent']
        tp_max_percent = result['tp_max_percent']
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
            'lo_row_tm1:close',
            'hi_row_tm1:ema_close',
            'lo_row_tm1:atr', 'lo_row_tm1:atr_bps'
        ]

    @staticmethod
    def get_strategy_algo_params() -> List[Dict[str, Any]]:
        return [
            {
                'key' : 'one_shot_only',
                'val' : True
            }
        ]