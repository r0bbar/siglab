from abc import ABC, abstractmethod

from typing import List, Dict, Any, Union

from siglab_py.algo.strategy_base import StrategyBase

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
        lo_row_tm1,
        hi_row_tm1,
        last_candles
    )  -> Dict[str, bool]:
        allow_long : bool = (
                lo_row_tm1['macd_cross'] == 'bullish' 
                # use 'macd_cross_last' instead in combinations with 'macd_bullish_cross_last_id' if you want to make more entries
                '''
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bullish_cross_last_id']
                      and 
                      (lo_row_tm1.name - lo_row_tm1['macd_bullish_cross_last_id']) < 5
                )
                '''
                and lo_row_tm1['rsi_trend']=="up"
                and lo_row_tm1['close']>hi_row_tm1['ema_close']
        )
        allow_short : bool = (
                lo_row_tm1['macd_cross'] == 'bearish'
                '''
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bearish_cross_last_id']
                      and 
                      (lo_row_tm1.name - lo_row_tm1['macd_bearish_cross_last_id']) < 5
                )
                '''
                and lo_row_tm1['rsi_trend']=="down"
                and lo_row_tm1['close']<hi_row_tm1['ema_close']
        )
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
        return StrategyBase.tp_eval(mid, tp_max_target, pos_side)

    @staticmethod
    def get_strategy_indicators() -> List[str]:
        return [ 
            'lo_row_tm1:macd_cross', 'lo_row_tm1:macd_bullish_cross_last_id', 'lo_row_tm1:macd_bearish_cross_last_id', 
            'lo_row_tm1:rsi_trend',  'lo_row_tm1:rsi', 'lo_row_tm1:rsi_max', 'lo_row_tm1:rsi_min', 'lo_row_tm1:rsi_idmax', 'lo_row_tm1:rsi_idmin' 
        ]