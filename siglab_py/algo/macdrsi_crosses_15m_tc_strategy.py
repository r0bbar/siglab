from abc import ABC, abstractmethod

from typing import List, Dict

from algo.strategy_base import StrategyBase

class MACDRSICrosses15mTCStrategy(StrategyBase):
    def __init__(self, *args: object) -> None:
        pass
    
    @staticmethod
    def allow_entry(
        lo_row_tm1,
        hi_row_tm1,
        last_candles
    )  -> Dict[str, bool]:
        allow_long : bool = (
                lo_row_tm1['macd_cross'] == 'bullish'
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bullish_cross_last_id']
                      and 
                      (lo_row_tm1.name - lo_row_tm1['macd_bullish_cross_last_id']) < 5
                )
                and lo_row_tm1['rsi_trend']=="up"
                and lo_row_tm1['close']>hi_row_tm1['ema_close']
        )
        allow_short : bool = (
                lo_row_tm1['macd_cross'] == 'bearish'
                and (
                      lo_row_tm1.name >= lo_row_tm1['macd_bearish_cross_last_id']
                      and 
                      (lo_row_tm1.name - lo_row_tm1['macd_bearish_cross_last_id']) < 5
                )
                and lo_row_tm1['rsi_trend']=="down"
                and lo_row_tm1['close']<hi_row_tm1['ema_close']
        )
        return {
            'long' : allow_long,
            'short' : allow_short
        }
    
    @staticmethod
    def get_strategy_indicators() -> List[str]:
        return [ 
            'lo_row_tm1:macd_cross', 'lo_row_tm1:macd_bullish_cross_last_id', 'lo_row_tm1:macd_bearish_cross_last_id', 
            'lo_row_tm1:rsi_trend',  'lo_row_tm1:rsi', 'lo_row_tm1:rsi_max', 'lo_row_tm1:rsi_min', 'lo_row_tm1:rsi_idmax', 'lo_row_tm1:rsi_idmin' 
        ]