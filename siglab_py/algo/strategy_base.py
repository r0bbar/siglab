from abc import ABC, abstractmethod

from typing import List, Dict, Any

class StrategyBase(ABC):
    def __init__(self, *args: object) -> None:
        pass
    
    @staticmethod
    def reversal(
        direction : str,  # up or down
        last_candles
    ) -> bool:
        if direction == "down" and all([ candle[1]<candle[4] for candle in last_candles ]): # All green?
            return True
        elif direction == "up" and all([ candle[1]>candle[4] for candle in last_candles ]): # All red?
            return True
        else:
            return False
        
    @staticmethod
    def order_notional_adj(
        algo_param : Dict,
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        target_order_notional = algo_param['amount_base_ccy']
        return {
            'target_order_notional' : target_order_notional
    }

    @staticmethod
    def allow_entry(
        *args: Any, **kwargs: Any
    )  -> Dict[str, bool]:
        return {
            'long' : False,
            'short' : False
        }

    @staticmethod
    def sl_adj(
        algo_param : Dict,
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        running_sl_percent_hard = algo_param['sl_hard_percent']
        return {
            'running_sl_percent_hard' : running_sl_percent_hard
        }

    @staticmethod
    def trailing_stop_threshold_eval(
        algo_param : Dict,
        *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        tp_min_percent = algo_param['tp_min_percent']
        tp_max_percent = algo_param['tp_max_percent']
        return {
            'tp_min_percent' : tp_min_percent,
            'tp_max_percent' : tp_max_percent
        }

    # List of TA/indicators you wish to include in POSITION_CACHE_COLUMNS from strategy_executor (Display concern only)
    @staticmethod
    def get_strategy_indicators() -> List[str]:
        return []