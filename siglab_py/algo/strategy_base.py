from abc import ABC, abstractmethod

from typing import List, Dict, Any

from siglab_py.backtests.backtest_core import generic_tp_eval

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

    @staticmethod
    def tp_eval (
            lo_row,
            this_ticker_open_trades : List[Dict],
            algo_param : Dict
    ) -> bool:
        '''
        Be very careful, backtest_core 'generic_pnl_eval' may use a) some indicator (tp_indicator_name), or b) target_price to evaluate 'unrealized_pnl_tp'.
        'tp_eval' only return True or False but it needs be congruent with backtest_core 'generic_pnl_eval', otherwise incorrect rosy pnl may be reported.
        '''
        return generic_tp_eval(lo_row, this_ticker_open_trades)

    # List of TA/indicators you wish to include in POSITION_CACHE_COLUMNS from strategy_executor (Display concern only)
    @staticmethod
    def get_strategy_indicators() -> List[str]:
        return []