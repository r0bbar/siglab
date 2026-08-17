from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import pandas as pd

from siglab_py.constants import OrderSide 
from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.util.market_data_util import fetch_candles
from siglab_py.util.analytic_util import compute_volume_profile, compute_candles_stats

class StrategyBase(ABC):
    def __init__(self, *args: object) -> None:
        pass

    '''
    strategy_executor already supply pd_hi_candles_w_ta and pd_lo_candles_w_ta with TA precompuited, and also hi_volume_profile, lo_volume_profile.
    These can be referenced from lambdas.
    So why we need a "Trading Context"?
    Generally pd_hi_candles_w_ta and pd_lo_candles_w_ta are calculated with smaller number of bars due to performance constraints.
    "Trading Context" is evaluated over a much longer period of time, the motivation is to give a "Bird eye" view of trading environment to strategies.
    "evaluate_trading_context" can be invoked from "stage_strat_specific_preentry_data" (Generally overridden in strategy sub classes) on say daily basis during quiet hours (Check "evaluation_timestamp_ms"), and when there's no open position.
    '''
    @staticmethod
    def evaluate_trading_context(
        exchange : AnyExchange,
        ticker : str,
        start_ts : int = (datetime.now() + timedelta(days=-30)).timestamp(), # Default three months ago. Also timestamp in seconds (not in ms).
        end_ts : int = datetime.now().timestamp(),
        candle_size : str = '1h',
        sliding_window_how_many_candles : int = 24*7, # Default: TAs are calculated using one week sliding window
        volume_profile_2_num_intervals : int = 24*30,
        volume_profile_3_num_intervals : int = 24*7
    ):
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts,
            end_ts,
            exchange=exchange,
            normalized_symbols = [ ticker ],
            candle_size = candle_size
        )[ticker]
        
        compute_candles_stats(
                pd_candles=pd_candles,
                boillenger_std_multiples=2,
                sliding_window_how_many_candles=sliding_window_how_many_candles, 
                pypy_compat=True
            )
        last_row =  pd_candles.iloc[-1]
        adx = last_row['adx']
        atr = last_row['atr']
        
        volume_profile_3m = compute_volume_profile(
                            pd_candles = pd_candles,
                            level_granularity = 0.1, # i.e. 10%
                            ohlc = 'close'
                        )
        volume_profile_1m = compute_volume_profile(
                            pd_candles = pd_candles.iloc[-volume_profile_2_num_intervals:],
                            level_granularity = 0.1, # i.e. 10%
                            ohlc = 'close'
                        )
        volume_profile_1w = compute_volume_profile(
                            pd_candles = pd_candles.iloc[-volume_profile_3_num_intervals:],
                            level_granularity = 0.1, # i.e. 10%
                            ohlc = 'close'
                        )
        
        return {
            'adx' : adx, # trending vs rangebound
            'atr' : atr, # volatility measures
            'volume_profiles' : {   # @todo: previous APAC, London, US session range 
                'volume_profile_1' : volume_profile_3m,
                'volume_profile_2' : volume_profile_1m,
                'volume_profile_3' : volume_profile_1w
            },
            'evaluation_timestamp_ms' : int(datetime.now().timestamp() *1000)
        }

    @staticmethod
    def stage_strat_specific_preentry_data(
        algo_param : Dict,
        
        pd_hi_candles_w_ta : pd.DataFrame,
        pd_lo_candles_w_ta : pd.DataFrame,
        ob : Dict[str, Any],

        mid : float,
        best_ask : float,
        best_bid : float,
        lo_row_tm1,

        dt_targettz : datetime,

        this_ticker_open_trades : List[Dict],

        strategy_specific_data_cache : Dict[str, Any],

        exchange : AnyExchange
    ) -> Dict[str, str]:
        '''
        a. pd_hi_candles_w_ta and pd_lo_candles_w_ta are candles from strategy_executor. 
        b. ob is order book.
        Strategy specific data pre-processing can be done here: Stick them into data_cache where needed.
        
        Return a List[str] remarks if any, otherwise null. 
        '''
        return None

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
    def allow_entry_initial(
        *args: Any, **kwargs: Any
    )  -> Dict[str, bool]:
        return {
            'long' : False,
            'short' : False
        }

    @staticmethod
    def allow_entry_final(
        lo_row,
        algo_param : Dict,
        *args: Any, **kwargs: Any
    ) -> Dict[str, Union[bool, float, None]]:
        open : float = lo_row['open']

        entry_price_long, entry_price_short = open, open
        allow_long, allow_short = False, False
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

    # List of TA/indicators you wish to include in POSITION_CACHE_COLUMNS from strategy_executor (Display concern only)
    @staticmethod
    def get_strategy_indicators() -> List[str]:
        return []

    @staticmethod
    def get_strategy_algo_params() -> List[Dict[str, Any]]:
        '''
        [
            {
                'key' : 'rsi_lower',
                'val' : 30
            }
        ]
        '''
        return [
            
        ]