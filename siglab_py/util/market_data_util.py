import tzlocal
from datetime import datetime, timezone
from typing import List, Dict, Union, NoReturn, Any, Tuple
from pathlib import Path
import math
import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm # in-compatible with pypy
from hurst import compute_Hc # compatible with pypy

from ccxt.base.exchange import Exchange as CcxtExchange
from ccxt import deribit

# https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/
from yahoofinancials import YahooFinancials

# yfinance allows intervals '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'. yahoofinancials not as flexible
import yfinance as yf

def timestamp_to_datetime_cols(pd_candles : pd.DataFrame):
    pd_candles['datetime'] = pd_candles['timestamp_ms'].apply(
        lambda x: datetime.fromtimestamp(int(x.timestamp()) if isinstance(x, pd.Timestamp) else int(x / 1000))
    )
    pd_candles['datetime'] = pd.to_datetime(pd_candles['datetime'])
    pd_candles['datetime'] = pd_candles['datetime'].dt.tz_localize(None)
    pd_candles['datetime_utc'] = pd_candles['timestamp_ms'].apply(
        lambda x: datetime.fromtimestamp(int(x.timestamp()) if isinstance(x, pd.Timestamp) else int(x / 1000), tz=timezone.utc)
    )
    
    # This is to make it easy to do grouping with Excel pivot table
    pd_candles['year'] = pd_candles['datetime'].dt.year
    pd_candles['month'] = pd_candles['datetime'].dt.month
    pd_candles['day'] = pd_candles['datetime'].dt.day
    pd_candles['hour'] = pd_candles['datetime'].dt.hour
    pd_candles['minute'] = pd_candles['datetime'].dt.minute
    pd_candles['dayofweek'] = pd_candles['datetime'].dt.dayofweek  # dayofweek: Monday is 0 and Sunday is 6

def fix_column_types(pd_candles : pd.DataFrame):
    pd_candles['open'] = pd_candles['open'].astype(float)
    pd_candles['high'] = pd_candles['high'].astype(float)
    pd_candles['low'] = pd_candles['low'].astype(float)
    pd_candles['close'] = pd_candles['close'].astype(float)
    pd_candles['volume'] = pd_candles['volume'].astype(float)

    timestamp_to_datetime_cols(pd_candles)

    '''
    The 'Unnamed: 0', 'Unnamed : 1'... etc columns often appears in a DataFrame when it is saved to a file (e.g., CSV or Excel) and later loaded. 
    This usually happens if the DataFrame's index was saved along with the data, and then pandas automatically treats it as a column during the file loading process.
    We want to drop them as it'd mess up idmin, idmax calls, which will take values from 'Unnamed' instead of actual pandas index.
    '''
    pd_candles.drop(pd_candles.columns[pd_candles.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    pd_candles.reset_index(drop=True, inplace=True)
    pd_candles.sort_values("datetime", inplace=True)

'''
https://polygon.io/docs/stocks
'''
class PolygonMarketDataProvider:
    pass

class NASDAQExchange:
    def __init__(self, data_dir : Union[str, None]) -> None:
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = Path(__file__).resolve().parents[2] / "data/nasdaq" 

    def fetch_ohlcv(
        self,
        symbol : str,
        since : int,
        timeframe : str,
        limit : int = 1
    ) -> List:
        pd_candles = self.fetch_candles(
            symbols=[symbol],
            start_ts=int(since/1000),
            end_ts=None,
            candle_size=timeframe
        )[symbol]
        if pd_candles is not None:
            return pd_candles.values.tolist()
        else:
            return []
    
    def fetch_candles(
        self,
        start_ts,
        end_ts,
        symbols,
        candle_size
    ) -> Dict[str, Union[pd.DataFrame, None]]:
        exchange_candles : Dict[str, Union[pd.DataFrame, None]] = {}

        start_date = datetime.fromtimestamp(start_ts)
        end_date = datetime.fromtimestamp(end_ts) if end_ts else None
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None
        local_tz = datetime.now().astimezone().tzinfo

        for symbol in symbols:
            # CSV from NASDAQ: https://www.nasdaq.com/market-activity/quotes/historical
            pd_daily_candles = pd.read_csv(f"{self.data_dir}\\NASDAQ_hist_{symbol.replace('^','')}.csv")
            pd_daily_candles.rename(columns={'Date' : 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close/Last' : 'close', 'Adj Close' : 'adj_close', 'Volume' : 'volume' }, inplace=True)
            pd_daily_candles['open'] = pd_daily_candles['open'].astype(str).str.replace('$','')
            pd_daily_candles['high'] = pd_daily_candles['high'].astype(str).str.replace('$','')
            pd_daily_candles['low'] = pd_daily_candles['low'].astype(str).str.replace('$','')
            pd_daily_candles['close'] = pd_daily_candles['close'].astype(str).str.replace('$','')
            pd_daily_candles['datetime']= pd.to_datetime(pd_daily_candles['datetime'])
            pd_daily_candles['timestamp_ms'] = pd_daily_candles.datetime.values.astype(np.int64) // 10 ** 6
            pd_daily_candles['symbol'] = symbol
            pd_daily_candles['exchange'] = 'nasdaq'
            fix_column_types(pd_daily_candles)

            if candle_size=="1h":
                # Fill forward (i.e. you dont actually have hourly candles)
                start = pd_daily_candles["datetime"].min().normalize()
                end = pd_daily_candles["datetime"].max().normalize() + pd.Timedelta(days=1)
                hourly_index = pd.date_range(start=start, end=end, freq="h") # FutureWarning: 'H' is deprecated and will be removed in a future version, please use 'h' instead.
                pd_hourly_candles = pd.DataFrame({"datetime": hourly_index})
                pd_hourly_candles = pd.merge_asof(
                    pd_hourly_candles.sort_values("datetime"),
                    pd_daily_candles.sort_values("datetime"),
                    on="datetime",
                    direction="backward"
                )

                # When you fill foward, a few candles before start date can have null values (open, high, low, close, volume ...)
                first_candle_dt = pd_hourly_candles[(~pd_hourly_candles.close.isna())  & (pd_hourly_candles['datetime'].dt.time == pd.Timestamp('00:00:00').time())].iloc[0]['datetime']
                pd_hourly_candles = pd_hourly_candles[pd_hourly_candles.datetime>=first_candle_dt]
                exchange_candles[symbol] = pd_hourly_candles

            elif candle_size=="1d":
                exchange_candles[symbol] = pd_daily_candles

        return exchange_candles

class YahooExchange:
    def fetch_ohlcv(
        self,
        symbol : str,
        since : int,
        timeframe : str,
        limit : int = 1
    ) -> List:
        pd_candles = self.fetch_candles(
            symbols=[symbol],
            start_ts=int(since/1000),
            end_ts=None,
            candle_size=timeframe
        )[symbol]
        if pd_candles is not None:
            return pd_candles.values.tolist()
        else:
            return []

    def fetch_candles(
        self,
        start_ts,
        end_ts,
        symbols,
        candle_size
    ) -> Dict[str, Union[pd.DataFrame, None]]:
        exchange_candles : Dict[str, Union[pd.DataFrame, None]] = {}

        start_date = datetime.fromtimestamp(start_ts)
        end_date = datetime.fromtimestamp(end_ts)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        local_tz = datetime.now().astimezone().tzinfo

        for symbol in symbols:
            # From yf, "DateTime" in UTC
            # The requested range must be within the last 730 days. Otherwise API will return empty DataFrame.
            pd_candles = yf.download(tickers=symbol, start=start_date_str, end=end_date_str, interval=candle_size)
            pd_candles.reset_index(inplace=True)
            pd_candles.rename(columns={ 'Date' : 'datetime', 'Datetime' : 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close' : 'close', 'Adj Close' : 'adj_close', 'Volume' : 'volume' }, inplace=True)
            pd_candles['datetime'] = pd.to_datetime(pd_candles['datetime'])
            if pd_candles['datetime'].dt.tz is None:
                pd_candles['datetime'] = pd.to_datetime(pd_candles['datetime']).dt.tz_localize('UTC')
            pd_candles['datetime'] = pd_candles['datetime'].dt.tz_convert(local_tz)
            pd_candles['datetime'] = pd_candles['datetime'].dt.tz_localize(None)
            pd_candles['timestamp_ms'] = pd_candles.datetime.values.astype(np.int64) // 10**6
            pd_candles = pd_candles.sort_values(by=['timestamp_ms'], ascending=[True])
            
            fix_column_types(pd_candles)
            pd_candles['symbol'] = symbol
            pd_candles['exchange'] = 'yahoo'
            exchange_candles[symbol] = pd_candles

        return exchange_candles

def fetch_historical_price(
    exchange,
    normalized_symbol : str,
    timestamp_ms : int,
    ref_timeframe : str = '1m'
):
    one_candle = fetch_ohlcv_one_candle(exchange=exchange, normalized_symbol=normalized_symbol, timestamp_ms=timestamp_ms, ref_timeframe=ref_timeframe)
    reference_price = abs(one_candle['close'] + one_candle['open'])/2 if one_candle else None
    return reference_price

def fetch_ohlcv_one_candle(
    exchange,
    normalized_symbol : str,
    timestamp_ms : int,
    ref_timeframe : str = '1m'
):
    candles = exchange.fetch_ohlcv(symbol=normalized_symbol, since=int(timestamp_ms), timeframe=ref_timeframe, limit=1)
    one_candle = {
            'timestamp_ms' : candles[0][0],
            'open' : candles[0][1],
            'high' : candles[0][2],
            'low' : candles[0][3],
            'close' : candles[0][4],
            'volume' : candles[0][5] 
        } if candles and len(candles)>0 else None
    
    return one_candle
    
def fetch_candles(
    start_ts, # in sec
    end_ts, # in sec
    exchange,
    normalized_symbols,
    candle_size,

    logger = None,

    num_candles_limit : int = 100,

    cache_dir : Union[str, None] = None,

    list_ts_field : Union[str, None] = None,

    validation_max_gaps : int = 10,
    validation_max_end_date_intervals : int = 1
) -> Dict[str, Union[pd.DataFrame, None]]:
    if type(exchange) is YahooExchange:
        return exchange.fetch_candles(
                            start_ts=start_ts,
                            end_ts=end_ts,
                            symbols=normalized_symbols,
                            candle_size=candle_size
                        )
    elif type(exchange) is NASDAQExchange:
        return exchange.fetch_candles(
                            start_ts=start_ts,
                            end_ts=end_ts,
                            symbols=normalized_symbols,
                            candle_size=candle_size
                        )
    elif issubclass(exchange.__class__, CcxtExchange):
        return _fetch_candles_ccxt(
            start_ts=start_ts,
            end_ts=end_ts,
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size=candle_size,
            logger=logger,
            num_candles_limit=num_candles_limit,
            cache_dir=cache_dir,
            list_ts_field=list_ts_field
        )
    return { '' : None }

def _fetch_candles_ccxt(
    start_ts : int,
    end_ts : int,
    exchange,
    normalized_symbols : List[str],
    candle_size : str,
    num_candles_limit : int = 100,
    logger = None,
    cache_dir : Union[str, None] = None,
    list_ts_field : Union[str, None] = None
) -> Dict[str, Union[pd.DataFrame, None]]:
  ticker = normalized_symbols[0]
  pd_candles = _fetch_candles(
              symbol = ticker,
              exchange = exchange,
              start_ts = start_ts,
              end_ts = end_ts,
              candle_size = candle_size,
          )
  return {
      ticker : pd_candles
  }

def _fetch_candles(
    symbol : str,
    exchange : CcxtExchange,
    start_ts : int,
    end_ts : int,
    candle_size : str = '1d',
    num_candles_limit : int = 100
):
    def _fetch_ohlcv(exchange, symbol, timeframe, since, limit, params) -> Union[List, NoReturn]:
        one_timeframe = f"1{timeframe[-1]}"
        candles = exchange.fetch_ohlcv(symbol=symbol, timeframe=one_timeframe, since=since, limit=limit, params=params)
        if candles and len(candles)>0:
            candles.sort(key=lambda x : x[0], reverse=False)

        return candles

    all_candles = []
    params = {}
    this_cutoff = start_ts
    while this_cutoff<=end_ts:
        candles = _fetch_ohlcv(exchange=exchange, symbol=symbol, timeframe=candle_size, since=int(this_cutoff * 1000), limit=num_candles_limit, params=params)
        if candles and len(candles)>0:
            all_candles = all_candles + [[ int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]) ] for x in candles if x[1] and x[2] and x[3] and x[4] and x[5] ]

            record_ts = max([int(record[0]) for record in candles])
            record_ts_str : str = str(record_ts)
            if len(record_ts_str)==13:
                record_ts = int(int(record_ts_str)/1000) # Convert from milli-seconds to seconds

            this_cutoff = record_ts  + 1
    columns = ['exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume']
    pd_all_candles = pd.DataFrame([ [ exchange.name, symbol, x[0], x[1], x[2], x[3], x[4], x[5] ] for x in all_candles], columns=columns)
    fix_column_types(pd_all_candles)
    pd_all_candles['pct_chg_on_close'] = pd_all_candles['close'].pct_change()
    return pd_all_candles


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

'''
compute_candles_stats will calculate typical/basic technical indicators using in many trading strategies:
    a. Basic SMA/EMAs (And slopes)
    b. ATR
    c. Boillenger bands (Yes incorrect spelling sorry)
    d. FVG
    e. Hurst Exponent
    f. RSI, MFI
    g. MACD
    h. Fibonacci
    i. Inflections points: where 'close' crosses EMA from above or below.

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
        rsi_sliding_window_how_many_candles : Union[int, None] = None, # RSI standard 14
        hurst_exp_window_how_many_candles : Union[int, None] = None, # Hurst exp standard 100-200
        boillenger_std_multiples_for_aggressive_moves_detect : int = 3 # Aggressive moves if candle low/high breaches boillenger bands from 3 standard deviations.
        ):
    pd_candles['candle_height'] = pd_candles['high'] - pd_candles['low']

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

    pd_candles['pct_change_close'] = pd_candles['close'].pct_change() * 100
    pd_candles['sma_short_periods'] = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).mean()
    pd_candles['sma_long_periods'] = pd_candles['close'].rolling(window=sliding_window_how_many_candles).mean()
    pd_candles['ema_short_periods'] = pd_candles['close'].ewm(span=int(sliding_window_how_many_candles/slow_fast_interval_ratio), adjust=False).mean()
    pd_candles['ema_long_periods'] = pd_candles['close'].ewm(span=sliding_window_how_many_candles, adjust=False).mean()
    pd_candles['ema_close'] = pd_candles['ema_long_periods'] # Alias, shorter name
    pd_candles['std'] = pd_candles['close'].rolling(window=sliding_window_how_many_candles).std()

    pd_candles['ema_volume_short_periods'] = pd_candles['volume'].ewm(span=sliding_window_how_many_candles/slow_fast_interval_ratio, adjust=False).mean()
    pd_candles['ema_volume_long_periods'] = pd_candles['volume'].ewm(span=sliding_window_how_many_candles, adjust=False).mean()

    pd_candles['max_short_periods'] = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).max()
    pd_candles['max_long_periods'] = pd_candles['close'].rolling(window=sliding_window_how_many_candles).max()
    pd_candles['idmax_short_periods'] = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(lambda x : x.idxmax())
    pd_candles['idmax_long_periods'] = pd_candles['close'].rolling(window=sliding_window_how_many_candles).apply(lambda x : x.idxmax())

    pd_candles['min_short_periods'] = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).min()
    pd_candles['min_long_periods'] = pd_candles['close'].rolling(window=sliding_window_how_many_candles).min()
    pd_candles['idmin_short_periods'] = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(lambda x : x.idxmin())
    pd_candles['idmin_long_periods'] = pd_candles['close'].rolling(window=sliding_window_how_many_candles).apply(lambda x : x.idxmin())


    # ATR https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58 
    pd_candles.loc[:,'h_l'] = pd_candles['high'] - pd_candles['low']
    pd_candles.loc[:,'h_pc'] = abs(pd_candles['high'] - pd_candles['close'].shift(1))
    pd_candles.loc[:,'l_pc'] = abs(pd_candles['low'] - pd_candles['close'].shift(1))
    pd_candles.loc[:,'tr'] = pd_candles[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    pd_candles.loc[:,'atr'] = pd_candles['tr'].rolling(window=sliding_window_how_many_candles).mean()


    '''
    @hardcode @todo
    Hurst https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
    Smaller Windows (e.g., 50–100)
    Larger Windows (e.g., 200+)

    Sometimes you may encounter "Exception has occurred: FloatingPointError invalid value encountered in scalar divide"
    And for example adjusting window size from 120 to 125 will resolve the issue.
    '''
    pd_candles['hurst_exp'] = pd_candles['close'].rolling(
        window=(
            hurst_exp_window_how_many_candles if hurst_exp_window_how_many_candles else (sliding_window_how_many_candles if sliding_window_how_many_candles>=125 else 125)
            )
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
        
        if up_or_down:
            aggressive_mask = window['close'] >= window['boillenger_upper_agg']
            if aggressive_mask.any():
                first_breach_index = aggressive_mask.idxmax()
        else:
            aggressive_mask = window['close'] <= window['boillenger_lower_agg']
            if aggressive_mask.any():
                first_breach_index = aggressive_mask.idxmax()

        return {
            'aggressive_move': aggressive_mask.any(),
            'first_breach_index': first_breach_index
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


    # RSI - https://www.youtube.com/watch?v=G9oUTi-PI18&t=809s 
    pd_candles.loc[:,'close_delta'] = pd_candles['close'].diff()
    pd_candles.loc[:,'close_delta_percent'] = pd_candles['close'].pct_change()
    lo_up = pd_candles['close_delta'].clip(lower=0)
    lo_down = -1 * pd_candles['close_delta'].clip(upper=0)
    pd_candles.loc[:,'up'] = lo_up
    pd_candles.loc[:,'down'] = lo_down

    if rsi_ema == True:
        # Use exponential moving average
        lo_ma_up = lo_up.ewm(
            com = (rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles) - 1, 
            adjust=True, 
            min_periods = rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).mean()
        lo_ma_down = lo_down.ewm(
            com = (rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles) - 1, 
            adjust=True, 
            min_periods = rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).mean()

    else:
        # Use simple moving average
        lo_ma_up = lo_up.rolling(window = rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).mean()
        lo_ma_down = lo_down.rolling(window = rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles).mean()
        
    lo_rs = lo_ma_up / lo_ma_down
    pd_candles.loc[:,'rsi'] = 100 - (100/(1 + lo_rs))
    pd_candles['ema_rsi'] = pd_candles['rsi'].ewm(
        span=rsi_sliding_window_how_many_candles if rsi_sliding_window_how_many_candles else sliding_window_how_many_candles, 
        adjust=False).mean()


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
    

    # MACD https://www.investopedia.com/terms/m/macd.asp
    pd_candles['macd'] = pd_candles['ema_short_periods'] - pd_candles['ema_long_periods']
    pd_candles['signal'] = pd_candles['macd'].ewm(span=9, adjust=False).mean()
    pd_candles['macd_minus_signal'] = pd_candles['macd'] - pd_candles['signal']


    # Slopes
    X = sm.add_constant(range(len(pd_candles['close'])))
    rolling_slope = pd_candles['close'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['close_short_slope'] = rolling_slope

    X = sm.add_constant(range(len(pd_candles['close'])))
    rolling_slope = pd_candles['close'].rolling(window=sliding_window_how_many_candles).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['close_long_slope'] = rolling_slope
    
    X = sm.add_constant(range(len(pd_candles['ema_short_periods'])))
    rolling_slope = pd_candles['ema_short_periods'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['ema_short_slope'] = rolling_slope

    X = sm.add_constant(range(len(pd_candles['ema_long_periods'])))
    rolling_slope = pd_candles['ema_long_periods'].rolling(window=sliding_window_how_many_candles).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['ema_long_slope'] = rolling_slope

    X = sm.add_constant(range(len(pd_candles['boillenger_upper'])))
    rolling_slope = pd_candles['boillenger_upper'].rolling(window=int(sliding_window_how_many_candles/slow_fast_interval_ratio)).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['boillenger_upper_slope'] = rolling_slope

    X = sm.add_constant(range(len(pd_candles['boillenger_lower'])))
    rolling_slope = pd_candles['boillenger_lower'].rolling(window=sliding_window_how_many_candles).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['boillenger_lower_slope'] = rolling_slope

    X = sm.add_constant(range(len(pd_candles['ema_rsi'])))
    rolling_slope = pd_candles['ema_rsi'].rolling(window=sliding_window_how_many_candles).apply(lambda x: sm.OLS(x, X[:len(x)]).fit().params[1], raw=False)
    pd_candles['ema_rsi_slope'] = rolling_slope

    pd_candles['regular_divergence'] = (
        (pd_candles['ema_long_slope'] > 0) & (pd_candles['ema_rsi_slope'] < 0) |
        (pd_candles['ema_long_slope'] < 0) & (pd_candles['ema_rsi_slope'] > 0)
    )
    

    # Fibonacci
    TARGET_FIB_LEVEL = 0.618
    pd_candles['fib_618_short_periods'] = pd_candles.apply(lambda rw : estimate_fib_retracement(rw['min_short_periods'], rw['idmin_short_periods'], rw['max_short_periods'], rw['idmax_short_periods'], TARGET_FIB_LEVEL), axis=1)
    pd_candles['fib_618_long_periods'] = pd_candles.apply(lambda rw : estimate_fib_retracement(rw['min_long_periods'], rw['idmin_long_periods'], rw['max_long_periods'], rw['idmax_long_periods'], TARGET_FIB_LEVEL), axis=1)


    # Inflection points
    pd_candles['gap_close_vs_ema'] = pd_candles['close'] - pd_candles['ema_long_periods']
    pd_candles['close_above_or_below_ema'] = None
    pd_candles.loc[pd_candles['gap_close_vs_ema'] > 0, 'close_above_or_below_ema'] = 'above'
    pd_candles.loc[pd_candles['gap_close_vs_ema'] < 0, 'close_above_or_below_ema'] = 'below'

    pd_candles.loc[
        (pd_candles['close_above_or_below_ema'] != pd_candles['close_above_or_below_ema'].shift(-1)),
        'close_vs_ema_inflection'
    ] = np.sign(pd_candles['close'] - pd_candles['ema_long_periods'])

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

def fetch_deribit_btc_option_expiries(
    market: str = 'BTC'
) -> Dict[
    str, Union[
        Dict[str, float],
        Dict[str, Dict[str, Union[str, float]]]
    ]
]:
    exchange = deribit()
    instruments = exchange.public_get_get_instruments({
        'currency': market,
        'kind': 'option',
        # 'expired': 'true'
    })['result']
    
    index_price = exchange.public_get_get_index_price({
        'index_name': f"{market.lower()}_usd"
    })['result']['index_price']
    index_price = float(index_price)
    
    expiry_data : Dict[str, float] = {}
    expiry_data_breakdown_by_strike : Dict[str, Dict] = {}
    for instrument in instruments:
        expiry_timestamp = int(instrument["expiration_timestamp"]) / 1000
        expiry_date = datetime.utcfromtimestamp(expiry_timestamp)

        strike = float(instrument['strike'])

        option_type = instrument['instrument_name'].split('-')[-1]  # Last part is 'C' or 'P'
        is_call = option_type == 'C'
    
        ticker = exchange.public_get_ticker({
            'instrument_name': instrument['instrument_name']
        })['result']
        
        open_interest = ticker.get("open_interest", 0)  # Open interest in BTC
        open_interest = float(open_interest)
        notional_value : float = open_interest * index_price  # Convert to USD
        
        expiry_str : str = expiry_date.strftime("%Y-%m-%d")
        if expiry_str not in expiry_data:
            expiry_data[expiry_str] = 0
        expiry_data[expiry_str] += notional_value

        if f"{expiry_str}-{strike}" not in expiry_data_breakdown_by_strike:
            expiry_data_breakdown_by_strike[f"{expiry_str}-{strike}"] = {
                'expiry' : expiry_str,
                'strike' : strike,
                'option_type': 'call' if is_call else 'put',
                'notional_value' : notional_value
            }
        else:
            expiry_data_breakdown_by_strike[f"{expiry_str}-{strike}"]['notional_value'] += notional_value
    
    sorted_expiry_data = sorted(expiry_data.items())

    return {
        'index_price' : index_price,
        'by_expiry' : sorted_expiry_data, # type: ignore Otherwise, Error: Type "dict[str, list[tuple[str, float]] | dict[str, Dict[Unknown, Unknown]]]" is not assignable to return type "Dict[str, Dict[str, float] | Dict[str, Dict[str, str | float]]]"
        'by_expiry_and_strike' : expiry_data_breakdown_by_strike
    }