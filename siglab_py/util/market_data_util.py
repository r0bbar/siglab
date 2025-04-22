import incremental
import tzlocal
from datetime import datetime, timezone
from typing import List, Dict, Union, NoReturn, Any, Tuple
from pathlib import Path
import math
import pandas as pd
import numpy as np
import asyncio

from ccxt.base.exchange import Exchange as CcxtExchange
from ccxt import deribit
import ccxt
import ccxt.pro as ccxtpro

# https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/
from yahoofinancials import YahooFinancials

# yfinance allows intervals '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'. yahoofinancials not as flexible
import yfinance as yf

from siglab_py.exchanges.futubull import Futubull
from siglab_py.exchanges.any_exchange import AnyExchange

def instantiate_exchange(
    exchange_name : str,
    api_key : Union[str, None] = None,
    secret : Union[str, None]  = None,
    passphrase : Union[str, None] = None,
    default_type : Union[str, None] = 'spot',
    rate_limit_ms : float = 100
) -> Union[AnyExchange, None]:
    exchange_name = exchange_name.lower().strip()

    # Look at ccxt exchange.describe. under 'options' \ 'defaultType' (and 'defaultSubType') for what markets the exchange support.
    # https://docs.ccxt.com/en/latest/manual.html#instantiation
    exchange_params : Dict[str, Any]= {
                        'apiKey' : api_key,
                        'secret' : secret,
                        'enableRateLimit'  : True,
                        'rateLimit' : rate_limit_ms,
                        'options' : {
                            'defaultType' : default_type
                        }
                    }
    if api_key:
        exchange_params['apiKey'] = api_key
    if secret:
        exchange_params['secret'] = secret
    if passphrase:
        exchange_params['passphrase'] = passphrase

    if exchange_name=='binance':
        exchange = ccxt.binance(exchange_params)  # type: ignore
    elif exchange_name=='bybit':
        exchange = ccxt.bybit(exchange_params) # type: ignore
    elif exchange_name=='okx':
        exchange = ccxt.okx(exchange_params) # type: ignore
    elif exchange_name=='deribit':
        exchange = ccxt.deribit(exchange_params)  # type: ignore
    elif exchange_name=='hyperliquid':
        exchange = ccxt.hyperliquid(
            {
                "walletAddress" : api_key, # type: ignore
                "privateKey" : secret, 
                'enableRateLimit'  : True,
                'rateLimit' : rate_limit_ms
            }
        )  
    else:
        raise ValueError(f"Unsupported exchange {exchange_name}.")

    exchange.load_markets() # type: ignore

    return exchange # type: ignore

async def async_instantiate_exchange(
    gateway_id : str,
    api_key : str,
    secret : str,
    passphrase : str,
    default_type : str,
    rate_limit_ms : float = 100
) -> Union[AnyExchange, None]:
    exchange : Union[AnyExchange, None] = None
    exchange_name : str = gateway_id.split('_')[0]
    exchange_name =exchange_name.lower().strip()

    # Look at ccxt exchange.describe. under 'options' \ 'defaultType' (and 'defaultSubType') for what markets the exchange support.
    # https://docs.ccxt.com/en/latest/manual.html#instantiation
    exchange_params : Dict[str, Any]= {
                        'apiKey' : api_key,
                        'secret' : secret,
                        'enableRateLimit'  : True,
                        'rateLimit' : rate_limit_ms,
                        'options' : {
                            'defaultType' : default_type
                        }
                    }

    if exchange_name=='binance':
        # spot, future, margin, delivery, option
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/binance.py#L1298
        exchange = ccxtpro.binance(exchange_params)  # type: ignore
    elif exchange_name=='bybit':
        # spot, linear, inverse, futures
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/bybit.py#L1041
        exchange = ccxtpro.bybit(exchange_params) # type: ignore
    elif exchange_name=='okx':
        # 'funding', spot, margin, future, swap, option
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/okx.py#L1144
        exchange_params['password'] = passphrase
        exchange = ccxtpro.okx(exchange_params) # type: ignore
    elif exchange_name=='deribit':
        # spot, swap, future
        # https://github.com/ccxt/ccxt/blob/master/python/ccxt/deribit.py#L360
        exchange = ccxtpro.deribit(exchange_params)  # type: ignore
    elif exchange_name=='hyperliquid':
        '''
        https://app.hyperliquid.xyz/API
        
        defaultType from ccxt: swap
            https://github.com/ccxt/ccxt/blob/master/python/ccxt/hyperliquid.py#L225
        
        How to integrate? You can skip first 6 min: https://www.youtube.com/watch?v=UuBr331wxr4&t=363s

        Example, 
            API credentials created under "\ More \ API":
                    Ledger Arbitrum Wallet Address: 0xAAAAA <-- This is your Ledger Arbitrum wallet address with which you connect to Hyperliquid. 
                    API Wallet Address 0xBBBBB <-- Generated
                    privateKey 0xCCCCC

        Basic connection via CCXT:
            import asyncio
            import ccxt.pro as ccxtpro

            async def main():
                rate_limit_ms = 100
                exchange_params = {
                    "walletAddress" : "0xAAAAA", # Ledger Arbitrum Wallet Address here! Not the generated address.
                    "privateKey" : "0xCCCCC"
                }
                exchange = ccxtpro.hyperliquid(exchange_params) 
                balances = await exchange.fetch_balance()
                print(balances)

            asyncio.run(main())
        '''
        exchange = ccxtpro.hyperliquid(
            {
                "walletAddress" : api_key,
                "privateKey" : secret,
                'enableRateLimit'  : True,
                'rateLimit' : rate_limit_ms
            }
        )  # type: ignore
    else:
        raise ValueError(f"Unsupported exchange {exchange_name}, check gateway_id {gateway_id}.")

    await exchange.load_markets() # type: ignore

    '''
    Is this necessary? The added trouble is for example bybit.authenticate requires arg 'url'. binance doesn't. And fetch_balance already test credentials.

    try:
        await exchange.authenticate() # type: ignore
    except Exception as swallow_this_error:
        pass
    '''

    return exchange

def timestamp_to_datetime_cols(pd_candles : pd.DataFrame):
    def _fix_timestamp_ms(x):
        if isinstance(x, pd.Timestamp):
            return int(x.value // 10**6)
        elif isinstance(x, np.datetime64):
            return int(x.astype('int64') // 10**6)
        elif isinstance(x, (int, float)):
            x = int(x)
            if len(str(abs(x))) == 13:
                return x
            else:
                return int(x * 1000)
        else:
            raise ValueError(f"Unsupported type {type(x)} for timestamp conversion")
    pd_candles['timestamp_ms'] = pd_candles['timestamp_ms'].apply(_fix_timestamp_ms)
    pd_candles['datetime'] = pd_candles['timestamp_ms'].apply(lambda x: datetime.fromtimestamp(int(x/1000)))
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

    pd_candles['week_of_month'] = pd_candles['timestamp_ms'].apply(
        lambda x: timestamp_to_week_of_month(int(x/1000))
    )

    pd_candles['apac_trading_hr'] = pd_candles['timestamp_ms'].apply(
        lambda x: "APAC" in timestamp_to_active_trading_regions(int(x/1000))
    )
    pd_candles['emea_trading_hr'] = pd_candles['timestamp_ms'].apply(
        lambda x: "EMEA" in timestamp_to_active_trading_regions(int(x/1000))
    )
    pd_candles['amer_trading_hr'] = pd_candles['timestamp_ms'].apply(
        lambda x: "AMER" in timestamp_to_active_trading_regions(int(x/1000))
    )

    pd_candles['timestamp_ms_gap'] = pd_candles['timestamp_ms'] - pd_candles['timestamp_ms'].shift(1)
    timestamp_ms_gap = pd_candles.iloc[-1]['timestamp_ms_gap']
    assert(pd_candles[~pd_candles.timestamp_ms_gap.isna()][pd_candles.timestamp_ms_gap!=timestamp_ms_gap].shape[0]==0)
    pd_candles.drop(columns=['timestamp_ms_gap'], inplace=True)

def timestamp_to_active_trading_regions(
        timestamp_ms : int
) -> List[str]:
    
    '''
    APAC (Asia-Pacific) Trading Hours
        UTC 22:00 - 09:00 (approximate range)
        Major financial centers: Tokyo, Hong Kong, Singapore, Sydney

    EMEA (Europe, Middle East, Africa) Trading Hours
        UTC 07:00 - 16:00 (approximate range)
        Major financial centers: London, Frankfurt, Paris, Zurich, Dubai

    US Trading Hours
        UTC 13:30 - 20:00 (approximate range)
        Major financial centers: New York, Chicago
        Key markets: NYSE, NASDAQ

    utcnow and utcfromtimestamp been deprecated in Python 3.12 
    https://www.pythonmorsels.com/converting-to-utc-time/
    '''
    active_trading_regions : List[str] = []

    dt_utc = datetime.fromtimestamp(int(timestamp_ms / 1000), tz=timezone.utc)
    utc_hour = dt_utc.hour
    if (utc_hour >= 22) or (utc_hour <= 9):
        active_trading_regions.append("APAC") 

    if 7 <= utc_hour <= 16:
        active_trading_regions.append("EMEA")

    if 13 <= utc_hour <= 20:
        active_trading_regions.append("AMER")

    return active_trading_regions

def timestamp_to_week_of_month(timestamp_ms: int) -> int:
    """
    Returns:
        int: Week of the month (0 = first week, 1 = second week, etc.).
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    day_of_month = dt.day
    week_of_month = (day_of_month - 1) // 7
    return week_of_month

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
            pd_candles.reset_index(inplace=True) # type: ignore
            pd_candles.rename(columns={ 'Date' : 'datetime', 'Datetime' : 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close' : 'close', 'Adj Close' : 'adj_close', 'Volume' : 'volume' }, inplace=True) # type: ignore
            pd_candles['datetime'] = pd.to_datetime(pd_candles['datetime']) # type: ignore
            if pd_candles['datetime'].dt.tz is None: # type: ignore
                pd_candles['datetime'] = pd.to_datetime(pd_candles['datetime']).dt.tz_localize('UTC') # type: ignore
            pd_candles['datetime'] = pd_candles['datetime'].dt.tz_convert(local_tz) # type: ignore
            pd_candles['datetime'] = pd_candles['datetime'].dt.tz_localize(None)  # type: ignore
            pd_candles['timestamp_ms'] = pd_candles.datetime.values.astype(np.int64) // 10**6 # type: ignore
            pd_candles = pd_candles.sort_values(by=['timestamp_ms'], ascending=[True]) # type: ignore
            
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
    
    if end_ts>datetime.now().timestamp():
        end_ts = int(datetime.now().timestamp())

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
    elif type(exchange) is Futubull:
        exchange_candles = exchange.fetch_candles(
                            start_ts=start_ts,
                            end_ts=end_ts,
                            symbols=normalized_symbols,
                            candle_size=candle_size
                        )
        for symbol in exchange_candles:
            pd_candles = exchange_candles[symbol]
            if not pd_candles is None:
                fix_column_types(pd_candles) # You don't want to do this from Futubull as you'd need import Futubull from there: Circular references
        return exchange_candles
    elif issubclass(exchange.__class__, CcxtExchange):
        return _fetch_candles_ccxt(
            start_ts=start_ts,
            end_ts=end_ts,
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size=candle_size,
            num_candles_limit=num_candles_limit
        )
    return { '' : None }

def _fetch_candles_ccxt(
    start_ts : int,
    end_ts : int,
    exchange,
    normalized_symbols : List[str],
    candle_size : str,
    num_candles_limit : int = 100
) -> Dict[str, Union[pd.DataFrame, None]]:
    ticker = normalized_symbols[0]

    def _fetch_ohlcv(exchange, symbol, timeframe, since, limit, params) -> Union[List, NoReturn]:
            one_timeframe = f"1{timeframe[-1]}"
            candles = exchange.fetch_ohlcv(symbol=symbol, timeframe=one_timeframe, since=since, limit=limit, params=params)
            if candles and len(candles)>0:
                candles.sort(key=lambda x : x[0], reverse=False)

            return candles
        
    def _calc_increment(candle_size):
        increment = 1
        num_intervals = int(candle_size[0])
        interval_type = candle_size[-1]
        if interval_type == "m":
            increment = 60
        elif interval_type == "h":
            increment = 60*60
        elif interval_type == "d":
            increment = 60*60*24
        else:
            raise ValueError(f"Invalid candle_size {candle_size}")
        return num_intervals * increment
    
    all_candles = []
    params = {}
    this_cutoff = start_ts
    while this_cutoff<end_ts:
        candles = _fetch_ohlcv(exchange=exchange, symbol=ticker, timeframe=candle_size, since=int(this_cutoff * 1000), limit=num_candles_limit, params=params)
        if candles and len(candles)>0:
            all_candles = all_candles + [[ int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]) ] for x in candles if x[1] and x[2] and x[3] and x[4] and x[5] ]

            record_ts = max([int(record[0]) for record in candles])
            record_ts_str : str = str(record_ts)
            if len(record_ts_str)==13:
                record_ts = int(int(record_ts_str)/1000) # Convert from milli-seconds to seconds
            
            this_cutoff = record_ts  + _calc_increment(candle_size)
        else:
            this_cutoff += _calc_increment(candle_size)

    columns = ['exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume']
    pd_all_candles = pd.DataFrame([ [ exchange.name, ticker, x[0], x[1], x[2], x[3], x[4], x[5] ] for x in all_candles], columns=columns)
    fix_column_types(pd_all_candles)
    pd_all_candles['pct_chg_on_close'] = pd_all_candles['close'].pct_change()

    return {
        ticker : pd_all_candles
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

def build_pair_candles(
    pd_candles1 : pd.DataFrame,
    pd_candles2 : pd.DataFrame
) -> pd.DataFrame:
    min_timestamp_ms1 = int(pd_candles1.iloc[0]['timestamp_ms'])
    max_timestamp_ms1 = int(pd_candles1.iloc[-1]['timestamp_ms'])
    min_timestamp_ms2 = int(pd_candles2.iloc[0]['timestamp_ms'])
    max_timestamp_ms2 = int(pd_candles2.iloc[-1]['timestamp_ms'])

    pd_candles1 = pd_candles1[(pd_candles1.timestamp_ms>=min_timestamp_ms2) & (pd_candles1.timestamp_ms<=max_timestamp_ms2) & (~pd_candles1.timestamp_ms.isna()) ]
    pd_candles2 = pd_candles2[(pd_candles2.timestamp_ms>=min_timestamp_ms1) & (pd_candles2.timestamp_ms<=max_timestamp_ms1) & (~pd_candles2.timestamp_ms.isna())]
    assert(pd_candles1.shape[0]==pd_candles2.shape[0])

    pd_candles1['timestamp_ms_gap'] = pd_candles1['timestamp_ms'] - pd_candles1['timestamp_ms'].shift(1)
    timestamp_ms_gap = pd_candles1.iloc[-1]['timestamp_ms_gap']
    
    assert(pd_candles1[~pd_candles1.timestamp_ms_gap.isna()][pd_candles1.timestamp_ms_gap!=timestamp_ms_gap].shape[0]==0)
    pd_candles1.drop(columns=['timestamp_ms_gap'], inplace=True)

    pd_candles2['timestamp_ms_gap'] = pd_candles2['timestamp_ms'] - pd_candles2['timestamp_ms'].shift(1)
    timestamp_ms_gap = pd_candles2.iloc[-1]['timestamp_ms_gap']
    assert(pd_candles2[~pd_candles2.timestamp_ms_gap.isna()][pd_candles2.timestamp_ms_gap!=timestamp_ms_gap].shape[0]==0)
    pd_candles2.drop(columns=['timestamp_ms_gap'], inplace=True)

    min_timestamp_ms1 = int(pd_candles1.iloc[0]['timestamp_ms'])
    max_timestamp_ms1 = int(pd_candles1.iloc[-1]['timestamp_ms'])
    min_timestamp_ms2 = int(pd_candles2.iloc[0]['timestamp_ms'])
    max_timestamp_ms2 = int(pd_candles2.iloc[-1]['timestamp_ms'])
    assert(min_timestamp_ms1==min_timestamp_ms2)
    assert(max_timestamp_ms1==max_timestamp_ms2)
    assert(pd_candles1.shape[0]==pd_candles2.shape[0])

    if len([ col for col in pd_candles1.columns if col[-2:]=='_1' ]) == 0:
        pd_candles1.columns = [str(col) + '_1' for col in pd_candles1.columns]

    if len([ col for col in pd_candles2.columns if col[-2:]=='_2' ]) == 0:
        pd_candles2.columns = [str(col) + '_2' for col in pd_candles2.columns]

    pd_candles1.reset_index(drop=True, inplace=True)
    pd_candles2.reset_index(drop=True, inplace=True)
    pd_candles = pd.concat([pd_candles1, pd_candles2], axis=1)
    pd_candles['timestamp_ms_gap'] = pd_candles['timestamp_ms_1'] - pd_candles['timestamp_ms_2']
    assert(pd_candles[pd_candles.timestamp_ms_gap!=0].shape[0]==0)

    pd_candles.drop(pd_candles.columns[pd_candles.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    return pd_candles

    