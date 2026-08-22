import logging
import tzlocal
from datetime import datetime, timezone, timedelta
import time
from dateutil import parser
from typing import List, Dict, Union, NoReturn, Any, Tuple
from types import MethodType
from pathlib import Path
import math
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from tabulate import tabulate
import inspect

import feedparser # RSS feed parser
from massive import RESTClient as PolygonRestClient # polygon.io: pip install -U massive
from ccxt.base.exchange import Exchange as CcxtExchange
import ccxt
import ccxt.pro as ccxtpro

# https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/
from yahoofinancials import YahooFinancials

# yfinance allows intervals '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'. yahoofinancials not as flexible
import yfinance as yf

from siglab_py.util.retry_util import retry
from siglab_py.util.datetime_util import timestamp_to_active_trading_regions 
from siglab_py.exchanges.futubull import Futubull
from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.exchanges.deribit import Deribit, DeribitAsync
from siglab_py.exchanges.lighter import Lighter, LighterAsync

def instantiate_exchange(
    exchange_name : str,
    api_key : Union[str, None] = None,
    secret : Union[str, None]  = None,
    passphrase : Union[str, None] = None,
    default_type : Union[str, None] = 'spot',
    default_sub_type : Union[str, None] = None,
    default_max_slippage_bps : int = 300, # This is for market orders
    rate_limit_ms : float = 100,
    exchange_specific_options: Union[Dict[str, Any], None] = None,
) -> Union[AnyExchange, None]:
    exchange_name = exchange_name.lower().strip()

    # Look at ccxt exchange.describe. under 'options' \ 'defaultType' (and 'defaultSubType') for what markets the exchange support.
    # https://docs.ccxt.com/en/latest/manual.html#instantiation
    _exchange_specific_options = {
                'defaultType' : default_type,
                'defaultSlippage' : default_max_slippage_bps
            }
    if exchange_specific_options:
        _exchange_specific_options = _exchange_specific_options | exchange_specific_options
    exchange_params : Dict[str, Any]= {
                        'apiKey' : api_key,
                        'secret' : secret,
                        'enableRateLimit'  : True,
                        'rateLimit' : rate_limit_ms,
                        'options' : _exchange_specific_options
                    }
    if secret=="DUMMY_SECRET":
        # Lighter DEX: secret actually not passed to Lighter, if you do, upon create_order you'd have error "invalid signature"
        exchange_params.pop("secret")
    if default_sub_type:
        exchange_params['defaultSubType'] = default_sub_type

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
        exchange = Deribit(exchange_params)  # type: ignore
    elif exchange_name=='hyperliquid':
        '''
        Discord: https://discord.com/channels/1029781241702129716/1180183595109847070
        '''
        exchange = ccxt.hyperliquid(
            {
                "walletAddress" : api_key, # type: ignore
                "privateKey" : secret, 
                'enableRateLimit'  : True,
                'rateLimit' : rate_limit_ms
            }
        )
        def patch_create_order_response(
                self,            
                average_price : float,
                create_order_response : Dict,
                order_type : str = "market"
            ):
                if order_type=='market':
                    create_order_response['type'] = order_type # Hyperliquid tag market orders as limit orders (but with very wide limit prices)
                    if (
                        'average' not in create_order_response 
                        or ('average' in create_order_response and not create_order_response['average'])
                        or ('average' in create_order_response and create_order_response['average']==0)
                    ):
                        create_order_response['average'] = average_price # For market orders, they tag 'average' null

        exchange.patch_create_order_response = MethodType(
            patch_create_order_response,
            exchange
        )  
    elif exchange_name=='lighter':
        '''
        For Lighter, pass your Ethereum wallet private key to 'api_key'.
        
        https://github.com/ccxt/ccxt/wiki/FAQ#how-to-use-the-lighter-exchange-in-ccxt

            lighter = ccxt.lighter({
                'privateKey': '0xYOUR_API_PRIVATE_KEY_HERE',                        # # It is not Ligher private key under menu \ Tools \ API keys (https://app.lighter.xyz/apikeys), it's your Ethereum Wallet private key!
                'options': {
                    'apiKeyIndex': 0,                                               # Integer (0–254) corresponding to the specific API key you created.
                    'accountIndex': 12345,                                          # https://mainnet.zklighter.elliot.ai/api/v1/accountsByL1Address?l1_address=0x1234567890abcdef...
                    'libraryPath': r'C:\lighter\lighter-signer-windows-amd64.dll'   # signer dll: https://github.com/elliottech/lighter-go/releases
                }
            })

        To test:
            from siglab_py.util.market_data_util import async_instantiate_exchange

            api_key : str = "xxxxx" # Your Ethereum Wallet's private key (not address)! This is NOT public key or private key from under menu \ Tools \ API keys (https://app.lighter.xyz/apikeys)

            # create_order go thru with NO exception. But from Order History you will find the trade actually cancelled by Lighter: "Order canceled due to excessive slippage beyond allowed limit"
            # Lighter very strict with market order, first create_order need specify price. Don't use mid price, very often your order will be canceled.
            default_max_slippage_bps : int = 30

            exchange_specific_options = {
                    'apiKeyIndex': 0,
                    'accountIndex': 687361,
                    'libraryPath': r'D:\lighter\lighter-signer-windows-amd64.dll'
                }

            exchange : Union[AnyExchange, None] = await async_instantiate_exchange(
                gateway_id='lighter',
                api_key=api_key,
                secret=secret,
                passphrase=passphrase,
                default_type=default_type,
                default_sub_type=default_sub_type,
                rate_limit_ms=rate_limit_ms,
                default_max_slippage_bps=default_max_slippage_bps,
                exchange_specific_options=exchange_specific_options,
                verbose=verbose
            )

            normalized_ticker = 'SOL/USDC:USDC'
            amount = 0.3
            price = 85
            side = 'sell'
            order_type = 'market'

            entry_order = await exchange.create_order(
                symbol = normalized_ticker,
                amount = amount,
                price = price, # This is NOT optional: ccxt.base.errors.ArgumentsRequired: lighter createOrder() requires a price argument
                type=order_type,
                side=side
            )
        '''
        if exchange_specific_options:
            exchange_specific_options['libraryPath'] = exchange_specific_options['libraryPath'].replace('/', '\\') # Forward vs back slash!!! Otherwise, on Windows, Lighter will complain "ExchangeError('lighter {"code":21120,"message":"invalid signature"}')"
        lighter_params = {
            'privateKey' : api_key
        }
        if exchange_specific_options:
            lighter_params['options'] = exchange_specific_options
        exchange = Lighter(lighter_params)
    elif exchange_name=='aster':
        '''
        @todo how to pass apikey/secret to aster constructor 
        '''
        exchange = ccxt.aster()
    else:
        raise ValueError(f"Unsupported exchange {exchange_name}.")
    
    exchange.options['default_max_slippage_bps'] = default_max_slippage_bps

    exchange.load_markets() # type: ignore

    if not hasattr(exchange, "patch_create_order_response"):
        def default_patch_create_order_response(
            self,
            average_price : float,
            create_order_response : Dict,
            order_type : str = "market"
        ):
            pass

        exchange.patch_create_order_response = MethodType(
            default_patch_create_order_response,
            exchange
        )

    exchange.name = exchange_name

    return exchange # type: ignore

async def async_instantiate_exchange(
    gateway_id : str,
    api_key : str,
    secret : str,
    passphrase : str,
    default_type : Union[str, None] = 'spot',
    default_sub_type : Union[str, None] = None,
    default_max_slippage_bps : int = 300, # This is for market orders
    rate_limit_ms : float = 100,
    exchange_specific_options: Union[Dict[str, Any], None] = None,
    verbose : bool = False,
    pass_aiohttp_session : bool = False # Set to True, if you run on Windows, and error after exchange instantiate but first CCXT calls such as load_markets: aiodns.error.DNSError: (11, 'Could not contact DNS servers')
) -> Union[AnyExchange, None]:
    exchange : Union[AnyExchange, None] = None
    exchange_name : str = gateway_id.split('_')[0]
    exchange_name =exchange_name.lower().strip()

    # Look at ccxt exchange.describe. under 'options' \ 'defaultType' (and 'defaultSubType') for what markets the exchange support.
    # https://docs.ccxt.com/en/latest/manual.html#instantiation
    _exchange_specific_options = {
                'defaultType' : default_type,
                'defaultSlippage' : default_max_slippage_bps
            }
    if exchange_specific_options:
        _exchange_specific_options = _exchange_specific_options | exchange_specific_options
    exchange_params : Dict[str, Any]= {
                        'apiKey' : api_key,
                        'secret' : secret,
                        'enableRateLimit'  : True,
                        'rateLimit' : rate_limit_ms,
                        'options' : _exchange_specific_options,
                        'verbose': verbose
                    }
    if secret=="DUMMY_SECRET":
        # Lighter DEX: secret actually not passed to Lighter, if you do, upon create_order you'd have error "invalid signature"
        exchange_params.pop("secret")
    if default_sub_type:
        exchange_params['defaultSubType'] = default_sub_type
    if pass_aiohttp_session:
        connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        exchange_params['session'] = session

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
        exchange = DeribitAsync(exchange_params)  # type: ignore
    elif exchange_name=='kraken':
        exchange = ccxtpro.kraken(exchange_params) # type: ignore
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
                'rateLimit' : rate_limit_ms,
                'verbose': verbose
            }  # type: ignore
        )
        def patch_create_order_response(
                self,
                average_price : float,
                create_order_response : Dict,
                order_type : str = "market"
            ):
                if order_type=='market':
                    create_order_response['type'] = order_type # Hyperliquid tag market orders as limit orders (but with very wide limit prices)
                    if order_type=='market':
                        create_order_response['type'] = order_type # Hyperliquid tag market orders as limit orders (but with very wide limit prices)
                        if (
                            'average' not in create_order_response 
                            or ('average' in create_order_response and not create_order_response['average'])
                            or ('average' in create_order_response and create_order_response['average']==0)
                        ):
                            create_order_response['average'] = average_price # For market orders, they tag 'average' null

        exchange.patch_create_order_response = MethodType(
            patch_create_order_response,
            exchange
        )
    elif exchange_name=='lighter':
        '''
        For Lighter, pass your Ethereum wallet private key to 'api_key'.
        
        https://github.com/ccxt/ccxt/wiki/FAQ#how-to-use-the-lighter-exchange-in-ccxt

            lighter = ccxt.lighter({
                'privateKey': '0xYOUR_API_PRIVATE_KEY_HERE',                        # # It is not Ligher private key under menu \ Tools \ API keys (https://app.lighter.xyz/apikeys), it's your Ethereum Wallet private key!
                'options': {
                    'apiKeyIndex': 0,                                               # Integer (0–254) corresponding to the specific API key you created.
                    'accountIndex': 12345,                                          # https://mainnet.zklighter.elliot.ai/api/v1/accountsByL1Address?l1_address=0x1234567890abcdef...
                    'libraryPath': r'C:\lighter\lighter-signer-windows-amd64.dll'   # signer dll: https://github.com/elliottech/lighter-go/releases
                }
            })

        To test:
            from siglab_py.util.market_data_util import async_instantiate_exchange

            api_key : str = "xxxxx" # Your Ethereum Wallet's private key (not address)! This is NOT public key or private key from under menu \ Tools \ API keys (https://app.lighter.xyz/apikeys)

            # create_order go thru with NO exception. But from Order History you will find the trade actually cancelled by Lighter: "Order canceled due to excessive slippage beyond allowed limit"
            # Lighter very strict with market order, first create_order need specify price. Don't use mid price, very often your order will be canceled.
            default_max_slippage_bps : int = 100

            exchange_specific_options = {
                    'apiKeyIndex': 0,
                    'accountIndex': 687361,
                    'libraryPath': r'D:\lighter\lighter-signer-windows-amd64.dll'
                }

            exchange : Union[AnyExchange, None] = await async_instantiate_exchange(
                gateway_id='lighter',
                api_key=api_key,
                secret=secret,
                passphrase=passphrase,
                default_type=default_type,
                default_sub_type=default_sub_type,
                rate_limit_ms=rate_limit_ms,
                default_max_slippage_bps=default_max_slippage_bps,
                exchange_specific_options=exchange_specific_options,
                verbose=verbose
            )

            normalized_ticker = 'SOL/USDC:USDC'
            amount = 0.3
            price = 85
            side = 'sell'
            order_type = 'market'

            entry_order = await exchange.create_order(
                symbol = normalized_ticker,
                amount = amount,
                price = price, # This is NOT optional: ccxt.base.errors.ArgumentsRequired: lighter createOrder() requires a price argument
                type=order_type,
                side=side
            )
        '''
        if exchange_specific_options:
            exchange_specific_options['libraryPath'] = exchange_specific_options['libraryPath'].replace('/', '\\') # Forward vs back slash!!! Otherwise, on Windows, Lighter will complain "ExchangeError('lighter {"code":21120,"message":"invalid signature"}')"
        lighter_params = {
            'privateKey' : api_key
        }
        if exchange_specific_options:
            lighter_params['options'] = exchange_specific_options
        exchange = LighterAsync(lighter_params)
    elif exchange_name=='aster':
        '''
        @todo how to pass apikey/secret to aster constructor 
        '''
        exchange = ccxtpro.aster()
    else:
        raise ValueError(f"Unsupported exchange {exchange_name}, check gateway_id {gateway_id}.")

    exchange.options['default_max_slippage_bps'] = default_max_slippage_bps

    await exchange.load_markets() # type: ignore

    if not hasattr(exchange, "patch_create_order_response"):
        def default_patch_create_order_response(
            self,
            average_price : float,
            create_order_response : Dict,
            order_type : str = "market"
        ):
            pass

        exchange.patch_create_order_response = MethodType(
            default_patch_create_order_response,
            exchange
        )

    '''
    Is this necessary? The added trouble is for example bybit.authenticate requires arg 'url'. binance doesn't. And fetch_balance already test credentials.

    try:
        await exchange.authenticate() # type: ignore
    except Exception as swallow_this_error:
        pass
    '''

    exchange.name = exchange_name

    return exchange

def timestamp_to_datetime_cols(
        pd_candles : pd.DataFrame,
        validation_max_gaps : int = 10
    ):
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
    pd_candles['datetime'] = pd_candles['datetime'].dt.tz_localize(None)  # type: ignore
    pd_candles['datetime_utc'] = pd_candles['timestamp_ms'].apply(
        lambda x: datetime.fromtimestamp(int(x.timestamp()) if isinstance(x, pd.Timestamp) else int(x / 1000), tz=timezone.utc)
    )
    
    # This is to make it easy to do grouping with Excel pivot table
    pd_candles['year'] = pd_candles['datetime'].dt.year  # type: ignore
    pd_candles['month'] = pd_candles['datetime'].dt.month  # type: ignore
    pd_candles['day'] = pd_candles['datetime'].dt.day  # type: ignore
    pd_candles['hour'] = pd_candles['datetime'].dt.hour  # type: ignore
    pd_candles['minute'] = pd_candles['datetime'].dt.minute  # type: ignore
    pd_candles['dayofweek'] = pd_candles['datetime'].dt.dayofweek  # type: ignore dayofweek: Monday is 0 and Sunday is 6

    pd_candles['week_of_month'] = pd_candles['timestamp_ms'].apply(
        lambda x: timestamp_to_week_of_month(x)
    )

    pd_candles['apac_trading_hr'] = pd_candles['timestamp_ms'].apply(
        lambda x: "APAC" in timestamp_to_active_trading_regions(x)
    )
    pd_candles['emea_trading_hr'] = pd_candles['timestamp_ms'].apply(
        lambda x: "EMEA" in timestamp_to_active_trading_regions(x)
    )
    pd_candles['amer_trading_hr'] = pd_candles['timestamp_ms'].apply(
        lambda x: "AMER" in timestamp_to_active_trading_regions(x)
    )

    pd_candles['timestamp_ms_gap'] = pd_candles['timestamp_ms'] - pd_candles['timestamp_ms'].shift(1)
    
    # Depending on asset, minutes bar may have gaps
    if validation_max_gaps: # if validation_max_gaps set to None, skip validation 
        timestamp_ms_gap_median = pd_candles['timestamp_ms_gap'].median()
        NUM_MS_IN_1HR = 60*60*1000
        if timestamp_ms_gap_median>=NUM_MS_IN_1HR:
            num_rows_with_expected_gap = pd_candles[~pd_candles.timestamp_ms_gap.isna()][pd_candles.timestamp_ms_gap==timestamp_ms_gap_median].shape[0]
            assert(num_rows_with_expected_gap/pd_candles.shape[0] > (100 - validation_max_gaps) / 100)
    pd_candles.drop(columns=['timestamp_ms_gap'], inplace=True)

def timestamp_to_week_of_month(timestamp_ms: int) -> int:
    """
    Returns:
        int: Week of the month (0 = first week, 1 = second week, etc.).
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    day_of_month = dt.day
    week_of_month = (day_of_month - 1) // 7
    return week_of_month

def fix_column_types(
        pd_candles : pd.DataFrame,
        validation_max_gaps : int = 10
    ):
    pd_candles['open'] = pd_candles['open'].astype(float)
    pd_candles['high'] = pd_candles['high'].astype(float)
    pd_candles['low'] = pd_candles['low'].astype(float)
    pd_candles['close'] = pd_candles['close'].astype(float)
    pd_candles['volume'] = pd_candles['volume'].astype(float)

    timestamp_to_datetime_cols(pd_candles=pd_candles, validation_max_gaps=validation_max_gaps)

    '''
    The 'Unnamed: 0', 'Unnamed : 1'... etc columns often appears in a DataFrame when it is saved to a file (e.g., CSV or Excel) and later loaded. 
    This usually happens if the DataFrame's index was saved along with the data, and then pandas automatically treats it as a column during the file loading process.
    We want to drop them as it'd mess up idmin, idmax calls, which will take values from 'Unnamed' instead of actual pandas index.
    '''
    pd_candles.drop(pd_candles.columns[pd_candles.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    pd_candles.reset_index(drop=True, inplace=True)
    pd_candles.sort_values("datetime", inplace=True)

def interval_to_ms(interval : str) -> int:
    interval_ms : int = 0
    if interval=="d":
        interval_ms = 24*60*60*1000
    elif interval=="h":
        interval_ms = 60*60*1000
    elif interval=="m":
        interval_ms = 60*1000

    return interval_ms

def candle_size_to_interval_sec(candle_size : str) -> int:
    increment = 1
    num_intervals = int(candle_size.replace(candle_size[-1],''))
    interval_type = candle_size[-1]
    single_interval_ms = interval_to_ms(interval_type)
    return num_intervals * int(single_interval_ms/1000)

'''
API doc https://polygon.io/docs
API Pricing https://massive.com/pricing
Dashboard https://massive.com/dashboard
'''
class PolygonMarketDataProvider:
    def __init__(
        self, 
        api_key : Union[str, None] = None,
        rate_limit_ms : int = 12*1000 # For free tiers, it's very restrictive 5 calls per minute (or 12 sec between calls)
    ):
        self.name = "polygonio"
        self.rest_client = PolygonRestClient(api_key=api_key)
        self.rate_limit_ms = rate_limit_ms

    def fetch_ohlcv(
        self,
        symbol : str,
        since : int, # in sec
        timeframe : str = '1h',
        limit : int = 5000, # default 5k, maximum 50k
     ) -> List:
        multiplier : int = int(timeframe.replace(timeframe[-1], ""))
        from_timestamp_ms : int = int(since * 1000)
        to_timestamp_ms : int = int(from_timestamp_ms + limit * multiplier * interval_to_ms(timeframe[-1]))
        # polygon.io _timeframe enumeration: minute, hour, day, week, month, quarter, year
        if timeframe[-1]=="d":
            _timeframe = "day"
        if timeframe[-1]=="h":
            _timeframe = "hour"
        elif timeframe[-1]=="m":
            _timeframe = "minute"
        else:
            _timeframe = "hour"

        '''
        polygon.io from_/to accept two formats:
            a) Date string, example: "2026-01-01", or even "2026-01-01T14:30:00Z" (ISO 8601 string)
            b) timestamp in ms
        '''
        candles = []
        for agg in self.rest_client.list_aggs(ticker=symbol, multiplier=multiplier, timespan=_timeframe, from_=from_timestamp_ms, to=to_timestamp_ms, limit=limit):
            timestamp_ms = agg.timestamp
            open = agg.open
            high = agg.high
            low = agg.low
            close = agg.close
            volume = agg.volume
            candles.append(
                (timestamp_ms, open, high, low, close, volume)
            )
            time.sleep(int(self.rate_limit_ms/1000))

        return candles

    def fetch_candles(
        self,
        start_ts, # in sec
        end_ts, # in sec
        symbols,
        candle_size : str = '1h',
        limit : int = 5000,
        validation_max_gaps : int = 10,
        logger = None
    ) -> Dict[str, Union[pd.DataFrame, None]]:
        rsp = {}

        num_tickers = len(symbols)
        i = 0
        for ticker in symbols:
            all_candles = []
            
            this_cutoff = start_ts
            while this_cutoff<end_ts:
                _ticker = ticker # @todo: This allows for ticker changes mapping later on

                if logger:
                    logger.info(f"{i}/{num_tickers} Fetching {candle_size} candles for {ticker}.")

                candles = self.fetch_ohlcv(symbol=_ticker, since=this_cutoff, timeframe=candle_size, limit=limit)
                if candles and len(candles)>0:
                    all_candles = all_candles + [[ int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]) ] for x in candles if x[1] and x[2] and x[3] and x[4] and x[5] ]

                    record_ts = max([int(record[0]) for record in candles])
                    record_ts_str : str = str(record_ts)
                    if len(record_ts_str)==13:
                        record_ts = int(int(record_ts_str)/1000) # Convert from milli-seconds to seconds
                    
                    this_cutoff = record_ts  + candle_size_to_interval_sec(candle_size)
                else:
                    this_cutoff += candle_size_to_interval_sec(candle_size)

            i+=1

            columns = ['exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume']
            pd_all_candles = pd.DataFrame([ [ "polygon.io", ticker, x[0], x[1], x[2], x[3], x[4], x[5] ] for x in all_candles], columns=columns)
            fix_column_types(pd_candles=pd_all_candles, validation_max_gaps=validation_max_gaps)
            pd_all_candles['pct_chg_on_close'] = pd_all_candles['close'].pct_change()

            rsp[ticker] = pd_all_candles

        rsp[ticker] = pd_all_candles

        return rsp

def aggregate_candles(
    interval : str,
    pd_candles : pd.DataFrame
) -> pd.DataFrame:
    if interval[-1]=='m':
        # 'm' for pandas means months!
        interval = interval.replace('m','min')
    pd_candles.set_index('datetime', inplace=True)
    pd_candles_aggregated = pd_candles.resample(interval).agg({
        'exchange' : 'first',
        'symbol' : 'first',
        'timestamp_ms' : 'first',
        
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',

        'datetime_utc' : 'first',
        'year' : 'first',
        'month' : 'first',
        'day' : 'first',
        'hour' : 'first',
        'minute' : 'first',
        'dayofweek' : 'first',
        'week_of_month' : 'first',

        'apac_trading_hr' : 'first',
        'emea_trading_hr' : 'first',
        'amer_trading_hr' : 'first',

        'pct_chg_on_close' : 'sum',

    })
    pd_candles.reset_index(inplace=True)
    pd_candles_aggregated.reset_index(inplace=True)
    return pd_candles_aggregated
    
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

    ticker_change_map : List[Dict[str, Union[str, int]]] = [],

    cache_dir : Union[str, None] = None,

    list_ts_field : Union[str, None] = None,

    validation_max_gaps : int = 10,
    validation_max_end_date_intervals : int = 1
) -> Dict[str, Union[pd.DataFrame, None]]:
    exchange_candles = { '' : None }
    num_intervals = int(candle_size.replace(candle_size[-1],''))

    if end_ts>datetime.now().timestamp():
        end_ts = int(datetime.now().timestamp())

    if issubclass(exchange.__class__, CcxtExchange):
        exchange_candles = _fetch_candles_ccxt(
            start_ts=start_ts,
            end_ts=end_ts,
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size=candle_size,
            num_candles_limit=num_candles_limit,
            ticker_change_map=ticker_change_map,
            validation_max_gaps=validation_max_gaps,
            logger=logger
        )
    
    elif type(exchange) is Futubull or type(exchange) is PolygonMarketDataProvider:
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

    if num_intervals!=1:
            for symbol in exchange_candles:
                if not exchange_candles[symbol] is None:
                    exchange_candles[symbol] = aggregate_candles(candle_size, exchange_candles[symbol]) #  type: ignore
                    
    # For invalid rows missing timestamps, o/h/l/c/v, fill forward close, set volume to zero.
    for symbol in exchange_candles:
        pd_candles = exchange_candles[symbol]
        
        if pd_candles is not None:
            mask_invalid_candles = pd_candles["timestamp_ms"].isna()
            if mask_invalid_candles.any():
                pd_invalid_candles = pd_candles[mask_invalid_candles]

                if logger is not None:
                    logger.warning(f"Dropping {pd_invalid_candles.shape[0]}/{pd_candles.shape[0]} rows from {symbol} candles (null timestamp_ms)") # type: ignore
                    logger.warning(f"{tabulate(pd_invalid_candles, headers='keys', tablefmt='psql')}") # type: ignore
                    
                def _to_timestamp_ms(dt):
                    if pd.isna(dt):
                        return pd.NA
                    if isinstance(dt, str):
                        dt = pd.to_datetime(dt)
                    return int(dt.timestamp() * 1000)
                
                pd_candles.loc[mask_invalid_candles, "timestamp_ms"] = pd_candles.loc[mask_invalid_candles, "datetime"].apply(_to_timestamp_ms)

                pd_candles["close"] = pd_candles["close"].ffill()
                pd_candles.loc[mask_invalid_candles, ["open", "high", "low"]] = pd_candles.loc[
                                                                                    mask_invalid_candles, ["close"]
                                                                                ]
                pd_candles.loc[mask_invalid_candles, "volume"] = 0.0

            # Mark trading sessions open/close
            for reg in ['apac', 'emea', 'amer']:
                th = f'{reg}_trading_hr'
                pd_candles[th] = pd_candles[th].fillna(False).astype(bool) # tradfi data such column can null all null, need ensure it's casted into boolean column
                start_mask = pd_candles[th] & ~pd_candles[th].shift(1).fillna(False)
                end_mask = pd_candles[th] & ~pd_candles[th].shift(-1).fillna(False)
                starts = pd_candles.loc[start_mask, ['timestamp_ms', 'open']].reset_index(drop=True)
                ends = pd_candles.loc[end_mask, ['timestamp_ms', 'close']].reset_index(drop=True)
                n = min(len(starts), len(ends))
                sessions = starts.iloc[:n].assign(end_ts=ends['timestamp_ms'].values[:n], close=ends['close'].values[:n])
                if len(starts) > n:
                    sessions = pd.concat([sessions, starts.iloc[n:].assign(end_ts=np.nan, close=np.nan)], ignore_index=True)
                tmp = pd.merge_asof(pd_candles[['timestamp_ms']].reset_index(drop=True), sessions.rename(columns={'start_ts': 'timestamp_ms'}), on='timestamp_ms', direction='backward')
                pd_candles[f'{reg}_session_open'] = tmp['open'].values
                pd_candles[f'{reg}_session_close'] = np.where((tmp['end_ts'].notna()) & (tmp['end_ts'] <= pd_candles['timestamp_ms'].values), tmp['close'], np.nan)
                mask = (pd_candles['timestamp_ms'] - tmp['timestamp_ms']) > 86400000 # 86400000 is 24 hours in milliseconds (24 × 60 × 60 × 1000).
            
    return exchange_candles # type: ignore

'''
Find listing date https://gist.github.com/mr-easy/5185b1dcdd5f9f908ff196446f092e9b

Usage:
    listing_ts = find_start_time(exchange, 'HYPE/USDT:USDT', int(datetime(2024,1,1).timestamp()*1000), int(datetime(2025,5,1).timestamp()*1000), '1h')

Caveats: 
1) If listing date lies outside [start_time, end_time], this function will stackoverflow, 
2) Even if not, it's still very time consuming.

Alternative: market['created']
'''
def search_listing_ts(exchange, symbol, start_time, end_time, timeframe):
    mid_time = (start_time + end_time)//2
    if(mid_time == start_time): return mid_time+1
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, mid_time, limit=1)
    time.sleep(1)
    if(len(ohlcv) == 0):
        return search_listing_ts(exchange, symbol, mid_time, end_time, timeframe)
    else:
        return search_listing_ts(exchange, symbol, start_time, mid_time, timeframe)
    
def _fetch_candles_ccxt(
    start_ts : int,
    end_ts : int,
    exchange,
    normalized_symbols : List[str],
    candle_size : str,
    num_candles_limit : int = 100,
    ticker_change_map : List[Dict[str, Union[str, int]]] = [],
    validation_max_gaps : int = 10,
    logger = None
) -> Dict[str, Union[pd.DataFrame, None]]:
    rsp = {}

    exchange.load_markets()
    
    num_tickers = len(normalized_symbols)
    i = 0
    for ticker in normalized_symbols:
        old_ticker = get_old_ticker(ticker, ticker_change_map)
        ticker_change_mapping = get_ticker_map(ticker, ticker_change_map)

        @retry(num_attempts=3, pause_between_retries_ms=1000, logger=logger)
        def _fetch_ohlcv(exchange, symbol, timeframe, since, limit, params) -> Union[List, NoReturn]:
            one_timeframe = f"1{timeframe[-1]}"
            candles = exchange.fetch_ohlcv(symbol=symbol, timeframe=one_timeframe, since=since, limit=limit, params=params)
            if candles and len(candles)>0:
                candles.sort(key=lambda x : x[0], reverse=False)

            return candles
            
        def _calc_increment(candle_size):
            increment = 1
            num_intervals = int(candle_size.replace(candle_size[-1],''))
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
        
        if logger:
            logger.info(f"{i}/{num_tickers} Fetching {candle_size} candles for {ticker}.")

        '''
        It uses a while loop to implement a sliding window to download candles between start_ts and end_ts. 
        However, start_ts for example can be 1 Jan 2021 for a given ticker. 
        But if that ticker listing date is 1 Jan 2025, this while loop would waste a lot of time loop between 1 Jan 2021 thru 31 Dec 2024, slowly incrementing this_cutoff += _calc_increment(candle_size).
        A more efficient way is to find listing date. Start looping from there.
        '''
        market = exchange.markets[ticker] if ticker in exchange.markets else None
        if not market:
            market = exchange.markets[old_ticker] if old_ticker else None
            if not market:
                raise ValueError(f"market {ticker} not support by exchange {exchange.name}!")

        this_ticker_start_ts = start_ts
        if market['created']:
            this_ticker_start_ts = max(this_ticker_start_ts, int(market['created']/1000))

        all_candles = []
        params = {}
        this_cutoff = this_ticker_start_ts
        while this_cutoff<end_ts:
            _ticker = ticker
            if ticker_change_mapping:
                ticker_change_cutoff_sec = int(ticker_change_mapping['cutoff_ms']) / 1000
                if this_cutoff<ticker_change_cutoff_sec:
                    _ticker = old_ticker
            candles = _fetch_ohlcv(exchange=exchange, symbol=_ticker, timeframe=candle_size, since=int(this_cutoff * 1000), limit=num_candles_limit, params=params)
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
        fix_column_types(pd_candles=pd_all_candles, validation_max_gaps=validation_max_gaps)
        pd_all_candles['pct_chg_on_close'] = pd_all_candles['close'].pct_change()

        rsp[ticker] = pd_all_candles

        i+=1

    return rsp

def fetch_deribit_btc_option_expiries(
    market: str = 'BTC'
) -> Dict[
    str, Union[
        Dict[str, float],
        Dict[str, Dict[str, Union[str, float]]]
    ]
]:
    exchange = ccxt.deribit()
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

def fetch_funding_rate(
    exchange,
    normalized_symbols : List[str],
    start_ts : int,
    end_ts : int,
    limit : int = 100
) -> Dict[str, pd.DataFrame]:
    results : Dict = {}
    markets = exchange.load_markets()

    funding_rate_annualized_buckets = {
        '< -15' : (float('-inf'), -15),
        '-15 - -10' : (-15, -10),
        '-10 - -8' : (-10, -8),
        '-8 - -5' : (-8, 5),
        '-5 - 0' : (-5, 0),
        '0 - 5' : (0, 5),
        '5 - 8' : (5, 8), 
        '8 - 10' : (8, 10),
        '10  - 15' : (10, 15),
        '> 15' : (15, float('inf'))
    }
    for ticker in normalized_symbols:
        market = exchange.markets[ticker] if ticker in markets else None
        this_ticker_start_ts = start_ts
        if market and market['created']:
            this_ticker_start_ts = max(this_ticker_start_ts, int(market['created']/1000))
        all_funding = []
        params = {}
        this_cutoff = this_ticker_start_ts
        while this_cutoff < end_ts:
            @retry(num_attempts=3, pause_between_retries_ms=1000, logger=None)
            def _fetch_funding_rate_history(exchange, symbol, since, limit, params):
                return exchange.fetchFundingRateHistory(symbol=symbol, since=since, limit=limit, params=params)
            funding = _fetch_funding_rate_history(exchange=exchange, symbol=ticker, since=int(this_cutoff * 1000), limit=limit, params=params)
            if funding and len(funding)>0:
                all_funding = all_funding + funding
                record_ts = max([int(entry['timestamp']) for entry in funding])
                record_ts_str : str = str(record_ts)
                if len(record_ts_str)==13:
                    record_ts = int(record_ts / 1000)
                this_cutoff = record_ts + 1
            else:
                break
        funding_history = [{
            'datetime_utc': datetime.fromtimestamp(int(entry['timestamp']/1000), tz=timezone.utc),
            'timestamp_ms': entry['timestamp'],
            'funding_rate_interval': round(entry['fundingRate'] * 100, 2),
            'funding_rate_annualized': round(entry['fundingRate'] * 100 * 3 * 365, 2),
        } for entry in all_funding]
        pd_funding_history = pd.DataFrame(funding_history)
        pd_funding_history['funding_rate_annualized_bucket'] = pd_funding_history['funding_rate_annualized'].apply(lambda x: next((k for k, (lo, hi) in funding_rate_annualized_buckets.items() if lo <= x < hi), None))
        pd_funding_history['datetime_utc'] = pd_funding_history['datetime_utc'].dt.tz_convert(None)
        results[ticker] = pd_funding_history
    return results

def build_pair_candles(
    pd_candles1 : pd.DataFrame,
    pd_candles2 : pd.DataFrame,
    left_columns_postfix : str = "_1",
    right_columns_postfix : str = "_2"
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

    if len([ col for col in pd_candles1.columns if col[-2:]==left_columns_postfix ]) == 0:
        pd_candles1.columns = [str(col) + left_columns_postfix for col in pd_candles1.columns]

    if len([ col for col in pd_candles2.columns if col[-2:]==right_columns_postfix ]) == 0:
        pd_candles2.columns = [str(col) + right_columns_postfix for col in pd_candles2.columns]

    pd_candles1.reset_index(drop=True, inplace=True)
    pd_candles2.reset_index(drop=True, inplace=True)
    pd_candles = pd.concat([pd_candles1, pd_candles2], axis=1)
    pd_candles['timestamp_ms_gap'] = pd_candles[f'timestamp_ms{left_columns_postfix}'] - pd_candles[f'timestamp_ms{right_columns_postfix}']
    assert(pd_candles[pd_candles.timestamp_ms_gap!=0].shape[0]==0)

    pd_candles.drop(pd_candles.columns[pd_candles.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    return pd_candles

def get_old_ticker(
        ticker : str,
        ticker_change_map : List[Dict[str, Union[str, int]]]
    ) -> Union[str, None]:
        if not ticker_change_map:
            return None

        mapping = get_ticker_map(ticker, ticker_change_map)
        if mapping:
            return str(mapping['old_ticker'])
        
        return None

def get_ticker_map(
    ticker : str,
    ticker_change_map : List[Dict[str, Union[str, int]]]
) -> Union[None, Dict[str, Union[str, int]]]:
    if not ticker_change_map:
        return None

    for mapping in ticker_change_map:
        new_ticker = mapping['new_ticker']
        if new_ticker==ticker:
            return mapping
    
    return None

def fetch_headlines_from_rss(
    rss_feeds : Dict[str, str],
    pd_old_headlines : pd.DataFrame, # For purpose of de-duplication
    top_lines : int = 20 # For each feed, we'd only go thru top number of lines
) -> List[Dict[str, Union[str, datetime, int]]]:
    logger: logging.Logger = logging.getLogger()

    new_headlines : List[Dict[str, Union[str, datetime, int]]] = []
    for source, feed_url in rss_feeds.items():
        try:
            feed = feedparser.parse(feed_url)
            logger.info(f"{source}: {len(feed.entries)} headlines found, we're going thru top {top_lines}")

            for entry in feed.entries[:top_lines]:
                try:
                    published_dt = None
                    published = entry.get('published')
                    if published:
                        published_dt = parser.parse(entry.get('published'))
                except Exception as dateparse_err:
                    published_dt = None
                    logger.info(f"{source} {entry.title}: Date parse error {dateparse_err}")

                new_fetch_row = {
                    'source': source,
                    'title': entry.title,
                    'published_utc_dt': published_dt,
                    'published_local_dt': published_dt.astimezone() if published_dt else None,
                    'published_timestamp_ms': int(published_dt.timestamp() * 1000) if published_dt else None,
                    'created_timestamp_ms' : int(datetime.now().timestamp() * 1000),
                    'url': entry.link,
                }
                if (
                        not ((pd_old_headlines['source'] == new_fetch_row['source']) & 
                        (pd_old_headlines['title'] == new_fetch_row['title'])).any()
                ):
                    # Try not append duplicates
                    new_headlines.append(new_fetch_row)
            
        except Exception as e:
            logger.info(f"{source}: feedparser error {e}")
        
    return new_headlines
