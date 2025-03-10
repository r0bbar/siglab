import unittest
from datetime import datetime, timedelta
from typing import Union
from pathlib import Path

from util.market_data_util import *
from exchanges.futubull import Futubull

from ccxt.binance import binance
from ccxt.bybit import bybit
from ccxt.okx import okx
from ccxt.deribit import deribit
from ccxt.base.exchange import Exchange

from futu import *


# @unittest.skip("Skip all integration tests.")
class MarketDataUtilTests(unittest.TestCase):

    def test_fetch_candles_yahoo(self):
        start_date : datetime = datetime(2024, 1,1)
        end_date : datetime = datetime(2024,12,31)

        exchange = YahooExchange()
        normalized_symbols = [ 'AAPL' ]
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp(),
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size='1h'
        )[normalized_symbols[0]]

        assert pd_candles is not None

        if pd_candles is not None:
            assert len(pd_candles) > 0, "No candles returned."
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute'}
            assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
            assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
            assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."
        
    def test_fetch_candles_nasdaq(self):
        start_date : datetime = datetime(2023,1,1)
        end_date : datetime = datetime(2023,12,31)

        '''
        Folder structure:
            \ siglab
                \ siglab_py	<-- python project root
                    \ sigab_py
                        __init__.py
                        \ util
                            __init__.py
                            market_data_util.py
                        \ tests
                            \ integration
                                __init__.py
                                market_data_util_tests.py <-- Tests here
                            
                \ siglab_rs <-- Rust project root
                \ data	 <-- Data files here!
        '''
        data_dir : Union[str, None] = str(Path(__file__).resolve().parents[3] / "data/nasdaq")
        exchange : NASDAQExchange = NASDAQExchange(data_dir = data_dir)

        # CSV from NASDAQ: https://www.nasdaq.com/market-activity/quotes/historical
        normalized_symbols = [ 'AAPL' ]
        # normalized_symbols = [ 'BABA', 'SPY', 'VOO', 'IVV', 'AAPL', 'AMZN', 'GOOG_classC', 'META', 'MSFT', 'MSTR', 'TSLA' ]

        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp(),
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size='1h'
        )[normalized_symbols[0]]

        assert pd_candles is not None

        if pd_candles is not None:
            assert len(pd_candles) > 0, "No candles returned."
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute'}
            assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
            assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
            assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."
        
    def test_fetch_candles_ccxt(self):
        start_date : datetime = datetime(2024,1,1)
        end_date : datetime = datetime(2024,12,31)

        param = {
            'apiKey' : None,
            'secret' : None,
            'password' : None,
            'subaccount' : None,
            'rateLimit' : 100,    # In ms
            'options' : {
                'defaultType': 'swap'            }
        }

        exchange : Exchange = okx(param)
        normalized_symbols = [ 'BTC/USDT:USDT' ]
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp(),
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size='1h'
        )[normalized_symbols[0]]

        assert pd_candles is not None

        if pd_candles is not None:
            assert len(pd_candles) > 0, "No candles returned."
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute'}
            assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
            assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
            assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."
        
    def test_fetch_candles_futubull(self):
        end_date : datetime = datetime.today()
        start_date : datetime = end_date - timedelta(days=365*3)

        params : Dict = {
            'trdmarket' : TrdMarket.HK,
            'security_firm' : SecurityFirm.FUTUSECURITIES,
            'market' : Market.HK, 
            'security_type' : SecurityType.STOCK,
            'daemon' : {
                'host' : '127.0.0.1',
                'port' : 11111
            }
        }
        exchange = Futubull(params)
        
        symbol = "HK.00700"
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
                start_ts=int(start_date.timestamp()),
                end_ts=int(end_date.timestamp()),
                exchange=exchange,
                normalized_symbols=[ symbol ],
                candle_size='1d'
            )[symbol]

        assert pd_candles is not None

        if pd_candles is not None:
            assert len(pd_candles) > 0, "No candles returned."
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute'}
            assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
            assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
            assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."
        
