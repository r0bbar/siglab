import unittest
from datetime import datetime, timedelta
from typing import Union
import json
import logging
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
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute', 'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr'}
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
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute', 'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr'}
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

        exchange : Exchange = okx(param) # type: ignore
        normalized_symbols = [ 'BTC/USDT:USDT' ]
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp(),
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size='1h',
            logger=logging.getLogger()
        )[normalized_symbols[0]]

        assert pd_candles is not None

        if pd_candles is not None:
            assert len(pd_candles) > 0, "No candles returned."
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute', 'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr'}
            assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
            assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
            assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."

    def test_fetch_candles_ccxt_with_ticker_change_map(self):
        ticker_change_map : List[Dict[str, Union[str, int]]] = [
            {
                'new_ticker' : 'XAU/USDT:USDT',
                'old_ticker' : 'XAUT/USDT:USDT',
                'cutoff_ms' : 1768464300000
            }
        ]

        start_date : datetime = datetime(2026,1,13,0,0,0)
        end_date : datetime = datetime(2026,1,15,18,0,0)

        param = {
            'apiKey' : None,
            'secret' : None,
            'password' : None,
            'subaccount' : None,
            'rateLimit' : 100,    # In ms
            'options' : {
                'defaultType': 'swap'            }
        }

        exchange : Exchange = okx(param) # type: ignore
        normalized_symbols = [ 'XAU/USDT:USDT' ]
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp(),
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size='1h',
            ticker_change_map=ticker_change_map,
            logger=logging.getLogger()
        )[normalized_symbols[0]]

        assert pd_candles is not None

    def test_fetch_trades_ccxt(self):
        param = {
            'apiKey' : None,
            'secret' : None,
            'password' : None,
            'subaccount' : None,
            'rateLimit' : 100,    # In ms
            'options' : {
                'defaultType': 'swap'            }
        }

        exchange : Exchange = okx(param) # type: ignore

        @retry(num_attempts=3, pause_between_retries_ms=1000)
        def _fetch_trades(exchange, symbol, since, limit) -> Union[List, NoReturn]:
            trades = exchange.fetch_trades(symbol=symbol, since=since, limit=limit)
            if trades and len(trades)>1:
                trades.sort(key=lambda x : x['timestamp'], reverse=False)
                assert(trades[0]['timestamp']<trades[-1]['timestamp'])
            return trades
        
        '''
        Caveats:
            since: Many exchanges don't actually respect 'since' parameter and only give you recent trades.
            limit: Generally exchanges give you recent hundred(s) trades and not more.
        
        Format:
            [
                ...
                {'info': {'instId': 'BTC-USDT-SWAP', 'side': 'buy', 'sz': '0.01', 'px': '73720.8', 'source': '0', 'tradeId': '2637932250', 'ts': '1780286404649'}, 'timestamp': 1780286404649, 'datetime': '2026-06-01T04:00:04.649Z', 'symbol': 'BTC/USDT:USDT', 'id': '2637932250', 'order': None, 'type': None, 'takerOrMaker': None, 'side': 'buy', 'price': 73720.8, 'amount': 0.01, 'cost': 7.37208, 'fee': {'cost': None, 'currency': None}, 'fees': []}
                ...
            ]
        '''
        trades = _fetch_trades(exchange=exchange, symbol="BTC/USDT:USDT", since=None, limit=1000)
        assert(trades)

    def test_aggregate_candles(self):
        end_date : datetime = datetime.today()
        start_date : datetime = end_date + timedelta(hours=-8)

        param = {
            'apiKey' : None,
            'secret' : None,
            'password' : None,
            'subaccount' : None,
            'rateLimit' : 100,    # In ms
            'options' : {
                'defaultType': 'swap'            }
        }

        exchange : Exchange = okx(param) # type: ignore
        normalized_symbols = [ 'BTC/USDT:USDT' ]
        pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp(),
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            candle_size='15m' # <---- aggregate 1m into 15m candles
        )[normalized_symbols[0]]

        assert pd_candles is not None
        pd_candles['timestamp_ms_gap'] = pd_candles['timestamp_ms'].diff()
        timestamp_ms_gap_median = pd_candles['timestamp_ms_gap'].median()
        NUM_MS_IN_1HR = 60*60*1000
        expected_15m_gap_ms = NUM_MS_IN_1HR/4
        assert(timestamp_ms_gap_median==expected_15m_gap_ms)
        total_num_rows = pd_candles.shape[0]
        num_rows_with_15min_gaps = pd_candles[pd_candles.timestamp_ms_gap!=timestamp_ms_gap_median].shape[0]
        assert(num_rows_with_15min_gaps/total_num_rows <= 0.4) # Why not 100% match? minute bars may have gaps (Also depends on what ticker)

    def test_fetch_funding_rates(self):
        NUM_DAYS : int = 365
        end_date : datetime = datetime.today()
        start_date : datetime = end_date + timedelta(days=-NUM_DAYS)
        param = {
            'apiKey' : None,
            'secret' : None,
            'password' : None,
            'subaccount' : None,
            'rateLimit' : 100,    # In ms
            'options' : {
                'defaultType': 'swap'            }
        }

        exchange : Exchange = binance(param) # type: ignore
        normalized_symbols = [ 'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT' ]
        results = fetch_funding_rate(
            exchange=exchange,
            normalized_symbols=normalized_symbols,
            start_ts=start_date.timestamp(),
            end_ts=end_date.timestamp()
        )
        assert(len(results)==3)
        for ticker in normalized_symbols:
            assert(ticker in results)
            assert(isinstance(results[ticker], pd.DataFrame))
            assert(results[ticker].shape[0]>=NUM_DAYS*3)

    def test_fetch_candles_futubull(self):
        # You need Futu OpenD running and you need entitlements
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
            expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute', 'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr'}
            assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
            assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
            assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."
        
