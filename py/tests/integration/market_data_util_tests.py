import unittest
from datetime import datetime
from typing import Union

from util.market_data_util import *

from ccxt.binance import binance
from ccxt.bybit import bybit
from ccxt.okx import okx
from ccxt.deribit import deribit
from ccxt.base.exchange import Exchange

# @unittest.skip("Skip all integration tests.")
class MarketDataGizmoTests(unittest.TestCase):

    def test_fetch_candles_yahoo(self):
        start_date : datetime = datetime(2023,1,1)
        end_date : datetime = datetime(2023,12,31)

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
        

    def test_fetch_candles_ccxt(self):
        start_date : datetime = datetime(2023,1,1)
        end_date : datetime = datetime(2023,12,31)

        param = {
            'apiKey' : None,
            'secret' : None,
            'password' : None,    # Other exchanges dont require this! This is saved in exchange.password!
            'subaccount' : None,
            'rateLimit' : 100,    # In ms
            'options' : {
                'defaultType': 'swap', # Should test linear instead
                'leg_room_bps' : 5,
                'trade_fee_bps' : 3,

                'list_ts_field' : 'listTime' # list_ts_field: Response field in exchange.markets[symbol] to indiate timestamp of symbol's listing date in ms. For bybit, markets['launchTime'] is list date. For okx, it's markets['listTime'].
            }
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
        

