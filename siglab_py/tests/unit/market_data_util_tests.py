import unittest
from datetime import datetime, timedelta
from typing import Union
import json
from pathlib import Path

from util.market_data_util import *

from futu import *


# @unittest.skip("Skip all integration tests.")
class MarketDataUtilTests(unittest.TestCase):
    def test_timestamp_to_week_of_month(self):
        test_timestamps = [
            1672531200000,  # 2023-01-01 (Week 0)
            1673136000000,  # 2023-01-08 (Week 1)
            1673740800000,  # 2023-01-15 (Week 2)
            1674345600000,  # 2023-01-22 (Week 3)
            1674950400000,  # 2023-01-29 (Week 4)
            1675468800000,  # 2023-02-01 (Week 0)
            1676073600000,  # 2023-02-08 (Week 1)
            1676678400000,  # 2023-02-15 (Week 2)
            1677283200000,  # 2023-02-22 (Week 3)
            1677888000000,  # 2023-03-01 (Week 0)
        ]

        expectations = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0]
        
        for i, ts in enumerate(test_timestamps):
            expectation = expectations[i]
            actual = timestamp_to_week_of_month(ts)
            assert expectation == actual, f"Test failed for timestamp {ts}. Expected: {expectation}, Actual: {actual}"
            
    def test_timestamp_to_active_trading_regions(self):
        test_timestamps = [
            1672531200000,  # 2023-01-01 00:00:00 UTC (APAC)
            1672563600000,  # 2023-01-01 09:00:00 UTC (APAC, EMEA)
            1672574400000,  # 2023-01-01 12:00:00 UTC (EMEA)
            1672588800000,  # 2023-01-01 16:00:00 UTC (EMEA, AMER)
            1672599600000,  # 2023-01-01 19:00:00 UTC (AMER)
            1672610400000,  # 2023-01-01 22:00:00 UTC (APAC)
        ]

        expectations = [ ['APAC'], ['APAC', 'EMEA'], ['EMEA'], ['EMEA','AMER'], ['AMER'], ['APAC']]

        i = 0
        for ts in test_timestamps:
            expectation = expectations[i]
            actual = timestamp_to_active_trading_regions(ts)
            assert(expectation==actual)
            i+=1
        
    def test_ticker_change_map_util(self):
        '''
        Example OKX managers decide to give work to customers https://www.okx.com/help/okx-will-rename-xautusdt-perpetual-to-xauusdt-perpetual
            OKX to rename XAUTUSDT perpetual to XAUUSDT perpetual📣 

                🗓 8:05 am on Jan 15, 2026 (UTC)    --> Timestamp in sec: 1768464300
                1️⃣Trading of XAUTUSDT perpetual will be suspended from 8:05 am to 8:25 am on Jan 15, 2026 (UTC)
                2️⃣Following aspects may also be affected:
                • Margin requirements (if you are using PM mode to trade this perpetual)
                • Index & funding fee
                • Trading bots & strategy orders
                • OpenAPI & WebSocket
        '''
        ticker_change_map_file : str = 'ticker_change_map.json'

        ticker_change_map : List[Dict[str, Union[str, int]]] = [
            {
                'new_ticker' : 'XAU/USDT:USDT',
                'old_ticker' : 'XAUT/USDT:USDT',
                'cutoff_ms' : 1768464300000
            }
        ]

        ticker : str = 'XAU/USDT:USDT'
        old_ticker : Union[None, str] = get_old_ticker(ticker, ticker_change_map)
        mapping : Union[None, Dict[str, Union[str, int]]] = get_ticker_map(ticker, ticker_change_map)
        assert(old_ticker)
        self.assertEqual(old_ticker, "XAUT/USDT:USDT")
        assert(mapping)

        '''
        with open(ticker_change_map_file, 'w', encoding='utf-8') as f:
            json.dump(ticker_change_map, f, indent=2)

        with open(ticker_change_map_file, 'r', encoding='utf-8') as f:
            ticker_change_map_from_disk : List[Dict[str, Union[str, int]]] = json.load(f)
        '''

            