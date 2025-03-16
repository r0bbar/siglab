import unittest
from datetime import datetime, timedelta
from typing import Union
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
        