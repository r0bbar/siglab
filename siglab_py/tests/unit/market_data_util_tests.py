import unittest
from datetime import datetime, timedelta
from typing import Union
from pathlib import Path

from util.market_data_util import *

from futu import *


# @unittest.skip("Skip all integration tests.")
class MarketDataUtilTests(unittest.TestCase):

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
        