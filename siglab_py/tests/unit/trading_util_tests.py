import unittest
from datetime import datetime, timedelta
from typing import Union
from pathlib import Path

from util.trading_util import *

'''
Have a look at this for a visual explaination how "Gradually tightened stops" works:
    https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/trading_util_tests.ipynb
    https://norman-lm-fung.medium.com/gradually-tightened-trailing-stops-f7854bf1e02b
'''

# @unittest.skip("Skip all integration tests.")
class TradingUtilTests(unittest.TestCase):
    def test_timestamp_to_active_trading_regions_case1(self):
        tp_min_percent : float = 1.5
        tp_max_percent : float = 2.5
        sl_percent_trailing : float = 50 # Trailing stop loss in percent
        default_effective_tp_trailing_percent : float = 50

        pnl_percent_notional : float = 0.5 # Trade's current pnl in percent.

        effective_tp_trailing_percent = calc_eff_trailing_sl(
            tp_min_percent = tp_min_percent,
            tp_max_percent = tp_max_percent,
            sl_percent_trailing = sl_percent_trailing,
            pnl_percent_notional = pnl_percent_notional,
            default_effective_tp_trailing_percent = default_effective_tp_trailing_percent
        )
        assert(effective_tp_trailing_percent==50) # Generous trailing SL when trading starting out and pnl small.

    def test_timestamp_to_active_trading_regions_case2(self):
        tp_min_percent : float = 1.5
        tp_max_percent : float = 2.5
        sl_percent_trailing : float = 50 # Trailing stop loss in percent
        default_effective_tp_trailing_percent : float = 50

        pnl_percent_notional : float = 2 # Trade's current pnl in percent.

        effective_tp_trailing_percent = calc_eff_trailing_sl(
            tp_min_percent = tp_min_percent,
            tp_max_percent = tp_max_percent,
            sl_percent_trailing = sl_percent_trailing,
            pnl_percent_notional = pnl_percent_notional,
            default_effective_tp_trailing_percent = default_effective_tp_trailing_percent
        )
        assert(effective_tp_trailing_percent==25) # Intermediate trailing SL

    def test_timestamp_to_active_trading_regions_case3(self):
        tp_min_percent : float = 1.5
        tp_max_percent : float = 2.5
        sl_percent_trailing : float = 50 # Trailing stop loss in percent
        default_effective_tp_trailing_percent : float = 50

        pnl_percent_notional : float = 2.5 # Trade's current pnl in percent.

        effective_tp_trailing_percent = calc_eff_trailing_sl(
            tp_min_percent = tp_min_percent,
            tp_max_percent = tp_max_percent,
            sl_percent_trailing = sl_percent_trailing,
            pnl_percent_notional = pnl_percent_notional,
            default_effective_tp_trailing_percent = default_effective_tp_trailing_percent
        )
        assert(effective_tp_trailing_percent==0) # Most tight trailing SL

    def test_round_to_level(self):
        prices = [ 
            { 'price' : 15080, 'rounded' : 15000}, 
            { 'price' : 15180, 'rounded' : 15200}, 
            { 'price' : 25080, 'rounded' : 25200}, 
            { 'price' : 25180, 'rounded' : 25200}, 
            { 'price' : 25380, 'rounded' : 25500}, 
            { 'price' : 95332, 'rounded' : 95000}, 
            { 'price' : 95878, 'rounded' : 96000}, 
            { 'price' : 103499, 'rounded' : 103000}, 
            { 'price' : 103500, 'rounded' : 104000}, 
            { 'price' : 150800, 'rounded' : 150000}, 
            { 'price' : 151800, 'rounded' : 152000}
        ]
        for entry in prices:
            price = entry['price']
            expected = entry['rounded']
            rounded_price = round_to_level(price)
            print(f"{price} rounded to: {rounded_price}")
            assert(rounded_price==expected)