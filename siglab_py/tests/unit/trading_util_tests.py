import unittest
from datetime import datetime, timedelta
from typing import Union
from pathlib import Path

from util.trading_util import *


# @unittest.skip("Skip all integration tests.")
class TradingUtilTests(unittest.TestCase):
    def test_timestamp_to_active_trading_regions_case1(self):
        tp_min_percent : float = 1.5
        tp_max_percent : float = 2.5
        sl_percent_trailing : float = 50 # Trailing stop loss in percent
        default_effective_tp_percent_trailing : float = 50

        pnl_percent_notional : float = 0.5 # Trade's current pnl in percent.

        effective_tp_trailing_percent = calc_eff_trailing_sl(
            tp_min_percent = tp_min_percent,
            tp_max_percent = tp_max_percent,
            sl_percent_trailing = sl_percent_trailing,
            pnl_percent_notional = pnl_percent_notional,
            default_effective_tp_percent_trailing = default_effective_tp_percent_trailing
        )
        assert(effective_tp_trailing_percent==50) # Generous trailing SL when trading starting out and pnl small.

    def test_timestamp_to_active_trading_regions_case2(self):
        tp_min_percent : float = 1.5
        tp_max_percent : float = 2.5
        sl_percent_trailing : float = 50 # Trailing stop loss in percent
        default_effective_tp_percent_trailing : float = 50

        pnl_percent_notional : float = 2 # Trade's current pnl in percent.

        effective_tp_trailing_percent = calc_eff_trailing_sl(
            tp_min_percent = tp_min_percent,
            tp_max_percent = tp_max_percent,
            sl_percent_trailing = sl_percent_trailing,
            pnl_percent_notional = pnl_percent_notional,
            default_effective_tp_percent_trailing = default_effective_tp_percent_trailing
        )
        assert(effective_tp_trailing_percent==25) # Intermediate trailing SL

    def test_timestamp_to_active_trading_regions_case3(self):
        tp_min_percent : float = 1.5
        tp_max_percent : float = 2.5
        sl_percent_trailing : float = 50 # Trailing stop loss in percent
        default_effective_tp_percent_trailing : float = 50

        pnl_percent_notional : float = 2.5 # Trade's current pnl in percent.

        effective_tp_trailing_percent = calc_eff_trailing_sl(
            tp_min_percent = tp_min_percent,
            tp_max_percent = tp_max_percent,
            sl_percent_trailing = sl_percent_trailing,
            pnl_percent_notional = pnl_percent_notional,
            default_effective_tp_percent_trailing = default_effective_tp_percent_trailing
        )
        assert(effective_tp_trailing_percent==0) # Most tight trailing SL