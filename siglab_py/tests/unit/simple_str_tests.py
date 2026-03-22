import unittest
from typing import List, Dict, Union
import time

from numpy import equal

from util.simple_str import classify_ticker

class SimpleStrTests(unittest.TestCase):

    def test_classify_ticker(self):
        cases = [
            # Crypto spot
            ("BTC/USDT",                "crypto.spot"),
            ("ETH/BTC",                 "crypto.spot"),
            ("1000PEPE/USDT",           "crypto.spot"),
            ("NVDA/USDC",               "crypto.spot"),

            # Crypto perpetual
            ("BTC/USDT:USDT",           "crypto.perpetual"),
            ("NVDA/USDC:USDC",          "crypto.perpetual"),
            ("XRP/USDT:1000XRP",        "crypto.perpetual"),

            # Crypto dated future & option
            ("BTC-26MAR26",             "crypto.dated_future"),
            ("BTC-22MAR26-68500-C",     "crypto.option"),
            ("ETH-25DEC25",             "crypto.dated_future"),
            ("BTC-USD-251226-100000-C", "crypto.option"),

            # TradFi / IBKR style
            ("AAPL",                    "tradfi.stock"),
            ("TSLA",                    "tradfi.stock"),
            ("IBM-STK-SMART-USD",       "tradfi.stock"),
            ("ESU6",                    "tradfi.future"),       # Sep 2026
            ("NQZ5",                    "tradfi.future"),
            ("CLM6",                    "tradfi.future"),
            ("AAPL-OPT-20260320-250-C-SMART-100-USD", "tradfi.option"),
            ("SPY-OPT-20260618-550-P-CBOE", "tradfi.option"),
            ("ES-FOP-20260320-5000-C-CME",  "tradfi.option"),

            # Edge / invalid
            ("BTCUSDT",                 "tradfi.stock"),
            ("BTC-USDT",                "unknown"),
            ("@ES",                     "unknown"),       # continuous future
            ("SPY",                     "tradfi.stock"),  # ETF
        ]

        for ticker, expected in cases:
            result = classify_ticker(ticker)
            assert result == expected, \
                f"Failed: {ticker!r} → got {result!r}, expected {expected!r}"