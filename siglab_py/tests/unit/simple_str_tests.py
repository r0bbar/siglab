import unittest
from typing import List, Dict, Union
import time
import json
from numpy import equal
from pathlib import Path

from util.simple_str import keywords_match, classify_ticker

class SimpleStrTests(unittest.TestCase):
    
    def test_keywords_match(self):
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
                            \ unit
                                __init__.py
                                analytic_util_tests.py <-- Tests here
                            
                \ siglab_rs <-- Rust project root
                \ data	 <-- Data files here!
        '''
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        with open(f"{data_dir}\\market_impact_keywords.json", 'r') as f:
            keywords_cache = json.load(f)
        fuzzy_threshold = 80
        # Note: Israel -> Isreel
        example = "Iran and Isreel are in tense negotiations. Oil prices surged after a missile strike near the Strait of Hormuz. The Fed might pause rates cuts."
        matches = keywords_match(
            sentence=example, 
            keywords_cache=keywords_cache, 
            fuzzy=True, 
            fuzzy_threshold=fuzzy_threshold
        )
        print(json.dumps(matches, indent=2))

        assert('num_matches' in matches)
        assert('word_count' in matches)
        assert('matches_percent' in matches)
        assert(matches['num_matches']>=0)
        assert(matches['word_count']>0)
        assert(matches['matches_percent']<=100)
        assert('nouns' in matches)
        assert('actions' in matches)
        assert('adjectives' in matches)

        assert(matches['matches_percent']>40)

        assert(matches['nouns']['countries']==['israel', 'iran'])
        assert(matches['nouns']['financial_institutions']==['fed'])
        assert(matches['nouns']['infrastructure']==['strait of hormuz'])
        assert(matches['nouns']['commodities']==['oil'])

        assert(matches['actions']['diplomacy']==['negotiation'])
        assert(matches['actions']['attack']==['strike', 'missile'])
        assert(matches['actions']['economic_policy']==['cut'])
        assert(matches['actions']['market_movement']==['surge'])

        assert(matches['adjectives']['geopolitical']==['tense'])

    def test_classify_ticker(self):
        cases = [
            # Crypto spot
            ("BTC/USDT",                "spot"),
            ("ETH/BTC",                 "spot"),
            ("1000PEPE/USDT",           "spot"),
            ("NVDA/USDC",               "spot"),
            ("EUR/JPY",                 "spot"),
            ("AUD/NZD",                 "spot"),

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