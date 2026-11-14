import sys
import traceback
import argparse
import time
from datetime import datetime
from typing import Dict, Union
import logging
import json
from pprint import pformat
from tabulate import tabulate
import asyncio
import pandas as pd
import numpy as np
from redis import StrictRedis

from ccxt.base.exchange import Exchange
from siglab_py.util.retry_util import retry
from siglab_py.util.market_data_util import instantiate_exchange, candle_size_to_interval_sec, fetch_candles, fetch_tape, fetch_candles
from siglab_py.util.analytic_util import compute_candles_stats
from siglab_py.util.trading_util import detect_sweeps_and_absorption

VALIDATION_MAX_GAPS = 30

'''
From command prompt:
    python tape_provider.py --exchange_name binance --default_type linear --ticker BTC --candles_size 1h --how_many_candles 2160 --loop_freq_ms 900000 --redis_ttl_ms 7776000000

        {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python Debugger: Current File",
                    "type": "debugpy",
                    "request": "launch",
                    "justMyCode": false,
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "args" : [
                            "--exchange_name", "binance",
                            "--default_type", "linear",
                            "--ticker", "BTC",
                            "--candle_size", "1h",
                            "--how_many_candles", "2160",
                            "--loop_freq_ms", "900000",
                            "--redis_ttl_ms", "7776000000"
                    ],
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}"
                    }
                }
            ]
        }
'''
param : Dict = {
    'ticker' : 'BTC',
    'depth' : 500,

    'exchange_name_2_quote_ccy_map' : {
        'binance' : 'USDT',
        'bybit' : 'USDT',
        'okx' : 'USDT',
        'deribit' : 'USDT',
        'kraken' : 'USDT',

        'hyperliquid' : 'USDC',
        'lighter' : 'USDC',
    },

    'refetch_historical_candles_every_num_days' : 365/12,
    'candles_size' : '1h',
    'how_many_candles' : 24*9,
    'history_depth_num_intervals' : {
        'd' : 365,
        'h' : 24 *(365/12),
        'm' : 4*24 *7
    },

    'loop_freq_ms' : 1000*30, # Loop frequency
    'redis_ttl_ms' : 1000*60*15,
    
    'mds' : {
        'redis' : {
            'host' : 'localhost',
            'port' : 6379,
            'db' : 0,
            'ttl_ms' : 1000*60*15 # 15 min?
        }

    },
    
    # Keep track of latency issues: ts_delta_observation_ms: Keep track of server clock vs timestamp from exchange
    'ts_delta_observation_ms_threshold' : 100
}

depth : int = param['depth']

logging.Formatter.converter = time.gmtime
logger = logging.getLogger()
log_level = logging.INFO # DEBUG --> INFO --> WARNING --> ERROR
logger.setLevel(log_level)
format_str = '%(asctime)s %(message)s'
formatter = logging.Formatter(format_str)
sh = logging.StreamHandler()
sh.setLevel(log_level)
sh.setFormatter(formatter)
logger.addHandler(sh)
fh = logging.FileHandler(f"{__file__}.log")
fh.setLevel(log_level)
fh.setFormatter(formatter)     
logger.addHandler(fh)

async def main():
    def parse_args():
        parser = argparse.ArgumentParser() # type: ignore

        parser.add_argument("--exchange_name", help="For example hyperliquid, binance, bybit, okx ...etc", default=None)
        parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
        parser.add_argument("--ticker", help="For crypto or FX, this is base ccy. For example, BTC/USDT:USDT is perpetual contract. Base ccy is BTC.", default="BTC")
        
        parser.add_argument("--candle_size", help="Fetches historical candles for basics like atr_bps and volume averages, which serves as threshold in determination of sweeps and absorptions.", default="1h")
        parser.add_argument("--how_many_candles", help="Sliding window size for compute_candles_stats.", default=24*90)

        parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster. Default: 900000 ms (15 min)", default=900000)
        parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: 90 days.", default=7776000000)

        args = parser.parse_args()

        param['exchange_name'] = args.exchange_name
        param['default_type'] = args.default_type
                
        param['ticker'] = args.ticker
        
        param['candle_size'] = args.candle_size
        history_depth_num_intervals = param['history_depth_num_intervals'][param['candle_size'][-1]]
        param['history_depth_sec'] = history_depth_num_intervals * candle_size_to_interval_sec(param['candle_size'])
        
        param['loop_freq_ms'] = int(args.loop_freq_ms)
        
        param['mds']['redis']['ttl_ms'] = int(args.redis_ttl_ms) if args.redis_ttl_ms else int(param['candle_interval_sec']*1000)*24*7
        
    def init_redis_client():
        redis_client = StrictRedis(
                        host = param['mds']['redis']['host'],
                        port = param['mds']['redis']['port'],
                        db = 0,
                        ssl = False
                    )
        try:
            redis_client.keys()
        except ConnectionError as redis_conn_error:
            err_msg = f"Failed to connect to redis: {param['mds']['redis']['host']}, port: {param['mds']['redis']['port']}"
            raise ConnectionError(err_msg)
        
        return redis_client
    
    parse_args()

    logger.info(f"startup parameters: {json.dumps(param, indent=4)}")

    redis_client = init_redis_client()
    
    exchange = instantiate_exchange(
        exchange_name = param['exchange_name'],
        default_type = param['default_type']
    )
    ticker : str = param['ticker']
    quote_ccy : str = param['exchange_name_2_quote_ccy_map'][param['exchange_name']]
    normalized_symbol : str = f"{ticker}/{quote_ccy}" if param['default_type']=='spot' else f"{ticker}/{quote_ccy}:{quote_ccy}"

    candle_size : str = param['candle_size']
    how_many_candles : int = param['how_many_candles']
    refetch_historical_candles_every_sec : float = 1000 *60*60*24 * param['refetch_historical_candles_every_num_days']

    historical_candles_fetched_timestamp_sec = None
    i = 0
    total_loop_elapsed_ms : int = 0
    while True:
        dt_now = datetime.now()
        loop_start = dt_now.timestamp()
        
        try:
            # Historical candles to define what atr_bps or volume is normal, what's not. 
            if historical_candles_fetched_timestamp_sec is None or (datetime.now().timestamp() - historical_candles_fetched_timestamp_sec) >= refetch_historical_candles_every_sec:
                historical_candles_fetched_timestamp_sec = loop_start
                start_fetch_ts = time.time()
                pd_historical_candles = fetch_candles(
                                        start_ts=(loop_start - param['history_depth_sec']), 
                                        end_ts=loop_start, 
                                        exchange=exchange, normalized_symbols=[normalized_symbol], 
                                        candle_size = candle_size, 
                                        num_candles_limit = 100,
                                        validation_max_gaps = VALIDATION_MAX_GAPS, # some newly listed FX pairs gaps
                                        logger = None
                                    )[normalized_symbol]
                finish_fetch_ts = time.time()
                elapsed_sec = round(finish_fetch_ts - start_fetch_ts)
                logger.info(f"fetch_candles elapsed_sec: {elapsed_sec}")

                start_compute_ts = time.time()
                compute_candles_stats(
                            pd_candles=pd_historical_candles,
                            boillenger_std_multiples=2, # We don't need BB
                            sliding_window_how_many_candles=how_many_candles,
                            validation_max_gaps = VALIDATION_MAX_GAPS, # some newly listed FX pairs gaps
                            pypy_compat=True # We don't need slope calc
                        )
                finish_compute_ts = time.time()
                elapsed_sec = round(finish_compute_ts - start_compute_ts)
                logger.info(f"compute_candles_stats elapsed_sec: {elapsed_sec}")

            _pd_candles = pd_historical_candles[
                (pd_historical_candles['dayofweek']==dt_now.weekday()) # You can't compare Monday against Fri after close
            ]
            avg_volume_base_ccy = _pd_candles['volume'].mean()
            volume_std_base_ccy = _pd_candles['volume'].std()
            volume_threshold_base_ccy = avg_volume_base_ccy + volume_std_base_ccy

            start_fetch_ts = time.time()
            tape = fetch_tape(exchange=exchange, symbol=normalized_symbol, max_batches=10, limit=1000)
            finish_fetch_ts = time.time()
            elapsed_sec = round(finish_fetch_ts - start_fetch_ts)
            logger.info(f"fetch_tape elapsed_sec: {elapsed_sec}")
    
            if tape.empty:
                logger.info("No trades fetched — check symbol or network.")
                raise SystemExit(0)
    
            logger.info(f"\nFetched {len(tape)} unique trades.")
            logger.info(f"Time range: {tape['datetime'].min()}  →  {tape['datetime'].max()}")
            logger.info(f"Last price: {tape['price'].iloc[-1]:.2f}")
    
            result = detect_sweeps_and_absorption(
                tape,
                volume_threshold_base_ccy=volume_threshold_base_ccy,
                reference_level=float(tape["price"].iloc[-1]),
                level_tolerance_pct=0.001,
            )

            bursts = result["bursts"]
            residual = (bursts["buy_vol"] + bursts["sell_vol"] - bursts["volume"]).abs().max()
            logger.info(f"Bursts:      {len(bursts)}  (max volume residual: {residual:.2e})")
            logger.info(f"Sweeps:      {len(result['sweeps'])}")
            logger.info(f"Absorption:  {len(result['absorption'])}\n")
    
            if not result["sweeps"].empty:
                logger.info("Top sweeps by volume:")
                top_sweeps = (
                    result["sweeps"]
                    .sort_values("volume", ascending=False)
                    .head(5)
                    .sort_values("start_time")
                )
                logger.info(
                    top_sweeps[["start_time", "end_price", "volume", "direction", "near_reference"]]
                    .to_string(index=False)
                )
                tape_key : str = f"tape:{ticker}:sweeps:{int(time.time()*1000)}"
                payload = top_sweeps.to_dict(orient="records")
                redis_client.set(tape_key, json.dumps(payload, default=str).encode('utf-8'), ex=int(param['mds']['redis']['ttl_ms']/1000))
    
            if not result["absorption"].empty:
                logger.info("Top absorption by volume:")
                top_absorption = (
                    result["absorption"]
                    .sort_values("volume", ascending=False)
                    .head(5)
                    .sort_values("start_time")
                )
                logger.info(
                    top_absorption[["start_time", "end_price", "volume", "absorbed_side", "near_reference"]]
                    .to_string(index=False)
                )
                tape_key : str = f"tape:{ticker}:sweeps:{int(time.time()*1000)}"
                payload = top_absorption.to_dict(orient="records")
                redis_client.set(tape_key, json.dumps(payload, default=str).encode('utf-8'), ex=int(param['mds']['redis']['ttl_ms']/1000))
    
        except Exception as loop_err:
            logger.error(f"#{i} Error: {loop_err}. {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")

        finally:
            this_loop_elapsed_ms : int = int((time.time()-loop_start)*1000)
            total_loop_elapsed_ms += this_loop_elapsed_ms
            avg_loop_elapsed_ms : int = int(total_loop_elapsed_ms / (i+1))
            logger.info(f"#{i} this_loop_elapsed_ms: {this_loop_elapsed_ms}, avg_loop_elapsed_ms: {avg_loop_elapsed_ms}")

            time.sleep(int(param['loop_freq_ms']/1000))
        
        i += 1
        
asyncio.run(main())