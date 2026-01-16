import sys
import logging
import argparse
from datetime import datetime, timedelta
import time
from typing import Dict, Union, Any
from enum import Enum
import asyncio
import pandas as pd

from ccxt.base.exchange import Exchange as CCXTExchange
from ccxt.binance import binance
from ccxt.bybit import bybit
from ccxt.okx import okx
from ccxt.deribit import deribit
from ccxt.kraken import kraken
from ccxt.hyperliquid import hyperliquid

from siglab_py.exchanges.futubull import Futubull
from siglab_py.util.market_data_util import fetch_candles
# from util.market_data_util import fetch_candles # For debug only
from siglab_py.util.analytic_util import compute_candles_stats
# from util.analytic_util import compute_candles_stats # For debug only

'''
Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python ccxt_candles_ta_to_csv.py --exchange_name okx --symbol BTC/USDT:USDT --candle_size 1h --end_date "2025-04-22 0:0:0" --start_date "2024-01-01 0:0:0" --default_type linear --compute_ta Y --pypy_compatible N
    
    (Remember: python -mpip install siglab_py)

This script is pypy compatible. Set "pypy_compatible" to True, in which case "compute_candles_stats" will skip calculation for TAs which requires: scipy, statsmodels, scikit-learn, sklearn.preprocessing
    pypy ccxt_candles_ta_to_csv.py --exchange_name bybit --symbol BTC/USDT:USDT --end_date "2025-03-11 0:0:0" --start_date "2024-03-11 0:0:0" --default_type linear --compute_ta Y --pypy_compatible Y

    (Remember: pypy -mpip install siglab_py)

Other arguments:
    candle_size: default 1h (Hourly candles). You can specify 1d, 1m ...etc
    ma_long_intervals (default 24), ma_short_intervals (default 8): 
        analytic_util.compute_candles_stats employ sliding windows to calculate things like std (Standard Deviation), EMA/SMAs, and actually most other technical indicators.
        compute_candles_stats calculate certain things, for example EMA, in two levels: 'long' vs 'short'
        'long' refers to 'higher timeframe' - this uses a bigger sliding window specified by 'ma_long_intervals'
        'short' refers to 'lower timeframes' - this uses a smaller sliding window specified by 'ma_short_intervals'

    compute_ta: Whether you wish to compute technical indicators? Y or N (Default)
    pypy_compatible: Some technical indicators requires python libraries that's not pypy compatible, such as statsmodels.api (slopes and divergence calc) and scipy.stats.linregress. Set to Y, then analytic_util.compute_candles_stats will skip calculations which requires these libraries. 

If debugging from VSCode, launch.json:

    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python Debugger: Current File",
                "type": "debugpy",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "args" : [
                        "--exchange_name", "bybit",
                        "--symbol", "BTC/USDT:USDT",
                        "--end_date", "2025-04-22 0:0:0",
                        "--start_date", "2024-01-01 0:0:0",
                        "--default_type", "linear",
                        "--compute_ta", "Y",
                        "--pypy_compatible", "N"
                    ],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            }
        ]
    }
'''
end_date : datetime = datetime.today()
end_date = datetime(end_date.year, end_date.month, end_date.day)
start_date : datetime = end_date - timedelta(days=365)

param : Dict = {
    'exchange' : 'bybit',
    'symbol' : None,
    'start_date' : start_date,
    'end_date' : end_date,
    'exchange_params' : {
                        'rateLimit' : 100, # in ms
                        'options' : {
                            'defaultType' : "linear"
                        }
                    },
    'output_filename' : 'candles_ta_$SYMBOL$.csv'
}

class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

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

def log(message : str, log_level : LogLevel = LogLevel.INFO):
    if log_level.value<LogLevel.WARNING.value:
        logger.info(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(f"{datetime.now()} {message}")

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore
    parser.add_argument("--exchange_name", help="Exchange name. bybit, okx, bybit, deribit, hyperliquid ...etc, add whatever you want top of script, import them. Then add to instantiate_exchange.", default="bybit")
    parser.add_argument("--symbol", help="symbol, CEX example BTC/USDT for spot. BTC/USDT:USDT for perpetuals. Many DEXes offer USDC pairs.", default="BTC/USDT:USDT")
    parser.add_argument("--start_date", help="Format: %Y-%m-%d %H:%M:%S", default=None)
    parser.add_argument("--end_date", help="Format: %Y-%m-%d %H:%M:%S", default=None)

    '''
    Enums here: 
    https://openapi.futunn.com/futu-api-doc/en/quote/quote.html#66
    https://openapi.futunn.com/futu-api-doc/en/trade/trade.html#9434
    '''
    parser.add_argument("--default_type", help="Depends on exchange. Very often, spot, linear/swap for perpetuals. Have a look at gateway.py instantiate_exchange https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py", default="linear")

    parser.add_argument("--compute_ta", help="Compute technical indicators?. Y or N (default).", default='N')
    parser.add_argument("--candle_size", help="candle interval: 1m, 1h, 1d... etc", default='1h')
    parser.add_argument("--ma_long_intervals", help="Sliding Window size in number of intervals for higher timeframe", default=24)
    parser.add_argument("--ma_short_intervals", help="Sliding Window size in number of intervals for lower timeframe", default=8)
    parser.add_argument("--boillenger_std_multiples", help="Boillenger bands: # std", default=2)

    parser.add_argument("--pypy_compatible", help="pypy_compatible: If Y, analytic_util will import statsmodels.api (slopes and divergence calc). In any case, partition_sliding_window requires scipy.stats.linregress and cannot be used with pypy. Y or N (default).", default='N')

    args = parser.parse_args()
    param['exchange_name'] = args.exchange_name.strip().lower()
    param['symbol'] = args.symbol.strip().upper()

    param['start_date'] = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S") if args.start_date else start_date
    param['end_date'] = datetime.strptime(args.end_date, "%Y-%m-%d %H:%M:%S") if args.end_date else end_date
    
    param['exchange_params']['options']['defaultType'] = args.default_type

    param['output_filename'] = param['output_filename'].replace('$SYMBOL$', param['symbol'].replace(":",".").replace("/","."))

    if args.compute_ta:
        if args.compute_ta=='Y':
            param['compute_ta'] = True
        else:
            param['compute_ta'] = False
    else:
        param['compute_ta'] = False
    param['candle_size'] = args.candle_size
    param['ma_long_intervals'] = int(args.ma_long_intervals)
    param['ma_short_intervals'] = int(args.ma_short_intervals)
    param['boillenger_std_multiples'] = int(args.boillenger_std_multiples)

    if args.pypy_compatible:
        if args.pypy_compatible=='Y':
            param['pypy_compatible'] = True
        else:
            param['pypy_compatible'] = False
    else:
        param['pypy_compatible'] = False

def instantiate_exchange(
    exchange_name : str,
    exchange_params : Dict[str, Any]
) -> CCXTExchange:
    if exchange_name=='binance':
        return binance(exchange_params)
    elif exchange_name=='bybit':
        return bybit(exchange_params)
    elif exchange_name=='okx':
        return okx(exchange_params)
    elif exchange_name=='deribit':
        return deribit(exchange_params)
    else:
        raise ValueError(f"Unsupported exchange {exchange_name}. Please import top of script and add to instantiate_exchange.")
    
async def main():
    parse_args()

    fh = logging.FileHandler(f"ccxt_candles_ta_to_csv.log")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)     
    logger.addHandler(fh) # type: ignore

    exchange = instantiate_exchange(param['exchange_name'], param['exchange_params'])
    markets = exchange.load_markets()
    if param['symbol'] not in markets:
        raise ValueError(f"{param['symbol']} not support by {param['exchange_name']}")
    
    pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=int(param['start_date'].timestamp()),
            end_ts=int(param['end_date'].timestamp()),
            exchange=exchange,
            normalized_symbols=[ param['symbol'] ],
            candle_size=param['candle_size']
        )[param['symbol']]

    assert pd_candles is not None

    if pd_candles is not None:
        assert len(pd_candles) > 0, "No candles returned."
        expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute', 'week_of_month', 'apac_trading_hr', 'emea_trading_hr', 'amer_trading_hr'}
        assert set(pd_candles.columns) >= expected_columns, "Missing expected columns."
        assert pd_candles['timestamp_ms'].notna().all(), "timestamp_ms column contains NaN values."
        assert pd_candles['timestamp_ms'].is_monotonic_increasing, "Timestamps are not in ascending order."

    if param['compute_ta']:
        start = time.time()
        compute_candles_stats(
                                    pd_candles=pd_candles, 
                                    boillenger_std_multiples=param['boillenger_std_multiples'], 
                                    sliding_window_how_many_candles=param['ma_long_intervals'], 
                                    slow_fast_interval_ratio=(param['ma_long_intervals']/param['ma_short_intervals']),
                                    pypy_compat=param['pypy_compatible']
                                )
        compute_candles_stats_elapsed_ms = int((time.time() - start) *1000)
        log(f"TA calculated, took {compute_candles_stats_elapsed_ms} ms")

    log(f"Candles (# rows: {pd_candles.shape[0]}) written to {param['output_filename']}")
    pd_candles.to_csv(param['output_filename'])

    sys.exit()

asyncio.run(main())
