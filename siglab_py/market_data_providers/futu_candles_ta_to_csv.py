import sys
import logging
import argparse
from datetime import datetime, timedelta
import time
from typing import Dict, Union
from enum import Enum
import asyncio
import pandas as pd

from futu import *

from siglab_py.exchanges.futubull import Futubull
from siglab_py.util.market_data_util import fetch_candles
from siglab_py.util.analytic_util import compute_candles_stats

'''
Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python futu_candles_ta_to_csv.py --symbol HK.00700 --end_date "2025-03-11 0:0:0" --start_date "2024-03-11 0:0:0" --market HK --trdmarket HK --security_firm FUTUSECURITIES --security_type STOCK --compute_ta Y --pypy_compatible N

    python futu_candles_ta_to_csv.py --symbol AAPL --end_date "2025-03-11 0:0:0" --start_date "2024-03-11 0:0:0" --market US --trdmarket US --security_firm FUTUSECURITIES --security_type STOCK --compute_ta Y --pypy_compatible N

This script is pypy compatible. Set "pypy_compatible" to True, in which case "compute_candles_stats" will skip calculation for TAs which requires: scipy, statsmodels, scikit-learn, sklearn.preprocessing
    python futu_candles_ta_to_csv.py --symbol HK.00700 --end_date "2025-03-11 0:0:0" --start_date "2024-03-11 0:0:0" --market HK --trdmarket HK --security_firm FUTUSECURITIES --security_type STOCK --compute_ta Y --pypy_compatible Y

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
                        "--symbol", "HK.00700",
                        "--end_date", "2025-03-11 0:0:0",
                        "--start_date", "2024-03-11 0:0:0",
                        "--market", "HK",
                        "--trdmarket", "HK",
                        "--security_firm", "FUTUSECURITIES",
                        "--security_type", "STOCK",
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
    'symbol' : None,
    'start_date' : start_date,
    'end_date' : end_date,
    'trdmarket' : TrdMarket.HK,
    'security_firm' : SecurityFirm.FUTUSECURITIES,
    'market' : Market.HK, 
    'security_type' : SecurityType.STOCK,
    'daemon' : {
        'host' : '127.0.0.1',
        'port' : 11111
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
    parser.add_argument("--symbol", help="symbol, example HK.00700", default=None)
    parser.add_argument("--start_date", help="Format: %Y-%m-%d %H:%M:%S", default=None)
    parser.add_argument("--end_date", help="Format: %Y-%m-%d %H:%M:%S", default=None)

    '''
    Enums here: 
    https://openapi.futunn.com/futu-api-doc/en/quote/quote.html#66
    https://openapi.futunn.com/futu-api-doc/en/trade/trade.html#9434
    '''
    parser.add_argument("--market", help="market: HK SH SZ US AU CA FX", default=Market.HK)
    parser.add_argument("--trdmarket", help="trdmarket: HK, HKCC, HKFUND, FUTURES, CN, CA, AU, JP, MY, SG, US, USFUND", default=TrdMarket.HK)
    parser.add_argument("--security_firm", help="security_firm: FUTUSECURITIES (HK), FUTUINC (US), FUTUSG (SG), FUTUAU (AU)", default=SecurityFirm.FUTUSECURITIES)
    parser.add_argument("--security_type", help="STOCK, BOND, ETF, FUTURE, WARRANT, IDX ... ", default=SecurityType.STOCK)

    parser.add_argument("--compute_ta", help="Compute technical indicators?. Y or N (default).", default='N')
    parser.add_argument("--candle_size", help="candle interval: 1m, 1h, 1d... etc", default='1h')
    parser.add_argument("--ma_long_intervals", help="Window size in number of intervals for higher timeframe", default=24)
    parser.add_argument("--ma_short_intervals", help="Window size in number of intervals for lower timeframe", default=8)
    parser.add_argument("--boillenger_std_multiples", help="Boillenger bands: # std", default=2)

    parser.add_argument("--pypy_compatible", help="pypy_compatible: If Y, analytic_util will import statsmodels.api (slopes and divergence calc). In any case, partition_sliding_window requires scipy.stats.linregress and cannot be used with pypy. Y or N (default).", default='N')

    args = parser.parse_args()
    param['symbol'] = args.symbol.strip().upper()

    param['start_date'] = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S") if args.start_date else start_date
    param['end_date'] = datetime.strptime(args.end_date, "%Y-%m-%d %H:%M:%S") if args.end_date else end_date
    
    param['market'] = args.market
    param['trdmarket'] = args.trdmarket
    param['security_firm'] = args.security_firm

    param['output_filename'] = param['output_filename'].replace('$SYMBOL$', param['symbol'])

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

async def main():
    parse_args()

    fh = logging.FileHandler(f"futu_candles_ta_to_csv.log")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)     
    logger.addHandler(fh) # type: ignore

    exchange = Futubull(param)

    pd_candles: Union[pd.DataFrame, None] = fetch_candles(
            start_ts=int(param['start_date'].timestamp()),
            end_ts=int(param['end_date'].timestamp()),
            exchange=exchange,
            normalized_symbols=[ param['symbol'] ],
            candle_size='1d'
        )[param['symbol']]

    assert pd_candles is not None

    if pd_candles is not None:
        assert len(pd_candles) > 0, "No candles returned."
        expected_columns = {'exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime_utc', 'datetime', 'year', 'month', 'day', 'hour', 'minute'}
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
