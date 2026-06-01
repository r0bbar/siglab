import asyncio
import os
import sys
import traceback
import logging
from typing import List, Dict, Optional, Union, Any, NoReturn
from datetime import datetime, timedelta
import time
import arrow
from dateutil import parser
import argparse
import json
from enum import Enum
import feedparser
import pandas as pd
from pprint import pformat
from redis import StrictRedis
from requests.exceptions import HTTPError

from siglab_py.util.retry_util import retry
from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.util.market_data_util import instantiate_exchange, timestamp_to_datetime_cols
from siglab_py.util.market_data_util import fetch_candles
from siglab_py.util.notification_util import dispatch_notification

current_filename = os.path.basename(__file__)
current_dir : str = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

logging.Formatter.converter = time.gmtime
logger: logging.Logger = logging.getLogger()
log_level: int = logging.INFO
logger.setLevel(log_level)
format_str: str = '%(asctime)s %(message)s'
formatter: logging.Formatter = logging.Formatter(format_str)
sh: logging.StreamHandler = logging.StreamHandler()
sh.setLevel(log_level)
sh.setFormatter(formatter)
logger.addHandler(sh)
fh = logging.FileHandler(f"simple_price_alert.log")
fh.setLevel(log_level)
fh.setFormatter(formatter)     
logger.addHandler(fh)

'''
Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python historical_trades_provider.py --tickers "BTC/USDT:USDT,SOL/USDT:USDT,ETH/USDT:USDT,XAG/USDT:USDT" --exchange_name binance --notification_info_url https://xxx.com/xxx --notification_critical_url https://xxx.com/xxx --notification_alert_url https://xxx.com/xxx
    
    --tickers: comma separated list of tickers you want to monitor
    --exchange_name: defaults to binance

    --notification_info_url/notification_critial_url/notification_alert_url: optional.

launch.json for Debugging from VSCode:
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
                "args": [
                    "--tickers" , "BTC/USDT:USDT,SOL/USDT:USDT,ETH/USDT:USDT,XAG/USDT:USDT,HYPE/USDT:USDT",
                    "--exchange_name", "lighter",
                    "--notification_info_url", "https://xxx.com/xxx",
                    "--notification_critical_url", "https://xxx.com/xxx",
                    "--notification_alert_url", "https://xxx.com/xxx",
                ],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            }
        ]
'''
param : Dict = {
    'loop_freq_ms' : 1000, 
    'current_filename' : current_filename,

    'cache_filename' : r"historical_trades_$EXCHANGE_NAME$_$BASE_CCY$_$YYYYMMDD$.csv",

    'notification' : {
        'footer' : None,

        # notification webhook url's for notifications
        'notification' : {
            'info' : { 'webhook_url' : None },
            'critical' : { 'webhook_url' : None },
            'alert' : { 'webhook_url' : None },
        }
    },
    
    'mds': {
        'topics': {
            
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl_ms': 1000 * 60 * 15
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore
    
    parser.add_argument("--exchange_name", help="Exchange to monitor, default is Binance", default='binance')
    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')
    parser.add_argument("--default_sub_type", help="default_sub_type", default=None)

    parser.add_argument("--tickers", help="Comma seperated list, example: BTC/USDT:USDT,SOL/USDT:USDT,ETH/USDT:USDT,XAG/USDT:USDT,HYPE/USDT:USDT", default="BTC/USDT:USDT,SOL/USDT:USDT,ETH/USDT:USDT,XAG/USDT:USDT,HYPE/USDT:USDT")
    
    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)

    parser.add_argument("--notification_info_url", help="Webhook url for INFO", default=None)
    parser.add_argument("--notification_critical_url", help="Webhook url for CRITICAL", default=None)
    parser.add_argument("--notification_alert_url", help="Webhook url for ALERT", default=None)
    parser.add_argument("--enable_notification", help="Enable notification on new headline arrival? Y or N (default).", default='Y')

    args = parser.parse_args()
    
    param['exchange_name'] = args.exchange_name
    param['default_type'] = args.default_type
    param['default_sub_type'] = args.default_sub_type

    param['tickers'] = args.tickers.split(',')
    param['rate_limit_ms'] = int(args.rate_limit_ms)

    param['notification']['notification']['info']['webhook_url'] = args.notification_info_url
    param['notification']['notification']['critical']['webhook_url'] = args.notification_critical_url
    param['notification']['notification']['alert']['webhook_url'] = args.notification_alert_url

    if args.enable_notification:
        if args.enable_notification=='Y':
            param['enable_notification'] = True
        else:
            param['enable_notification'] = False
    else:
        param['enable_notification'] = True

    param['notification']['footer'] = f"From {param['current_filename']}"

    logger.info(f"Startup args: {args}") # Dont use logger, not yet setup yet.
    logger.info(f"param: {json.dumps(param, indent=2)}")

def init_redis_client() -> StrictRedis:
    redis_client : StrictRedis = StrictRedis(
                    host = param['mds']['redis']['host'],
                    port = param['mds']['redis']['port'],
                    db = 0,
                    ssl = False
                )
    try:
        redis_client.keys()
    except ConnectionError as redis_conn_error:
        err_msg = f"Failed to connect to redis: {param['mds']['redis']['host']}, port: {param['mds']['redis']['port']}"
        logger.info(f"Failed to init redis connection. Will skip publishes to redis. {err_msg}")
        redis_client = None # type: ignore
    
    return redis_client

async def main() -> None:
    parse_args()

    notification_params : Dict[str, Any] = param['notification']

    try:        
        redis_client : Optional[StrictRedis] = init_redis_client()
    except Exception as redis_err:
        redis_client = None
        logger.info(f"Failed to connect to redis. Still run but not publishing to it. {redis_err}")

    exchange : Union[AnyExchange, None] = instantiate_exchange(
        exchange_name=param['exchange_name'],
        api_key=None,
        secret=None,
        passphrase=None,
        default_type=param['default_type'],
        rate_limit_ms=param['rate_limit_ms'],
    )

    @retry(num_attempts=3, pause_between_retries_ms=1000)
    def _fetch_trades(exchange, symbol, since, limit) -> List:
        trades = exchange.fetch_trades(symbol=symbol, since=since, limit=limit)
        if trades and len(trades)>1:
            trades.sort(key=lambda x : x['timestamp'], reverse=False)
            assert(trades[0]['timestamp']<trades[-1]['timestamp'])
        return trades

    loop_counter : int = 0
    while True:
        try:
            start_ts_sec = time.time()

            try:
                markets = exchange.load_markets() 
                for ticker in param['tickers']:
                    if ticker in markets:
                        base_ccy = ticker.split('/')[0]
                        cache_filename = param['cache_filename'].replace("$BASE_CCY$", base_ccy)
                        cache_filename = cache_filename.replace("$EXCHANGE_NAME$", exchange.name)
                        cache_filename = cache_filename.replace("$YYYYMMDD$", datetime.now().strftime("%Y%m%d"))

                        if os.path.exists(cache_filename):
                            pd_historical_trades = pd.read_csv(cache_filename)
                            pd_historical_trades.drop(pd_historical_trades.columns[pd_historical_trades.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
                            pd_historical_trades['timestamp_ms'] = pd_historical_trades['timestamp_ms'].astype('Int64')
                        else:
                            pd_historical_trades = pd.DataFrame()

                        trades = _fetch_trades(exchange=exchange, symbol="BTC/USDT:USDT", since=None, limit=1000)
                        pd_new_trades = pd.DataFrame(trades)
                        pd_new_trades = pd_new_trades.rename(columns={"timestamp": "timestamp_ms", "amount": "amount_base_ccy"})
                        pd_new_trades["amount_quote_ccy"] = pd_new_trades["amount_base_ccy"] * pd_new_trades["price"]
                        timestamp_to_datetime_cols(pd_new_trades)

                        pd_historical_trades = (
                            pd.concat([pd_new_trades, pd_historical_trades])
                            .drop_duplicates(subset=['id'], keep='last')
                            .sort_values('timestamp_ms', ascending=True)
                            .reset_index(drop=True)
                        )

                        pd_historical_trades.to_csv(cache_filename)

            except HTTPError as fetch_err:
                err_msg = f'[{exchange.name}] fetch error, {fetch_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}'
                logger.error(err_msg)

            elapsed_ms = int((time.time() - start_ts_sec) *1000)
            logger.info(f"[loop# {loop_counter}] fetch elapsed_ms: {elapsed_ms:,}")

        except Exception as loop_err:
            err_msg = f'loop error {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}'
            logger.error(err_msg)
            dispatch_notification(title=f"{param['current_filename']} error.", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)
                
        finally:
            await asyncio.sleep(int(param['loop_freq_ms'] / 1000))

            loop_counter += 1

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.info(f"Unexpected error: {e} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")