import os
import sys
import traceback
import logging
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Any, Union, Callable
from datetime import datetime, timedelta, timezone
import time
import arrow
from zoneinfo import ZoneInfo
from tabulate import tabulate
from pprint import pformat
import plotext as plt
import asyncio
import pandas as pd

from redis import StrictRedis

from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.util.market_data_util import instantiate_exchange
from siglab_py.util.market_data_util import fetch_funding_rate
from siglab_py.util.notification_util import dispatch_notification
from siglab_py.constants import INVALID, JSON_SERIALIZABLE_TYPES, LogLevel

current_filename = os.path.basename(__file__)
current_dir : str = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

'''
Error: RuntimeError: aiodns needs a SelectorEventLoop on Windows.
Hack, by far the filthest hack I done in my career: Set SelectorEventLoop on Windows
'''
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

param : Dict = {
    "num_days" : 365 *3,
    "loop_freq_ms" : 60000 * 60, 

    'current_filename' : current_filename,
    'current_dir' : parent_dir,
    'raw_funding_rate_cache_filename' : "funding_history_$BASE_CCY$.csv",
    'bucketed_funding_rate_cache_filename' : "bucketed_funding_history_$BASE_CCY$.csv",
    'tickers_summary' : "tickers_summary_$EXCHANGE_NAME$.csv",

    'notification' : {
        'footer' : None,

        # notification webhook url's for notifications
        'notification' : {
            'info' : { 'webhook_url' : None },
            'critical' : { 'webhook_url' : None },
            'alert' : { 'webhook_url' : None },
        }
    },

    'mds' : {
        'redis' : {
            'host' : 'localhost',
            'port' : 6379,
            'db' : 0,
            'ttl_ms' : 1000*60*15 # 15 min?
        }
    }
}


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
        logger.info(message)

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(message)

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(message)

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
        raise ConnectionError(err_msg)
    
    return redis_client

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exchange_name", help="Exchange name", default='binance')
    parser.add_argument("--tickers", help="Comma seperated list, example: BTC/USDT:USDT,ETH/USDT:USDT,XRP/USDT:USDT ", default="BTC/USDT:USDT")
    parser.add_argument("--default_type", help="default_type: spot, linear, inverse, futures ...etc", default='linear')

    parser.add_argument("--rate_limit_ms", help="rate_limit_ms: Check your exchange rules", default=100)
    parser.add_argument("--verbose", help="logging verbosity, Y or N (default).", default='N')
    parser.add_argument("--loop_freq_ms", help="Loop delays. Remember funding rates generally updated every 8 hours, so no need fast.", default=60000 * 60)

    parser.add_argument("--notification_info_url", help="Webhook url for INFO", default=None)
    parser.add_argument("--notification_critical_url", help="Webhook url for CRITICAL", default=None)
    parser.add_argument("--notification_alert_url", help="Webhook url for ALERT", default=None)

    args, additional_args = parser.parse_known_args()

    param['exchange_name'] = args.exchange_name
    param['tickers'] = args.tickers.split(',')
    param['default_type'] = args.default_type

    param['rate_limit_ms'] = int(args.rate_limit_ms)

    if args.verbose:
        if args.verbose=='Y':
            param['verbose'] = True
        else:
            param['verbose'] = False
    else:
        param['verbose'] = False

    param['loop_freq_ms'] = int(args.loop_freq_ms)

    param['notification']['notification']['info']['webhook_url'] = args.notification_info_url
    param['notification']['notification']['critical']['webhook_url'] = args.notification_critical_url
    param['notification']['notification']['alert']['webhook_url'] = args.notification_alert_url

    param['notification']['footer'] = f"From {param['current_filename']} {param['exchange_name']}"

async def main():
    parse_args()

    redis_client : StrictRedis = init_redis_client()

    notification_params : Dict[str, Any] = param['notification']

    exchange : Union[AnyExchange, None] = instantiate_exchange(
        exchange_name=param['exchange_name'],
        api_key=None,
        secret=None,
        passphrase=None,
        default_type=param['default_type'],
        rate_limit_ms=param['rate_limit_ms'],
    )
    if exchange:
        loop_counter = 0
        while True:
            try:
                end_date = datetime.now()
                start_date = end_date + timedelta(days=-90)

                tickers_summary : List[Dict[str, Union[str, float]]] = []

                markets = exchange.load_markets() 
                log(f"# tickers: {len(param['tickers'])}")
                for ticker in param['tickers']:
                    if ticker in markets:
                        base_ccy = ticker.split('/')[0]
                        market = markets[ticker]

                        raw_funding_rate_cache_filename : str = param['raw_funding_rate_cache_filename'].replace("$BASE_CCY$", base_ccy)
                        bucketed_funding_rate_cache_filename : str = param['bucketed_funding_rate_cache_filename'].replace("$BASE_CCY$", base_ccy)

                        if os.path.exists(raw_funding_rate_cache_filename):
                            pd_old_funding_history = pd.read_csv(raw_funding_rate_cache_filename)
                            pd_old_funding_history['timestamp_ms'] = pd_old_funding_history['timestamp_ms'].astype('Int64')
                        else:
                            pd_old_funding_history = pd.DataFrame()

                        results = fetch_funding_rate(
                            exchange=exchange,
                            normalized_symbols = [ ticker ],
                            start_ts=start_date.timestamp(),
                            end_ts=end_date.timestamp(),
                            limit=param['num_days']
                        )
                        pd_funding_history = results[ ticker ]

                        pd_funding_history = (
                            pd.concat([pd_funding_history, pd_old_funding_history])
                            .drop_duplicates(subset=['timestamp_ms'], keep='first')
                            .sort_values('timestamp_ms', ascending=False)
                            .reset_index(drop=True)
                        )

                        pd_bucketed_funding_history = (
                            pd_funding_history
                            .groupby('funding_rate_annualized_bucket')
                            .agg(
                                count=('funding_rate_annualized', 'count'),
                                avg_funding_rate_annualized=('funding_rate_annualized', 'mean')
                            )
                            .reset_index()
                            .sort_values('count', ascending=False)
                        )

                        total_interval_count = int(pd_bucketed_funding_history['count'].sum())
                        top = pd_bucketed_funding_history.iloc[0]
                        funding_rate_annualized_bucket = top['funding_rate_annualized_bucket']
                        top_bucket_count = int(top['count'])
                        top_bucket_avg_funding_rate_annualized = round(float(top['avg_funding_rate_annualized']), 2)
                        top_bucket_dominance_percent = round(top_bucket_count/total_interval_count *100, 2)

                        summary = {
                                'ticker' : ticker,
                                'funding_rate_annualized_bucket' : funding_rate_annualized_bucket,
                                'top_bucket_avg_funding_rate_annualized' : top_bucket_avg_funding_rate_annualized,
                                'top_bucket_count' : top_bucket_count,
                                'total_interval_count' : total_interval_count,
                                'top_bucket_dominance_percent' : top_bucket_dominance_percent,
                                'score' :  round(top_bucket_dominance_percent * top_bucket_avg_funding_rate_annualized,  2)
                            }
                        tickers_summary.append(summary)
                        
                        pd_funding_history.to_csv(raw_funding_rate_cache_filename, index=False)
                        pd_bucketed_funding_history.to_csv(bucketed_funding_rate_cache_filename, index=False)
                        
                        log(f"[{loop_counter}] {ticker} #rows: {pd_funding_history.shape[0]} written to {raw_funding_rate_cache_filename}")
                        log(f"bucketed summary written to {bucketed_funding_rate_cache_filename}")
                        log(f"{pformat(summary, indent=2, width=100)}")
                    else:
                        log(f"{ticker} not in markets, skipping.")

                pd_summary = pd.DataFrame(tickers_summary)
                pd_summary.sort_values(
                    by=['score'],
                    ascending=[False],
                    inplace=True
                )
                pd_summary.reset_index(drop=True, inplace=True)
                ticker_summary_filename = param['tickers_summary'].replace("$EXCHANGE_NAME$", param['exchange_name'])
                pd_summary.to_csv(ticker_summary_filename)
                log(f"tickers summary written to {ticker_summary_filename}")

            except Exception as loop_err:
                err_msg = f"Error: {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}"
                log(err_msg, log_level=LogLevel.ERROR)
                dispatch_notification(title=f"{param['current_filename']} {param['exchange_name']} error.", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)
                
            finally:
                time.sleep(int(param['loop_freq_ms']/1000))

                loop_counter += 1

asyncio.run(
    main()
)