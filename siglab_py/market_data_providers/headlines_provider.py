import asyncio
import os
import logging
from typing import Dict, Optional
from datetime import datetime
import time
import arrow
from dateutil import parser
import argparse
import json
from enum import Enum
import feedparser
import pandas as pd
from pprint import pformat
from tabulate import tabulate
from redis import StrictRedis

from siglab_py.util.market_data_util import fetch_headlines_from_rss
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
fh = logging.FileHandler(f"rss_headlines_provider.log")
fh.setLevel(log_level)
fh.setFormatter(formatter)     
logger.addHandler(fh)

'''
RSS feeds to fetch Financial and Geopolitical headlines:
* impact stock prices, crypto, metal and commodities prices
* macro (For example FEDs rate cut decisions, dot plot, yield curves ..etc)
* geopolitics (For example Ukraine war, US/Israel - Iran conflict Strait of Hormuz, sanctions)
* crypto exchanges, dexes hacks, security breaches, vulnerabilities may impact some crypto prices
* tokens unlocks may impact some crypto prices

Events which may move asset prices.

Usage:
    set PYTHONPATH=%PYTHONPATH%;D:\dev\siglab\siglab_py
    python headlines_provider.py --urls_list_filename headlines_rss_source.txt --focus_keywords "war, oil, trump, israel, tehran, iran, kharg, military, strike, explode, explosion, negotiate, negotiation, sanctions, nuclear, uranium" --slack_info_url https://hooks.slack.com/services/xxx --slack_critial_url https://hooks.slack.com/services/xxx --slack_alert_url https://hooks.slack.com/services/xxx

    --slack_info_url/slack_critial_url/slack_alert_url are optional (If you want to receive slack notifications upon arrival of new headlines)

launch.json for Debugging from VSCode:
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "justMyCode": false,
                "args" : [
                        "--urls_list_filename", "headlines_rss_source.txt",
                        "--focus_keywords", "war, oil, trump, israel, tehran, iran, kharg, military, strike, explode, explosion, negotiate, negotiation, sanctions, nuclear, uranium"
                    ],
            }
        ]
    }

"headlines_rss_source.txt" takes this format:
    bloomberg|https://feeds.bloomberg.com/markets/news.rss
    bbc|http://feeds.bbci.co.uk/news/rss.xml
    cnn|http://rss.cnn.com/rss/edition.rss

"headlines_provider.py" put headlines from RSS in DataFrame, then publish to redis (optionally). Algo's or sentiment analysis can pick up from there and react to the head lines.
'''
rss_feeds = {}

param : Dict = {
    'headlines_cache_filename' : None,
    'loop_freq_ms' : 1000*60, 
    'current_filename' : current_filename,

    'notification' : {
        'footer' : None,

        # slack webhook url's for notifications
        'slack' : {
            'info' : { 'webhook_url' : None },
            'critical' : { 'webhook_url' : None },
            'alert' : { 'webhook_url' : None },
        }
    },
    
    'mds': {
        'topics': {
            'headlines': 'trading.headlines'
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
    parser.add_argument("--urls_list_filename", help="File containing list of RSS url's", default=None)
    parser.add_argument("--focus_keywords", help="Comma separated list of focused keywords", default=None)
    parser.add_argument("--headlines_cache_filename", help="Export headers to csv file? Export don't filter by focus_keywords, whole data set is dumped to csv.", default='rss_headlines.csv')

    parser.add_argument("--slack_info_url", help="Slack webhook url for INFO", default=None)
    parser.add_argument("--slack_critial_url", help="Slack webhook url for CRITICAL", default=None)
    parser.add_argument("--slack_alert_url", help="Slack webhook url for ALERT", default=None)

    args = parser.parse_args()
    
    urls_list_filename : str = f"{parent_dir}\\{args.urls_list_filename}"
    print(f"Trying to read RSS url list from: {urls_list_filename}")
    with open(urls_list_filename, 'r') as urls_list:
        for line in urls_list:
            source = line.rstrip('\n').split('|')[0]
            url = line.rstrip('\n').split('|')[1]
            rss_feeds[source] = url

    if args.focus_keywords:
        param['focus_keywords'] = [ keyword.strip().lower() for keyword in args.focus_keywords.split(',') ]

    headlines_cache_filename : str = f"{parent_dir}\\{args.headlines_cache_filename}"
    param['headlines_cache_filename'] = headlines_cache_filename
    print(f"headlines cache here: {headlines_cache_filename}")

    param['notification']['slack']['info']['webhook_url'] = args.slack_info_url
    param['notification']['slack']['critical']['webhook_url'] = args.slack_critial_url
    param['notification']['slack']['alert']['webhook_url'] = args.slack_alert_url

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

    headlines_data = []

    loop_counter : int = 0
    while True:
        try:
            start_ts_sec = time.time()

            pd_old_headlines = pd.DataFrame(columns=[ 'source', 'title', 'published_timestamp_ms', 'published_utc_dt', 'published_local_dt', 'created_timestamp_ms', 'url' ])
            if os.path.exists(param['headlines_cache_filename']):
                pd_old_headlines = pd.read_csv(param['headlines_cache_filename'], parse_dates=['published_utc_dt', 'published_local_dt'])
                pd_old_headlines.drop(pd_old_headlines.columns[pd_old_headlines.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
                pd_old_headlines['published_timestamp_ms'] = pd_old_headlines['published_timestamp_ms'].astype(float).astype('Int64')
                if not pd_old_headlines.empty:
                    headlines_data = pd_old_headlines.to_dict('records')
                    logger.info(f"Loaded {len(headlines_data)} existing headlines from cache")
            
            new_headlines = fetch_headlines_from_rss(rss_feeds, pd_old_headlines)
            for new_headline in new_headlines:
                try:
                    logger.info(f"new row appended: {pformat(new_headline, indent=2, width=100)}")
                    if loop_counter>0 and (any([ keyword for keyword in param['focus_keywords'] if keyword in new_headline['title'] ])): # Don't send flood of slacks in first iteration
                        dispatch_notification(title=f"{param['current_filename']}]", message=new_headline, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.CRITICAL, logger=logger)
                except Exception as notification_err:
                    pass # Just swallow
            
            num_new_headlines = len(new_headlines)
            num_old_headlines = len(headlines_data)
            headlines_data = headlines_data + new_headlines

            pd_headlines = pd.DataFrame(headlines_data)
            pd_headlines['published_timestamp_ms'] = pd_headlines['published_timestamp_ms'].fillna(0)
            pd_headlines = pd_headlines.sort_values(by=['published_timestamp_ms', 'created_timestamp_ms'], ascending=False)

            try:
                pd_headlines.to_csv(param['headlines_cache_filename'])
            except Exception as writecsv_err:
                logger.error(f"If you want to update {param['headlines_cache_filename']}, don't lock it.")

            filtered_headlines = pd_headlines[pd_headlines['title'].str.contains('|'.join(param['focus_keywords']), case=False, na=False)]

            if not filtered_headlines.empty:
                logger.info(f"{tabulate(filtered_headlines.loc[:, 'source':'created_timestamp_ms'], headers='keys', tablefmt='psql')}")

            if redis_client:
                try:
                    json_str = filtered_headlines.to_json(orient='records', date_format='iso', indent=2)

                    publish_topic = param['mds']['topics']['headlines']
                    redis_client.publish(publish_topic, json_str)
                    redis_client.setex(publish_topic, param['mds']['redis']['ttl_ms'] // 1000, json_str)
                    logger.info(f"Published filtered headlines to Redis topic {publish_topic}")

                except Exception as e:
                    logger.info(f"Failed to publish to Redis: {str(e)}")

            elapsed_ms = int((time.time() - start_ts_sec) *1000)
            logger.info(f"[loop# {loop_counter} (elapsed_ms: {elapsed_ms:,})] {pd_headlines.shape[0]} rows headlines exported to {param['headlines_cache_filename']}. # new_headlines: {num_new_headlines}, # old headlines: {num_old_headlines}")

        except Exception as loop_err:
            err_msg = f'loop error {loop_err}'
            logger.error(err_msg)
            dispatch_notification(title=f"{param['current_filename']} error. {_ticker}", message=err_msg, footer=param['notification']['footer'], params=notification_params, log_level=LogLevel.ERROR, logger=logger)
                
        finally:
            await asyncio.sleep(int(param['loop_freq_ms'] / 1000))

            loop_counter += 1

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")