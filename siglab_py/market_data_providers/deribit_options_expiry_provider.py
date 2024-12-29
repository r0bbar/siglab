import os
import sys
import traceback
from enum import Enum
import argparse
from datetime import datetime, timedelta
import time
from typing import Dict, Union
import json
import asyncio
import logging
from ccxt import deribit
from redis import StrictRedis
import pandas as pd

from util.market_data_util import fetch_deribit_btc_option_expiries, fetch_ohlcv_one_candle

param : Dict = {
    'market' : 'BTC',

    # Provider ID is part of mds publish topic. 
    'provider_id' : 'b0f1b878-c281-43d7-870a-0347f90e6ece',

    'archive_file_name' : "deribit_options_expiry.csv",

    # Publish to message bus
    'mds' : {
        'topics' : {
            'deribit_options_expiry_publish_topic' : 'deribit-options-expiry'
        },
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
# fh = logging.FileHandler(f"{param['job_name']}.log")
# fh.setLevel(log_level)
# fh.setFormatter(formatter)     
# logger.addHandler(fh)

class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

def log(message : str, log_level : LogLevel = LogLevel.INFO):
    if log_level.value<LogLevel.WARNING.value:
        logger.info(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.WARNING.value:
        logger.warning(f"{datetime.now()} {message}")

    elif log_level.value==LogLevel.ERROR.value:
        logger.error(f"{datetime.now()} {message}")

def parse_args():
    parser = argparse.ArgumentParser() # type: ignore

    parser.add_argument("--provider_id", help="candle_provider will go to work if from redis a matching topic partition_assign_topic with provider_id in it.", default=None)
    parser.add_argument("--market", help="Default BTC", default='BTC')
    parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: 1000*60*60 (i.e. 1hr)",default=1000*60*60)

    args = parser.parse_args()
    if args.provider_id:
        param['provider_id'] = args.provider_id
    param['market'] = args.market
    param['redis_ttl_ms'] = int(args.redis_ttl_ms)


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


def _fetch_historical_daily_candle_height(
        exchange,
        normalized_symbol : str,
        timestamp_ms : int,
        offset_days : int,
        candle_height : float,
        reload_candle_height : bool = False
    ):
    if not candle_height or reload_candle_height:
        dt = datetime.fromtimestamp(int(timestamp_ms/1000)) + timedelta(days=offset_days)
        dt = datetime(dt.year, dt.month, dt.day)
        timestamp_ms = int(dt.timestamp()) * 1000
        if dt < datetime(datetime.today().year, datetime.today().month, datetime.today().day):
            historical_day_candle = fetch_ohlcv_one_candle(exchange=exchange, normalized_symbol=normalized_symbol, timestamp_ms=timestamp_ms, ref_timeframe='1d')
            if historical_day_candle:
                return historical_day_candle['close'] - historical_day_candle['open']
            else:
                return None
        else:
            return None
    else:
        return None

async def main():
    parse_args()
    
    param['job_name'] = f'candles_provider_{param["provider_id"]}'
    redis_client : StrictRedis = init_redis_client()

    exchange = deribit()

    i = 0
    while True:
        try:
            pd_old_expiry_data = None
            max_datetime_from_old_expiry_data = None
            if os.path.isfile(param['archive_file_name']):
                pd_old_expiry_data = pd.read_csv(param['archive_file_name'])
                pd_old_expiry_data['datetime'] = pd.to_datetime(pd_old_expiry_data['datetime'])
                pd_old_expiry_data['datetime'] = pd_old_expiry_data['datetime'].dt.tz_localize(None)
                max_datetime_from_old_expiry_data = pd_old_expiry_data['datetime'].max()

            start = time.time()
            expiry_data = fetch_deribit_btc_option_expiries(market = "BTC")
            elapsed_sec = int((time.time() - start))
            log(f"#{i} Took {elapsed_sec} sec to fetch option expiry data from Deribit")

            publish_key = param['mds']['topics']['deribit_options_expiry_publish_topic']
            expiry_sec = int(int(param['mds']['redis']['ttl_ms'])/1000)
            redis_client.set(name=publish_key, value=json.dumps(expiry_data).encode('utf-8'), ex=expiry_sec)

            pd_new_expiry_data = pd.DataFrame([ { 'datetime' : x[0], 'notional_usd' : x[1] } for x in expiry_data ])

            pd_new_expiry_data['symbol'] = f"{param['market']}/USDT"
            pd_new_expiry_data['datetime'] = pd.to_datetime(pd_new_expiry_data['datetime'])
            pd_new_expiry_data['datetime'] = pd_new_expiry_data['datetime'].dt.tz_localize(None)
            pd_new_expiry_data['timestamp_sec'] = pd_new_expiry_data['datetime'].apply(lambda dt: int(dt.timestamp()))
            pd_new_expiry_data['timestamp_ms'] = pd_new_expiry_data['timestamp_sec'] * 1000
            
            # This is to make it easy to do grouping with Excel pivot table
            pd_new_expiry_data['year'] = pd_new_expiry_data['datetime'].dt.year
            pd_new_expiry_data['month'] = pd_new_expiry_data['datetime'].dt.month
            pd_new_expiry_data['day'] = pd_new_expiry_data['datetime'].dt.day
            pd_new_expiry_data['hour'] = pd_new_expiry_data['datetime'].dt.hour
            pd_new_expiry_data['minute'] = pd_new_expiry_data['datetime'].dt.minute
            pd_new_expiry_data['dayofweek'] = pd_new_expiry_data['datetime'].dt.dayofweek  # dayofweek: Monday is 0 and Sunday is 6

            if pd_old_expiry_data is not None:
                pd_new_expiry_data = pd_new_expiry_data[
                    pd_new_expiry_data['datetime'] > max_datetime_from_old_expiry_data
                ]

                pd_merged_expiry_data = pd.concat([pd_old_expiry_data, pd_new_expiry_data], axis=0, ignore_index=True)
            else:
                pd_merged_expiry_data = pd_new_expiry_data

            if not 'daily_candle_height_tm0' in pd_merged_expiry_data.columns:
                pd_merged_expiry_data['daily_candle_height_tm0'] = None
                pd_merged_expiry_data['daily_candle_height_tm1'] = None
                pd_merged_expiry_data['daily_candle_height_tm2'] = None
                pd_merged_expiry_data['daily_candle_height_tm3'] = None

            # candle_height = close - open (Can be positive, can be negative)
            pd_merged_expiry_data['daily_candle_height_tm0'] = pd_merged_expiry_data.apply(lambda rw : _fetch_historical_daily_candle_height(exchange, rw['symbol'], rw['timestamp_ms'], 0, rw['daily_candle_height_tm0']), axis=1) # type: ignore
            pd_merged_expiry_data['daily_candle_height_tm1'] = pd_merged_expiry_data.apply(lambda rw : _fetch_historical_daily_candle_height(exchange, rw['symbol'], rw['timestamp_ms'], -1, rw['daily_candle_height_tm1']), axis=1) # type: ignore
            pd_merged_expiry_data['daily_candle_height_tm2'] = pd_merged_expiry_data.apply(lambda rw : _fetch_historical_daily_candle_height(exchange, rw['symbol'], rw['timestamp_ms'], -2, rw['daily_candle_height_tm2']), axis=1) # type: ignore
            pd_merged_expiry_data['daily_candle_height_tm3'] = pd_merged_expiry_data.apply(lambda rw : _fetch_historical_daily_candle_height(exchange, rw['symbol'], rw['timestamp_ms'], -3, rw['daily_candle_height_tm3']), axis=1) # type: ignore

            pd_merged_expiry_data.to_csv(param['archive_file_name'])
            
        except Exception as loop_err:
            log(f"Loop error {loop_err} {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}", log_level=LogLevel.ERROR)
        finally:
            i += 1

async def _run_jobs():
    await main()
asyncio.get_event_loop().run_until_complete(_run_jobs())