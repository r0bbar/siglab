import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict
import logging
from redis import StrictRedis

param : Dict = {
    'mds' : {
        'topics' : {
            "source_orderbooks_topic" : "orderbooks:*",
        },
        'redis' : {
            'host' : 'localhost',
            'port' : 6379,
            'db' : 0,
            'ttl_ms' : 1000,
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

def archive_hist_ob(
    redis_client,
    archive_files_folder : str,
    pattern: str,
    loop_freq_ms : int = 1000 *60 *60 *24
):
    orderbooks: dict[str, dict] = {}
    buffer: Dict[str, list] = {}
    cursor = 0

    min_timestamp_ms = float('inf')
    max_timestamp_ms = float('-inf')

    i = 0
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match=pattern)
        for k in keys:
            i += 1
            key = k.decode('utf-8')

            logger.info(f"key: {key}")

            tokens = key.split(":")
            if len(tokens)==3:
                ticker: str = key.split(':')[1]
                exchange_name: str = key.split(':')[2]
                timestamp_ms: int = int(key.split(':')[-1])
            else:
                # strategy_executor also logs ob to redis, but format: f"orderbooks:entry|exit:{_ticker}:{exchange_name}:{ob['timestamp']}"
                ticker: str = key.split(':')[2]
                exchange_name: str = key.split(':')[3]
                timestamp_ms: int = int(key.split(':')[-1])

            if timestamp_ms<min_timestamp_ms:
                min_timestamp_ms = timestamp_ms
            if timestamp_ms>max_timestamp_ms:
                max_timestamp_ms = timestamp_ms

            if (ticker, exchange_name) not in buffer:
                buffer[(ticker, exchange_name)] = []

            raw = redis_client.get(k)
            ob = json.loads(raw)
            buffer[(ticker, exchange_name)].append({
                'ticker' : ticker,
                'exchange_name' : exchange_name,
                'timestamp_ms' : timestamp_ms,
                'key' : k,
                'ob' :  ob
            })

        if cursor == 0:
            break

    from_dt = datetime.fromtimestamp(int(min_timestamp_ms/1000))
    to_dt = datetime.fromtimestamp(int(max_timestamp_ms/1000))
    logger.info(f"{i} order books fetched from {from_dt} to {to_dt}")

    try:
        for key in buffer:
            grouped: Dict[tuple, list] = {}
            for item in buffer[key]:
                yyyyMM = datetime.fromtimestamp(
                    item['timestamp_ms'] / 1000, tz=timezone.utc
                ).strftime('%Y%m')
                grouped.setdefault((item['ticker'], item['exchange_name'], yyyyMM), []).append(item)

            for (ticker, exchange_name, yyyyMM), items in grouped.items():
                archive_filename: str = f"{archive_files_folder}\orderbooks_{ticker}_{exchange_name}_{yyyyMM}.json"

                existing = []
                if os.path.exists(archive_filename):
                    try:
                        with open(archive_filename, 'r') as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        existing = []

                merged = existing + [it['ob'] for it in items if it['ob'] is not None]
                with open(archive_filename, 'w') as f:
                    json.dump(merged, f)
                    logger.info(f"Orderbooks from redis dumped (appended) to {archive_filename}")

                orderbooks.setdefault(ticker, {})[f"{exchange_name}_{yyyyMM}"] = archive_filename
    except Exception as loop_err:
        logger.error(loop_err)
    finally:
        logger.info(f"Sleeping for {int(loop_freq_ms/1000)} seconds.")
        time.sleep(int(loop_freq_ms/1000))


if __name__ == "__main__":
    redis_client : StrictRedis = init_redis_client()

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder containing orderbooks JSON files.', default=".")
    parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=1000 *60 *60 *24)
    args = parser.parse_args()

    archive_files_folder : str = args.folder
    if archive_files_folder[-1]=="\\":
        archive_files_folder = archive_files_folder[:-1]

    loop_freq_ms : int = int(args.loop_freq_ms)

    archive_hist_ob(redis_client=redis_client, archive_files_folder=archive_files_folder, pattern=param['mds']['topics']['source_orderbooks_topic'], loop_freq_ms=loop_freq_ms)