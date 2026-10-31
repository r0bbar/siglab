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
    pattern: str
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

            ticker: str = key.split(':')[1]
            exchange_name: str = key.split(':')[2]
            timestamp_ms: int = int(key.split(':')[3])

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

    for key in buffer:
        grouped: Dict[tuple, list] = {}
        for item in buffer[key]:
            yyyyMMdd = datetime.fromtimestamp(
                item['timestamp_ms'] / 1000, tz=timezone.utc
            ).strftime('%Y%m%d')
            grouped.setdefault((item['ticker'], item['exchange_name'], yyyyMMdd), []).append(item)

        for (ticker, exchange_name, yyyyMMdd), items in grouped.items():
            archive_filename: str = f"orderbooks_{ticker}_{exchange_name}_{yyyyMMdd}.json"
            print(archive_filename)

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

            orderbooks.setdefault(ticker, {})[f"{exchange_name}_{yyyyMMdd}"] = archive_filename

if __name__ == "__main__":
    redis_client : StrictRedis = init_redis_client()

    archive_hist_ob(redis_client, pattern=param['mds']['topics']['source_orderbooks_topic'])