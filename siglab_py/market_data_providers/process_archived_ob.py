import argparse
import time
import csv
import json
import re
from typing import Dict
from pathlib import Path
from pprint import pformat
import logging
from redis import StrictRedis

from siglab_py.util.orderbook_util import OrderBookFeatures, extract_ob_features

'''
Usage:
    python process_archived_ob.py .
        The "." means current folder, this is where orderbook json's are sourced from.

    python process_archived_ob.py . --order_notional 1000000 --price_level_separation_bps 50 --atr_bps 90 --focus_atr_multiple 5 --wall_threshold_multiplier 3

Debug from vscode, launch.json:

    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python Debugger: Current File",
                "type": "debugpy",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "justMyCode": false,
                "args": [
                    "${workspaceFolder}/market_data_providers"
                ],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}/.."
                }
            }
        ]
    }

'''
param : Dict = {
    'mds' : {
        'topics' : {
            "orderbook_features_topic" : f"orderbook-features:$TICKER$:$EXCHANGE_NAME$:$TIMESTAMP_MS$",
        },
        'redis' : {
            'host' : 'localhost',
            'port' : 6379,
            'db' : 0,
            'ttl_ms' : 1000,
        },
        'archive' : {
            'filename_pattern' : r"^orderbooks_(?P<ticker>[A-Za-z0-9]+)_(?P<exchange_name>[A-Za-z0-9]+)_(?P<date>\d{8})\.json$",
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

def _process_ob(
    ob,
    ticker,
    exchange_name,
    timestamp_ms,
    order_notional=1_000_000,
    price_level_separation_bps=50,
    atr_bps=90,
    focus_atr_multiple=5,
    wall_threshold_multiplier=None
):
    features = dict(
        extract_ob_features(
            ob=ob,
            order_notional=order_notional,
            price_level_separation_bps=price_level_separation_bps,
            atr_bps=atr_bps,
            focus_atr_multiple=focus_atr_multiple,
            wall_threshold_multiplier=wall_threshold_multiplier
        )
    )
    features['ticker'] = ticker
    features['exchange_name'] = exchange_name
    features['timestamp_ms'] = timestamp_ms
    return features


def _extract_timestamp_ms(ob, key):
    if ob.get('timestamp_ms') is not None:
        return int(ob['timestamp_ms'])
    if ob.get('timestamp') is not None:
        return int(ob['timestamp']) * 1000
    if key is not None:
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        return int(key.split(':')[3])
    return None


def parse_archive_json(
        path, 
        redis_client,
        **kwargs
    ):
    path = Path(path)
    with path.open('r') as f:
        orderbooks = json.load(f)

    stem_parts = path.stem.split('_')
    default_ticker = stem_parts[1] if len(stem_parts) > 1 else None
    default_exchange = stem_parts[2] if len(stem_parts) > 2 else None

    summaries = []
    for ob in orderbooks:
        ticker = ob.get('ticker', default_ticker)
        exchange_name = ob.get('exchange_name', default_exchange)
        timestamp_ms = _extract_timestamp_ms(ob, ob.get('key'))
        book = ob.get('ob', ob)
        features = _process_ob(
            book,
            ticker=ticker,
            exchange_name=exchange_name,
            timestamp_ms=timestamp_ms,
            **kwargs
        )
        summaries.append(features)
        
        key : str = param['mds']['topics']['orderbook_features_topic']
        key = key.replace("$TICKER$", ticker)
        key = key.replace("$EXCHANGE_NAME$", exchange_name)
        key = key.replace("$TIMESTAMP_MS$", str(timestamp_ms))
        redis_client.set(name=key, value=json.dumps(features).encode('utf-8'), ex=60)
        logger.info(f"Orderbook processed, features published to redis key: {key}")
    return summaries

def find_archive_files(folder) -> list:
    folder = Path(folder)
    pattern = re.compile(param['mds']['archive']['filename_pattern'])
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and pattern.match(p.name):
            files.append(str(p))
    return files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder containing orderbooks JSON files to process (from archive_redis_ob.py)')

    parser.add_argument('--order_notional', help="intended trading order notional. spread_bps_by_order_notional from extract_ob_features is the spread given the order_notional amount you specified.", type=float, default=1_000_000)
    parser.add_argument('--price_level_separation_bps', type=float, default=50)
    parser.add_argument('--atr_bps', help="atr_bps is used in combination with --focus_atr_multiple. You determine this from separate ATR analysis, for the trading session/time of day/day of week/month you intended to trade. focus_spread_bps from extract_ob_features is the spread given (atr_bps and focus_atr_multiple)", type=float, default=90)
    parser.add_argument('--focus_atr_multiple', type=float, default=5)
    parser.add_argument('--wall_threshold_multiplier', help="This is for identification of potential buy/sell walls on orderbook based on volume multiples at given price level. Set to like 3. Default None - i.e. Don't bother compute.", type=float, default=None)
    
    args = parser.parse_args()

    kwargs = dict(
        order_notional=args.order_notional,
        price_level_separation_bps=args.price_level_separation_bps,
        atr_bps=args.atr_bps,
        focus_atr_multiple=args.focus_atr_multiple,
        wall_threshold_multiplier=args.wall_threshold_multiplier
    )

    redis_client : StrictRedis = init_redis_client()

    all_rows = []
    for file in find_archive_files(args.folder):
        all_rows.extend(parse_archive_json(file, redis_client, **kwargs))
        logger.info(f"Processed {file}")