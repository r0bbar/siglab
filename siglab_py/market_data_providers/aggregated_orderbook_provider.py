from enum import Enum
import argparse
import time
from datetime import datetime
from typing import Any, Dict, Union
import logging
import json
from tabulate import tabulate
import asyncio
import pandas as pd
import numpy as np
from redis import StrictRedis

'''
From command prompt:
    python aggregated_orderbook.py --normalized_symbol BTC/USDT:USDT --sliding_window_num_intervals 1200 --update_imabalce_csv_intervals 100 --dump_imbalance_to_disk Y --publish_imbalance_to_redis N
    or
    pypy aggregated_orderbook.py --normalized_symbol BTC/USDT:USDT --sliding_window_num_intervals 1200 --update_imabalce_csv_intervals 100 --dump_imbalance_to_disk Y --publish_imbalance_to_redis N

This script is pypy compatible.

Spot orderbooks REST API from exchanges
    Binance https://binance-docs.github.io/apidocs/spot/en/#order-book
    OKX https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-order-book
        'sz':  Order book depth per side. Maximum 400, e.g. 400 bids + 400 asks
    Bybit https://bybit-exchange.github.io/docs/v5/market/orderbook
    Coinbase https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductbook
    Kraken https://docs.kraken.com/api/docs/rest-api/get-order-book/

Key parameters you may want to modify:
    normalized_symbol: Which ticker you wish monitor
    sliding_window_num_intervals: We calc EMAs on pct_imbalance, bids_amount_usdt and asks_amount_usdt. Window size will impact this. It's quoted in # intervals.
                                  It's measured in # intervals (or # loops), default 1200.
    update_imabalce_csv_intervals:  We'd update, or publish, imbalance data only at multiples of update_imabalce_csv_intervals.
                                  It's measured in # intervals (or # loops), default 100.
    topic_imbalance_data: Imbalance data is published to redis. This is published topic.
                           Since redis has special treatment for ':' (it'd consider it a folder), we'd replace ':' with '|'. 
                           Example BTC/USDT:USDT will become BTC/USDT|USDT.
    redis_ttl_ms: Imbalance data is published to redis. TTL of published data.
    dump_imbalance_to_disk: Dump imbalance data to disk?
    publish_imbalance_to_redis: Publish imbalance data to redis?

Launch.json if you wish to debug from VSCode:
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
                                    "--normalized_symbol", "BTC/USDT:USDT",
                                    "--sliding_window_num_intervals", "100",
                                    "--update_imabalce_csv_intervals", "100",
                                    "--topic_imbalance_data", "imbalance_BTC/USDT:USDT",
                                    "--redis_ttl_ms", "3600000",

                                    "--dump_imbalance_to_disk", "Y",
                                    "--publish_imbalance_to_redis", "N"
                ]
            }
        ]
    }

'''
from ccxt.binance import binance
from ccxt.okx import okx
from ccxt.bybit import bybit
from ccxt.coinbase import coinbase
from ccxt.kraken import kraken
from ccxt.base.exchange import Exchange

param : Dict = {
    'normalized_symbol' : 'BTC/USDT:USDT',
    'market_type' : 'linear', # For spots, set to "spot". For perpectual, you need to look at ccxt doc, for most exchanges, it's 'linear' or 'swap' for perpetuals. Example, https://github.com/ccxt/ccxt/blob/master/python/ccxt/okx.py?plain=1#L1110
    'depth' : 1000,
    'price_level_increment' : 10,
    'sliding_window_num_intervals' : 1200, # For example if each iteration takes 2 sec. 90 intervals = 180 sec (i.e. three minutes)
    'update_imabalce_csv_intervals' : 100,
    'imbalance_output_file' : 'imbalance.csv',
    'dump_imbalance_to_disk' : True,
    'publish_imbalance_to_redis' : False,

    # Provider ID is part of mds publish topic. 
    'provider_id' : 1,


    # Publish to message bus
    'mds' : {
        'mds_topic' : 'ccxt_rest_ob_$PROVIDER_ID$', 
        'redis' : {
            'host' : 'localhost',
            'port' : 6379,
            'db' : 0,
            'ttl_ms' : 1000*60*15 # 15 min?
        }

    },
    
    # Keep track of latency issues: ts_delta_observation_ms: Keep track of server clock vs timestamp from exchange
    'ts_delta_observation_ms_threshold' : 150
}

depth : int = param['depth']
market_type : str = param['market_type'] 
price_level_increment : float = param['price_level_increment']

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


coinbase_param : Dict[str, int] = { }

binance_param : Dict[str, int] = {
    'limit' : depth if depth <=5000 else 5000
}

okx_param : Dict[str, int] = {
    'sz' : depth if depth <= 400 else 400
}


bybit_param : Dict[str, int] = {
    'limit' : depth if depth <= 200 else 200
}


kraken_param : Dict[str, int] = {
    'depth' : depth if depth <= 500 else 500
}


coinbase_exchange = coinbase({
    'defaultType' : market_type
})

binance_exchange = binance({
    'defaultType' : market_type
})

okx_exchange = okx({
    'defaultType' : market_type
})

bybit_exchange = bybit({
    'defaultType' : market_type
})

kraken_exchange = kraken({
    'defaultType' : market_type
})

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
    
async def _fetch_orderbook(symbol : str, exchange : Exchange, fetch_ob_params : Dict):
    try:
        ob = exchange.fetch_order_book(symbol=symbol, params=fetch_ob_params)
        is_valid = True
        ts_delta_observation_ms : int = 0
        if 'timestamp' in ob and ob['timestamp']:
            update_ts_ms = ob['timestamp']
            ts_delta_observation_ms = int(datetime.now().timestamp()*1000) - update_ts_ms
            is_valid = True if ts_delta_observation_ms<=param['ts_delta_observation_ms_threshold'] else False

        bid_prices = [ x[0] for x in ob['bids'] ]
        ask_prices = [ x[0] for x in ob['asks'] ]
        min_bid_price = min(bid_prices)
        max_bid_price = max(bid_prices)
        min_ask_price = min(ask_prices)
        max_ask_price = max(ask_prices)

        mid = (max([ x[0] for x in ob['bids'] ]) + min([ x[0] for x in ob['asks'] ])) / 2

        log(f"{exchange.name} mid: {mid}, min_bid_price: {min_bid_price}, max_bid_price: {max_bid_price}, min_ask_price: {min_ask_price}, max_ask_price: {max_ask_price}, range: {int(max_ask_price-min_bid_price)}, ts_delta_observation_ms: {ts_delta_observation_ms}")

        return {
                    'source' : exchange.name,
                    'orderbook' : ob,
                    'mid' : mid,
                    'min_bid_price' : min_bid_price,
                    'max_bid_price' : max_bid_price,
                    'min_ask_price' : min_ask_price,
                    'max_ask_price' : max_ask_price,
                    'is_valid' : is_valid,
                    'ts_delta_observation_ms' : ts_delta_observation_ms
                }
    except Exception as fetch_err:
        print(f"_fetch_orderbook failed for {exchange.name}: {fetch_err}")
        return {
            'source' : exchange.name,
            'is_valid' : False
        }
 

async def main():
    def parse_args():
        parser = argparse.ArgumentParser() # type: ignore

        parser.add_argument("--normalized_symbol", help="Example BTC/USDT for spot. BTC/USDT:USDT for perps.",default="BTC/USDT")
        parser.add_argument("--sliding_window_num_intervals", help="Sliding window is used for EMA's calculation. It's measured in # intervals (or # loops)",default=1200)
        parser.add_argument("--update_imabalce_csv_intervals", help="We'd update, or publish, imbalance data only at multiples of update_imabalce_csv_intervals. Again, it's measured in # intervals.",default=100)

        parser.add_argument("--topic_imbalance_data", help="Publish topic for imbalance data. Since redis has special treatment for ':' (it'd consider it a folder), we'd replace ':' with '|'. Example BTC/USDT:USDT will become BTC/USDT|USDT.",default=None)
        parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: 1000*60*60 (i.e. 1hr)",default=1000*60*60)

        parser.add_argument("--dump_imbalance_to_disk", help="Y or N (default).", default='Y')
        parser.add_argument("--publish_imbalance_to_redis", help="Y or N (default).", default='N')
        
        args = parser.parse_args()
        param['normalized_symbol'] = args.normalized_symbol
        param['sliding_window_num_intervals'] = int(args.sliding_window_num_intervals)
        param['update_imabalce_csv_intervals'] = int(args.update_imabalce_csv_intervals)
        param['mds']['mds_topic'] = args.topic_imbalance_data if args.topic_imbalance_data else f"imbalance_data_{param['normalized_symbol'].replace(':','|')}"
        param['redis_ttl_ms'] = int(args.redis_ttl_ms)

        if args.dump_imbalance_to_disk:
            if args.dump_imbalance_to_disk=='Y':
                param['dump_imbalance_to_disk'] = True
            else:
                param['dump_imbalance_to_disk'] = False
        else:
            param['dump_imbalance_to_disk'] = False

        if args.publish_imbalance_to_redis:
            if args.publish_imbalance_to_redis=='Y':
                param['publish_imbalance_to_redis'] = True
            else:
                param['publish_imbalance_to_redis'] = False
        else:
            param['publish_imbalance_to_redis'] = False

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
    redis_client = init_redis_client()

    normalized_symbol : str = param['normalized_symbol']
    param['job_name'] = f'ccxt_agg_ob_provider_{normalized_symbol.replace(":","_").replace("/","-")}'
    sliding_window_num_intervals : int = param['sliding_window_num_intervals']
    update_imabalce_csv_intervals : int = param['update_imabalce_csv_intervals']

    pd_imbalances = pd.DataFrame(columns=['timestamp_ms', 'datetime', 'mid', 'imbalance', 'bids_amount_usdt', 'asks_amount_usdt', 'total_amount', 'pct_imbalance', 'ema_pct_imbalance', 'ema_bids_amount_usdt', 'ema_asks_amount_usdt'])
    mid : Union[float, None] = None
    imbalance : Union[float, None] = None
    pct_imbalance : Union[float, None] = None
    sum_bids_amount_usdt : Union[float, None] = None
    sum_asks_amount_usdt : Union[float, None] = None
    total_amount : Union[float, None] = None
    last_ema_pct_imbalance : Union[float, None] = None
    last_ema_bids_amount_usdt : Union[float, None] = None
    last_ema_asks_amount_usdt : Union[float, None] = None

    i = 0
    total_loop_elapsed_ms : int = 0
    while True:
        loop_start = time.time()

        try:
            # Comment out Coinbase and Kraken spot only exchanges.
            orderbooks = await asyncio.gather(
                # _fetch_orderbook(symbol=normalized_symbol, exchange=coinbase_exchange, fetch_ob_params=coinbase_param),
                _fetch_orderbook(symbol=normalized_symbol, exchange=binance_exchange, fetch_ob_params=binance_param),
                _fetch_orderbook(symbol=normalized_symbol, exchange=bybit_exchange, fetch_ob_params=bybit_param),
                # _fetch_orderbook(symbol=normalized_symbol, exchange=kraken_exchange, fetch_ob_params=kraken_param),
                _fetch_orderbook(symbol=normalized_symbol, exchange=okx_exchange, fetch_ob_params=okx_param)
                )
            valid_orderbooks = [ ob for ob in orderbooks if ob['is_valid'] ]
            invalid_orderbooks = [ ob for ob in orderbooks if not ob['is_valid'] ]
            invalid_orderbooks_names = " ".join([ ob['source'] for ob in invalid_orderbooks ] )

            max_min_bid_price = max([ ob['min_bid_price'] for ob in valid_orderbooks if ob])
            best_bid_price = max([ob['max_bid_price'] for ob in valid_orderbooks if ob])
            min_max_ask_price = min([ob['max_ask_price'] for ob in valid_orderbooks if ob])
            best_ask_price = min([ob['min_ask_price'] for ob in valid_orderbooks if ob])

            elapsed_ms = (time.time() - loop_start) * 1000
            logger.info(f"orderbooks fetch elapsed (ms): {elapsed_ms}, # orderbooks: {len(valid_orderbooks)}, max_min_bid_price: {max_min_bid_price}, min_max_ask_price: {min_max_ask_price}, best_bid_price: {best_bid_price}, best_ask_price: {best_ask_price}. Invalid books: {invalid_orderbooks_names}")

            aggregated_orderbooks = {
                'bids' : {},
                'asks' : {}
            }

            def round_to_nearest(price, increment):
                return round(price / increment) * increment

            mid = [ x['mid'] for x in valid_orderbooks if x['source']=='Binance'][0] # use Binance as mid reference
            for orderbook in valid_orderbooks:
                bids = orderbook['orderbook']['bids']
                asks = orderbook['orderbook']['asks']

                for bid in bids:
                    price = round_to_nearest(bid[0], price_level_increment)
                    amount = bid[1]
                    if bid[0] > max_min_bid_price:
                        existing_amount = 0
                        if price in aggregated_orderbooks['bids']:
                            existing_amount = aggregated_orderbooks['bids'][price]['amount']
                        amount_in_base_ccy = existing_amount + amount
                        amount_in_usdt = amount_in_base_ccy * mid
                        aggregated_orderbooks['bids'][price] = {
                            'price' : price,
                            'amount' : amount_in_base_ccy,
                            'amount_usdt' : amount_in_usdt
                        }

                for ask in asks:
                    price = round_to_nearest(ask[0], price_level_increment)
                    amount = ask[1]
                    if ask[0] < min_max_ask_price:
                        existing_amount = 0
                        if price in aggregated_orderbooks['asks']:
                            existing_amount = aggregated_orderbooks['asks'][price]['amount']
                        amount_in_base_ccy = existing_amount + amount
                        amount_in_usdt = amount_in_base_ccy * mid
                        aggregated_orderbooks['asks'][price] = {
                            'price' : price,
                            'amount' : amount_in_base_ccy,
                            'amount_usdt' : amount_in_usdt
                        }
            
            sorted_asks = dict(sorted(aggregated_orderbooks['asks'].items(), key=lambda item: item[0], reverse=True))
            sorted_bids = dict(sorted(aggregated_orderbooks['bids'].items(), key=lambda item: item[0], reverse=True))

            pd_aggregated_orderbooks_asks = pd.DataFrame(sorted_asks)
            pd_aggregated_orderbooks_bids = pd.DataFrame(sorted_bids)

            pd_aggregated_orderbooks_asks = pd_aggregated_orderbooks_asks.transpose()
            pd_aggregated_orderbooks_bids = pd_aggregated_orderbooks_bids.transpose()

            sum_asks_amount_usdt = pd.to_numeric(pd_aggregated_orderbooks_asks['amount_usdt']).sum()
            sum_bids_amount_usdt = pd.to_numeric(pd_aggregated_orderbooks_bids['amount_usdt']).sum()

            pd_aggregated_orderbooks_asks['str_amount_usdt'] = pd_aggregated_orderbooks_asks['amount_usdt'].apply(lambda x: f'{x:,.2f}')
            pd_aggregated_orderbooks_bids['str_amount_usdt'] = pd_aggregated_orderbooks_bids['amount_usdt'].apply(lambda x: f'{x:,.2f}')

            ask_resistance_price_level = pd_aggregated_orderbooks_asks['amount_usdt'].idxmax()
            bid_support_price_level = pd_aggregated_orderbooks_bids['amount_usdt'].idxmax()

            pd_aggregated_orderbooks_asks['is_max_amount_usdt'] = pd_aggregated_orderbooks_asks.index == ask_resistance_price_level
            pd_aggregated_orderbooks_bids['is_max_amount_usdt'] = pd_aggregated_orderbooks_bids.index == bid_support_price_level

            pd_aggregated_orderbooks_asks_ = pd_aggregated_orderbooks_asks[['price', 'amount', 'str_amount_usdt', 'is_max_amount_usdt']]
            pd_aggregated_orderbooks_asks_.rename(columns={'str_amount_usdt': 'amount_usdt'}, inplace=True)
            pd_aggregated_orderbooks_bids_ = pd_aggregated_orderbooks_bids[['price', 'amount', 'str_amount_usdt', 'is_max_amount_usdt']]
            pd_aggregated_orderbooks_bids_.rename(columns={'str_amount_usdt': 'amount_usdt'}, inplace=True)

            spread_bps = (best_ask_price-best_bid_price) / mid * 10000
            spread_bps = round(spread_bps, 0)
            imbalance = sum_bids_amount_usdt - sum_asks_amount_usdt if sum_bids_amount_usdt and sum_asks_amount_usdt else None
            total_amount = sum_bids_amount_usdt + sum_asks_amount_usdt if sum_bids_amount_usdt and sum_asks_amount_usdt else None
            pct_imbalance = (imbalance/total_amount) * 100 if imbalance and total_amount else None

            log(f"mid: {mid}, imbalance (bids - asks): {imbalance:,.0f}, pct_imbalance: {pct_imbalance:,.2f}, last_ema_pct_imbalance: {last_ema_pct_imbalance if last_ema_pct_imbalance else '--'}, spread_bps between bests: {spread_bps} (If < 0, arb opportunity). Range {max_min_bid_price} - {min_max_ask_price} (${int(min_max_ask_price-max_min_bid_price)})")
            
            if last_ema_bids_amount_usdt and last_ema_asks_amount_usdt:
                log(f"ema_bids_amount_usdt: {last_ema_bids_amount_usdt:,.0f}, ema_asks_amount_usdt: {last_ema_asks_amount_usdt:,.0f}")

            log(f"asks USD {sum_asks_amount_usdt:,.0f}, best: {best_ask_price:,.2f}")
            log(f"{tabulate(pd_aggregated_orderbooks_asks_.reset_index(drop=True), headers='keys', tablefmt='psql', colalign=('right', 'right', 'right'), showindex=False)}") # type: ignore Otherwise error: tabulate Argument of type "DataFrame" cannot be assigned to parameter "tabular_data" of type "Mapping[str, Iterable[Any]] | Iterable[Iterable[Any]]" in function "tabulate"

            log(f"bids USD {sum_bids_amount_usdt:,.0f}, best: {best_bid_price:,.2f}")
            log(f"{tabulate(pd_aggregated_orderbooks_bids_.reset_index(drop=True), headers='keys', tablefmt='psql', colalign=('right', 'right', 'right'), showindex=False)}") # type: ignore Otherwise error: tabulate Argument of type "DataFrame" cannot be assigned to parameter "tabular_data" of type "Mapping[str, Iterable[Any]] | Iterable[Iterable[Any]]" in function "tabulate"

        except Exception as loop_err:
            log(f"#{i} Error: {loop_err}")

        finally:
            this_loop_elapsed_ms : int = int((time.time()-loop_start)*1000)
            total_loop_elapsed_ms += this_loop_elapsed_ms
            avg_loop_elapsed_ms : int = int(total_loop_elapsed_ms / (i+1))
            sliding_window_num_sec : int = int(sliding_window_num_intervals*avg_loop_elapsed_ms/1000)
            log(f"#{i} this_loop_elapsed_ms: {this_loop_elapsed_ms}, avg_loop_elapsed_ms: {avg_loop_elapsed_ms}, sliding_window_num_intervals: {sliding_window_num_intervals}, sliding_window_num_sec: {sliding_window_num_sec}")
        
        pct_imbalance = imbalance/total_amount * 100 if imbalance and total_amount else None
        pd_imbalances.loc[i] = [ int(loop_start*1000), datetime.fromtimestamp(loop_start), mid, imbalance, sum_bids_amount_usdt, sum_asks_amount_usdt, total_amount, pct_imbalance, np.nan, np.nan, np.nan ]
        
        if i%update_imabalce_csv_intervals==0:
            if pd_imbalances.shape[0]>sliding_window_num_intervals:
                pd_imbalances['ema_pct_imbalance'] = pd_imbalances['pct_imbalance'].ewm(span=sliding_window_num_intervals, adjust=False).mean()
                pd_imbalances['ema_bids_amount_usdt'] = pd_imbalances['bids_amount_usdt'].ewm(span=sliding_window_num_intervals, adjust=False).mean()
                pd_imbalances['ema_asks_amount_usdt'] = pd_imbalances['asks_amount_usdt'].ewm(span=sliding_window_num_intervals, adjust=False).mean()
                last_ema_pct_imbalance = pd_imbalances['ema_pct_imbalance'].iloc[-1]
                last_ema_bids_amount_usdt = pd_imbalances['ema_bids_amount_usdt'].iloc[-1]
                last_ema_asks_amount_usdt = pd_imbalances['ema_asks_amount_usdt'].iloc[-1]
            
                data : Dict[str, Union[str, float, int, None]] = {
                    'normalized_symbol' : normalized_symbol,
                    'timestamp_ms' : int(datetime.now().timestamp() * 1000),
                    
                    'ema_pct_imbalance' : last_ema_pct_imbalance,
                    'ema_bids_amount_usdt' : last_ema_bids_amount_usdt,
                    'ema_asks_amount_usdt' : last_ema_asks_amount_usdt,

                    'avg_loop_elapsed_ms' : avg_loop_elapsed_ms,
                    'sliding_window_num_intervals' : sliding_window_num_intervals,
                    'sliding_window_num_sec' : sliding_window_num_sec
                }
                redis_client.set(name=param['mds']['mds_topic'], value=json.dumps(data), ex=int(param['mds']['redis']['ttl_ms']/1000))
                
                if param['dump_imbalance_to_disk']:
                    pd_imbalances.to_csv(param['imbalance_output_file'])
            
        i += 1


asyncio.run(main())