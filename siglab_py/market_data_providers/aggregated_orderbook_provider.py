import sys
import traceback
import argparse
import time
from datetime import datetime
from typing import Dict, Union
import logging
import json
from pprint import pformat
from tabulate import tabulate
import asyncio
import pandas as pd
import numpy as np
from redis import StrictRedis

from ccxt.base.exchange import Exchange
from siglab_py.util.orderbook_util import OrderBookFeatures, extract_ob_features
from siglab_py.util.market_data_util import instantiate_exchange, candle_size_to_interval_sec, fetch_candles
from siglab_py.util.analytic_util import compute_candles_stats

VALIDATION_MAX_GAPS = 30

'''
From command prompt:
    python aggregated_orderbook_provider.py --ticker BTC --price_level_separation_bps 5 --order_notional 1000000 --focus_atr_multiple 5 --wall_threshold_multiplier 3 --ts_delta_observation_ms_threshold 100 --loop_freq_ms 10000 --redis_ttl_ms 60000

Every iteration, it will spits out a summary like one below:
    +----+------------+------------+------------+---------+------------+------------+------------+------------+-------------------+--------------------------------+-------------------+-----------------------------+-----------------------------+------------------------------------+------------------------------------+--------------------+--------------------+
    |    | source     |   num_bids |   num_asks |     mid |   best_bid |   best_ask |   bid_knee |   ask_knee |   best_spread_bps |   spread_bps_by_order_notional |   knee_spread_bps |   slippage_target_bid_price |   slippage_target_ask_price |   slippage_target_bid_volume_quote |   slippage_target_ask_volume_quote |   total_bid_volume |   total_ask_volume |
    |----+------------+------------+------------+---------+------------+------------+------------+------------+-------------------+--------------------------------+-------------------+-----------------------------+-----------------------------+------------------------------------+------------------------------------+--------------------+--------------------|
    |  0 | binance    |       1000 |       1000 | 80901.8 |    80901.7 |    80901.8 |    80820.8 |    80982.7 |           0.01236 |                       15.1253  |        20.032     |                     80857.7 |                     80980   |                          1,408,904 |                          3,797,043 |            516.783 |            631.819 |
    |  1 | kraken        |        400 |        400 | 80907.1 |    80907   |    80907.1 |    80866.5 |    80947.6 |           0.01236 |                        0.01236 |        10.0289    |                     80907   |                     80907.1 |                      1,275,158,215 |                      1,727,918,922 |          16076.9   |          21679.5   |
    |  2 | AGGREGATED |       1097 |       1045 | 80904.4 |    80907   |    80901.8 |    80904.3 |    80904.4 |          -0.64271 |                        0.59331 |         0.0123603 |                     80902.3 |                     80907.1 |                      1,211,669,924 |                      1,739,855,837 |          16593.7   |          22311.3   |
    +----+------------+------------+------------+---------+------------+------------+------------+------------+-------------------+--------------------------------+-------------------+-----------------------------+-----------------------------+------------------------------------+------------------------------------+--------------------+--------------------+

It will also publish to redis individual order book from supported_exchanges under key:
    ob_key : str = f"orderbooks:{ticker}:{ob['source']}:{ob['orderbook']['timestamp']}"

Aggregated orderbook is also published under key:
    ob_key : str = f"orderbooks:{ticker}:aggregated:{int(time.time()*1000)}"

Key config:
    ticker: For crypto or FX, this is base ccy. For example, BTC/USDT:USDT is perpetual contract. Base ccy is BTC.

    price_level_separation_bps: Prices are binned and aggregated.
    order_notional: Position notional in USD. extract_ob_features will calculate spread based on size you want to trade.
    focus_atr_multiple: Focus range is defined in ATR multiples. Related config: candles_size and how_many_candles
    wall_threshold_multiplier: This is threshold to detect bid/ask wall on orderbook based on avg volume at particular price level. Set to like 3. Default None - i.e. Don't bother compute
    ts_delta_observation_ms_threshold: If orderbook age, in ms, breaches this threshold, the orderbook is deemed invalid and will be discarded.
    
    candles_size: We fetch trailing candles for things like ATR. Default: 1h
    how_many_candles: Sliding window size for TA calculations. Default: 24

    loop_freq_ms: Loop frequency in ms. 
        900,000: orderbook snapshot per 15m
        3,600,000: orderbook snapshot per hr 
    redis_ttl_ms: TTL, in ms, of published data in redis.

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
                                "--ticker", "BTC",
                                "--price_level_separation_bps", "5",
                                "--order_notional", "1000000",
                                "--focus_atr_multiple", "5",
                                "--wall_threshold_multiplier", "3",
                                "--ts_delta_observation_ms_threshold", "100",
                                "--loop_freq_ms", "10000",
                                "--redis_ttl_ms", "60000",
                ],
            }
        ]
    }

    Orderbooks REST API from exchanges
    Binance https://binance-docs.github.io/apidocs/spot/en/#order-book
    OKX https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-order-book
        'sz':  Order book depth per side. Maximum 400, e.g. 400 bids + 400 asks
    Bybit https://bybit-exchange.github.io/docs/v5/market/orderbook
    Coinbase https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductbook
    Kraken https://docs.kraken.com/api/docs/rest-api/get-order-book/
'''
param : Dict = {
    'ticker' : 'BTC',
    'depth' : 500,

    'price_level_separation_bps' : 20,
    'order_notional' : 1_000_000,
    'focus_atr_multiple' : 5,
    'wall_threshold_multiplier' : None,
    
    'candles_size' : '1h',
    'how_many_candles' : 24,

    'loop_freq_ms' : 1000*30, # Loop frequency
    'redis_ttl_ms' : 1000*60*15,
    
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
    'ts_delta_observation_ms_threshold' : 100
}

depth : int = param['depth']

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

binance_linear_exchange = instantiate_exchange(
    exchange_name="binance",
    default_type="linear",
    exchange_specific_options={
        'fetch_ob_params' : {
            'limit' : depth if depth <=5000 else 5000,
        },
        'quote_ccy' : 'USDT',
    }
)

bybit_linear_exchange = instantiate_exchange(
    exchange_name="bybit",
    default_type="linear",
    exchange_specific_options={
        'fetch_ob_params' : {
            'limit' : depth if depth <= 200 else 200,
        },
        'quote_ccy' : 'USDT',
    }
)

okx_linear_exchange = instantiate_exchange(
    exchange_name="okx",
    default_type="linear",
    exchange_specific_options={
        'fetch_ob_params' : {
            'sz' : depth if depth <= 400 else 400,
        },
        'quote_ccy' : 'USDT'
    }
)

deribit_linear_exchange = instantiate_exchange(
    exchange_name="deribit",
    default_sub_type="linear",
    exchange_specific_options={
        'fetch_ob_params' : {
            'limit' : depth if depth <= 200 else 200,
        },
        'quote_ccy' : 'USDT'
    }
)

supported_exchanges = [
    binance_linear_exchange,
    bybit_linear_exchange,
    okx_linear_exchange,
    deribit_linear_exchange
]

async def _fetch_orderbook(ticker : str, exchange : Exchange):
    try:
        normalized_symbol : str = f"{ticker}/{exchange.options['quote_ccy']}" if exchange.options['defaultType']=='spot' else f"{ticker}/{exchange.options['quote_ccy']}:{exchange.options['quote_ccy']}"
        ob = exchange.fetch_order_book(symbol=normalized_symbol, params=exchange.options['fetch_ob_params'])
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

        logger.info(f"{exchange.name} mid: {mid}, min_bid_price: {min_bid_price}, max_bid_price: {max_bid_price}, min_ask_price: {min_ask_price}, max_ask_price: {max_ask_price}, range: {int(max_ask_price-min_bid_price)}, ts_delta_observation_ms: {ts_delta_observation_ms}, #bids: {len(ob['bids'])}, #asks: {len(ob['asks'])}")

        return {
                    'source' : exchange.name,
                    'orderbook' : ob,
                    'is_valid' : is_valid,
                    'ts_delta_observation_ms' : ts_delta_observation_ms
                }
    except Exception as fetch_err:
        logger.error(f"_fetch_orderbook failed for {exchange.name}: {fetch_err}")
        return {
            'source' : exchange.name,
            'is_valid' : False
        }
 

async def main():
    def parse_args():
        parser = argparse.ArgumentParser() # type: ignore

        parser.add_argument("--ticker", help="For crypto or FX, this is base ccy. For example, BTC/USDT:USDT is perpetual contract. Base ccy is BTC.", default="BTC")
        parser.add_argument("--market_type", help="Spot or linear.", default="linear")

        parser.add_argument("--price_level_separation_bps", help="Prices are rounded and binned before aggregation.", default=20)
        parser.add_argument("--order_notional", help="Prices are rounded and binned before aggregation.", default=1_000_000)
        parser.add_argument("--focus_atr_multiple", help="Focus range is defined in ATR multiples. Related config: candles_size and how_many_candles", default=None)
        parser.add_argument("--wall_threshold_multiplier", help="This is threshold to detect bid/ask wall on orderbook based on avg volume at particular price level. Set to like 3. Default None - i.e. Don't bother compute", default=None)
        parser.add_argument("--ts_delta_observation_ms_threshold", help="If orderbook age, in ms, breaches this threshold, the orderbook is deemed invalid and will be discarded.", default=100)

        parser.add_argument("--candle_size", help="We fetch trailing candles for things like ATR", default="1h")
        parser.add_argument("--how_many_candles", help="Sliding window size for TA calculations", default=24)

        parser.add_argument("--loop_freq_ms", help="Loop delays. Reduce this if you want to trade faster.", default=5000)
        parser.add_argument("--redis_ttl_ms", help="TTL for items published to redis. Default: assuming candle_size 1h, then redis_ttl_ms defaults to 24*7 hours.", default=None)

        args = parser.parse_args()
        param['ticker'] = args.ticker
        
        param['price_level_separation_bps'] = int(args.price_level_separation_bps)
        param['order_notional'] = float(args.order_notional)
        param['focus_atr_multiple'] = float(args.focus_atr_multiple)
        param['wall_threshold_multiplier'] = float(args.wall_threshold_multiplier) if args.wall_threshold_multiplier else None
        param['ts_delta_observation_ms_threshold'] = int(args.ts_delta_observation_ms_threshold)

        param['candle_size'] = args.candle_size
        param['how_many_candles'] = int(args.how_many_candles)
        param['candle_interval_sec'] = candle_size_to_interval_sec(param['candle_size'])

        param['loop_freq_ms'] = int(args.loop_freq_ms)
        
        param['redis_ttl_ms'] = int(args.redis_ttl_ms) if args.redis_ttl_ms else int(param['candle_interval_sec']*1000)*24*7
        
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

    primary_exchange = supported_exchanges[0]

    ticker : str = param['ticker']
    param['job_name'] = f'ccxt_agg_ob_provider_{ticker.replace(":","_").replace("/","-")}'

    price_level_separation_bps : int = param['price_level_separation_bps']
    order_notional : float = param['order_notional']
    focus_atr_multiple : float = param['focus_atr_multiple']
    wall_threshold_multiplier : Union[float, None] = param['wall_threshold_multiplier']

    candle_size : str = param['candle_size']
    how_many_candles : int = param['how_many_candles']
    candle_interval_sec : int = param['candle_interval_sec']

    i = 0
    total_loop_elapsed_ms : int = 0
    while True:
        loop_start = time.time()

        try:
            # Fetch trailing candles for basic TA like ATR
            normalized_symbol : str = f"{ticker}/{primary_exchange.options['quote_ccy']}" if primary_exchange.options['defaultType']=='spot' else f"{ticker}/{primary_exchange.options['quote_ccy']}:{primary_exchange.options['quote_ccy']}"
            pd_trailing_candles = fetch_candles(
                                    start_ts=(loop_start -candle_interval_sec*how_many_candles), 
                                    end_ts=loop_start, 
                                    exchange=primary_exchange, normalized_symbols=[normalized_symbol], 
                                    candle_size = candle_size, 
                                    num_candles_limit = 100,
                                    validation_max_gaps = VALIDATION_MAX_GAPS, # some newly listed FX pairs gaps
                                    logger = None
                                )[normalized_symbol]
            compute_candles_stats(
                        pd_candles=pd_trailing_candles,
                        boillenger_std_multiples=2, # We don't need BB
                        sliding_window_how_many_candles=how_many_candles,
                        validation_max_gaps = VALIDATION_MAX_GAPS, # some newly listed FX pairs gaps
                        pypy_compat=True # We don't need slope calc
                    )
            last_candle = pd_trailing_candles.iloc[-1]
            atr_bps : float = last_candle['atr_bps']

            orderbooks = await asyncio.gather(
                *[_fetch_orderbook(ticker=ticker, exchange=exchange) for exchange in supported_exchanges]
            )
            
            valid_orderbooks = [ ob for ob in orderbooks if ob['is_valid'] ]
            invalid_orderbooks = [ ob for ob in orderbooks if not ob['is_valid'] ]
            invalid_orderbooks_names = " ".join([ ob['source'] for ob in invalid_orderbooks ] )

            elapsed_ms = (time.time() - loop_start) * 1000
            logger.info(f"orderbooks fetch elapsed (ms): {elapsed_ms}, # orderbooks: {len(valid_orderbooks)}. Invalid books: {invalid_orderbooks_names}")
            logger.info(f"atr_bps: {atr_bps} from {candle_size} bars, over sliding window size of {how_many_candles} intervals")

            def _extract_summary(
                source : str,
                features : Dict):
                return {
                    'source' : source,
                    'num_bids' : features['num_bids'],
                    'num_asks' : features['num_asks'],
                    'mid' : features['mid'],
                    'best_bid' : features['best_bid'],
                    'best_ask' : features['best_ask'],
                    'bid_knee' : features['bid_knee'][0],
                    'ask_knee' : features['ask_knee'][0],
                    'best_spread_bps' : features['best_spread_bps'],
                    'spread_bps_by_order_notional' : features['spread_bps_by_order_notional'],
                    'knee_spread_bps' : features['knee_spread_bps'],
                    'slippage_target_bid_price' : features['slippage_target_bid'][0] if features['slippage_target_bid'] else None,
                    'slippage_target_ask_price' : features['slippage_target_ask'][0] if features['slippage_target_ask'] else None,
                    'slippage_target_bid_volume_quote' : f"{features['slippage_target_bid'][1] * features['mid']:,.0f}" if features['slippage_target_bid'] else None, # volume in USD
                    'slippage_target_ask_volume_quote' : f"{features['slippage_target_ask'][1] * features['mid']:,.0f}" if features['slippage_target_ask'] else None,
                    'total_bid_volume' : features['total_bid_volume'],
                    'total_ask_volume' : features['total_ask_volume'],
                }
                
            features = []
            aggregated_orderbooks = {
                'bids' : {},
                'asks' : {}
            }

            for ob in valid_orderbooks:
                ob_key : str = f"orderbooks:{ticker}:{ob['source']}:{ob['orderbook']['timestamp']}"
                redis_client.set(name=ob_key, value=json.dumps(ob['orderbook']).encode('utf-8'), ex=int(param['mds']['redis']['ttl_ms']/1000))
                features.append(
                    _extract_summary(
                        source=ob['source'],
                        features=extract_ob_features(
                                ob=ob['orderbook'],
                                order_notional=order_notional,
                                price_level_separation_bps=price_level_separation_bps,
                                atr_bps=atr_bps,
                                focus_atr_multiple=focus_atr_multiple, 
                                wall_threshold_multiplier=wall_threshold_multiplier, 
                            )
                    )
                )

                bids = ob['orderbook']['bids']
                asks = ob['orderbook']['asks']

                for bid in bids:
                    price = bid[0]
                    amount = bid[1]

                    existing_amount = 0
                    if price in aggregated_orderbooks['bids']:
                        existing_amount = aggregated_orderbooks['bids'][price]['amount']
                    amount_in_base_ccy = existing_amount + amount
                    aggregated_orderbooks['bids'][price] = {
                        'price' : price,
                        'amount' : amount_in_base_ccy
                    }

                for ask in asks:
                    price = ask[0]
                    amount = ask[1]
                
                    existing_amount = 0
                    if price in aggregated_orderbooks['asks']:
                        existing_amount = aggregated_orderbooks['asks'][price]['amount']
                    amount_in_base_ccy = existing_amount + amount
                    aggregated_orderbooks['asks'][price] = {
                        'price' : price,
                        'amount' : amount_in_base_ccy
                    }

            # First element closest to mid
            aggregated_orderbooks = {
                'bids' : [ (aggregated_orderbooks['bids'][x]['price'], aggregated_orderbooks['bids'][x]['amount']) for x in aggregated_orderbooks['bids'] ],
                'asks' : [ (aggregated_orderbooks['asks'][x]['price'], aggregated_orderbooks['asks'][x]['amount']) for x in aggregated_orderbooks['asks'] ]
            }

            features.append(
                        _extract_summary(
                            source="AGGREGATED",
                            features=extract_ob_features(
                                    ob=aggregated_orderbooks,
                                    order_notional=order_notional,
                                    price_level_separation_bps=price_level_separation_bps,
                                    atr_bps=atr_bps,
                                    focus_atr_multiple=focus_atr_multiple, 
                                    wall_threshold_multiplier=wall_threshold_multiplier, 
                                )
                        )
                    )

            ob_key : str = f"orderbooks:{ticker}:aggregated:{int(time.time()*1000)}"
            redis_client.set(name=ob_key, value=json.dumps(aggregated_orderbooks).encode('utf-8'), ex=int(param['mds']['redis']['ttl_ms']/1000))

            pd_orderbook_features = pd.DataFrame(features)
            logger.info(f"{tabulate(pd_orderbook_features, headers='keys', tablefmt='psql')}")
            
        except Exception as loop_err:
            logger.error(f"#{i} Error: {loop_err}. {str(sys.exc_info()[0])} {str(sys.exc_info()[1])} {traceback.format_exc()}")

        finally:
            this_loop_elapsed_ms : int = int((time.time()-loop_start)*1000)
            total_loop_elapsed_ms += this_loop_elapsed_ms
            avg_loop_elapsed_ms : int = int(total_loop_elapsed_ms / (i+1))
            logger.info(f"#{i} this_loop_elapsed_ms: {this_loop_elapsed_ms}, avg_loop_elapsed_ms: {avg_loop_elapsed_ms}")

            time.sleep(int(param['loop_freq_ms']/1000))
        
        i += 1


asyncio.run(main())