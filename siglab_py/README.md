# [**siglab_py**](https://pypi.org/project/siglab-py) 
... allows engineers/traders to quickly setup a trading desk: From back tests to trading.

![alt text](https://github.com/r0bbar/siglab/blob/master/siglab_py/siglab_py.jpg)

It consists of two primary components.

## 1. Under [**market_data_providers**](https://github.com/r0bbar/siglab/tree/master/siglab_py/market_data_providers)

[**orderbooks_provider.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/orderbooks_provider.py): fetches orderbooks from exchanges. Orderbooks are published to redis under topic 'orderbooks_$SYMBOL$_$EXCHANGE$'

[**candles_provider.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/candles_provider.py): fetches candles from exchanges (currrent implementation only crypto exchanges supported). Candles are published to redis under topic 'candles-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$'. Please look at **market_data_util.py**.

[**candles_ta_provider.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/candles_ta_provider.py): calculate technical indicators from candles_provider (Fetched from redis). TA calculations are published back to redis for strategy consumption. Please look at analytic_util.py

TAs computed (This is an expanding list):

+ Basic SMA/EMAs (And slopes)

+ ATR

+ Boillenger bands (Yes incorrect spelling sorry)

+ FVG

+ Hurst Exponent

+ RSI, MFI

+ MACD

+ Fibonacci

+ Inflections points: where 'close' crosses EMA from above or below.


[market_data_util](https://github.com/r0bbar/siglab/blob/master/siglab_py/util/market_data_util.py) **fetch_candles** contains implementation to grab candles from exchanges/market data providers (Yahoo Finance for example) - With sliding window implementation, as all exchanges restrict how many candles you can get in a single fetch. **Example on Usage?** [market_data_util_tests.py](https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/integration/market_data_util_tests.py)

```
from siglab_py.util.market_data_util import fetch_candles

start_date : datetime = datetime(2024,1,1)
end_date : datetime = datetime(2024,12,31)

param = {
    'apiKey' : None,
    'secret' : None,
    'password' : None,
    'subaccount' : None,
    'rateLimit' : 100,    # In ms
    'options' : {
        'defaultType': 'swap'
    }
}

exchange : Exchange = okx(param)
normalized_symbols = [ 'BTC/USDT:USDT' ]
pd_candles: Union[pd.DataFrame, None] = fetch_candles(
    start_ts=start_date.timestamp(),
    end_ts=end_date.timestamp(),
    exchange=exchange,
    normalized_symbols=normalized_symbols,
    candle_size='1h'
)[normalized_symbols[0]]
```

The code which computes technical indicators is in [analytic_util.compute_candles_stats](https://github.com/r0bbar/siglab/blob/master/siglab_py/util/analytic_util.py). **Example on Usage?** [analytic_util_tests.py](https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/unit/analytic_util_tests.py)

```
from siglab_py.util.analytic_util import compute_candles_stats

pd_candles : pd.DataFrame = pd.read_csv(csv_path)
compute_candles_stats(
    pd_candles=pd_candles,
    boillenger_std_multiples=2,
    sliding_window_how_many_candles=20,
    pypy_compat=True
)

expected_columns : List[str] = ['exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'datetime', 'datetime_utc', 'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'pct_chg_on_close', 'candle_height', 'is_green', 'pct_change_close', 'sma_short_periods', 'sma_long_periods', 'ema_short_periods', 'ema_long_periods', 'ema_close', 'std', 'ema_volume_short_periods', 'ema_volume_long_periods', 'max_short_periods', 'max_long_periods', 'idmax_short_periods', 'idmax_long_periods', 'min_short_periods', 'min_long_periods', 'idmin_short_periods', 'idmin_long_periods', 'h_l', 'h_pc', 'l_pc', 'tr', 'atr', 'hurst_exp', 'boillenger_upper', 'boillenger_lower', 'boillenger_channel_height', 'boillenger_upper_agg', 'boillenger_lower_agg', 'boillenger_channel_height_agg', 'aggressive_up', 'aggressive_up_index', 'aggressive_up_candle_height', 'aggressive_up_candle_high', 'aggressive_up_candle_low', 'aggressive_down', 'aggressive_down_index', 'aggressive_down_candle_height', 'aggressive_down_candle_high', 'aggressive_down_candle_low', 'fvg_low', 'fvg_high', 'fvg_gap', 'fvg_mitigated', 'close_delta', 'close_delta_percent', 'up', 'down', 'rsi', 'ema_rsi', 'typical_price', 'money_flow', 'money_flow_positive', 'money_flow_negative', 'positive_flow_sum', 'negative_flow_sum', 'money_flow_ratio', 'mfi', 'macd', 'signal', 'macd_minus_signal', 'fib_618_short_periods', 'fib_618_long_periods', 'gap_close_vs_ema', 'close_above_or_below_ema', 'close_vs_ema_inflection']
assert(pd_candles.columns.to_list()==expected_columns)
```

Two examples shows usage of market_data_util and analytic_util in back tests.

### Examples 1: Fibonacci
    https://medium.com/@norman-lm-fung/debunking-myth-fibonacci-618-ea957c795d5a

### Example 2: Trading breakouts
    https://medium.com/@norman-lm-fung/debunking-myths-trading-breakouts-f73db8006f44

**partition_sliding_window** segments time series based on inflection points: Where price crosses MAs.
    https://medium.com/@norman-lm-fung/time-series-slicer-and-price-pattern-extractions-81f9dd1108fd


## 2. Under [**ordergateway**](https://github.com/r0bbar/siglab/tree/master/siglab_py/ordergateway)

[**gateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py): This is a standalone order gateway. Current implementation supports a couple crypto exchanges. But if you look at [any_exchange.py](https://github.com/r0bbar/siglab/blob/master/siglab_py/exchanges/any_exchange.py), the ultimate goal is to support trading via tradfi brokerages like IBKR. To trade exchanges not supported by ccxt or tradfi brokerages of your choice, extend AnyExchange.

The idea is, strategies (separate service that you'd build), see send orders to [**gateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py) via redis, using **DivisiblePosition** and **execute_positions** exposed in [**client.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/client.py).

The simplest example [**test_ordergateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/test_ordergateway.py) on what you need to implement strategy side to send orders, and wait for fills is:

```
from typing import List, Union, Dict
from redis import StrictRedis
from siglab_py.ordergateway.client import DivisiblePosition, execute_positions

redis_client : StrictRedis = StrictRedis(
                    host = 'localhost',
                    port = 6379,
                    db = 0,
                    ssl = False
                )

positions : List[DivisiblePosition] = [
        DivisiblePosition(
            ticker = 'SUSHI/USDT:USDT',
            side = 'sell',
            amount = 10,
            leg_room_bps = 5,
            order_type = 'limit',
            slices=5,
            wait_fill_threshold_ms=15000
        ),
        DivisiblePosition(
            ticker = 'DYDX/USDT:USDT',
            side = 'buy',
            amount = 10,
            leg_room_bps = 5,
            order_type = 'limit',
            slices=5,
            wait_fill_threshold_ms=15000
        )
    ]

gateway_id : str = "hyperliquid_01" 
ordergateway_pending_orders_topic = f"ordergateway_pending_orders_$GATEWAY_ID$"
ordergateway_pending_orders_topic = ordergateway_pending_orders_topic.replace("$GATEWAY_ID$", gateway_id)

ordergateway_executions_topic = "ordergateway_executions_$GATEWAY_ID$"
ordergateway_executions_topic = ordergateway_executions_topic.replace("$GATEWAY_ID$", gateway_id)

executed_positions : Union[Dict, None] = execute_positions(
        redis_client=redis_client,
        positions=positions,
        ordergateway_pending_orders_topic=ordergateway_pending_orders_topic,
        ordergateway_executions_topic=ordergateway_executions_topic
        )
if executed_positions:
    for position in executed_positions:
        print(f"{position['ticker']} {position['side']} amount: {position['amount']} leg_room_bps: {position['leg_room_bps']} slices: {position['slices']}, filled_amount: {position['filled_amount']}, average_cost: {position['average_cost']}, # executions: {len(position['executions'])}") # type: ignore
```

(Remember: set PYTHONPATH=C:\dev\siglab\siglab_py)

From strategy code, you can also access ordergateway.client.**DivisiblePosition** and **execute_positions** by first installing [**siglib_py**](https://pypi.org/project/siglab-py)

```
pip install siglab-py
```

[**gateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py) has logic to ...

+ Execute orders in slices

+ Round order price and amount to exchange precision 
    (ccxt price_to_precision and amount_to_precision)

+ If you're trading contracts, [gateway.py](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py) will handle contract multipliers for you.

+ Hung limit orders will be cancelled. Remainder are sent as market orders.

The spirit of the implementation is to have a very very simple standalone order gateway, which is separate from strategy implementation. Strategies implementation should only have entry/exit logic. Strategy concerns, and Execution concerns should be separate.
    
A note on position slicing ... When strategies want to enter into position(s), you don't send "Orders". From [client side](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/test_ordergateway.py) you should actually be sending a list of DivisiblePosition ("slice" is a property of "DivisiblePosition"). While "positions" are executed in parallel (Think of delta neutral spread trades? You'd like the two legs to be executed concurrently.), "slices" are executed sequentially. And also, amount on last slice need be >= min amount specified by exchanges, otherwise that last slice will be skipped.

Further, a note on entry vs exit:

+ When you Enter into a position: You are thinking in USD. You'd want to deploy $xxx to buy BTC for example. Traders generally think of "amount" in USD. 

+ When you Exit from a position: You'd unwind the BTC you have, back into USD. Traders generally think of amount in base currency. BTC for example.

**DivisiblePosition.amount** is always in base currency. Not USD. Not number of contracts.

That's a strategy concern, and gateway.py don't handle that for you.