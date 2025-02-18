**siglab_py** allows engineers/traders to quickly setup a trading desk: From back tests to trading.

![alt text](https://github.com/r0bbar/siglab/blob/master/siglab_py/siglab_py.jpg)

It consists of two primary components.

1. Under [**market_data_providers**](https://github.com/r0bbar/siglab/tree/master/siglab_py/market_data_providers)

    [**orderbooks_provider.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/orderbooks_provider.py): fetches orderbooks from exchanges. Orderbooks are published to redis under topic 'orderbooks_$SYMBOL$_$EXCHANGE$'

    [**candles_provider.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/candles_provider.py): fetches candles from exchanges (currrent implementation only crypto exchanges supported). Candles are published to redis under topic 'candles-$DENORMALIZED_SYMBOL$-$EXCHANGE_NAME$-$INTERVAL$'. Please look at **market_data_util.py**.

    [**candles_ta_provider.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/market_data_providers/candles_ta_provider.py): calculate technical indicators from candles_provider (Fetched from redis). TA calculations are published back to redis for strategy consumption. Please look at analytic_util.py

    TAs computed (This is an expanding list):

        a. Basic SMA/EMAs (And slopes)

        b. ATR

        c. Boillenger bands (Yes incorrect spelling sorry)

        d. FVG
        
        e. Hurst Exponent

        f. RSI, MFI

        g. MACD

        h. Fibonacci
        
        i. Inflections points: where 'close' crosses EMA from above or below.


    [market_data_util](https://github.com/r0bbar/siglab/blob/master/siglab_py/util/market_data_util.py) fetches the candles.

    The code which computes the TA is in [analytic_util](https://github.com/r0bbar/siglab/blob/master/siglab_py/util/analytic_util.py).

    Two examples shows usage of market_data_util and analytic_util in back tests.

        Examples 1: Fibonacci
            https://medium.com/@norman-lm-fung/debunking-myth-fibonacci-618-ea957c795d5a

        Example 2: Trading breakouts
            https://medium.com/@norman-lm-fung/debunking-myths-trading-breakouts-f73db8006f44

            **partition_sliding_window** segments time series based on inflection points: Where price crosses MAs.
            https://medium.com/@norman-lm-fung/time-series-slicer-and-price-pattern-extractions-81f9dd1108fd


2. Under [**ordergateway**](https://github.com/r0bbar/siglab/tree/master/siglab_py/ordergateway)

    [**gateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py): This is a standalone order gateway. Current implementation supports a couple crypto exchanges. But if you look at [any_exchange.py](https://github.com/r0bbar/siglab/blob/master/siglab_py/exchanges/any_exchange.py), the ultimate goal is to support trading via tradfi brokerages like IBKR. To trade exchanges not supported by ccxt or tradfi brokerages of your choice, extend AnyExchange.
    
    The idea is, strategies (separate service that you'd build), see send orders to [**gateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py) via redis, using API exposed in [**client.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/client.py). Take a look at [**test_ordergateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/test_ordergateway.py) for an example what you need to implement strategy side to send orders, and wait for fills.

    From strategy code, you can also access ordergateway.client.DivisiblePosition by first installing **siglib_py**
    
    ```
    pip install siglab-py
    ```

    [siglab-py](https://pypi.org/project/siglab-py)

    [**gateway.py**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/gateway.py) has logic to ...
    
    a. Execute orders in slices

    b. Round order price and amount to exchange precision 
       (ccxt price_to_precision and amount_to_precision)

    c. Hung limit orders will be cancelled. Remainder are sent as market orders.

    The spirit of the implementation is to have a very very simple standalone order gateway, which is separate from strategy implementation. Strategies implementation should only have entry/exit logic. Strategy concerns, and Execution concerns should be separate.

    Take look at [**DivisiblePosition**](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/client.py)
       
    A note on position slicing ... When strategies want to enter into position(s), you don't send "Orders". From [client side](https://github.com/r0bbar/siglab/blob/master/siglab_py/ordergateway/test_ordergateway.py) you should actually be sending a list of DivisiblePosition ("slice" is a property of "DivisiblePosition"). While "positions" are executed in parallel (Think of delta neutral spread trades? You'd like the two legs to be executed concurrently.), "slices" are executed sequentially. And also, amount on last slice need be >= min amount specified by exchanges, otherwise that last slice will be skipped.

    Further, a note on entry vs exit:
    
    a. When you Enter into a position: You are thinking in USD. You'd want to deploy $xxx to buy BTC for example.

    b. When you Exit from a position: You'd unwind the BTC you have, back into USD.

    That's a strategy concern, and gateway.py don't handle that for you.