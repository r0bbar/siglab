**siglab.py** allows engineers/traders to quickly setup a trading desk: From back tests to trading.
It consists of two primary components.
1. Under market_data_providers
    **candles_provider.py**: fetches candles from exchanges (currrent implementation only crypto exchanges supported). Candles are published to Redis. Please look at market_data_util.py.
    **candles_ta_provider.py**: calculate tecnical indicators from candles_provider (Fetched from redis). TA calculations are published back to redis for strategy consumption. Please look at analytic_util.py

    Two examples shows usage of market_data_util and analytic_util in back tests.
        Examples 1: Fibonacci
            https://medium.com/@norman-lm-fung/debunking-myth-fibonacci-618-ea957c795d5a
        Example 2: Trading breakouts
            https://medium.com/@norman-lm-fung/debunking-myths-trading-breakouts-f73db8006f44

2. Under ordergateway
    **gateway.py**: This is a standalone order gateway. Current implementation supports a couple crypto exchanges. But if you look at any_exchange.py, the ultimate goal is to support trading via tradfi brokerages like IKBR.
    The idea is, strategies (separate service that you'd build), see send orders to **gateway.py** via redis, using API exposed in **client.py**.
