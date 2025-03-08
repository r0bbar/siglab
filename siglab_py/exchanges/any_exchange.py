from ccxt.base.exchange import Exchange

'''
The idea here is, if we were to trade thru for example IKBR, which is a tradfi broker not supported by CCXT, we would implement IBKR like CCXT implement for crypto exchanges.
Concrete implementation will:
a. Inherit from AnyExchange, thus ccxt Exchange class
b. Override ccxt basic functions
    - load_markets
    - fetch_position
    - fetch_balance
    - create_order
    - update_order
    - cancel_order
    - fetch_order (REST) vs watch_orders (websocket)
    - order amount rounding: amount_to_precision
    - order price rounding: price_to_precision
    ... etc

    Besides above, it's very common below are required from algo's
    - fetch_time()
    - fetch_order_book
    - fetch_positions
    
'''
class AnyExchange(Exchange):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
