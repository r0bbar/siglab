from typing import Dict

import ccxt

'''
basic_exchange_connectivity_test will test following:
    1. load_markets
    2. fetch_balance
    3. create_order (send a small order)
    4. fetch_positions
    5. create_order (close out the small order sent in step 3)

Arguments:
    1. exchange
    2. normailized_ticker
        Example,
            'XAU-USDT-SWAP' = denormalized ticker (Exchange specific)
            'XAU/USDT:USDT' = normalized ticker (same across different exchanges)
    3. amount - test order amount in # contracts, not base ccy (Be aware of multiplier)
        Example,
            buy 0.001 XAU means 10 XAUT/USDT:USDT contract (0.0001 multipler)
    4. side - test order buy or sell?
    5. order_type - market (default) or limit? 
'''
def basic_exchange_connectivity_test(
    exchange,
    normalized_ticker : str,
    amount : float,
    side : str,

    order_type : str = 'market'
):
    markets = exchange.load_markets()

    balances = exchange.fetch_balance()
    print(balances)

    found = next(( market for market in markets if market == normalized_ticker), None)
    if found:
        market = markets[normalized_ticker]
        print(market)

        trailing_candles = await exchange.fetch_ohlcv(symbol=normalized_ticker, timeframe='1m', limit=10)
        price = trailing_candles[-1][4]

        entry_order = exchange.create_order(
            symbol = normalized_ticker,
            amount = amount,
            price = price,
            type=order_type,
            side=side
        )

        print(entry_order)

        positions = exchange.fetch_positions()
        for pos in positions:
            '''
            For longs, 'side' = 'long', 'contracts' = 100.0 (positive integer)
            For shorts, 'side' = 'short', 'contracts' = 100.0 (positive integer)
            '''
            print(pos)

            exit_order = exchange.create_order(
                symbol = normalized_ticker,
                amount = pos['contracts'],
                price = price,
                type=order_type,
                side='buy' if pos['side']=='short' else 'sell'
            )

            print(exit_order)


if __name__ == '__main__':
    apiKey : str = None
    secret : str = None 
    passphrase : str = None
    subaccount : str = None
    
    params : Dict = {
        'apiKey' : apiKey,
        'secret' : secret,
        'password' : passphrase,
        'subaccount' : subaccount,
        'rateLimit' : 100,                    # In ms
        'options' : {
            'defaultType': 'swap', # 'funding', 'spot', 'margin', 'future', 'swap', 'option'
        }
    }
    exchange = ccxt.binance(params)
    normalized_ticker = 'SOL/USDT:USDT'
    amount = 10
    side = 'sell'
    order_type = 'market'

    basic_exchange_connectivity_test(
        exchange = exchange,
        normalized_ticker = normalized_ticker,
        amount = amount,
        side = side,
        order_type = order_type
    )