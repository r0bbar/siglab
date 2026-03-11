import asyncio
import ccxt.pro as ccxtpro

from siglab_py.util.market_data_util import async_instantiate_exchange

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
async def basic_exchange_connectivity_test(
    exchange,
    normalized_ticker : str,
    amount : float,
    side : str,

    order_type : str = 'market'
):
    markets = await exchange.load_markets()

    balances = await exchange.fetch_balance()
    print(balances)

    found = next(( market for market in markets if market == normalized_ticker), None)
    if found:
        market = markets[normalized_ticker]
        print(market)

        trailing_candles = await exchange.fetch_ohlcv(symbol=normalized_ticker, timeframe='1m', limit=10)
        price = trailing_candles[-1][4]

        entry_order = await exchange.create_order(
            symbol = normalized_ticker,
            price = price,
            amount = amount,
            type=order_type,
            side=side
        )

        print(entry_order)

        positions = await exchange.fetch_positions()
        for pos in positions:
            '''
            For longs, 'side' = 'long', 'contracts' = 100.0 (positive integer)
            For shorts, 'side' = 'short', 'contracts' = 100.0 (positive integer)
            '''
            print(pos)

            exit_order = await exchange.create_order(
                symbol = normalized_ticker,
                price = price,
                amount = pos['contracts'],
                type=order_type,
                side='buy' if pos['side']=='short' else 'sell'
            )

            print(exit_order)

async def main():
    api_key : str = None
    secret : str = None
    passphrase : str = None
    default_type : str = "linear"
    default_sub_type : str = None
    rate_limit_ms : int = 100
    default_max_slippage_bps : int = 25
    verbose : bool = False

    exchange : Union[AnyExchange, None] = await async_instantiate_exchange(
        gateway_id='hyperliquid',
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        default_type=default_type,
        default_sub_type=default_sub_type,
        rate_limit_ms=rate_limit_ms,
        default_max_slippage_bps=default_max_slippage_bps,
        verbose=verbose
    )

    normalized_ticker = 'SOL/USDC:USDC'
    amount = 0.3
    side = 'sell'
    order_type = 'market'

    await basic_exchange_connectivity_test(
        exchange = exchange,
        normalized_ticker = normalized_ticker,
        amount = amount,
        side = side,
        order_type = order_type
    )

asyncio.run(main())