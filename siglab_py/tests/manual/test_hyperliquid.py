import asyncio
from pprint import pformat
import ccxt.pro as ccxtpro

async def main():
    rate_limit_ms = 100
    exchange_params = {
        "walletAddress" : "0x??????????", # Ledger Arbitrum Wallet Address here!
        "privateKey" : "0x??????????",",
        'enableRateLimit'  : True,
        'rateLimit' : rate_limit_ms
    }
    exchange = ccxtpro.hyperliquid(exchange_params) 
    balances = await exchange.fetch_balance()
    print(f"{pformat(balances, indent=2, width=100)}")

    '''
    [ 
    { 
        'amount': xxx,
        'cost': xxx,
        'datetime': '2025-01-01T00:00:00.000Z',
        'fee': {'cost': 0.01, 'currency': 'USDC'},
        'fees': [{'cost': 0.01, 'currency': 'USDC'}],
        'id': 'xxxxxxxxxxxxxxxx',
        'info': { 'closedPnl': '0.01',
                'coin': 'SOL',
                'crossed': True,
                'dir': 'Close Long',
                'fee': '0.01',
                'feeToken': 'USDC',
                'hash': '0x???',
                'oid': 'xxx',
                'px': '86',
                'side': 'A',
                'startPosition': 'xxx',
                'sz': 'xxx',
                'tid': 'xxxxxxxxxxxxxxx',
                'time': 'xxxxxxxxxxxxx',
                'twapId': None},
        'order': 'xxxxxxxxxxxxxxx',
        'price': 86,
        'side': 'sell',
        'symbol': 'SOL/USDC:USDC',
        'takerOrMaker': 'taker',
        'timestamp': xxxxxxxxxxxxx,
        'type': None}
    ]
    '''
    trades = await exchange.fetch_my_trades(
        symbol="SOL/USDC:USDC",
        limit=100,
        params={
            'subAccountAddress': None
        }
    )
    order_id = "xxxxxxxxxxxxxxx"
    filtered = [t for t in trades if str(t.get('order')) == str(order_id)]
    print(f"{pformat(filtered, indent=2, width=100)}")

asyncio.run(main())