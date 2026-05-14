import asyncio
from pprint import pformat

from siglab_py.util.market_data_util import async_instantiate_exchange

'''
Pass apikey/secret to lighter constructor: https://github.com/ccxt/ccxt/wiki/FAQ#how-to-use-the-lighter-exchange-in-ccxt

            lighter = ccxt.lighter({
                'privateKey': '0xYOUR_API_PRIVATE_KEY_HERE',                        # It is not Ligher private key under menu \ Tools \ API keys (https://app.lighter.xyz/apikeys), it's your Ethereum Wallet private key!
                'options': {
                    'apiKeyIndex': 0,                                               # Integer (0–254) corresponding to the specific API key you created.
                    'accountIndex': 12345,                                          # https://mainnet.zklighter.elliot.ai/api/v1/accountsByL1Address?l1_address=0x1234567890abcdef...
                    'libraryPath': r'C:\lighter\lighter-signer-windows-amd64.dll'   # signer dll: https://github.com/elliottech/lighter-go/releases
                }
            })
'''
async def main():
    private_key : str = 'xxxxx' # This is your Ethereum Wallet private key. It is not Ligher private key under menu \ Tools \ API keys (https://app.lighter.xyz/apikeys).

    ticker : str = "SOL/USDC:USDC"
    amount_base_ccy = 0.1
    side = 'buy'
    order_type = 'market'

    rate_limit_ms = 100
    default_type : str = "linear"
    default_sub_type = None
    # Lighter very strict with market order, first create_order need specify price. Don't use mid price, very often your order will be canceled: "Order canceled due to excessive slippage beyond allowed limit"
    # If you specify a slippage too wide, create_order will still go thru with NO exception. But from Order History you will find the trade actually cancelled by Lighter. 
    # Too tight? Again, "Order canceled due to excessive slippage beyond allowed limit".
    default_max_slippage_bps : int = 30 # If you specify a slippage too wide, create_order will still go thru with NO exception. But from Order History you will find the trade actually cancelled by Lighter. Too tight? "Order canceled due to excessive slippage beyond allowed limit".
    verbose : bool = False

    exchange_specific_options: Union[Dict[str, Any], None] = {
        'apiKeyIndex': 0,
        'accountIndex': 123456,
        'libraryPath': r'D:\lighter\lighter-signer-windows-amd64.dll'
    }

    # instantiate exchange using siglab_py market_data_util.async_instantiate_exchange
    exchange : Union[AnyExchange, None] = await async_instantiate_exchange(
        gateway_id='lighter',
        api_key=private_key,
        secret="DUMMY_SECRET", # secret not actually passed to Lighter
        passphrase=None,
        default_type=default_type,
        default_sub_type=default_sub_type,
        rate_limit_ms=rate_limit_ms,
        default_max_slippage_bps=default_max_slippage_bps,
        exchange_specific_options=exchange_specific_options,
        verbose=verbose
    )

    balances = await exchange.fetch_balance()
    print(f"{pformat(balances, indent=2, width=100)}")

    '''
    [
        'info' = { ... }
        'id' = None
        'symbol' = 'SOL/USDC:USDC'
        'timestamp' = None
        'datetime' = None
        'isolated' = False
        'hedged' = None
        'side' = 'long'
        'contracts' = 0.001
        'contractSize' = 1.0
        'entryPrice' = 84.222
        'markPrice' = None
        'notional' = 0.08417
        'leverage' = 16.666666666666668
        'collateral' = 0.0
        'initialMargin' = None
        'maintenanceMargin' = None
        'initialMarginPercentage' = None
        'maintenanceMarginPercentage' = None
        'unrealizedPnl' = -5.2e-05
        'liquidationPrice' = 0.0
        'marginMode' = 'cross'
        'percentage' = None
    ]
    '''
    positions = await exchange.fetch_positions()
    print(f"{pformat(positions, indent=2, width=100)}")

    orderbook = await exchange.fetch_order_book(symbol=ticker, limit=5)
    asks = [ ask[0] for ask in orderbook['asks'] ]
    best_ask = min(asks)
    bids = [ bid[0] for bid in orderbook['bids'] ]
    best_bid = max(bids)
    price = best_ask if side=='buy' else best_bid

    executed_order = await exchange.create_order(
                            symbol = ticker,
                            type = order_type,
                            amount = amount_base_ccy,
                            price = price,
                            side = side
    )

    order_id = executed_order['clientOrderId']

    '''
        [
            {
                'info' = { ... }
                'id' = 'xxxxx'
                'timestamp' = xxxxx
                'datetime' = '2025-12-20T23:00:00.000Z'
                'symbol' = 'SOL/USDC:USDC'
                'order' = 'xxx'
                'type' = 'trade'
                'side' = 'buy'
                'takerOrMaker' = 'taker'
                'price' = 84.222
                'amount' = 0.001
                'cost' = 0.084222
                'fee' = {'cost': None, 'currency': None}
                'fees' = []
                'subAccountAddress' = None
    '''
    trades = await exchange.fetch_my_trades(
        symbol="SOL/USDC:USDC",
        limit=100,
        params={
            'subAccountAddress': None
        }
    )
    order_id = executed_order['clientOrderId']
    filtered = [t for t in trades if str(t.get('order')) == str(order_id)]
    print(f"{pformat(filtered, indent=2, width=100)}")

asyncio.run(main())