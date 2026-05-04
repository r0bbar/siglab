import asyncio
from pprint import pformat
import ccxt.pro as ccxtpro

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

    rate_limit_ms = 100
    exchange_params = {
        'privateKey': private_key,
        'options': {
            'apiKeyIndex': 0,
            'accountIndex': 687361,
            'libraryPath': r'D:\lighter\lighter-signer-windows-amd64.dll'
        }
    }
    exchange = ccxtpro.lighter(exchange_params) 
    balances = await exchange.fetch_balance()
    print(f"{pformat(balances, indent=2, width=100)}")

    '''
    
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