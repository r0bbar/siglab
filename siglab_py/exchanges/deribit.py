from typing import Dict, Any

import ccxt
import ccxt.pro as ccxtpro

from exchanges.any_exchange import AnyExchange

'''
Why override load_markets?
    Deribit is one of the OG CEX. If you look at 'contractSize' for BTC/USDC:USDC, a linear perp, for example:
        exchange.markets['BTC/USDC:USDC']['contractSize'] 0.0001
    However, for Deribit, create_order actually expects 'amount' in base ccy, not in "# of contracts" as with most other exchanges supported by CCXT.
    Also note that 'filled' in response from create_order also in base ccy, not in # contracts.
    The general prevailing convention in CCXT is: 'amount' should be quoted in '# contracts'.
    Why CCXT not fix Deribit, so that it follows the prevailing convention? This is because this would be a breaking changes.
    Thus, we override 'contractSize' to 1 for all markets.
'''
class Deribit(ccxt.deribit):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore

    def load_markets(self, reload=False, params={}):
        self.markets = super().load_markets(reload=reload, params=params)

        for market in self.markets:
            self.markets[market]['contractSize'] = 1

        return self.markets

class DeribitAsync(ccxtpro.deribit):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore

    async def load_markets(self, reload=False, params={}):
        self.markets = await super().load_markets(reload=reload, params=params)

        for market in self.markets:
            self.markets[market]['contractSize'] = 1
            
        return self.markets