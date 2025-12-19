from typing import Dict, Any

import ccxt
import ccxt.pro as ccxtpro

'''
Why override load_markets?
    Deribit is one of the OG CEX. If you look at 'contractSize' for BTC/USDC:USDC, a linear perp, for example:
        exchange.markets['BTC/USDC:USDC']['contractSize'] 0.0001
    However, for Deribit, create_order actually expects 'amount' in base ccy, not in "# of contracts" as with most other exchanges supported by CCXT.
    Also note that 'filled' in response from create_order also in base ccy, not in # contracts.
    The general prevailing convention in CCXT is: 'amount' should be quoted in '# contracts'.
    Why CCXT not fix Deribit, so that it follows the prevailing convention? This is because this would be a breaking changes.
    Thus, we override 'contractSize' to 1 for all markets.

Additionally, we need to override 'fetch_position' as it swapped 'notional' with 'contracts'!!! Real ugly. Example below.
    'id' = None
    'symbol' = 'BTC/USDC:USDC'
    'timestamp' = None
    'datetime' = None
    'lastUpdateTimestamp' = None
    'initialMargin' = ???
    'initialMarginPercentage' = ???
    'maintenanceMargin' = ???
    'maintenanceMarginPercentage' = ???
    'entryPrice' = 85657.0
    'notional' = 0.0009	<-- This is NOT USD! And this is NOT # Contracts! This is # BTC!
    'leverage' = 50
    'unrealizedPnl' = ???
    'realizedPnl' = ???
    'contracts' = 77.081445	<-- This is NOT "# contracts"! 0.0009 BTC x markPrice 85646.05
    'contractSize' = 1.0
    'marginRatio' = None
    'liquidationPrice' = None
    'markPrice' = 85646.05  <-- They use 'markPrice' to calc 'contracts'
    'lastPrice' = None
    'collateral' = None
    'marginMode' = None
    'side' = 'long'
    'percentage' = None
    'hedged' = None
    'stopLossPrice' = None
    'takeProfitPrice' = None
'''
class Deribit(ccxt.deribit):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore

    def load_markets(self, reload=False, params={}):
        self.markets = super().load_markets(reload=reload, params=params)

        for market in self.markets:
            self.markets[market]['contractSize'] = 1

        return self.markets
    
    def fetch_position(self, symbol: str, params={}): # type: ignore
        position = super().fetch_position(symbol=symbol, params=params)
        pos_usdt = position['contracts']
        pos_baseccy = position['notional']
        position['contracts'] = pos_baseccy
        position['notional'] = pos_usdt
        return position

class DeribitAsync(ccxtpro.deribit):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore

    async def load_markets(self, reload=False, params={}):
        self.markets = await super().load_markets(reload=reload, params=params)

        for market in self.markets:
            self.markets[market]['contractSize'] = 1
            
        return self.markets

    async def fetch_position(self, symbol: str, params={}): # type: ignore
        position = await super().fetch_position(symbol=symbol, params=params)
        pos_usdt = position['contracts']
        pos_baseccy = position['notional']
        position['contracts'] = pos_baseccy
        position['notional'] = pos_usdt
        return position