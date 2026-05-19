from typing import Dict, Any

from ccxt.base.types import Str
import ccxt
import ccxt.pro as ccxtpro

'''
Why subclass Lighter? They don't support fetch_order
'''
class Lighter(ccxt.lighter):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore
    
    def fetch_orders(self, symbol: str, params={}): # type: ignore
        open_orders = super().fetch_open_orders(symbol)
        closed_orders = super().fetch_closed_orders(symbol)
        orders = open_orders + closed_orders
        return orders
    
    def fetch_order(self, id: str, symbol: Str = None, params={}):
        orders = self.fetch_orders(symbol)
        orders = [ order for order in orders if order['clientOrderId']==id ]
        return orders[-1] if orders else None

class LighterAsync(ccxtpro.lighter):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore

    async def load_markets(self, reload=False, params={}):
        self.markets = await super().load_markets(reload=reload, params=params)

        for market in self.markets:
            self.markets[market]['contractSize'] = 1
            
        return self.markets
    
    async def fetch_orders(self, symbol: str, params={}): # type: ignore
        '''
        Format:
            'id' = '1121111026571112'
            'clientOrderId' =
            '896750192'
            'timestamp' = 1771111174000
            'datetime' = '2026-01-1T23:00:00.000Z'
            'lastTradeTimestamp' = None
            'lastUpdateTimestamp' = 1771111174000
            'symbol' = 'SOL/USDC:USDC'
            'type' = 'limit'
            'timeInForce' = 'GTC'
            'postOnly' = False
            'reduceOnly' = False
            'side' = 'buy'
            'price' = 86.305
            'triggerPrice' = None
            'stopLossPrice' = None
            'takeProfitPrice' = None
            'amount' = 0.2
            'cost' = 0.0
            'average' = None
            'filled' = 0.0
            'remaining' = 0.2
            'status' = 'open'
            'fee' = None
            'trades' = []
            'fees' = []
            'stopPrice' = None
        '''
        open_orders = await super().fetch_open_orders(symbol)

        '''
        Format:
            'id' = '1121111026579011'
            'clientOrderId' = '11111111'
            'timestamp' = 11118973111000
            'datetime' = '2026-01-16T23:00:00.000Z'
            'lastTradeTimestamp' = None
            'lastUpdateTimestamp' = 1111173618000
            'symbol' = 'SOL/USDC:USDC'
            'type' = 'market'
            'timeInForce' = 'IOC'
            'postOnly' = False
            'reduceOnly' = False
            'side' = 'buy'
            'price' = 87.489
            'triggerPrice' = None
            'stopLossPrice' = None
            'takeProfitPrice' = None
            'amount' = 0.1
            'cost' = 8.6623
            'average' = 86.623
            'filled' = 0.1
            'remaining' = 0.0
            'status' = 'closed'
            'fee' = None
            'trades' = []
            'fees' = []
            'stopPrice' = None
        '''
        closed_orders = await super().fetch_closed_orders(symbol)
        orders = open_orders + closed_orders
        return orders
    
    async def fetch_order(self, id: str, symbol: Str = None, params={}):
        orders = await self.fetch_orders(symbol)
        orders = [ order for order in orders if order['clientOrderId']==id ]
        return orders[-1] if orders else None