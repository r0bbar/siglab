from typing import Dict, Any

import ccxt
from ccxt.base.types import Balances
import ccxt.pro as ccxtpro

'''
Why override fetch_balance?
    balances['total'] empty
But you can find that from exchange raw response under balances['info']['balances'] (verbose=True).
'''
def _populate_balance_total_if_missing(
        balances : Dict[str, Any]
    ):
    for ccy_balance in balances['info']['balances']:
        ccy = ccy_balance['asset']
        free = float(ccy_balance.get('free', 0))
        locked = float(ccy_balance.get('locked', 0))
        total = free + locked
        if total!=0 and ccy not in balances['total']:
            balances['total'][ccy] = total
class Binance(ccxt.binance):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore
    
    def fetch_balance(self, params={}) -> Balances: # type: ignore
        balances = super().fetch_balance(params=params)
        _populate_balance_total_if_missing(balances)
        return balances

class BinanceAsync(ccxtpro.binance):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args) # type: ignore

    async def fetch_balance(self, params={}) -> Balances: # type: ignore
        balances = await super().fetch_balance(params=params)
        _populate_balance_total_if_missing(balances)
        return balances