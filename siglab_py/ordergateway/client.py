
from typing import List, Dict, Any

from ccxt.base.types import Position

from constants import JSON_SERIALIZABLE_TYPES
from exchanges.any_exchange import AnyExchange

'''
Example,
    spot        'BTC/USDT'
    perpetual   'BTC/USDT:USDT'
In both cases, BTC is base ccy, and USDT is quote ccy.

crypto/CCXT convention is, 
    For spot, in base ccy. 
    For perpetual contracts, this is in # contracts, not base ccy.
Here, Order.amount is always in base ccy regardless whether you're trading spot or perpetual. 

leg_room_bps: 
    For limit orders, when order is executed, limit price be ...
        buy order: best ask * (1 + leg_room_bps/10000) 
        sell order: best bid * (1 - leg_room_bps/10000)
    Thus a positive leg room means you're more aggressive to trying to get the order filled: Buy at higher price, Sell at lower price. 
'''
class Order:
    def __init__(
        self,
        ticker : str,
        side : str,  # buy/sell
        amount : float,
        order_type : str, # market/limit
        leg_room_bps : float = 0
    ) -> None:
        self.ticker = ticker
        self.side = side.strip().lower()
        self.amount = amount
        self.order_type = order_type.strip().lower()
        self.leg_room_bps = leg_room_bps

    def to_dict(self) -> Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES]:
        return {
            "ticker" : self.ticker,
            "side" : self.side,
            "amount" : self.amount,
            "order_type" : self.order_type,
            "leg_room_bps" : self.leg_room_bps
        }

'''
For limit orders, if not filled within wait_fill_threshold_ms, we'd try cancel and resend un-filled amount as market order.
wait_fill_threshold_ms default to -1: Means wait forever until fully filled.
'''
class DivisiblePosition(Order):
    def __init__(
        self,
        ticker : str,
        side : str,  # buy/sell
        amount : float,
        order_type : str, # market/limit
        leg_room_bps : float,
        slices : int = 1,
        wait_fill_threshold_ms : float = -1
    ) -> None:
        super().__init__(ticker, side, amount, order_type, leg_room_bps)
        self.slices = slices
        self.wait_fill_threshold_ms = wait_fill_threshold_ms

        self.executions : Dict[str, Dict[str, Any]] = {}

    def to_slices(self) -> List[Order]:
        slices : List[Order] = []

        remaining_amount_in_base_ccy : float = self.amount
        slice_amount_in_base_ccy : float = self.amount / self.slices
        for i in range(self.slices):
            if remaining_amount_in_base_ccy>0:
                if remaining_amount_in_base_ccy >= slice_amount_in_base_ccy:
                    slice = Order(
                        ticker=self.ticker, 
                        side=self.side, 
                        amount=slice_amount_in_base_ccy, 
                        leg_room_bps=self.leg_room_bps, 
                        order_type=self.order_type)
                    
                else:
                    # Last slice
                    slice = Order(
                        ticker=self.ticker, 
                        side=self.side, 
                        amount=remaining_amount_in_base_ccy, 
                        leg_room_bps=self.leg_room_bps, 
                        order_type=self.order_type)

                slices.append(slice)
                    
        return slices

    def append_execution(
        self,
        order_id : str,
        execution : Dict[str, Any]
    ):
        self.executions[order_id] = execution

    def get_execution(
        self,
        order_id : str
    ) -> Dict[str, Any]:
        return self.executions[order_id]

    def get_filled_amount(self):
        # filled_amount is in base ccy
        filled_amount = sum([ self.executions[order_id]['filled'] * self.executions[order_id]['multipler'] for order_id in self.executions ])
        return filled_amount

    def to_dict(self) -> Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES]:
        rv = super().to_dict()
        rv['slices'] = self.slices
        rv['wait_fill_threshold_ms'] = self.wait_fill_threshold_ms
        rv['executions'] = self.executions
        return rv
