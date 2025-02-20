
from typing import List, Dict, Any, Union
import json
import time
from redis import StrictRedis

from exchanges.any_exchange import AnyExchange
from constants import JSON_SERIALIZABLE_TYPES

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
        self.multiplier = 1
        self.filled_amount : Union[float, None] = None
        self.average_cost : Union[float, None] = None
        self.pos : Union[float, None] = None # in base ccy, after execution. (Not in USDT or quote ccy, Not in # contracts)

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

    def get_executions(self) -> Dict[str, Dict[str, Any]]:
        return self.executions

    def get_filled_amount(self) -> float:
        # filled_amount is in base ccy
        filled_amount = sum([ self.executions[order_id]['filled'] * self.multiplier for order_id in self.executions ])
        return filled_amount

    def get_average_cost(self) -> float:
        average_cost = sum([ self.executions[order_id]['average'] * self.executions[order_id]['amount'] for order_id in self.executions ])
        average_cost = average_cost / sum([ self.executions[order_id]['amount'] for order_id in self.executions ])
        return average_cost

    def to_dict(self) -> Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES]:
        rv = super().to_dict()
        rv['slices'] = self.slices
        rv['wait_fill_threshold_ms'] = self.wait_fill_threshold_ms
        rv['executions'] = self.executions
        rv['filled_amount'] = self.filled_amount
        rv['average_cost'] = self.average_cost
        rv['pos'] = self.pos
        return rv

def execute_positions(
    redis_client : StrictRedis, 
    positions : List[DivisiblePosition], 
    ordergateway_pending_orders_topic : str,
    ordergateway_executions_topic : str
    ) -> Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None]:

    executed_positions : Union[Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES], None] = None

    # https://redis.io/commands/publish/
    _positions = [ position.to_dict() for position in positions ]
    redis_client.set(name=ordergateway_pending_orders_topic, value=json.dumps(_positions).encode('utf-8'), ex=60*15)

    print(f"{ordergateway_pending_orders_topic}: Orders sent {_positions}.")

    # Wait for fills
    fills_received : bool = False
    while not fills_received:
        try:
            keys = redis_client.keys()
            for key in keys:
                s_key : str = key.decode("utf-8")
                if s_key==ordergateway_executions_topic:
                    message = redis_client.get(key)
                    if message:
                        message = message.decode('utf-8')
                        executed_positions = json.loads(message)
                        fills_received = True
                        break

        except Exception as loop_err:
            raise loop_err
        finally:
            time.sleep(1)

    return executed_positions
