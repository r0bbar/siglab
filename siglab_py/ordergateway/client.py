
from typing import List, Dict, Any, Union
import json
from  datetime import datetime
import time
from redis import StrictRedis

from siglab_py.exchanges.any_exchange import AnyExchange
from siglab_py.constants import JSON_SERIALIZABLE_TYPES

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
        leg_room_bps : float = 0,
        reduce_only : bool = False,
        fees_ccy : Union[str, None] = None
    ) -> None:
        if amount<=0:
            raise ValueError(f"amount must be >0!")
            
        self.ticker = ticker
        self.side = side.strip().lower()
        self.amount = amount
        self.order_type = order_type.strip().lower()
        self.leg_room_bps = leg_room_bps
        self.reduce_only = reduce_only
        self.fees_ccy = fees_ccy

        self.dispatched_amount : float = 0 # This is amount in base ccy, with rounding + multiplier applied (So for contracts with multiplier!=1, this is no longer amount in base ccy.)
        self.dispatched_price : Union[None, float] = None

        self.timestamp_ms = int(datetime.now().timestamp() * 1000)

    def to_dict(self) -> Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES]:
        return {
            "ticker" : self.ticker,
            "side" : self.side,
            "amount" : self.amount,
            "order_type" : self.order_type,
            "leg_room_bps" : self.leg_room_bps,
            "reduce_only" : self.reduce_only,
            "fees_ccy" : self.fees_ccy,
            "dispatched_amount" : self.dispatched_amount,
            "dispatched_price" : self.dispatched_price,
            "timestamp_ms" : self.timestamp_ms
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
        reduce_only : bool = False,
        fees_ccy : Union[str, None] = None,
        slices : int = 1,
        wait_fill_threshold_ms : float = -1
    ) -> None:
        if amount<=0:
            raise ValueError(f"amount must be >0!")

        super().__init__(ticker, side, amount, order_type, leg_room_bps, reduce_only, fees_ccy)
        self.slices = slices
        self.wait_fill_threshold_ms = wait_fill_threshold_ms
        self.multiplier = 1
        self.filled_amount : Union[float, None] = None
        self.average_cost : Union[float, None] = None
        self.fees : Union[float, None] = None
        self.pos : Union[float, None] = None # in base ccy, after execution. (Not in USDT or quote ccy, Not in # contracts)

        self.done : bool = False
        self.execution_err : str = ""

        self.dispatched_slices : List[Order] = []
        self.executions : Dict[str, Dict[str, Any]] = {}

        self.timestamp_ms = int(datetime.now().timestamp() * 1000)

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
                        reduce_only=self.reduce_only,
                        fees_ccy=self.fees_ccy,
                        order_type=self.order_type)
                    
                else:
                    # Last slice
                    slice = Order(
                        ticker=self.ticker, 
                        side=self.side, 
                        amount=remaining_amount_in_base_ccy, 
                        leg_room_bps=self.leg_room_bps, 
                        reduce_only=self.reduce_only,
                        fees_ccy=self.fees_ccy,
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

    '''
    The purpose of patch_executions is to fix erroneous exchange response where filled, amount or average None when status already 'closed'.

    Example hyperliquid don't have 'average' in response JSON:
        {
            'info': {
                'order': {
                    'coin': 'SUSHI', 
                    'side': 'B', 
                    'limitPx': '0.79252', 
                    'sz': '0.0', 
                    'oid': xxx, 
                    'timestamp': xxx, 
                    'origSz': '30.0'
                }, 
                'status': 'filled', 
                'statusTimestamp': xxx}, 
                'id': 'xxx', 
                'clientOrderId': None, 
                'timestamp': xxx, 
                'datetime': 'xxx', 
                'lastTradeTimestamp': None, 
                'lastUpdateTimestamp': xxx, 
                'symbol': 'SUSHI/USDC:USDC', 
                'type': None, 
                'timeInForce': None, 
                'postOnly': None, 
                'reduceOnly': None, 
                'side': 'buy', 
                'price': 0.79252, 
                'triggerPrice': None, 
                'amount': 30.0, 
                'cost': 23.7756, 
                'average': None, <-- No average cost?
                'filled': 30.0, 
                'remaining': 0.0, 
                'status': 'closed', 
                'fee': None, 
                'trades': [], 
                'fees': [], 
                'stopPrice': None, 
                'takeProfitPrice': None, 
                'stopLossPrice': None
            }

        A second example, sometimes even 'amount', 'filled' and 'average' both None, but status is 'filled':
            [
                {
                    'info': {
                        'order': {
                            'coin': 'BTC', 
                            'side': 'A', 
                            'limitPx': '84389.0', 
                            'sz': '0.0',
                            'oid': xxx, 
                            'timestamp': xxx, 
                            'origSz': '0.01184'
                        },
                        'status': 'filled', 
                        'statusTimestamp': xxx
                    }, 
                    'id': 'xxx',
                    'clientOrderId': None, 
                    'timestamp': xxx, 
                    'datetime': 'xxx', 
                    'lastTradeTimestamp': None, 
                    'lastUpdateTimestamp': None, 
                    'symbol': 'BTC/USDC:USDC', 
                    'type': None, 
                    'timeInForce': None,
                    'postOnly': None, 
                    'reduceOnly': None, 
                    'side': 'sell', 
                    'price': 84389.0,
                    'triggerPrice': None, 
                    'amount': None, <-- No amount!?!
                    'cost': None, 
                    'average': None, <-- No average!?!
                    'filled': None,    <-- No filled!?!
                    'remaining': 0.0, 
                    'status': 'closed', 
                    'fee': None, 'trades': [], 'fees':
                    [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None}
                ]
        '''
    def patch_executions(self):
        for order_id in self.executions:
            execution = self.executions[order_id]
            execution['patch'] = {}
            execution['patch']['average'] = None
            execution['patch']['filled'] = 0
            execution['patch']['amount'] = 0

            slice_id = execution['slice_id']
            correspinding_slice = self.dispatched_slices[slice_id]

            status = execution['status']
            if status and status.strip().lower()=='closed':
                if 'average' in execution:
                    if not execution['average']:
                        if 'price' in execution and execution['price']:
                            execution['patch']['average'] = execution['price']
                        else:
                            execution['patch']['average'] = corresponding_slice.dispatched_price
                    else:
                        execution['patch']['average'] = execution['average']

                if 'filled' in execution:
                    if not execution['filled']:
                        execution['patch']['filled'] = corresponding_slice.dispatched_amount
                    else:
                        execution['patch']['filled'] = execution['filled']

                if 'amount' in execution:
                    if not execution['amount']:
                        execution['patch']['amount'] = corresponding_slice.dispatched_amount
                    else:
                        execution['patch']['amount'] = execution['amount']
        
    def get_filled_amount(self) -> float:
        # filled_amount is in base ccy
        filled_amount = sum(
                    [ 
                        (
                            self.executions[order_id]['filled'] 
                            if 'filled' in self.executions[order_id] and self.executions[order_id]['filled'] 
                            else self.executions[order_id]['patch']['dispatched_amount']
                        )  * self.multiplier 
                        for order_id in self.executions 
                    ]
                )
        if self.side=='sell':
            filled_amount = -1 * filled_amount
        return filled_amount

    def get_average_cost(self) -> float:
        total_amount : float = sum(
            [ 
                self.executions[order_id]['amount'] 
                if 'amount' in self.executions[order_id] and self.executions[order_id]['amount'] 
                else self.executions[order_id]['patch']['amount']
                for order_id in self.executions 
            ]
        )
        
        average_cost = sum(
            [ 
                (
                    self.executions[order_id]['average'] 
                    if 'average' in self.executions[order_id] and self.executions[order_id]['average'] 
                    else (
                        self.executions[order_id]['price'] 
                        if self.executions[order_id]['price'] 
                        else self.executions[order_id]['patch']['average']
                    )
                ) * 
                (
                    self.executions[order_id]['amount'] 
                    if 'amount' in self.executions[order_id] and self.executions[order_id]['amount'] 
                    else self.executions[order_id]['patch']['amount']
                )
                for order_id in self.executions 
            ]
        )
        average_cost = average_cost / total_amount if total_amount!=0 else 0
        return average_cost

    def get_fees(self) -> float:
        fees : float = 0
        if self.fees_ccy:
            fees = sum([ float(self.executions[order_id]['fee']['cost'] if self.executions[order_id]['fee'] and self.executions[order_id]['fee']['cost'] else 0)  for order_id in self.executions if self.executions[order_id]['fee'] and self.executions[order_id]['fee']['currency'].strip().upper()==self.fees_ccy.strip().upper() ])
        return fees

    def to_dict(self) -> Dict[JSON_SERIALIZABLE_TYPES, JSON_SERIALIZABLE_TYPES]:
        rv = super().to_dict()
        rv['slices'] = self.slices
        rv['wait_fill_threshold_ms'] = self.wait_fill_threshold_ms
        rv['executions'] = self.executions
        rv['filled_amount'] = self.filled_amount
        rv['average_cost'] = self.average_cost
        rv['fees'] = self.fees
        rv['pos'] = self.pos
        rv['done'] = self.done
        rv['execution_err'] = self.execution_err
        rv['timestamp_ms'] = self.timestamp_ms
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

                        redis_client.delete(key)
                        break

        except Exception as loop_err:
            raise loop_err
        finally:
            time.sleep(1)

    return executed_positions
