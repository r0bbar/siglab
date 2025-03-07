'''
https://github.com/FutunnOpen/py-futu-api/blob/master/README.md
https://www.futuhk.com/en/support/categories/909?global_content=%7B%22promote_id%22%3A13765%2C%22sub_promote_id%22%3A10%7D

Fees: https://www.futuhk.com/en/commissionnew#crypto

Investor Protection: https://www.futuhk.com/en

Margin Trading: https://www.futunn.com/en/learn/detail-what-is-margin-trading-62010-220679271

Download Futu OpenD
    https://www.futuhk.com/en/support/topic1_464?global_content=%7B%22promote_id%22%3A13765%2C%22sub_promote_id%22%3A10%7D

    If you run the installer version "Futu_OpenD-GUI_9.0.5008_Windows.exe", it'd be installed under:
        C:\\Users\\xxx\\AppData\\Roaming\\Futu_OpenD\\Futu_OpenD.exe

Architecture: https://openapi.futunn.com/futu-api-doc/en/intro/intro.html

python -mpip install futu-api

Whatsapp support: Moomoo OpenAPI 1群

API
    Fetches:
        market status https://openapi.futunn.com/futu-api-doc/en/quote/get-global-state.html
        stock basic info https://openapi.futunn.com/futu-api-doc/en/quote/get-static-info.html
        historical candles https://openapi.futunn.com/futu-api-doc/en/quote/request-history-kline.html
        realtime candles https://openapi.futunn.com/futu-api-doc/en/quote/get-kl.html
        real time quote https://openapi.futunn.com/futu-api-doc/en/quote/get-stock-quote.html
        open orders https://openapi.futunn.com/futu-api-doc/en/trade/get-order-list.html
        historical orders https://openapi.futunn.com/futu-api-doc/en/trade/get-history-order-list.html
        positions https://openapi.futunn.com/futu-api-doc/en/trade/get-position-list.html
        balances https://openapi.futunn.com/futu-api-doc/en/trade/get-funds.html

    Trading:
        create order https://openapi.futunn.com/futu-api-doc/en/trade/place-order.html
        amend order https://openapi.futunn.com/futu-api-doc/en/trade/modify-order.html

'''
from typing import List, Dict, Union, Any
import futu as ft
from futu import *

from any_exchange import AnyExchange

class Futubull(AnyExchange):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args)
        
        params : Dict[str, Any] = args[0] # Type "object" is not assignable to declared type "Dict[Unknown, Unknown]"
        self.host_addr = params['daemon']['host']
        self.port = params['daemon']['port']
        self.market = params['market']
        self.security_type = params['security_type']
        self.quote_ctx = OpenQuoteContext(host=self.host_addr, port=self.port)
        self.trd_ctx = OpenSecTradeContext(
                filter_trdmarket=params['trdmarket'], 
                host=self.host_addr, port=self.port, 
                security_firm=params['security_firm']
            )
        self.markets : Dict = {}

    def load_markets(self, reload=False, params={}):  # type: ignore
        '''
        Mimic CCXT load_markets https://github.com/ccxt/ccxt/blob/master/python/ccxt/base/exchange.py

        Examplem,
            {
                ... more pairs ...
                'ETH/USDC:USDC': {
                    'id': 'ETH-USDC-SWAP',
                    'lowercaseId': None,
                    'symbol': 'ETH/USDC:USDC',
                    'base': 'ETH',
                    'quote': 'USDC',
                    'settle': 'USDC',
                    'baseId': 'ETH',
                    'quoteId': 'USDC',
                    'settleId': 'USDC',
                    'type': 'swap',
                    'spot': False,
                    'margin': False,
                    'swap': True,
                    'future': False,
                    'option': False,
                    'index': None,
                    'active': True,
                    'contract': True,
                    'linear': True,
                    'inverse': False,
                    'subType': 'linear',
                    'taker': 0.0005,
                    'maker': 0.0002,
                    'contractSize': 0.001,
                    'expiry': None,
                    'expiryDatetime': None,
                    'strike': None,
                    'optionType': None,
                    'precision': {'amount': 1.0, 'price': 0.01, 'cost': None, 'base': None, 'quote': None},
                    'limits': {'leverage': {'min': 1.0, 'max': 50.0}, 'amount': {'min': 1.0, 'max': None}, 'price': {'min': None, 'max': None}, 'cost': {'min': None, 'max': None}},
                    'marginModes': {'cross': None, 'isolated': None},
                    'created': 1666076197702,
                    'info' : {
                        ... raw exchange response here ...
                    }
                },
                ... more pairs ...
            }

        gateway.py will call:
            1. multiplier = market['contractSize'] (Note this is contract size for futures, not same as 'Lot Size')
            2. price_to_precision -> market['precision']['price']
            3. amount_to_precision  -> market['precision']['amount']    <-- This is 'Lot Size'
        '''
        
        ret, data = self.quote_ctx.get_stock_basicinfo(self.market, self.security_type)
        if ret == RET_OK:
            for index, row in data.iterrows(): # type: ignore
                symbol = row['code']

                name = row['name']
                stock_type = row['stock_type']
                stock_child_type = row['stock_child_type']
                stock_owner = row['stock_owner']
                option_type = row['option_type']
                strike_time = row['strike_time']
                strike_price = row['strike_price']
                suspension = row['suspension']
                stock_id = row['stock_id']
                listing_date = row['listing_date']
                delisting = bool(row['delisting'])
                main_contract = bool(row['main_contract'])
                last_trade_time = row['last_trade_time']
                exchange_type = row['exchange_type']
                lot_size = row['lot_size']
                
                # No additional information from 'get_stock_basicinfo'
                # ret, detail = self.quote_ctx.get_stock_basicinfo(self.market, self.security_type, [ symbol ])

                info : Dict = {
                    'symbol' : symbol,
                    'name' : name,
                    'stock_type' : stock_type,
                    'stock_child_type' : stock_child_type,
                    'stock_owner' : stock_owner,
                    'option_type' : option_type,
                    'strike_time' : strike_time,
                    'strike_price' : strike_price,
                    'suspension' : suspension,
                    'stock_id' : stock_id,
                    'listing_date' : listing_date,
                    'delisting' : delisting,
                    'main_contract' : main_contract,
                    'last_trade_time' : last_trade_time,
                    'exchange_type' : exchange_type,
                    'lot_size' : lot_size
                }
                self.markets[symbol] = {
                    'symbol' : symbol,
                    'id' : symbol,

                    'base': None,
                    'quote': None,
                    'settle': None,
                    'baseId': None,
                    'quoteId': None,
                    'settleId': None,

                    'type': self.security_type,
                    'spot': False,
                    'margin': False,
                    'swap': False,
                    'future': False,
                    'option': False,
                    'index': None,
                    'active': not delisting,
                    'contract': False,
                    'linear': False,
                    'inverse': False,
                    'subType': False,
                    'taker': 0,
                    'maker': 0,
                    'contractSize': 0,
                    'expiry': strike_time,
                    'expiryDatetime': strike_time,
                    'strike': strike_price,
                    'optionType': option_type,

                    'precision': {
                        'amount': lot_size, 
                        'price': 0.01, 
                        'cost': None, 
                        'base': None, 
                        'quote': None
                    },

                    'limits': {
                        'leverage': {'min': 1, 'max': 5}, 
                        'amount': {'min': 1.0, 'max': None}, 
                        'price': {'min': None, 'max': None}, 
                        'cost': {'min': None, 'max': None}
                    },
                    'marginModes': {'cross': None, 'isolated': None},

                    'info' : info
                }
        return self.markets

    def fetch_ohlcv(self, symbol: str, timeframe='1m', since: Union[int, None] = None, limit: Union[int, None] = None, params={}) -> List[list]:  # type: ignore
        return [[]]

params : Dict = {
    'trdmarket' : TrdMarket.HK,
    'security_firm' : SecurityFirm.FUTUSECURITIES,
    'market' : Market.HK, 
    'security_type' : SecurityType.STOCK,
    'daemon' : {
        'host' : '127.0.0.1',
        'port' : 11111
    }
}
exchange = Futubull(params)
markets = exchange.load_markets()

'''
Examples from Futu https://openapi.futunn.com/futu-api-doc/en

'''
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

print(quote_ctx.get_global_state())

ret, data = quote_ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK)
if ret == RET_OK:
    print(data)
else:
    print('error:', data)
print('******************************************')
ret, data = quote_ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK, ['HK.06998', 'HK.00700'])
if ret == RET_OK:
    assert isinstance(data, pd.DataFrame)
    print(data)
    print(data['name'][0])
    print(data['name'].values.tolist())
else:
    print('error:', data)

'''
KLType: https://openapi.futunn.com/futu-api-doc/en/quote/quote.html#66
'''
ret, data, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2024-01-01', end='2025-03-07', ktype=KLType.K_60M, max_count=100)
if ret == RET_OK:
    assert isinstance(data, pd.DataFrame)
    print(data)
    print(data['code'][0])
    print(data['close'].values.tolist())
else:
    print('error:', data)
while page_req_key != None:
    print('*************************************')
    ret, data, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2019-09-11', end='2019-09-18', max_count=5,page_req_key=page_req_key) # Request the page after turning data
    if ret == RET_OK:
        print(data)
    else:
        print('error:', data)

# For any real time data, you need subscribe https://openapi.futunn.com/futu-api-doc/en/intro/authority.html#5331
ret_sub, err_message = quote_ctx.subscribe(['HK.00700'], [SubType.K_DAY], subscribe_push=False)
if ret_sub == RET_OK:
    ret, data = quote_ctx.get_cur_kline('HK.00700', 2, SubType.K_DAY, AuType.QFQ)
    if ret == RET_OK:
        assert isinstance(data, pd.DataFrame)
        print(data)
        print(data['turnover_rate'][0])
        print(data['turnover_rate'].values.tolist())
    else:
        print('error:', data)
else:
    # (-1, '无权限订阅HK.00005的行情，请检查香港市场股票行情权限')
    print('subscription failed', err_message)

ret_sub, err_message = quote_ctx.subscribe(['HK.00700'], [SubType.QUOTE], subscribe_push=False)
if ret_sub == RET_OK:
    ret, data = quote_ctx.get_stock_quote(['HK.00700'])
    if ret == RET_OK:
        assert isinstance(data, pd.DataFrame)
        print(data)
        print(data['code'][0])
        print(data['code'].values.tolist())
    else:
        print('error:', data)
else:
    # '无权限订阅HK.00700的行情，请检查香港市场股票行情权限'
    print('subscription failed', err_message)

ret_sub = quote_ctx.subscribe(['HK.00700'], [SubType.ORDER_BOOK], subscribe_push=False)[0]
if ret_sub == RET_OK:
    ret, data = quote_ctx.get_order_book('HK.00700', num=3)
    if ret == RET_OK:
        print(data)
    else:
        print('error:', data)
else:
    print('subscription failed')

trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
ret, data = trd_ctx.accinfo_query()
if ret == RET_OK:
    assert isinstance(data, pd.DataFrame)
    print(data)
    print(data['power'][0])
    print(data['power'].values.tolist())
else:
    print('accinfo_query error: ', data)

ret, data = trd_ctx.order_list_query()
if ret == RET_OK:
    assert isinstance(data, pd.DataFrame)
    print(data)
    if data.shape[0] > 0:
        print(data['order_id'][0])
        print(data['order_id'].values.tolist())
else:
    print('order_list_query error: ', data)

ret, data = trd_ctx.history_order_list_query()
if ret == RET_OK:
    assert isinstance(data, pd.DataFrame)
    print(data)
    if data.shape[0] > 0:
        print(data['order_id'][0])
        print(data['order_id'].values.tolist())
else:
    print('history_order_list_query error: ', data)

ret, data = trd_ctx.position_list_query()
if ret == RET_OK:
    assert isinstance(data, pd.DataFrame)
    print(data)
    if data.shape[0] > 0:
        print(data['stock_name'][0])
        print(data['stock_name'].values.tolist())
else:
    print('position_list_query error: ', data)

trd_ctx.close()