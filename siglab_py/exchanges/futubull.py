'''
https://github.com/FutunnOpen/py-futu-api/blob/master/README.md
https://www.futuhk.com/en/support/categories/909?global_content=%7B%22promote_id%22%3A13765%2C%22sub_promote_id%22%3A10%7D

Fees: https://www.futuhk.com/en/commissionnew#crypto

Subscribe L2 data: https://openapi.futunn.com/futu-api-doc/en/intro/authority.html

Investor Protection: https://www.futuhk.com/en

Margin Trading: 
    https://www.futunn.com/en/learn/detail-what-is-margin-trading-62010-220679271
    Margin rate 6.8% annual https://www.futuhk.com/en/support/topic2_417

Download Futu OpenD
    https://www.futuhk.com/en/support/topic1_464?global_content=%7B%22promote_id%22%3A13765%2C%22sub_promote_id%22%3A10%7D

    If you run the installer version "Futu_OpenD-GUI_9.0.5008_Windows.exe", it'd be installed under C-Drive:
        C:\\Users\\xxx\\AppData\\Roaming\\Futu_OpenD\\Futu_OpenD.exe
    Unfortunately, log folder also under C-drive as a result, and they are big.

    For command line version: https://openapi.futunn.com/futu-api-doc/opend/opend-cmd.html
        Binary under downloaded package (You can put it under for example D-drive): 
            ...\Futu_OpenD_9.4.5408_Windows\Futu_OpenD_9.4.5408_Windows
            
        Put a batch file "start_futu_opend.bat", if login_pwd include special characters, enclose pwd with double quotes:
            FutuOpenD -login_account=1234567 -login_pwd="... Your Secret here ..."
    Config file is "FutuOpenD.xml", you can adjust logging verbosity here.

Architecture: https://openapi.futunn.com/futu-api-doc/en/intro/intro.html

python -mpip install futu-api

Whatsapp support: Moomoo OpenAPI 1群

API
    Fetches:
        market status https://openapi.futunn.com/futu-api-doc/en/quote/get-global-state.html
        stock basic info https://openapi.futunn.com/futu-api-doc/en/quote/get-static-info.html
        historical candles https://openapi.futunn.com/futu-api-doc/en/quote/request-history-kline.html
        realtime candles https://openapi.futunn.com/futu-api-doc/en/quote/get-kl.html
        orderbook https://openapi.futunn.com/futu-api-doc/en/quote/get-order-book.html
        real time quote https://openapi.futunn.com/futu-api-doc/en/quote/get-stock-quote.html
        open orders https://openapi.futunn.com/futu-api-doc/en/trade/get-order-list.html
        historical orders https://openapi.futunn.com/futu-api-doc/en/trade/get-history-order-list.html
        positions https://openapi.futunn.com/futu-api-doc/en/trade/get-position-list.html
        balances https://openapi.futunn.com/futu-api-doc/en/trade/get-funds.html

    Trading:
        create order https://openapi.futunn.com/futu-api-doc/en/trade/place-order.html
        amend order https://openapi.futunn.com/futu-api-doc/en/trade/modify-order.html

'''
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Union, Any, NoReturn
import pandas as pd

import futu as ft
from futu import *

from siglab_py.exchanges.any_exchange import AnyExchange

class Futubull(AnyExchange):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args)
        
        self.name = 'futubull'

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

    '''
    With CCXT, 'params' is used to parse exchange specific parameters. Not actually used with Futu.

    Get Historical Candlesticks max_count: In API doc it advises<1000, but when I tested, it's 246 (1 year).
    To fetch more than this, use market_data_util.fetch_candles (Sliding window implemented).
    Under the hood, it'd use implementation below.

    CCXT fetch_ohlcv response format is:
        [
            [1704038400000, 42469.9, 42723.1, 42442.3, 42609.8, 3624.59]
            [1704042000000, 42609.8, 42672.4, 42473.9, 42630.6, 2829.69]
            [1704045600000, 42630.6, 42742.9, 42542.9, 42673.8, 2097.12]
            [1704049200000, 42673.8, 42705.0, 42593.7, 42627.7, 2055.54]
            [1704052800000, 42627.6, 42693.0, 42511.1, 42565.5, 2061.97]
            ... more candles ...
        ]
    
    Fields are: timestamp (in ms), open, high, low, close, volume.
    '''
    def fetch_ohlcv(self, symbol: str, timeframe : str ='1h', since: Union[int, None] = None, limit: int = 100, params={}) -> Union[List[List[Union[int, float]]], None]:  # type: ignore
        if not symbol or not since:
            return None
        
        ktype=KLType.K_DAY
        if timeframe=="1m":
            ktype=KLType.K_1M
        elif timeframe=="1h":
            ktype=KLType.K_60M
        elif timeframe=="1h":
            ktype=KLType.K_DAY

        candles = []
        dt_start = datetime.fromtimestamp(int(since/1000))
        s_start : str = dt_start.strftime("%Y-%m-%d")
        ret, data, page_req_key = self.quote_ctx.request_history_kline(code=symbol, start=s_start, ktype=ktype, max_count=limit)
        if ret == RET_OK:
            for index, row in data.iterrows(): # type: ignore

                '''
                From doc:
                    Format: yyyy-MM-dd HH:mm:ss
                    The default of HK stock market and A-share market is Beijing time, 
                    while that of US stock market is US Eastern time.
                '''
                time_key = row['time_key']
                dt = datetime.strptime(time_key, "%Y-%m-%d %H:%M:%S")
                
                if self.market in [ Market.HK, Market.SH, Market.SZ ]:
                    tz = pytz.timezone('Asia/Shanghai')
                    dt = tz.localize(dt)
                elif self.market == Market.US:
                    tz = pytz.timezone('US/Eastern')
                    dt = tz.localize(dt)
                elif self.market == Market.CA:
                    tz = pytz.timezone('US/Eastern')
                    dt = tz.localize(dt)
                elif self.market == Market.AU:
                    tz = pytz.timezone('Australia/Sydney')
                    dt = tz.localize(dt)
                else:
                    # @todo:  HK SH SZ US AU CA FX
                    raise ValueError(f"Unsupported market {self.market}")

                timestamp_ms = int(dt.timestamp() * 1000)
                open = row['open']
                high = row['high']
                low = row['low']
                close = row['close']
                volume = row['volume']
                candles.append(
                    [ timestamp_ms, open, high, low, close, volume ]
                )
        return candles

    def fetch_candles(
        self,
        start_ts,
        end_ts,
        symbols,
        candle_size
    ) -> Dict[str, Union[pd.DataFrame, None]]:
        exchange_candles : Dict[str, Union[pd.DataFrame, None]] = {}

        for symbol in symbols:
            pd_candles = self._fetch_candles(symbol=symbol, start_ts=start_ts, end_ts=end_ts, candle_size=candle_size)
            pd_candles['exchange'] = self.name
            exchange_candles[symbol] = pd_candles

        return exchange_candles

    def _fetch_candles(
        self,
        symbol : str,
        start_ts : int,
        end_ts : int,
        candle_size : str = '1d',
        num_candles_limit : int = 100
    ):
        def _fetch_ohlcv(symbol, timeframe, since, limit, params) -> Union[List[List[Union[int, float]]], None]:
            one_timeframe = f"1{timeframe[-1]}"
            candles = self.fetch_ohlcv(symbol=symbol, timeframe=one_timeframe, since=since, limit=limit, params=params)
            if candles and len(candles)>0:
                candles.sort(key=lambda x : x[0], reverse=False)

            return candles

        all_candles = []
        params = {}
        this_cutoff = start_ts
        while this_cutoff<=end_ts:
            candles = _fetch_ohlcv(symbol=symbol, timeframe=candle_size, since=int(this_cutoff * 1000), limit=num_candles_limit, params=params)
            if candles and len(candles)>0:
                all_candles = all_candles + [[ int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]) ] for x in candles if x[1] and x[2] and x[3] and x[4] and x[5] ]

                record_ts = max([int(record[0]) for record in candles])
                record_ts_str : str = str(record_ts)
                if len(record_ts_str)==13:
                    record_ts = int(int(record_ts_str)/1000) # Convert from milli-seconds to seconds

                if this_cutoff==record_ts+1:
                    break
                else:
                    this_cutoff = record_ts  + 1
        columns = ['exchange', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume']
        pd_all_candles = pd.DataFrame([ [ "futubull", symbol, x[0], x[1], x[2], x[3], x[4], x[5] ] for x in all_candles], columns=columns)
        # fix_column_types(pd_all_candles)
        pd_all_candles['pct_chg_on_close'] = pd_all_candles['close'].pct_change()
        return pd_all_candles

if __name__ == '__main__':
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

    dt_end = datetime.today()
    dt_start : datetime = dt_end - timedelta(days=365*3)
    timestamp_ms : int = int(dt_start.timestamp() * 1000)
    candles = exchange.fetch_ohlcv(symbol="HK.00700", timeframe="1h", since=timestamp_ms, limit=1000)

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