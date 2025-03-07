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
import futu as ft
from futu import *

from any_exchange import AnyExchange

class Futubull(AnyExchange):
    pass


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

ret, data, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2025-03-06', end='2025-03-07', max_count=5) # 5 per page, request the first page
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