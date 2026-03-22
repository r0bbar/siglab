import re

def is_int_string(s: str) -> bool:
    if not s:
        return False
    return s.lstrip('-+').isdigit()

def is_float_string(s: str) -> bool:
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def classify_ticker(ticker: str) -> str:
    """
    Classify ticker: 'spot', 'perpetual', 'option', 'dated_future',
    'tradfi_stock', 'tradfi_future', 'tradfi_option', or 'unknown'.
    """
    ticker = ticker.strip().upper()

    # 1. Crypto spot (example BTC/USDT) or tradfi FX (example EUR/USD, AUD/NZD) — requires exactly one / and nothing else suspicious
    if '/' in ticker and ':' not in ticker and '-' not in ticker and ticker.count('/') == 1:
        base, quote = ticker.split('/')
        if base.isalnum() and quote.isalnum():
            return 'spot'

    # 2. Crypto perpetual
    if ':' in ticker:
        return 'crypto.perpetual'

    # 3. Explicit TradFi option/futures suffixes
    if '-OPT-' in ticker or '-FOP-' in ticker:
        return 'tradfi.option'

    if '-STK-' in ticker:
        return 'tradfi.stock'

    # 4. Crypto dated future / option
    crypto_date_pat = r'(?:\d{1,2}[A-Z]{3}\d{2,4}|\d{6}|\d{2}[A-Z]{3}\d{2})'
    if re.search(crypto_date_pat, ticker):
        if re.search(r'[-_]\d+[-_]?[CP](?:$|[-_])', ticker) or ticker.endswith(('-C', '-P', 'C', 'P')):
            return 'crypto.option'
        return 'crypto.dated_future'

    # 5. Short futures codes (CME style: ESU6, NQZ5, ...)
    if re.match(r'^[A-Z]{1,4}[FGHJKMNQUVXZ]\d{1}$', ticker):
        return 'tradfi.future'

    # 6. TradFi verbose options fallback
    if re.search(r'\d{8}', ticker) and re.search(r'[CP](?:$|[-_])', ticker):
        return 'tradfi.option'

    # 7. TradFi stocks — NO dashes allowed (plain tickers only)
    #    This prevents misclassifying things like BTC-USDT
    if '-' not in ticker and \
       len(ticker) <= 10 and \
       ticker.isalnum() and \
       not any(c in ticker for c in '/:'):
        return 'tradfi.stock'

    # 8. Broader futures fallback
    if re.search(r'[FGHJKMNQUVXZ]\d$', ticker) and len(ticker) <= 6:
        return 'tradfi.future'

    # 9. Very last generic fallbacks
    if re.search(r'[CP]$', ticker):
        return 'tradfi.option'

    if re.search(r'\d{6,8}', ticker):
        return 'tradfi.future_or_option'

    return 'unknown'