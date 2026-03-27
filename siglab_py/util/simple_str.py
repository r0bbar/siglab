from typing import List, Dict, Any, Union
import re
import rapidfuzz.fuzz as fuzz

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

def keywords_match(
    sentence : str, 
    keywords_cache : Dict[str, Dict[str, Union[str, List[str]]]], 
    fuzzy=False, 
    fuzzy_threshold=30
) -> Dict[str, Union[None, float, Dict[str, Any]]]:
    sentence_clean = sentence.strip()
    sentence_lower = sentence_clean.lower()
    word_count = len(sentence_clean.split())
    result = {
        "num_matches": 0,
        "word_count": word_count,
        "matches_percent": 0.0,
        "nouns": {},
        "actions": {},
        "adjectives": {}
    }

    def exact_match(keyword, text):
        """Exact substring match after cleaning."""
        kw_clean = keyword.strip().lower()
        return kw_clean and kw_clean in text

    def fuzzy_match(keyword, text, threshold):
        """Return True if keyword approximately appears in text."""
        kw_lower = keyword.lower().strip()
        if not kw_lower:
            return False
        # Exact match first (fast path)
        if kw_lower in text:
            return True
        text_words = text.split()
        kw_words = kw_lower.split()
        # Single word: compare with each word in text
        if len(kw_words) == 1:
            for word in text_words:
                if fuzz.ratio(word, kw_lower) >= threshold:
                    return True
        # Multi-word: sliding window of equal length, average ratio
        else:
            for i in range(len(text_words) - len(kw_words) + 1):
                phrase = text_words[i:i + len(kw_words)]
                avg_score = sum(fuzz.ratio(phrase[j], kw_words[j]) for j in range(len(kw_words))) / len(kw_words)
                if avg_score >= threshold:
                    return True
        return False

    def process_category(category_list):
        category_result = {}
        total_in_category = 0
        for item in category_list:
            sub_type = item["sub_type"]
            matched_keywords = []
            for kw in item["keywords"]:
                if fuzzy:
                    if fuzzy_match(kw, sentence_lower, fuzzy_threshold):
                        matched_keywords.append(kw)
                        total_in_category += 1
                else:
                    if exact_match(kw, sentence_lower):
                        matched_keywords.append(kw)
                        total_in_category += 1
            if matched_keywords:
                category_result[sub_type] = matched_keywords
        return category_result, total_in_category

    for cat in ["nouns", "actions", "adjectives"]:
        cat_result, cat_total = process_category(keywords_cache[cat])
        result[cat] = cat_result
        result["num_matches"] += cat_total

    if result["word_count"] > 0:
        result["matches_percent"] = round((result["num_matches"] / result["word_count"]) * 100, 2)

    return result

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