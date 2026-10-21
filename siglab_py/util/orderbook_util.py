from typing import Dict, List, Tuple, TypedDict
import numpy as np

# Usage: https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/orderbook_exploratory.ipynb
class OrderBookFeatures(TypedDict):
    num_bids : int # Number of bids, before price levels binning in extract_ob_features
    num_asks : int # Number of asks, before price levels binning in extract_ob_features
    mid : float
    best_bid : float
    best_ask : float
    focus_bid : float # focus range
    focus_ask : float
    worst_bid : float # Why you want to know? You want to know how much depth you've got from fetch_orderbook given OB_DEPTH specified. And if that's sufficient for your purpose?
    worst_ask : float
    best_spread_bps : float
    spread_bps_by_order_notional : float
    focus_spread_bps : float
    knee_spread_bps : float # This is the actively traded region where market participants places bets
    worst_spread_bps : float
    buy_walls : List[Tuple[float, float]]  # List of (price, volume) tuples
    sell_walls : List[Tuple[float, float]]  # List of (price, volume) tuples
    bid_knee : Tuple[float, float]
    ask_knee : Tuple[float, float]
    slippage_target_bid : Tuple[float, float]
    slippage_target_ask : Tuple[float, float]
    total_bid_volume : float
    total_ask_volume : float
    imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)

def calc_slippages_by_notional(
    order_notional : float,
    mid : float,
    ob : Dict,
):
    result = calc_slippages(
        bids_in_scope = [ (lvl[0], lvl[1]) for lvl in ob['bids'] if lvl[0] < mid and lvl[1] * mid >= order_notional],
        asks_in_scope = [ (lvl[0], lvl[1]) for lvl in ob['asks'] if lvl[0] > mid and lvl[1] * mid >= order_notional]
    )
    return result

def calc_slippages_by_target(
    atr_bps : float,
    focus_atr_multiple : float,
    mid : float,
    ob : Dict,
):
    result = calc_slippages(
        bids_in_scope = [ (lvl[0], lvl[1]) for lvl in ob['bids'] if lvl[0] < mid and lvl[0] > mid * (1 - focus_atr_multiple * atr_bps / 10000)],
        asks_in_scope = [ (lvl[0], lvl[1]) for lvl in ob['asks'] if lvl[0] > mid and lvl[0] < mid * (1 + focus_atr_multiple * atr_bps / 10000)]
    )
    return result

def calc_slippages(
    bids_in_scope,
    asks_in_scope
):
    bids_in_scope_price = max([x[0] for x in bids_in_scope]) 
    bids_in_scope_vol = sum([x[1] for x in bids_in_scope])
    target_bid = (bids_in_scope_price, bids_in_scope_vol)
    asks_in_scope_price = min([x[0] for x in asks_in_scope])
    asks_in_scope_vol = sum([x[1] for x in asks_in_scope])
    target_ask = (asks_in_scope_price, asks_in_scope_vol)
    spread_bps = round((asks_in_scope_price/bids_in_scope_price-1) *10000, 5)
    return {
        'spread_bps' : spread_bps,
        'target_bid' : target_bid,
        'target_ask' : target_ask
    }

def extract_ob_features(
    ob: Dict,
    order_notional : float,
    price_level_separation_bps : float,
    atr_bps : float,
    focus_atr_multiple : float = 5,
    wall_threshold_multiplier: float = None # Set to for example 3. Default None, meaning: Don't bother compute
) -> OrderBookFeatures:
    def _aggregate_price_levels(
            levels: List[List[float]],
            side: str,
            mid_ref: float
        ) -> List[List[float]]:
        binned: Dict[int, List[float]] = {}
        for lvl in levels:
            price, qty = lvl[0], lvl[1]      # <-- was: for price, qty in levels
            dist_bps = (price - mid_ref) / mid_ref * 10000.0
            bucket = int(np.floor(dist_bps / price_level_separation_bps))
            if bucket not in binned:
                binned[bucket] = [0.0, 0.0]
            binned[bucket][1] += qty
            if binned[bucket][0] == 0.0:
                binned[bucket][0] = price
            elif side == 'bid':
                binned[bucket][0] = max(binned[bucket][0], price)
            else:
                binned[bucket][0] = min(binned[bucket][0], price)
        result = [[v[0], v[1]] for v in binned.values()]
        result.sort(key=lambda x: x[0], reverse=(side == 'bid'))
        return result

    def _find_knee(levels: List[List[float]]) -> Tuple[float, float]:
        if not levels:
            return (0.0, 0.0)
        if len(levels) < 3:
            return (levels[-1][0], sum(lvl[1] for lvl in levels))          # was: sum(v for _, v in levels)
        prices = np.array([lvl[0] for lvl in levels], dtype=float)         # was: [p for p, _ in levels]
        cum = np.cumsum([lvl[1] for lvl in levels])                        # was: [v for _, v in levels]
        x = (prices - prices.min()) / (prices.max() - prices.min()) if prices.max() > prices.min() else np.zeros_like(prices)
        y = (cum - cum.min()) / (cum.max() - cum.min()) if cum.max() > cum.min() else np.zeros_like(cum)
        x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
        denom = np.hypot(y2 - y1, x2 - x1)
        if denom == 0:
            return (float(levels[-1][0]), float(cum[-1]))
        dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        idx = int(np.argmax(dist))
        return (float(prices[idx]), float(cum[idx]))

    _raw_best_bid = max([x[0] for x in ob['bids']])
    _raw_best_ask = min([x[0] for x in ob['asks']])
    _raw_mid = (_raw_best_bid + _raw_best_ask) / 2.0

    _ob = {
        **ob,
        'bids': _aggregate_price_levels(ob['bids'], 'bid', _raw_mid),
        'asks': _aggregate_price_levels(ob['asks'], 'ask', _raw_mid),
    }

    num_bids = len(ob['bids'])
    num_asks = len(ob['asks'])
    best_bid = max([x[0] for x in ob['bids']])
    worst_bid = min([x[0] for x in ob['bids']])
    best_ask = min([x[0] for x in ob['asks']])
    worst_ask = max([x[0] for x in ob['asks']])

    mid = (best_bid + best_ask)/2
    best_spread_bps = round((best_ask/best_bid-1) *10000, 5)
    worst_spread_bps = round((worst_ask/worst_bid-1) *10000, 5)

    total_bid_vol = sum(lvl[1] for lvl in ob['bids'])
    total_ask_vol = sum(lvl[1] for lvl in ob['asks'])
    total_vol = total_bid_vol + total_ask_vol
    imbalance = (total_bid_vol - total_ask_vol) / total_vol if total_vol else 0.0

    result = calc_slippages_by_notional(mid=mid, order_notional=order_notional, ob=ob)
    spread_bps_by_order_notional = result['spread_bps']
    slippage_target_bid = result['target_bid']
    slippage_target_ask = result['target_ask']

    result = calc_slippages_by_target(mid=mid, atr_bps=atr_bps, focus_atr_multiple=focus_atr_multiple, ob=ob)
    focus_spread_bps = result['spread_bps']
    focus_bid = result['target_bid']
    focus_ask = result['target_ask']

    bids = _ob['bids']
    asks = _ob['asks']

    bid_knee = _find_knee(bids)
    ask_knee = _find_knee(asks)
    knee_spread_bps = (ask_knee[0]/bid_knee[0] -1) *10000

    buy_walls: List[Tuple[float, float]] = []
    sell_walls: List[Tuple[float, float]] = []

    buy_walls = None
    sell_walls = None
    if wall_threshold_multiplier:
        # --- Bid side (buy walls) ---
        if bids:
            bid_vols = [v for p, v in bids if p>bid_knee[0] and p<mid]
            avg_bid_vol = sum(bid_vols) / len(bid_vols)
            threshold = avg_bid_vol * wall_threshold_multiplier
            buy_walls = [(p, v) for p, v in bids if v >= threshold]

        # --- Ask side (sell walls) ---
        if asks:
            ask_vols = [v for p, v in asks if p<ask_knee[0] and p>mid]
            avg_ask_vol = sum(ask_vols) / len(ask_vols)
            threshold = avg_ask_vol * wall_threshold_multiplier
            sell_walls = [(p, v) for p, v in asks if v >= threshold]

    return OrderBookFeatures(
        num_bids=num_bids,
        num_asks=num_asks,
        best_bid=best_bid,
        best_ask=best_ask,
        focus_bid=focus_bid,
        focus_ask=focus_ask,
        worst_bid=worst_bid,
        worst_ask=worst_ask,
        mid=mid,
        best_spread_bps=best_spread_bps,
        spread_bps_by_order_notional=spread_bps_by_order_notional,
        focus_spread_bps=focus_spread_bps,
        knee_spread_bps=knee_spread_bps,
        worst_spread_bps=worst_spread_bps,
        buy_walls=buy_walls,
        sell_walls=sell_walls,
        bid_knee=bid_knee,
        ask_knee=ask_knee,
        slippage_target_bid=slippage_target_bid,
        slippage_target_ask=slippage_target_ask,
        total_bid_volume=total_bid_vol,
        total_ask_volume=total_ask_vol,
        imbalance=imbalance,
    )