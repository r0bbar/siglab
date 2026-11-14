import math
import pandas as pd
import numpy as np

'''
pnl_percent_notional = Trade's current pnl in percent.

Examples,
    y-axis:
        max (i.e most tight) = 0%
        sl_percent_trailing = 50% (Trailing stop loss in percent)

    x-axis:
        min TP = 1.5% <-- min TP
        max TP = 2.5% <-- max TP

    slope = (0-50)/(2.5-1.5) = -50/+1 = -50
    effective_tp_trailing_percent = slope * (pnl_percent_notional - 1.5%) + sl_percent_trailing

Case 1. pnl_percent_notional = 1.5% (Trade starting off, only +50bps pnl. i.e. min TP)
            effective_tp_trailing_percent = slope * (pnl_percent_notional - 1.5%) + sl_percent_trailing
                                            = -50 * (1.5-1.5) + 50%
                                            = 0 + 50
                                            = 50% (Most loose)

Case 2. pnl_percent_notional = 2% (Deeper into profit, +200bps pnl)
            effective_tp_trailing_percent = slope * (pnl_percent_notional - 1.5%) + sl_percent_trailing
                                            = -50 * (2-1.5) +50%
                                            = -25 + 50
                                            = 25% (Somewhat tight)

Case 3. pnl_percent_notional = 2.5% (Very deep in profit, +250bps pnl. i.e. max TP)
            effective_tp_trailing_percent = slope * (pnl_percent_notional - 1.5%) + sl_percent_trailing
                                            = -50 * (2.5-1.5) +50%
                                            = -50 + 50
                                            = 0 (Most tight)

So you see, effective_tp_trailing_percent gets smaller and smaller as pnl approach max TP, finally zero.

How to use it?
    if loss_trailing>=effective_tp_trailing_percent and pnl_percent_notional > tp_min_percent:
            Fire trailing stops and take profit.

What's 'loss_trailing'? 'loss_trailing' is essentially pnl drop from max_unrealized_pnl_live.

    Say, when trade started off:
        unrealized_pnl_live = $80
        max_unrealized_pnl_live = $100
        loss_trailing = (1 - unrealized_pnl_live/max_unrealized_pnl_live) = (1-80/100) = 0.2 (Or 20%)

    If pnl worsen:
        unrealized_pnl_live = $40
        max_unrealized_pnl_live = $100
        loss_trailing = (1 - unrealized_pnl_live/max_unrealized_pnl_live) = (1-40/100) = 0.6 (Or 60%)

Have a look at this for a visual explaination how "Gradually tightened stops" works:
    https://github.com/r0bbar/siglab/blob/master/siglab_py/tests/manual/trading_util_tests.ipynb
    https://norman-lm-fung.medium.com/gradually-tightened-trailing-stops-f7854bf1e02b
'''
def calc_eff_trailing_sl(
        tp_min_percent : float,
        tp_max_percent : float,
        sl_percent_trailing : float,
        pnl_percent_notional : float,
        default_effective_tp_trailing_percent : float = float('inf'), # inf: essentially saying, don't fire off trailing stop.
        linear : bool = True,
        pow : float = 5 # This is for non-linear trailing stops
) -> float:
    if pnl_percent_notional>tp_max_percent:
        return 0
    if pnl_percent_notional<tp_min_percent:
         return default_effective_tp_trailing_percent
    
    if linear:
        slope = (0 - sl_percent_trailing) / (tp_max_percent - tp_min_percent)
        effective_tp_trailing_percent = (
                                            slope * (pnl_percent_notional - tp_min_percent) + sl_percent_trailing 
                                            if pnl_percent_notional>=tp_min_percent 
                                            else default_effective_tp_trailing_percent
                                        )
    else:    
        def y(
				x : float,
				x_shift : float,
				pow : float
				) -> float:
            return -1 * ( (x+x_shift)**pow)
        
        y_min = y(
												x=tp_min_percent,
												x_shift=tp_min_percent,
												pow=pow
												)
        y_max = y(
												x=tp_max_percent,
												x_shift=tp_min_percent,
												pow=pow
												)
        y_shift = abs(y_max) - abs(y_min)
        
        if y_shift!=0:
            y_normalized = y(
                                                        x=pnl_percent_notional,
                                                        x_shift=tp_min_percent,
                                                        pow=pow
                                                    ) / y_shift
            effective_tp_trailing_percent = (
                                                y_normalized * sl_percent_trailing + sl_percent_trailing
                                                if pnl_percent_notional>=tp_min_percent 
                                                else default_effective_tp_trailing_percent
                                            )
        else:
            '''
            y_shift = 0 when tp_max_percent==tp_min_percent.
            If default_effective_tp_trailing_percent==float('inf'), essentially it means trailing stops won't fire.
            Client side needs handle this.
            '''
            effective_tp_trailing_percent = default_effective_tp_trailing_percent

    return effective_tp_trailing_percent

def detect_sweeps_and_absorption(
    tape: pd.DataFrame,

    volume_threshold_base_ccy : float,

    burst_gap_sec: float = 0.05,

    # percentile thresholds
    sweep_vol_pct: float = 0.90,
    sweep_move_pct: float = 0.90,
    absorb_vol_pct: float = 0.90,
    absorb_move_pct: float = 0.25,

    reference_level: float = None,
    level_tolerance_pct: float = 0.001,
) -> dict:
    def _burst_stats(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "start_time": g["datetime"].iloc[0],
            "end_time": g["datetime"].iloc[-1],
            "start_price": g["price"].iloc[0],
            "end_price": g["price"].iloc[-1],
            "volume": g["amount_base_ccy"].sum(),
            "buy_vol": g.loc[g["side"] == "buy", "amount_base_ccy"].sum(),
            "sell_vol": g.loc[g["side"] == "sell", "amount_base_ccy"].sum(),
            "num_trades": len(g),
        })

    tape = tape.copy().sort_values("datetime").reset_index(drop=True)

    if "side" not in tape.columns:
        raise ValueError("Tape must contain a 'side' column ('buy'/'sell').")
    if not set(tape["side"].dropna().unique()).issubset({"buy", "sell"}):
        raise ValueError("'side' must only contain 'buy' or 'sell'.")

    # Step 1: burst construction via time gaps
    tape["time_gap"] = tape["datetime"].diff().dt.total_seconds()
    tape["burst_id"] = (tape["time_gap"] > burst_gap_sec).cumsum()

    bursts = (
                tape.groupby("burst_id", group_keys=False)
                .apply(_burst_stats, include_groups=False)
                .reset_index()
            )

    bursts["price_move"] = (bursts["end_price"] - bursts["start_price"]).abs()
    bursts["price_move_bps"] = bursts["price_move"]/bursts["start_price"] *10000
    bursts["vol_rank"] = bursts["volume"].rank(pct=True)
    bursts["move_rank"] = bursts["price_move"].rank(pct=True)

    # Step 2a: sweeps
    sweeps = bursts[
        (bursts["vol_rank"] > sweep_vol_pct)
        & (bursts["move_rank"] > sweep_move_pct)
        & bursts['volume'] > volume_threshold_base_ccy
    ].copy()
    sweeps["direction"] = np.where(sweeps["buy_vol"] > sweeps["sell_vol"], "up", "down")
    
    # Step 2b: absorption
    absorption = bursts[
        (bursts["vol_rank"] > absorb_vol_pct)
        & (bursts["move_rank"] < absorb_move_pct)
        & bursts['volume'] > volume_threshold_base_ccy
    ].copy()
    absorption["absorbed_side"] = np.where(
        absorption["buy_vol"] > absorption["sell_vol"],
        "buyers_absorbed",
        "sellers_absorbed",
    )

    # Step 3: reference-level tagging
    if reference_level is not None:
        tol = reference_level * level_tolerance_pct
        sweeps["near_reference"] = (sweeps["end_price"] - reference_level).abs() < tol
        absorption["near_reference"] = (absorption["end_price"] - reference_level).abs() < tol
    else:
        sweeps["near_reference"] = False
        absorption["near_reference"] = False

    return {
        "tape": tape,
        "bursts": bursts,
        "sweeps": sweeps,
        "absorption": absorption,
    }

if __name__ == '__main__':
    from datetime import datetime, timedelta
    import time
    import logging

    import ccxt
    from siglab_py.util.market_data_util import fetch_tape, fetch_candles
    from siglab_py.util.analytic_util import compute_candles_stats

    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger()
    log_level = logging.INFO # DEBUG --> INFO --> WARNING --> ERROR
    logger.setLevel(log_level)
    format_str = '%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
    exchange.load_markets()
    symbol = "BTC/USDT:USDT"

    dt_end = datetime.now()
    dt_start = dt_end + timedelta(days=-365)

    candles = fetch_candles(
        exchange=exchange,
        normalized_symbols=[symbol],
        start_ts=dt_start.timestamp(),
        end_ts=dt_end.timestamp(),
        candle_size='1d'
    )
    pd_candles = candles[symbol]
    compute_candles_stats(
        pd_candles=pd_candles,
        boillenger_std_multiples=2,
        sliding_window_how_many_candles=int(365/12)
    )

    while True:
        _pd_candles = pd_candles[
            (pd_candles['dayofweek']==dt_end.weekday()) # You can't compare Monday against Fri after close
        ]
        avg_daily_volume_base_ccy = _pd_candles['volume'].mean()
        daily_volume_std_base_ccy = _pd_candles['volume'].std()
        daily_volume_threshold_base_ccy = avg_daily_volume_base_ccy + daily_volume_std_base_ccy
        volume_threshold_base_ccy = daily_volume_threshold_base_ccy / (24*4)
    
        start_fetch_ts = time.time()
        tape = fetch_tape(exchange=exchange, symbol=symbol, max_batches=10, limit=1000)
        finish_fetch_ts = time.time()
        elapsed_sec = round(finish_fetch_ts - start_fetch_ts)
        logger.info(f"elapsed_sec: {elapsed_sec}")

        if tape.empty:
            logger.info("No trades fetched — check symbol or network.")
            raise SystemExit(0)

        logger.info(f"\nFetched {len(tape)} unique trades.")
        logger.info(f"Time range: {tape['datetime'].min()}  →  {tape['datetime'].max()}")
        logger.info(f"Last price: {tape['price'].iloc[-1]:.2f}")

        result = detect_sweeps_and_absorption(
            tape,
            volume_threshold_base_ccy=volume_threshold_base_ccy,
            reference_level=float(tape["price"].iloc[-1]),
            level_tolerance_pct=0.001,
        )

        bursts = result["bursts"]
        residual = (bursts["buy_vol"] + bursts["sell_vol"] - bursts["volume"]).abs().max()
        logger.info(f"Bursts:      {len(bursts)}  (max volume residual: {residual:.2e})")
        logger.info(f"Sweeps:      {len(result['sweeps'])}")
        logger.info(f"Absorption:  {len(result['absorption'])}\n")

        if not result["sweeps"].empty:
            logger.info("Top sweeps by volume:")
            top_sweeps = (
                result["sweeps"]
                .sort_values("volume", ascending=False)
                .head(5)
                .sort_values("start_time")
            )
            logger.info(
                top_sweeps[["start_time", "end_price", "volume", "direction", "near_reference"]]
                .to_string(index=False)
            )

        if not result["absorption"].empty:
            logger.info("Top absorption by volume:")
            top_absorption = (
                result["absorption"]
                .sort_values("volume", ascending=False)
                .head(5)
                .sort_values("start_time")
            )
            logger.info(
                top_absorption[["start_time", "end_price", "volume", "absorbed_side", "near_reference"]]
                .to_string(index=False)
            )