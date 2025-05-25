
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
        default_effective_tp_trailing_percent : float = 50
) -> float:
    slope = (0 - sl_percent_trailing) / (tp_max_percent - tp_min_percent)
    effective_tp_trailing_percent = (
                                        slope * (pnl_percent_notional - tp_min_percent) + sl_percent_trailing 
                                        if pnl_percent_notional>tp_min_percent 
                                        else default_effective_tp_trailing_percent
                                    )
    return effective_tp_trailing_percent
