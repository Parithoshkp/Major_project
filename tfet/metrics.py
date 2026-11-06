"""Figure-of-merit (FoM) extraction from I_D-V_G curves."""
from __future__ import annotations

import numpy as np

I_TH = 1e-10   # constant-current threshold criterion [A/um]


def subthreshold_swing(V_G, I_D):
    """Point-wise SS = dV_G / dlog10(I_D) in mV/dec (midpoints of the sweep)."""
    logI = np.log10(np.clip(I_D, 1e-30, None))
    dlog = np.diff(logI)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = np.diff(V_G) / dlog * 1e3
    ss[dlog <= 0] = np.inf
    return ss


def figures_of_merit(V_G, I_D, I_th=I_TH):
    """Return I_ON, I_OFF, ON/OFF ratio, V_th, SS_min and SS_avg for one curve.

    SS_min: steepest point of the curve inside the subthreshold region
            (between 10x the off current and I_th*1e3).
    SS_avg: from I_OFF*10 to I_th (the usual TFET definition, averaged over the swing).
    """
    V_G, I_D = np.asarray(V_G), np.asarray(I_D)
    I_on, I_off = I_D[-1], I_D[0]
    logI = np.log10(np.clip(I_D, 1e-30, None))

    # V_th by constant current (interpolate in log space, curve must cross I_th)
    if logI.max() > np.log10(I_th) > logI.min():
        idx = np.argmax(logI >= np.log10(I_th))
        v_th = np.interp(np.log10(I_th), logI[idx - 1: idx + 1], V_G[idx - 1: idx + 1])
    else:
        v_th = np.nan

    ss = subthreshold_swing(V_G, I_D)
    I_mid = np.sqrt(I_D[:-1] * I_D[1:])
    mask = (I_mid > 10 * I_off) & (I_mid < I_th * 1e3)
    ss_min = ss[mask].min() if mask.any() else np.nan

    lo, hi = np.log10(10 * I_off), np.log10(I_th)
    if logI.max() > hi and hi > lo:
        v_lo = np.interp(lo, np.maximum.accumulate(logI), V_G)
        v_hi = np.interp(hi, np.maximum.accumulate(logI), V_G)
        ss_avg = (v_hi - v_lo) / (hi - lo) * 1e3
    else:
        ss_avg = np.nan

    return {"I_on": I_on, "I_off": I_off, "on_off": I_on / I_off,
            "V_th": v_th, "SS_min": ss_min, "SS_avg": ss_avg}
