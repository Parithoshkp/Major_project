"""Shared evaluation helpers so every notebook scores models the same way."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .metrics import figures_of_merit
from .physics import PARAM_NAMES


def curve_split(df: pd.DataFrame, test_frac=0.2, seed=42):
    """Split by whole curves (no V_G point of a test curve is ever seen in training)."""
    ids = df.curve_id.unique()
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, size=int(round(test_frac * len(ids))), replace=False)
    mask = df.curve_id.isin(test_ids)
    return df[~mask].copy(), df[mask].copy()


def regression_report(y_true, y_pred) -> dict:
    """The paper's metrics (linear-scale R2, RMSE as % of std) plus log-scale ones."""
    y_true, y_pred = np.asarray(y_true), np.clip(np.asarray(y_pred), 1e-20, None)
    rel = np.abs(y_pred - y_true) / y_true
    return {
        "R2_linear_%": 100 * r2_score(y_true, y_pred),
        "RMSE_%std": 100 * np.sqrt(np.mean((y_pred - y_true) ** 2)) / y_true.std(),
        "R2_log10_%": 100 * r2_score(np.log10(y_true), np.log10(y_pred)),
        "median_rel_err_%": 100 * np.median(rel),
        "p95_rel_err_%": 100 * np.percentile(rel, 95),
    }


def error_by_decade(y_true, y_pred) -> pd.DataFrame:
    """Median relative error binned by the decade of the true current."""
    y_true, y_pred = np.asarray(y_true), np.clip(np.asarray(y_pred), 1e-20, None)
    dec = np.floor(np.log10(y_true)).astype(int)
    rel = 100 * np.abs(y_pred - y_true) / y_true
    return (pd.DataFrame({"decade": dec, "rel_err_%": rel})
            .groupby("decade")["rel_err_%"].agg(["median", "count"]).reset_index())


def fom_per_curve(df: pd.DataFrame, col="I_D") -> pd.DataFrame:
    """Figures of merit for every curve in a long-format frame."""
    rows = []
    for cid, c in df.groupby("curve_id", sort=True):
        c = c.sort_values("V_G")
        rows.append({"curve_id": cid, **figures_of_merit(c.V_G.values, c[col].values)})
    return pd.DataFrame(rows).set_index("curve_id")


def fom_errors(df: pd.DataFrame, pred_col="I_pred") -> dict:
    """Mean absolute error of the FoMs extracted from predicted vs true curves."""
    t, p = fom_per_curve(df, "I_D"), fom_per_curve(df, pred_col)
    both = lambda k: (t[k].notna() & p[k].notna() & np.isfinite(t[k]) & np.isfinite(p[k]))
    m_th, m_ss = both("V_th"), both("SS_min")
    # NaN error = the FoM could not be extracted from any predicted curve
    mae = lambda a, b, m: np.mean(np.abs(a[m] - b[m])) if m.any() else np.nan
    return {
        "I_on_err_%": 100 * np.mean(np.abs(p.I_on - t.I_on) / t.I_on),
        "I_off_err_decades": np.mean(np.abs(np.log10(p.I_off) - np.log10(t.I_off))),
        "V_th_err_mV": 1e3 * mae(p.V_th, t.V_th, m_th),
        "V_th_missing_%": 100 * np.mean(t.V_th.notna() & ~m_th),
        "SS_min_err_mV/dec": mae(p.SS_min, t.SS_min, m_ss),
        "SS_min_missing_%": 100 * np.mean(t.SS_min.notna() & ~m_ss),
    }


def monotonic_violations(df: pd.DataFrame, pred_col="I_pred") -> float:
    """% of curves whose predicted I_D decreases somewhere along V_G (unphysical)."""
    bad = df.sort_values(["curve_id", "V_G"]).groupby("curve_id")[pred_col].apply(
        lambda s: bool((np.diff(s.values) < -1e-3 * np.abs(s.values[:-1])).any()))
    return 100 * bad.mean()


FEATURES = ["V_G"] + PARAM_NAMES          # the paper's five inputs
