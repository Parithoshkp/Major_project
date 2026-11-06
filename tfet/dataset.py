"""Builders for the datasets used in the paper (Section III) plus an extended one.

Every dataset is a long-format DataFrame: one row per (curve, V_G) point, with a
`curve_id` column so that splits can be done per curve instead of per point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .physics import NOMINAL, PARAM_NAMES, PARAM_RANGES, id_vg

V_G_SWEEP = np.linspace(0.0, 0.5, 51)
SOLVER_NOISE = 0.005        # 0.5 % multiplicative noise, mimics TCAD convergence noise


def simulate(devices: pd.DataFrame, V_G=V_G_SWEEP, noise=SOLVER_NOISE, seed=0) -> pd.DataFrame:
    """Run the simulator for every row of `devices` (columns: L_ov, WK, t_n, t_ox, [V_D], [T])."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid, dev in enumerate(devices.to_dict("records")):
        V_D, T = dev.get("V_D", 0.5), dev.get("T", 300.0)
        _, I = id_vg(dev, V_G=V_G, V_D=V_D, T=T, noise=noise, rng=rng)
        rows.append(pd.DataFrame({"curve_id": cid, **{k: dev[k] for k in PARAM_NAMES},
                                  "V_D": V_D, "T": T, "V_G": V_G, "I_D": I}))
    return pd.concat(rows, ignore_index=True)


def latin_hypercube(n, ranges: dict, seed=0) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
    lo, hi = zip(*ranges.values())
    return pd.DataFrame(qmc.scale(sampler.random(n), lo, hi), columns=list(ranges))


def one_at_a_time(n_per_param=100) -> pd.DataFrame:
    """Dataset (1): vary one parameter over its Table-1 range, others at nominal -> 400 curves."""
    out = []
    for name, (lo, hi) in PARAM_RANGES.items():
        dev = pd.DataFrame([NOMINAL] * n_per_param)
        dev[name] = np.linspace(lo, hi, n_per_param)
        dev["varied"] = name
        out.append(dev)
    return pd.concat(out, ignore_index=True)


def lov_tn_pairs(n=200, seed=1) -> pd.DataFrame:
    """Dataset (2): L_ov and t_n varied together, WK and t_ox fixed -> 200 curves."""
    dev = latin_hypercube(n, {k: PARAM_RANGES[k] for k in ("L_ov", "t_n")}, seed)
    dev["WK"], dev["t_ox"] = NOMINAL["WK"], NOMINAL["t_ox"]
    return dev[PARAM_NAMES]


def with_drain_bias(n=300, V_D=(0.5, 0.05, 0.005), seed=2) -> pd.DataFrame:
    """Dataset (3): random devices, each simulated at several drain voltages."""
    dev = latin_hypercube(n, PARAM_RANGES, seed)
    return pd.concat([dev.assign(V_D=v) for v in V_D], ignore_index=True)


def all_parameters(n=2500, seed=3) -> pd.DataFrame:
    """Dataset (4): ~2500 devices with all four parameters varied (V_D = 0.5 V)."""
    return latin_hypercube(n, PARAM_RANGES, seed)


def extended(n=3000, seed=4) -> pd.DataFrame:
    """Extension: also vary V_D (0.05-0.5 V) and temperature T (250-400 K) -- the PVT
    variation the paper lists as future work."""
    return latin_hypercube(n, {**PARAM_RANGES, "V_D": (0.05, 0.5), "T": (250.0, 400.0)}, seed)
