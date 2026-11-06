"""Analytical (compact) model of the scaled line TFET (SLTFET) from Akbar et al., IEEE Access 2022.

The paper's data came from 3-D TCAD (Sentaurus-class). That is not freely available,
so this module stands in as a cheap "device simulator". It keeps the physics the paper
describes in Section II:

  * vertical BTBT through the source/epitaxy overlap, area  A_tun = L_ov * 2 (W + t_n)
  * lateral BTBT at the source/channel junction (exists even when L_ov -> 0)
  * tunnelling length  lambda = sqrt(eps_ns * t_ox * t_n / eps_ox)
  * Kane-type generation  ~ F^2 exp(-B / F)  with the paper's Si0.6Ge0.4 B parameter
  * work function WK shifts the onset voltage (higher WK -> later turn-on, lower I_ON)
  * trap-assisted tunnelling (TAT) sets the off-state floor
  * temperature dependence (Varshni band gap + thermally activated TAT)  [extension]

The absolute numbers are calibrated to the ranges shown in the paper's figures
(I_ON ~ 1e-6..4e-5 A/um at V_D = V_G = 0.5 V, I_OFF ~ 1e-14 A/um, SS_min ~ 30-40 mV/dec).
It is a teaching/prototyping model, not a replacement for TCAD.
"""
from __future__ import annotations

import numpy as np

# ---- physical constants and material parameters ---------------------------------
K_B = 8.617333e-5          # Boltzmann constant [eV/K]
EPS_NS = 11.9              # nanosheet (Si) relative permittivity
EPS_OX = 22.0              # HfO2 gate oxide relative permittivity
W_SHEET = 14.0             # nanosheet width [nm] (paper: effective width > 14 nm)

# Si0.6Ge0.4 indirect-path Kane parameter from the paper (16.8 MV/cm)
B_KANE = 16.8              # [MV/cm]
EG_300 = 0.95              # band gap of Si0.6Ge0.4 at 300 K [eV]
VARSHNI_A, VARSHNI_B = 4.73e-4, 636.0   # Varshni coefficients (Si-like)

CAL = 1.0                  # BTBT calibration constant (nominal I_ON ~ 2.2e-5 A/um)

# ---- parameter ranges (Table 1 of the paper) ------------------------------------
PARAM_RANGES = {
    "L_ov": (0.1, 10.0),   # source overlap length [nm]
    "WK":   (4.2, 4.5),    # gate work function [eV]
    "t_n":  (1.0, 5.0),    # epitaxy thickness [nm]
    "t_ox": (1.0, 6.0),    # oxide thickness [nm]
}
NOMINAL = {"L_ov": 5.0, "WK": 4.4, "t_n": 2.0, "t_ox": 3.0}
PARAM_NAMES = list(PARAM_RANGES)


def band_gap(T):
    """Varshni temperature dependence, anchored to EG_300 at 300 K."""
    shift = lambda t: VARSHNI_A * t**2 / (t + VARSHNI_B)
    return EG_300 + shift(300.0) - shift(T)


def tunnelling_length(t_ox, t_n):
    """lambda = sqrt(eps_ns * t_ox * t_n / eps_ox)  [nm] (paper, Section II-C)."""
    return np.sqrt(EPS_NS * t_ox * t_n / EPS_OX)


def _softplus(x):
    return np.logaddexp(0.0, x)


def drain_current(V_G, L_ov, WK, t_n, t_ox, V_D=0.5, T=300.0):
    """Drain current I_D [A/um] of the line TFET.

    All arguments broadcast, so you can pass a V_G sweep with scalar device
    parameters, or flat arrays of equal length (one row per sample).
    """
    V_G, L_ov, WK, t_n, t_ox, V_D, T = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (V_G, L_ov, WK, t_n, t_ox, V_D, T))
    )
    kT = K_B * T
    Eg = band_gap(T)
    B = B_KANE * (Eg / EG_300) ** 1.5           # Kane B scales as Eg^1.5

    # gate-to-junction coupling: thicker oxide / thinner epitaxy -> weaker control
    m = 1.0 + 0.12 * (EPS_NS / EPS_OX) * t_ox / t_n
    # onset voltage: set by work function, lowered slightly by drain bias
    V_on = (WK - 4.20) - 0.04 * V_D
    # energy window that opens for tunnelling (smoothed band overlap), in volts
    s = 0.0125 * (T / 300.0) ** 0.5             # tail steepness of the window
    psi = s * _softplus((V_G - V_on) / (m * s))

    lam = tunnelling_length(t_ox, t_n)          # [nm]
    # effective tunnel path: the paper's lambda is a scale length; the real path
    # through the p++/n junction is longer and only partly gate-controlled
    lam = 3.0 + 2.0 * lam

    # vertical tunnelling (through L_ov), field across lambda in MV/cm
    F_v = (Eg + psi) / (lam * 1e-7) / 1e6
    A_v = L_ov * 2.0 * (W_SHEET + t_n) / 1e3    # normalised area
    I_v = A_v * psi * F_v**2 * np.exp(-B / F_v)

    # lateral tunnelling (source/channel), longer path, small area
    lam_l = 1.35 * lam
    F_l = (Eg + psi) / (lam_l * 1e-7) / 1e6
    A_l = t_n * W_SHEET / 1e3
    I_l = A_l * psi * F_l**2 * np.exp(-B / F_l)

    I_btbt = CAL * (I_v + I_l)

    # trap-assisted tunnelling floor (thermally activated, weak gate dependence)
    I_tat = 3.0e-9 * (1.0 + 0.05 * L_ov) * np.exp(-0.30 / kT) \
        * np.exp((V_G - V_on) / (m * 6.0 * kT * (T / 300.0)))
    I_tat = np.minimum(I_tat, 1e-11)

    # drain-voltage dependence (linear -> saturation)
    drain = 1.0 - np.exp(-V_D / 0.1)
    return (I_btbt + I_tat) * drain


def id_vg(params: dict, V_G=None, V_D=0.5, T=300.0, noise=0.0, rng=None):
    """One I_D-V_G sweep. `noise` adds multiplicative log-normal 'solver noise'."""
    if V_G is None:
        V_G = np.linspace(0.0, 0.5, 51)
    I = drain_current(V_G, params["L_ov"], params["WK"], params["t_n"], params["t_ox"], V_D, T)
    if noise:
        rng = np.random.default_rng() if rng is None else rng
        I = I * np.exp(rng.normal(0.0, noise, size=I.shape))
    return np.asarray(V_G), I
