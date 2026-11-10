"""Line-TFET ML surrogate builder (Streamlit).

  1. Data   : upload TCAD I_D-V_G curves, or (no TCAD) generate them with our analytical simulator
  2. Train  : fit a neural network on log10(I_D) (optionally also the paper's Random Forest),
              scored on held-out curves the model never saw
  3. Predict: pick new device parameters and get the predicted I_D-V_G curve + figures of merit

Run from the project root:
    .venv/bin/streamlit run app/app.py
"""
from __future__ import annotations

import hashlib
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfet import NOMINAL, PARAM_NAMES, PARAM_RANGES, drain_current, figures_of_merit  # noqa: E402
from tfet.dataset import latin_hypercube, simulate  # noqa: E402
from tfet.evaluate import curve_split, fom_errors, regression_report  # noqa: E402
from tfet.metrics import I_TH, subthreshold_swing  # noqa: E402
from tfet.surrogate import LogCurrentSurrogate  # noqa: E402

UNITS = {"L_ov": "nm", "WK": "eV", "t_n": "nm", "t_ox": "nm", "V_D": "V", "T": "K"}
PHYSICS_COLS = {"L_ov", "t_n", "t_ox"}          # needed for the lambda / A_tun features

st.set_page_config(page_title="TFET Surrogate Builder", page_icon="📈", layout="wide")


# =============================================================================== data
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\[.*?\]|\(.*?\)", "", str(name).lower()))


ALIASES = {
    "V_G": {"vg", "vgs", "gatevoltage", "vgate"},
    "I_D": {"id", "ids", "draincurrent", "idrain", "current"},
    "curve_id": {"curveid", "curve", "label", "device", "run", "name"},
    "L_ov": {"lov"}, "WK": {"wk", "wf", "workfunction"}, "t_n": {"tn", "tepi"},
    "t_ox": {"tox"}, "V_D": {"vd", "vds", "drainvoltage"}, "T": {"t", "temp", "temperature"},
}


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns (Vg, Ids, tox, ...) to the names the code uses."""
    ren = {}
    for c in df.columns:
        for std, al in ALIASES.items():
            if _norm(c) in al | {_norm(std)} and std not in ren.values():
                ren[c] = std
    return df.rename(columns=ren)


def prepare_upload(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Long-format frame with curve_id, input columns, V_G, I_D (>0) + list of candidate inputs."""
    df = standardise_columns(raw)
    for need in ("V_G", "I_D"):
        if need not in df.columns:
            raise ValueError(f"No {need} column found (accepted names: {need}, "
                             f"{', '.join(sorted(ALIASES[need]))}).")
    num = [c for c in df.columns if c not in ("V_G", "I_D", "curve_id")
           and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    if not num:
        raise ValueError("No device-parameter columns that vary (e.g. L_ov, WK, t_n, t_ox). "
                         "The model needs them as inputs to learn how the curve changes.")
    if "curve_id" not in df.columns:            # one curve = one unique parameter combination
        df["curve_id"] = df.groupby(num, sort=False).ngroup()
    df["curve_id"] = pd.factorize(df["curve_id"])[0]
    df["I_D"] = df["I_D"].abs()
    df = df[df["I_D"] > 0].dropna(subset=["V_G", "I_D", *num])
    return df.sort_values(["curve_id", "V_G"]).reset_index(drop=True), num


@st.cache_data(show_spinner="Running the analytical simulator...")
def generate_data(n_curves: int, vary_bias: bool, noise: float, seed: int) -> pd.DataFrame:
    ranges = dict(PARAM_RANGES)
    if vary_bias:
        ranges |= {"V_D": (0.05, 0.5), "T": (250.0, 400.0)}
    return simulate(latin_hypercube(n_curves, ranges, seed), noise=noise, seed=seed)


def template_csv() -> str:
    dev = latin_hypercube(5, PARAM_RANGES, seed=7)
    return simulate(dev, V_G=np.linspace(0, 0.5, 26), noise=0)[
        ["curve_id", *PARAM_NAMES, "V_G", "I_D"]].to_csv(index=False)


# =============================================================================== training
def train_models(df, inputs, physics, steps, test_frac, with_rf, seed=0):
    train, test = curve_split(df, test_frac, seed=42)
    cols = ["V_G", *inputs]
    X, Xt = train[cols].to_numpy(float), test[cols].to_numpy(float)

    t0 = time.perf_counter()
    batch = 4096
    epochs = int(np.ceil(steps / np.ceil(len(X) / batch)))
    nn_model = LogCurrentSurrogate(cols, physics=physics, seed=seed).fit(
        X, train.I_D.to_numpy(), epochs=epochs, batch=batch)
    out = {"nn": nn_model, "nn_s": time.perf_counter() - t0, "cols": cols,
           "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in cols},
           "n_train": train.curve_id.nunique(), "n_test": test.curve_id.nunique()}

    test = test.copy()
    scores = {}
    if len(test):
        test["I_nn"] = nn_model.predict(Xt)
        scores["Neural net (log I_D)"] = score(test, "I_nn")
    if with_rf:
        from sklearn.ensemble import RandomForestRegressor
        t0 = time.perf_counter()
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, train.I_D)
        out["rf"], out["rf_s"] = rf, time.perf_counter() - t0
        if len(test):
            test["I_rf"] = np.clip(rf.predict(Xt), 1e-20, None)
            scores["Random Forest (paper, linear I_D)"] = score(test, "I_rf")
    out["scores"], out["test"] = pd.DataFrame(scores).T, test
    return out


def score(test, col):
    r = regression_report(test.I_D, test[col])
    f = fom_errors(test, col)
    return {"median error [%]": r["median_rel_err_%"], "95th pct error [%]": r["p95_rel_err_%"],
            "R² linear [%]": r["R2_linear_%"], "R² log [%]": r["R2_log10_%"],
            "I_ON error [%]": f["I_on_err_%"], "I_OFF error [dec]": f["I_off_err_decades"],
            "V_th error [mV]": f["V_th_err_mV"], "SS_min error [mV/dec]": f["SS_min_err_mV/dec"]}


def predict_curve(res, which, params, V):
    X = np.column_stack([V] + [np.full_like(V, params[c]) for c in res["cols"][1:]])
    if which == "rf":
        return np.clip(res["rf"].predict(X), 1e-20, None)
    return res["nn"].predict(X)


# =============================================================================== plotting
def curve_fig(curves: dict, kind="log") -> go.Figure:
    """curves: {name: (V_G, I_D, dash)}"""
    fig = go.Figure()
    for name, (V, I, dash) in curves.items():
        if kind == "ss":
            ss = subthreshold_swing(V, I)
            x, y = np.sqrt(I[:-1] * I[1:]), np.where(np.isfinite(ss) & (ss < 500), ss, np.nan)
        elif kind == "gm":
            x, y = V, np.gradient(I, V)
        else:
            x, y = V, I
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                                 line=dict(dash=dash or "solid")))
    axes = {"log": ("V_G [V]", "I_D [A/µm]"), "linear": ("V_G [V]", "I_D [A/µm]"),
            "gm": ("V_G [V]", "g_m [S/µm]"), "ss": ("I_D [A/µm]", "SS [mV/dec]")}[kind]
    fig.update_layout(xaxis_title=axes[0], yaxis_title=axes[1], height=450,
                      margin=dict(l=10, r=10, t=30, b=10))
    if kind == "log":
        fig.update_yaxes(type="log", exponentformat="e")
        fig.add_hline(y=I_TH, line_dash="dot", line_color="gray", annotation_text="I_th (V_th)")
    if kind == "ss":
        fig.update_xaxes(type="log", exponentformat="e")
        fig.add_hline(y=60, line_dash="dot", line_color="gray", annotation_text="60 mV/dec")
    return fig


def curve_tabs(curves: dict):
    tabs = st.tabs(["log I_D–V_G", "linear I_D–V_G", "g_m", "SS"])
    for tab, kind in zip(tabs, ["log", "linear", "gm", "ss"]):
        tab.plotly_chart(curve_fig(curves, kind), width="stretch")


def fom_table(curves: dict) -> pd.DataFrame:
    rows = {}
    for name, (V, I, _) in curves.items():
        f = figures_of_merit(V, I)
        rows[name] = {"I_ON [A/µm]": f["I_on"], "I_OFF [A/µm]": f["I_off"],
                      "ON/OFF": f["on_off"], "V_th [V]": f["V_th"],
                      "SS_min [mV/dec]": f["SS_min"], "SS_avg [mV/dec]": f["SS_avg"]}
    return pd.DataFrame(rows).T


# =============================================================================== page
st.title("📈 TFET Surrogate Builder")
st.caption("Give it I_D–V_G curves (your TCAD runs, or our simulator if you have none). "
           "It trains a machine-learning model on them and then predicts the curve of "
           "any new device in milliseconds.")

# ------------------------------------------------------------------ 1. data
st.header("1 · Data")
source = st.radio("Where do the training curves come from?",
                  ["Upload my TCAD data", "No TCAD data: generate with our Python simulator"],
                  horizontal=True)
sim_mode = source.startswith("No TCAD")

if not sim_mode:
    c1, c2 = st.columns([3, 2])
    with c2:
        st.markdown("**Expected format** (CSV / TXT, long format, one row per point):\n\n"
                    "`curve_id, L_ov, WK, t_n, t_ox, V_G, I_D`\n\n"
                    "- any device / bias columns you swept work as inputs (e.g. `V_D`, `T`)\n"
                    "- `curve_id` is optional\n- units: V and A/µm")
        st.download_button("Download a template CSV", template_csv(), "tfet_template.csv")
    with c1:
        file = st.file_uploader("TCAD I_D–V_G data", type=["csv", "txt", "dat", "tsv"])
    if file is None:
        st.info("Upload a file to continue, or choose the simulator option above.")
        st.stop()
    try:
        raw = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8", "replace")),
                          sep=None, engine="python", comment="#")
        df, candidates = prepare_upload(raw)
    except Exception as e:                       # show the reason to the user
        st.error(f"Could not use this file: {e}")
        st.stop()
    inputs = st.multiselect("Input columns (device / bias parameters the model learns from)",
                            candidates, default=candidates)
    if not inputs:
        st.stop()
else:
    c1, c2, c3 = st.columns(3)
    n_curves = c1.slider("Number of simulated curves", 50, 1500, 300, 50)
    noise = c2.slider("Simulation noise [%]", 0.0, 5.0, 0.5, 0.1) / 100
    vary_bias = c3.checkbox("Also vary V_D and temperature", False)
    df = generate_data(n_curves, vary_bias, noise, seed=0)
    inputs = PARAM_NAMES + (["V_D", "T"] if vary_bias else [])
    st.caption("Data from our analytical line-TFET model (Kane BTBT, calibrated to "
               "Akbar et al. 2022), not from TCAD.")

n_c = df.curve_id.nunique()
st.write(f"**{n_c} curves**, {len(df)} points, inputs: `V_G`, "
         + ", ".join(f"`{c}`" for c in inputs))
with st.expander("Preview the data"):
    st.dataframe(df.head(200), width="stretch")
    show = df[df.curve_id.isin(df.curve_id.unique()[:8])]
    st.plotly_chart(curve_fig({f"curve {i}": (g.V_G.to_numpy(), g.I_D.to_numpy(), None)
                               for i, g in show.groupby("curve_id")}), width="stretch")
if n_c < 5:
    st.warning("Fewer than 5 curves: the model can't learn how parameters change the curve.")
    st.stop()

# ------------------------------------------------------------------ 2. train
st.header("2 · Train")
c1, c2, c3, c4 = st.columns(4)
steps = c1.select_slider("Training steps", [500, 1000, 2000, 3000, 5000], 2000,
                         help="More steps = more accurate but slower.")
test_frac = c2.slider("Held-out test curves [%]", 10, 40, 20, 5) / 100
can_phys = PHYSICS_COLS <= set(inputs)
physics = c3.checkbox("Physics features (λ, A_tun)", can_phys, disabled=not can_phys,
                      help="Adds the tunnelling length and tunnelling area as extra inputs. "
                           "Needs L_ov, t_n and t_ox.")
with_rf = c4.checkbox("Also train the paper's Random Forest", True)

key = hashlib.md5(pd.util.hash_pandas_object(df[["curve_id", "V_G", "I_D", *inputs]])
                  .values.tobytes() + str((inputs, steps, test_frac, physics, with_rf))
                  .encode()).hexdigest()
if st.button("🚀 Train model", type="primary"):
    with st.spinner("Training... (usually under a minute)"):
        st.session_state.res, st.session_state.key = train_models(
            df, inputs, physics, steps, test_frac, with_rf), key
res = st.session_state.get("res")
if res is None or st.session_state.get("key") != key:
    st.info("Press **Train model** (settings or data changed since the last training)."
            if res is not None else "Press **Train model**.")
    st.stop()

st.success(f"Trained on {res['n_train']} curves in {res['nn_s']:.1f} s; "
           f"tested on {res['n_test']} curves it never saw.")
if len(res["scores"]):
    st.dataframe(res["scores"].style.format("{:.3g}"), width="stretch")
    st.caption("Errors are per point (median / 95th percentile of |pred − true| / true) and on "
               "the figures of merit extracted from each predicted test curve.")
    test = res["test"]
    pick = st.selectbox("Check a test curve", sorted(test.curve_id.unique()))
    g = test[test.curve_id == pick]
    shown = {"true": (g.V_G.to_numpy(), g.I_D.to_numpy(), None),
             "neural net": (g.V_G.to_numpy(), g.I_nn.to_numpy(), "dash")}
    if "I_rf" in g:
        shown["random forest"] = (g.V_G.to_numpy(), g.I_rf.to_numpy(), "dot")
    st.plotly_chart(curve_fig(shown), width="stretch")

# ------------------------------------------------------------------ 3. predict
st.header("3 · Predict a new device")
params, cols = {}, st.columns(min(len(inputs), 3))
for i, c in enumerate(inputs):
    lo, hi = res["ranges"][c]
    default = NOMINAL.get(c, {"V_D": 0.5, "T": 300.0}.get(c, (lo + hi) / 2))
    params[c] = cols[i % len(cols)].slider(
        f"{c} [{UNITS.get(c, '')}]", lo, hi, float(np.clip(default, lo, hi)),
        (hi - lo) / 200, format="%.3f")
c1, c2, c3 = st.columns(3)
vlo, vhi = res["ranges"]["V_G"]
V = np.linspace(vlo, vhi, c1.slider("Points on the curve", 11, 301, 101, 10))
sweep = c2.selectbox("Sweep one parameter", ["none", *inputs])
n_sw = c3.slider("Curves in sweep", 2, 10, 5, disabled=sweep == "none")

devices = {"prediction": params} if sweep == "none" else {
    f"{sweep}={v:.3g}": {**params, sweep: v} for v in np.linspace(*res["ranges"][sweep], n_sw)}
curves = {}
for name, p in devices.items():
    curves[f"{name} (NN)"] = (V, predict_curve(res, "nn", p, V), None)
    if "rf" in res and sweep == "none":
        curves[f"{name} (RF)"] = (V, predict_curve(res, "rf", p, V), "dot")
    if sim_mode and sweep == "none":                 # we know the true answer for simulated data
        curves["simulator (truth)"] = (V, drain_current(V, p["L_ov"], p["WK"], p["t_n"],
                                                        p["t_ox"], p.get("V_D", 0.5),
                                                        p.get("T", 300.0)), "dash")
curve_tabs(curves)
foms = fom_table(curves)
st.dataframe(foms.style.format("{:.3g}"), width="stretch")
long = pd.concat([pd.DataFrame({"curve": n, "V_G": v, "I_D": i})
                  for n, (v, i, _) in curves.items()])
d1, d2 = st.columns(2)
d1.download_button("Download predicted curves (CSV)", long.to_csv(index=False),
                   "predicted_curves.csv")
d2.download_button("Download figures of merit (CSV)", foms.to_csv(), "predicted_foms.csv")
