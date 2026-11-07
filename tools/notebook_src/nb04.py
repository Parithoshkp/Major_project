# %% [markdown]
# # 04 · How sure is the model? Deep-ensemble uncertainty and active learning
#
# Notebook 03 gave us an accurate surrogate. Two questions a device engineer asks next:
# 1. **Can I trust a single prediction?** We train a **deep ensemble** (5 networks) and use their disagreement as an error bar.
#    We check whether the bars are honest (coverage), fix them if not (calibration), and see whether they grow outside the training range.
# 2. **Can we simulate less?** TCAD costs about 5 h per 100 curves. **Active learning** lets the model choose which devices to simulate next.
#    We compare it with random choice.
#
# We use **dataset 5**, which also varies the drain voltage $V_D$ (0.05–0.5 V) and the temperature $T$ (250–400 K). The paper listed
# that as future work. Main model: the log-target MLP with physics features. **The monotonicity penalty is off from now on**
# (notebook 03 found no benefit, see `docs/PROJECT_LOG.md`).
#
# **Honest note:** all data comes from our analytical simulator (`tfet/physics.py`), not TCAD.

# %% [markdown]
# ### 🧒 This notebook in simple words
# Ask five friends to guess the weight of a cake. If they all say "about 1 kg", you can trust the answer. If one says 0.5 kg and
# another says 2 kg, you know the guess is shaky. We do the same thing with five neural networks: **when they agree, we trust them; when
# they disagree, we are careful.**
#
# Then we use that disagreement to save work. Instead of simulating random devices, we simulate the ones the networks
# **disagree about most**, because that is where they have the most to learn. It's like a student asking the teacher about the
# questions they're unsure of, not the ones they already know.
#
# **Words you need for this notebook:**
#
# | Word | Simple meaning |
# |---|---|
# | **Deep ensemble** | several copies of the same network, each started from a different random point. Their average is the prediction |
# | **σ (sigma)** | how much the networks disagree (their standard deviation). Big σ = unsure |
# | **±2σ band** | the error bar. An honest bar should contain the true value about 95 % of the time |
# | **Coverage** | how often the true value really falls inside the band |
# | **Overconfident** | the band is too narrow: coverage is below 95 % |
# | **Calibration** | stretching the bands by one number so the coverage becomes honest |
# | **In-distribution / out-of-distribution** | inside / outside the range of settings the model learned from |
# | **Active learning** | the model picks which devices to simulate next |
# | **Acquisition rule** | the rule for that pick (random, most uncertain, …) |
# | **Pool** | the list of candidate devices we *could* simulate |
# | **p95 error** | the error that 95 % of points beat. It measures the *bad* cases, while the median measures the typical case |
# | **$V_D$, T** | drain voltage and temperature: the new inputs in dataset 5 |

# %% [markdown]
# ### 🧰 Block: tools, data, splits and the noise floor
# **What it does:**
# 1. Loads libraries and dataset 5 (3000 curves).
# 2. Splits by whole curves: **test** = 600 curves, never touched until scoring. From the rest, **validation** = 240 curves,
#    used only to calibrate the error bars. **Fit** = the 2160 curves the networks learn from.
# 3. Works out the **noise floor**: the score of a perfect model, which knows the clean simulator, against the noisy test data.
#
# **How to read it:** the printed dictionary is the best score anyone could get.
#
# **What to notice:** the floor is about 0.34 % median error, the same as on dataset 4.

# %%
# --- tools, data, splits, noise floor ---
import os, sys, time
sys.path.append("..")
os.environ["PYTHONPATH"] = os.path.abspath("..")   # parallel workers need to find tfet
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr

from tfet import NOMINAL, drain_current
from tfet.physics import PARAM_NAMES
from tfet.evaluate import FEATURES, curve_split, regression_report, fom_errors
from tfet.surrogate import DeepEnsemble

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
COLS = FEATURES + ["V_D", "T"]                      # 7 inputs: V_G, L_ov, WK, t_n, t_ox, V_D, T
DEV = PARAM_NAMES + ["V_D", "T"]                    # the 6 settings that define one curve
X = lambda d: d[COLS].values
SIGMA_NOISE = np.log10(1.005)                       # 0.5 % data noise, in decades of current

ds5 = pd.read_pickle("../data/ds5_extended.pkl")
train, test = curve_split(ds5, 0.2)                 # same rule as every notebook: seed 42
fit, val = curve_split(train, 0.1, seed=1)
print("fit / validation / test curves:", fit.curve_id.nunique(), val.curve_id.nunique(), test.curve_id.nunique())

def clean_current(d):
    """Noise-free simulator output for every row (what a perfect model would predict)."""
    return drain_current(d.V_G.values, d.L_ov.values, d.WK.values, d.t_n.values, d.t_ox.values,
                         V_D=d.V_D.values, T=d["T"].values)

floor = regression_report(test.I_D, clean_current(test))
print("noise floor (perfect model):", {k: round(float(v), 3) for k, v in floor.items()})

# %% [markdown]
# # Part A · Uncertainty
# ### 🧠 Block: train the deep ensemble
# **What it does:** trains **5 networks** with the same recipe (log target, physics features λ and A_tun, 5000 steps), each
# started from a different random seed. They run 4 at a time on the CPU, which takes about 3 minutes.
#
# **How to read it:** one line per network with its training time. Network 5 runs alone at the end, so it is faster.
#
# **What to notice:** nothing yet. This only builds the five guessers.

# %%
# --- 5 networks, different seeds ---
t0 = time.perf_counter()
ens = DeepEnsemble.train(X(fit), fit.I_D.values, COLS, n_members=5, steps=5000)
print(f"ensemble trained in {(time.perf_counter() - t0) / 60:.1f} min")
for i, s in enumerate(ens.train_s):
    print(f"  network {i}: {s:5.0f} s")

# %% [markdown]
# ### 🏆 Block: is the ensemble accurate?
# **What it does:** scores the ensemble average and each single network on the 600 test curves, with the same metrics as notebook 03,
# including the device metrics (I_ON, I_OFF, V_th, SS).
#
# **How to read it:** one row per model. Compare `median_rel_err_%` with the noise-floor row.
#
# **What to notice:** see the summary below the table.

# %%
# --- accuracy: ensemble mean vs single networks ---
mu, sd = ens.predict_log(X(test))                  # mean and spread of log10 I_D over the 5 networks
y = np.log10(test.I_D.values)
acc = {"noise floor": floor}
for i, m in enumerate(ens.members):
    acc[f"network {i}"] = regression_report(test.I_D, m.predict(X(test)))
acc["ensemble mean"] = {**regression_report(test.I_D, 10 ** mu), **fom_errors(test.assign(I_pred=10 ** mu))}
acc_table = pd.DataFrame(acc).T
cols_show = ["R2_linear_%", "R2_log10_%", "median_rel_err_%", "p95_rel_err_%", "I_on_err_%", "I_off_err_decades",
             "V_th_err_mV", "SS_min_err_mV/dec"]
display(acc_table[cols_show].round(3))

# %% [markdown]
# ACC_SUMMARY

# %% [markdown]
# ### 📏 Block: are the error bars honest? (coverage and calibration)
# **What it does:**
# 1. Builds an error bar for every test point: σ from the ensemble, combined with the known 0.5 % data noise.
# 2. Checks the **coverage**: how often the true value lies inside ±2σ. The goal is 95.4 %.
# 3. **Calibrates**: finds one stretch factor *k* on the **validation** curves so that ±2kσ covers 95.4 % there, then checks the test curves.
# 4. Draws a **reliability plot**: for every band width, the coverage we *asked for* against the coverage we *got*.
#
# **How to read it:** on the plot, the diagonal is perfect honesty. A curve **below** the diagonal means the bands are too narrow
# (overconfident).
#
# **What to notice:** see the summary below the plot.

# %%
# --- coverage, calibration, reliability plot ---
def total_sigma(s):
    return np.hypot(s, SIGMA_NOISE)                # model disagreement + known data noise

mv, sv = ens.predict_log(X(val))
z_val = np.abs(np.log10(val.I_D.values) - mv) / total_sigma(sv)
k = np.quantile(z_val, norm.cdf(2) - norm.cdf(-2)) / 2      # stretch so ±2kσ covers 95.4 % on validation
err = np.abs(y - mu)
cov = lambda scale, n=2: 100 * np.mean(err <= n * scale)
cal_table = pd.DataFrame({
    "±2σ coverage on test (%)": [cov(sd), cov(total_sigma(sd)), cov(k * total_sigma(sd))],
}, index=["ensemble σ only", "ensemble σ + data noise", f"calibrated (k = {k:.2f})"])
display(cal_table.round(1))

levels = np.linspace(0.05, 0.99, 30)
fig, ax = plt.subplots(figsize=(5, 4.5))
for name, s in [("ensemble σ + noise", total_sigma(sd)), (f"calibrated (k={k:.2f})", k * total_sigma(sd))]:
    got = [np.mean(err <= norm.ppf((1 + p) / 2) * s) for p in levels]
    ax.plot(100 * levels, 100 * np.array(got), "o-", ms=3, label=name)
ax.plot([0, 100], [0, 100], "k--", lw=1, label="perfectly honest")
ax.set(xlabel="coverage asked for (%)", ylabel="coverage obtained on test (%)", title="Reliability of the error bars")
ax.legend(fontsize=8)
fig.savefig("../figures/04_reliability.png", bbox_inches="tight")

# %% [markdown]
# CAL_SUMMARY

# %% [markdown]
# ### 🎯 Block: does a big σ really mean a big mistake?
# **What it does:** sorts the test points into 10 groups, from the smallest σ to the largest, and plots the typical actual error in each group.
# It also prints the rank correlation (Spearman) between σ and the error: 0 = unrelated, 1 = perfectly in step.
#
# **How to read it:** a rising line means σ is useful as a warning light. If the line ran flat, σ would tell us nothing.
#
# **What to notice:** see the summary below.

# %%
# --- σ against the real error ---
bins = pd.qcut(sd, 10, labels=False)
g = pd.DataFrame({"sd": sd, "err": err, "bin": bins}).groupby("bin").median()
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(g.sd, g.err, "o-", label="median actual error")
ax.loglog(g.sd, k * np.hypot(g.sd, SIGMA_NOISE), "k--", lw=1, label="calibrated σ (k·σ_total)")
ax.set(xlabel="ensemble σ (decades)", ylabel="|error| (decades)", title="Larger σ ↔ larger mistakes?")
ax.legend(fontsize=8)
fig.savefig("../figures/04_sigma_vs_error.png", bbox_inches="tight")
rho = spearmanr(sd, err)[0]
print(f"Spearman correlation σ vs error: {rho:.2f}")
print(f"error in the 10 % most-uncertain points is {g.err.iloc[-1] / g.err.iloc[0]:.1f}x the error in the 10 % least-uncertain")

# %% [markdown]
# SIG_SUMMARY

# %% [markdown]
# ### 🌡️ Block: does σ grow outside the training range? (hot devices)
# **What it does:** trains a second ensemble (5 networks) **only on devices at T ≤ 350 K**, then asks it about
# test devices at T > 350 K, which it never saw. This takes about 3 minutes.
#
# **How to read it:** the table compares the cool test devices (inside the training range) with the hot ones (outside). We want σ to
# **grow** on the hot devices, so the error bar warns us. Coverage uses the stretch factor *k* from the main model.
#
# **What to notice:** see the summary below.

# %%
# --- train cool, test hot ---
cool_fit = fit[fit["T"] <= 350]
ens_cool = DeepEnsemble.train(X(cool_fit), cool_fit.I_D.values, COLS, n_members=5, steps=5000)
ood = {}
for name, part in [("inside: test T ≤ 350 K", test[test["T"] <= 350]), ("outside: test T > 350 K", test[test["T"] > 350])]:
    m_, s_ = ens_cool.predict_log(X(part))
    e_ = np.abs(np.log10(part.I_D.values) - m_)
    ood[name] = {"curves": part.curve_id.nunique(), "median_rel_err_%": regression_report(part.I_D, 10 ** m_)["median_rel_err_%"],
                 "median σ (decades)": np.median(s_), "±2kσ coverage (%)": 100 * np.mean(e_ <= 2 * k * total_sigma(s_))}
ood_table = pd.DataFrame(ood).T
display(ood_table.round(4))

# %% [markdown]
# OOD_SUMMARY

# %% [markdown]
# ### 🔬 Block: error bars on real curves, and the new knobs V_D and T
# **What it does:** draws the predicted curve with its calibrated ±2σ band for the nominal device (L_ov 5, WK 4.4, t_n 2, t_ox 3) at
# three temperatures (left) and three drain voltages (right), on top of the noise-free simulator.
#
# **How to read it:** dots = simulator (the truth), line = ensemble average, shaded = ±2kσ band. The y-axis is logarithmic.
#
# **What to notice:** heating the device raises the **off-current** a lot (the left end of the curve moves up), while the on-current barely
# changes. Lowering $V_D$ cuts the on-current. The ensemble follows both effects, and the bands are thin wherever it is sure.

# %%
# --- nominal device at several T and V_D with bands ---
Vg = np.linspace(0, 0.5, 51)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, knob, values, fixed in [(axes[0], "T", (250, 300, 400), {"V_D": 0.5}), (axes[1], "V_D", (0.05, 0.2, 0.5), {"T": 300.0})]:
    for v, c in zip(values, ("C0", "C1", "C3")):
        d = pd.DataFrame({"V_G": Vg, **NOMINAL, **fixed, knob: v})
        m_, s_ = ens.predict_log(X(d))
        band = 2 * k * total_sigma(s_)
        ax.semilogy(Vg, clean_current(d), "o", ms=3, color=c, alpha=0.6)
        ax.semilogy(Vg, 10 ** m_, "-", color=c, label=f"{knob} = {v}")
        ax.fill_between(Vg, 10 ** (m_ - band), 10 ** (m_ + band), color=c, alpha=0.25)
    ax.set(xlabel="$V_G$ (V)", ylabel="$I_D$ (A/µm)", title=f"nominal device, varying {knob}  (dots = simulator)")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/04_bands_T_VD.png", bbox_inches="tight")

# %% [markdown]
# # Part B · Active learning: simulating less
# ### 🔁 Block: the active-learning loop
# **What it does:** pretends the 2160 fit curves have **not** been simulated yet. They are our **pool** of candidate devices. Then:
# 1. Start with the **same 50 random curves** for every rule.
# 2. Train a small ensemble (4 networks, 2000 steps) and score it on the 600 test curves.
# 3. Use the rule to pick **50 more** devices from the pool, "simulate" them (reveal their curves) and repeat until 400 curves.
#
# Three rules:
# * **random:** pick any 50. This is the baseline, like ordinary sampling.
# * **most uncertain:** pick the 50 devices with the largest average σ.
# * **uncertainty-weighted:** a lottery where each device's chance of being picked grows with σ² (more doubt = more tickets).
#   Unsure devices are favoured, but the picks still spread out. A prototype also tried "most uncertain device in each of 50 k-means
#   groups"; it did worse than this lottery, so we left it out to save time.
#
# This is the longest block (about 17 minutes).
#
# **How to read it:** one line per round: rule, number of simulated curves, typical (median) error and bad-case (p95) error.

# %%
# --- pool-based active learning ---
pool = fit
devs = pool.groupby("curve_id")[DEV].first()
start = list(np.random.default_rng(0).choice(devs.index, 50, replace=False))

def pick(rule, ens_, rest, rng, n=50):
    """Choose n new devices from the not-yet-simulated part of the pool."""
    if rule == "random":
        return list(rng.choice(rest.curve_id.unique(), n, replace=False))
    _, s_ = ens_.predict_log(X(rest))
    score = pd.Series(s_, index=rest.curve_id.values).groupby(level=0).mean()   # average σ along each curve
    if rule == "most uncertain":
        return list(score.nlargest(n).index)
    p = score.values ** 2                                                     # uncertainty-weighted lottery
    return list(rng.choice(score.index.values, n, replace=False, p=p / p.sum()))

RULES = ("random", "most uncertain", "uncertainty-weighted")
al_rows, chosen_by = [], {}
t0 = time.perf_counter()
for rule in RULES:
    chosen, rng = list(start), np.random.default_rng(1)
    while True:
        tr = pool[pool.curve_id.isin(chosen)]
        e = DeepEnsemble.train(X(tr), tr.I_D.values, COLS, n_members=4, steps=2000)
        rep = regression_report(test.I_D, e.predict(X(test)))
        al_rows.append({"rule": rule, "curves": len(chosen), **rep})
        print(f"{rule:24s} {len(chosen):4d} curves  median {rep['median_rel_err_%']:5.2f} %  p95 {rep['p95_rel_err_%']:6.2f} %"
              f"   ({(time.perf_counter() - t0) / 60:.1f} min)")
        if len(chosen) >= 400:
            break
        chosen += pick(rule, e, pool[~pool.curve_id.isin(chosen)], rng)
    chosen_by[rule] = chosen
al = pd.DataFrame(al_rows)
al.to_csv("../data/nb04_active_learning.csv", index=False)

# %% [markdown]
# ### 📈 Block: which rule needs the fewest simulations?
# **What it does:** plots the test error against the number of simulated curves for each rule. Left: typical (median) error.
# Right: bad-case (p95) error.
#
# **How to read it:** lower is better. A line that is lower at the same x-position reached that accuracy with fewer simulations.
#
# **What to notice:** see the summary below the plots.

# %%
# --- error vs number of simulations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in [(axes[0], "median_rel_err_%", "typical error"), (axes[1], "p95_rel_err_%", "bad-case (p95) error")]:
    for rule, gr in al.groupby("rule", sort=False):
        ax.plot(gr.curves, gr[metric], "o-", label=rule)
    ax.set(xlabel="simulated curves", ylabel=f"{metric}", title=title)
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/04_active_learning.png", bbox_inches="tight")
al.pivot_table(index="curves", columns="rule", values=["median_rel_err_%", "p95_rel_err_%"], sort=False).round(2)

# %% [markdown]
# AL_SUMMARY

# %% [markdown]
# ### 🗺️ Block: where did each rule choose to simulate?
# **What it does:** for every rule, draws histograms of the settings of the 350 devices it **added** (the starting 50 are left out).
#
# **How to read it:** a flat histogram means the rule spread its picks evenly. A tall bar at one edge means it crowded into that corner.
#
# **What to notice:** see the summary below.

# %%
# --- distribution of the chosen devices ---
fig, axes = plt.subplots(1, len(DEV), figsize=(16, 3))
for rule in RULES:
    added = devs.loc[chosen_by[rule][50:]]
    for ax, c in zip(axes, DEV):
        ax.hist(added[c], bins=8, range=(devs[c].min(), devs[c].max()), histtype="step", lw=1.5, label=rule)
for ax, c in zip(axes, DEV):
    ax.set(xlabel=c, yticks=[])
axes[0].legend(fontsize=7)
fig.suptitle("settings of the 350 devices each rule added"); fig.tight_layout()
fig.savefig("../figures/04_al_picks.png", bbox_inches="tight")
pd.DataFrame({rule: devs.loc[chosen_by[rule][50:]].mean() for rule in RULES}).T.round(2)

# %% [markdown]
# PICKS_SUMMARY

# %% [markdown]
# ### 💾 Block: save the ensemble
# **What it does:** saves the main 5-network ensemble with its calibration factor *k* to `models/`, then reloads it to check the file.
#
# **How to read it:** the last line should say the reloaded ensemble matches.
#
# **What to notice:** notebook 05 (SHAP, inverse design, explorer) can load this file and get error bars for free.

# %%
# --- save and check ---
ens.k_calibration, ens.cols = k, COLS
joblib.dump(ens, "../models/ensemble_ds5.joblib")
again = joblib.load("../models/ensemble_ds5.joblib")
print("reloaded ensemble matches:", np.allclose(again.predict_log(X(test))[0], mu), "| k =", round(again.k_calibration, 3))

# %% [markdown]
# CONCLUSIONS
