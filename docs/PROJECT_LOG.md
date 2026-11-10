# Project log: ML surrogates for line TFETs

This is the running record of everything done in the project: what was built, which decisions were
made and why, and the key results. **Update it at the end of every step.**

- **Base paper:** C. Akbar, Y. Li, N. Thoti, "Device-Simulation-Based Machine Learning Technique for the
  Characteristic of Line Tunnel Field-Effect Transistors", *IEEE Access* 10, 53098 (2022).
- **Honest data note:** we have no TCAD. All data comes from `tfet/physics.py`, an **analytical simulator**
  (Kane BTBT, tunnelling length λ, tunnelling area A_tun, WK-shifted onset, TAT floor, Varshni band gap).
  It is calibrated to the paper's figures, and 0.5 % multiplicative noise mimics TCAD solver noise.
  Every accuracy number below is for this analytical data, not TCAD.

---

## 1. What the paper did vs what this project adds

| | Paper | This project |
|---|---|---|
| Model | RandomForestRegressor (50/100 trees) | RFR reproduced, plus XGBoost, MLP and physics-informed MLP |
| Target | linear I_D | **log10 I_D** (the off-state matters) |
| Evaluation | R², RMSE on linear scale | adds log-space R², error per current decade, **device metrics** (I_ON, I_OFF, V_th, SS), monotonicity, extrapolation, learning curves |
| Inputs | V_G, L_ov, WK, t_n, t_ox (+V_D) | adds physics features λ and log A_tun; **V_D and temperature T** (paper's future work) |
| Usability | offline scripts | **Streamlit app** (`app/app.py`): upload TCAD curves (or generate them with our simulator), train the model in the browser, predict curves + FoMs for new devices, compare against the paper's RFR |

## 2. Status

| Step | Notebook | Status |
|---|---|---|
| Analytical simulator and datasets ds1–ds5 | 01 | ✅ done |
| Reproduce the paper's RFR, then show where it breaks | 02 | ✅ done |
| Improved surrogates: 7 models, ablation, extrapolation, learning curve | 03 | ✅ done |
| ~~Deep-ensemble uncertainty and active learning~~ | ~~04~~ | ❌ dropped 2026-11-10 (scope cut) |
| Streamlit app: TCAD upload or simulator → train → predict | `app/` | ✅ done (2026-11-10) |
| README, report, slides | n/a | ⏳ when asked |

## 3. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11 (nb01) | Use an analytical simulator instead of TCAD | No TCAD licence. It is calibrated to the paper's nominal device: I_ON ≈ 2.2e-5 A/µm, I_OFF ≈ 1e-14, SS_min 30–35 mV/dec |
| 2026-11 (nb01) | Add 0.5 % noise to all simulated data | Mimics TCAD convergence noise, and gives an honest **noise floor** (0.34 % median error on ds4) |
| 2026-11 (nb02) | Split by whole curves, never by points | A point split leaks neighbouring V_G points into training and inflates scores |
| 2026-11 (nb03) | Predict log10 I_D, not I_D | The single biggest fix: log-space R² goes from 63.7 % to 99.3 % with the same forest |
| 2026-11 (nb03) | Train MLPs with a fixed step budget (`train_run`), 3 seeds, 4 jobs in parallel | Small training sets would otherwise be under-trained. Seeds show the spread, and 4 jobs fit in ~7 GB RAM |
| **2026-11-07** | **Drop the monotonicity penalty from the main model** (`mono_weight` default 1.0 → 0.0) | No measurable benefit in any nb03 test (1.00 % vs 0.99 % median error, seed spread ~0.1 %). The log-target MLP was already monotonic (1 curve in 1500 goes down, with or without it), and the penalty costs ~30 % more training time. **Reported as a negative result.** The code is kept, and nb03 still reproduces because it passes `mono_weight` explicitly |
| 2026-11-07 | Main model from now on = **MLP-log+feat** (log target and physics features, no penalty) | Equal to PI-MLP within the seed spread, and faster |
| 2026-11-07 | `DeepEnsemble` rewritten to train members in parallel through `train_run` | The old version trained members one at a time and had never been run |
| **2026-11-10** | **Scope cut: drop notebook 04 (ensembles, active learning) and the planned nb05; the novelty is now a Streamlit app** | User wants a smaller project. The app turns the nb03 result (log-target MLP ≫ paper RFR) into a usable tool: train on your own TCAD data, predict new devices. `DeepEnsemble` and nb04 removed (they are still in git history, commit `8a43b19`) |

## 4. Change log

### Commit `94604cd`: baseline
- `tfet/` package: `physics.py` (simulator), `metrics.py` (FoM extraction), `dataset.py` (ds1–ds5 builders),
  `evaluate.py` (curve split, reports, FoM errors, monotonicity check), `surrogate.py` (first version).
- Notebooks 01 and 02, built from `tools/notebook_src/nb01.py` and `nb02.py` with `tools/build_nb.py`.
- Datasets `data/ds1…ds5`, `data/learning_curve_rfr.csv`, figures `figures/01_*`, `figures/02_*`.

### Commit `d289901`: notebook 03
- `surrogate.py`: added `train_run` (fixed step budget, joblib-safe) and `n_colloc` (collocation points per step for the penalty).
- Notebook 03 with source, figures `03_*`, result CSVs `nb03_comparison.csv`, `nb03_extrapolation.csv`,
  `learning_curve_nb03.csv`, and the saved model `models/best_surrogate_ds4.joblib` (PI-MLP, seed 0, trained **with** the penalty).

### 2026-11-07: penalty dropped, ensemble rewritten
- `tfet/surrogate.py`: `mono_weight` default is now `0.0` in `LogCurrentSurrogate` and `train_run`. The docstrings explain why.
- `tfet/surrogate.py`: `DeepEnsemble.train(X, I_D, cols, n_members, steps, n_jobs)` trains members in parallel. It has `predict_log` (mean, std) and `predict`.
- Added this file (`docs/PROJECT_LOG.md`).
- Timing check on ds5 (2400 curves, 4 jobs in parallel): ~25 ms per optimiser step per member, so 5000 steps ≈ 2 min.

### 2026-11-07: notebook 04 drafted
- `tools/notebook_src/nb04.py` and `notebooks/04_uncertainty_active_learning.ipynb` (31 cells): ensemble accuracy, coverage and calibration,
  σ against error, temperature out-of-distribution test, V_D/T bands, and active learning with 3 rules (random, most uncertain, uncertainty-weighted).
- Prototyped 4 active-learning rules. The k-means "spread-out" rule was dropped because it did worse than the uncertainty-weighted lottery.
- The full run was stopped at the user's request. Next session: execute it, write the summaries from real outputs, and update this log.

### 2026-11-10: scope cut, Streamlit app
- Removed `notebooks/04_uncertainty_active_learning.ipynb`, `tools/notebook_src/nb04.py` and `DeepEnsemble` from `tfet/surrogate.py`.
- Added `app/app.py` (Streamlit + Plotly; installed `streamlit 1.67`, `plotly 7.1` in `.venv`). Three steps on one page:
  1. **Data:** upload a CSV/TXT (long format `curve_id, L_ov, WK, t_n, t_ox, V_G, I_D`; column names like `Vg`, `Ids [A/um]`, `tox` are recognised; any extra numeric column such as `V_D`, `T` becomes an input) or generate N curves with the simulator (LHS, noise, optional V_D/T).
  2. **Train:** `LogCurrentSurrogate` on log10 I_D with a curve split (default 20 % held out), optional physics features, optional paper RFR for comparison. Shows the nb03 metrics (median/p95 error, R², I_ON/I_OFF/V_th/SS errors) and a true-vs-predicted test curve.
  3. **Predict:** sliders over the training range, optional parameter sweep, log/linear I_D–V_G, g_m, SS plots, FoM table, CSV downloads. In simulator mode the true simulator curve is overlaid.
- Tested headless with `streamlit.testing` (both modes, sweep) and the upload parser on a file with renamed columns.

### 2026-11-10: project report
- Project report written as a Claude doc: https://claude.ai/artifact/D7gWPrD7gHYndHT5VQP2cm (abstract, paper, simulator, RFR reproduction, improved models with a 7-model chart, the app with a workflow diagram, limitations, conclusion, how to reproduce, references).

### 2026-11-10: app runs on its own
- Confirmed `app/app.py` needs no notebook outputs (no `data/` or `models/` files): it imports only the `tfet/` package and makes its own data. `cd app && streamlit run app.py` works (tested with `streamlit.testing`, full train run).
- Added `requirements.txt` (app dependencies; notebook-only packages listed as optional).

## 5. Results

### Notebook 02: the paper's RFR (ds4 test set unless noted)
- Linear R² matches the paper: ds1 99.5 % (paper 99.9 %); ds2/ds3/ds4 98.5 / 76.1 / 93.3 %.
- But log-space R² is only **63.7 %** and the median relative error **62 %**.
- The RFR never predicts below 1.09e-9 A/µm (true minimum 5e-15), so **the off-state is lost**: I_OFF is off by
  3.34 decades, V_th can't be extracted from 77.4 % of curves, and the SS_min error is 20 mV/dec.
- 24.2 % of predicted curves go down somewhere, and the output is a staircase on a fine V_G grid.
- Extrapolation to t_ox < 2 nm gives linear R² 37 %. Learning curve: 25 → 1600 curves gives 451 % → 77 % median error.

### Notebook 03: improved surrogates (ds4 test; MLPs are the mean of 3 seeds; noise floor 0.34 %)

| Model | Median rel. error |
|---|---|
| RFR-linear (paper) | 62 % |
| RFR-log | 26 % |
| XGB-log | 17.7 % |
| XGB-log + physics features | 11.8 % |
| MLP-log | 1.88 % |
| MLP-log + physics features | 1.00 % |
| PI-MLP (features + penalty) | 0.99 % |

- PI-MLP: log R² 99.998 %, I_OFF error 0.016 decades, V_th missing 0 %, SS_min error 0.60 mV/dec.
- **Extrapolation** to t_ox < 2 nm: linear R² 37.3 % (RFR) vs 98.8 % (MLP+feat); median error 74.7 % vs 4.0 %.
- **Learning curve:** the MLP on 25 curves (6.3 %) beats the RFR on 1600 curves (76.8 %) and XGB-log on 1600 (20.1 %).
- **Negative result:** the monotonicity penalty gave no measurable gain (see the decision log).
- **Caveat:** λ and A_tun are formulas inside the simulator, so their gain may be smaller on real TCAD.

### Streamlit app (2026-11-10 test, default settings: 300 simulated curves, 2000 steps, 20 % test)
- Neural net: median error **2.2 %**, log R² 99.99 %, I_OFF error 0.05 dec, V_th error 0.6 mV, SS_min error 1.6 mV/dec. Trains in ~18 s.
- Paper RFR on the same split: median error 150 %, log R² 53 %, V_th not extractable.
- Upload path, 60 curves with renamed columns, 1000 steps: NN median error 4.1 % vs RFR 1232 %.

## 6. Known limitations (keep these in the report)
1. All data is analytical, not TCAD. It is smooth and easier to learn than real TCAD.
2. The physics features are partly circular: they are formulas inside the simulator.
3. `monotonic_violations` can't detect staircases (flat steps never go down). Use fine-grid plots for that.
4. MLP timings are rough because parallel runs share the CPU.
5. The 1.9 % plateau in the nb03 learning curve is probably the 3000-step budget. This has not been tested.

## 7. How to reproduce
```bash
cd notebooks
/home/kpp/miniconda3/bin/python ../tools/build_nb.py ../tools/notebook_src/nbXX.py XX_name.ipynb
/home/kpp/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=tfet-ml --ExecutePreprocessor.timeout=2400 XX_name.ipynb
```
Environment: `.venv` (Python 3.12, uv), Jupyter kernel `tfet-ml`, 16 CPU cores, ~7 GB RAM, no GPU.
