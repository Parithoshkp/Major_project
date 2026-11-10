# Prompt for the next Claude session

Copy everything below the line into a new Claude Code session started in
`/home/kpp/work/projects/Major_project`.

---

I'm continuing my major project. Please read this whole brief, then look at the files it mentions before doing anything.

## The project
- **Base paper:** `Research_paper.pdf`, C. Akbar, Y. Li, N. Thoti, "Device-Simulation-Based Machine Learning Technique for the Characteristic of Line Tunnel Field-Effect Transistors", IEEE Access 10, 53098 (2022). It's a **scanned PDF with no text layer**. Don't re-read it unless needed; the summary is below.
- **Paper summary:** 3-D TCAD of a scaled line TFET (Si0.6Ge0.4 source, n-epitaxy). Four device parameters: L_ov 0.1–10 nm, WK 4.2–4.5 eV, t_n 1–5 nm, t_ox 1–6 nm. A scikit-learn RandomForestRegressor (50 trees, max_depth None, MSE, random_state 42; 100 trees for the 2500-curve set) maps (V_G, L_ov, t_n, t_ox, WK[, V_D]) to **linear** I_D. It uses z-score normalisation and an 80/20 curve split. Claims: R² > 99%, error < 1%, ~99% less compute than TCAD (5 h per 100 curves). The authors' own future work: material parameters and PVT variation.
- **My project = the paper plus novelty:** log-target and FoM-based evaluation, a physics-informed neural network (physics features; the monotonicity loss was tried and **dropped** as a negative result), a temperature/V_D extension, and (**the novelty, since 2026-11-10**) a Streamlit app `app/app.py`: upload TCAD curves or generate them with our simulator, train the neural surrogate in the browser, predict curves and FoMs for new devices. Ensembles / active learning / SHAP / inverse design were **dropped** (scope cut by the user).
- **Data source:** we have no TCAD, so `tfet/physics.py` is an **analytical simulator** (Kane BTBT with the paper's B = 16.8 MV/cm, λ = sqrt(ε_ns·t_ox·t_n/ε_ox), A_tun = L_ov·2(W+t_n), WK-shifted onset, TAT floor, Varshni temperature dependence). It's calibrated to the paper's figures: nominal (L_ov 5, WK 4.4, t_n 2, t_ox 3) gives I_ON ≈ 2.2e-5 A/µm, I_OFF ≈ 1e-14, SS_min 30–35 mV/dec. Always state honestly that the data comes from an analytical model, not TCAD.

## How I like to work
- **One step at a time.** Do one notebook (or one clear step), run it, show me the key results, then **stop and wait** for my go-ahead. Don't run long experiments without telling me first.
- I use **JupyterLab** to see how everything works.
- **Easy explanations:** every notebook has a "🧒 In simple words" section with a mini-dictionary at the top, and a markdown cell `### <emoji> Block: <name>` before **every** code cell saying *what it does*, *how to read the plot/table* and *what to notice*. The code also gets short plain comments. Explain like I'm new to device physics and ML.
- Text in the notebooks must match the actual outputs. Check the results after running and fix any wording that no longer fits.

## Environment (already set up)
- Project venv: `.venv` (Python 3.12, created with `uv`) with numpy, scipy, pandas, scikit-learn, matplotlib, xgboost, torch (CPU), shap, optuna, ipywidgets, tqdm, ipykernel. Install extras with `uv pip install --python .venv/bin/python <pkg>`.
- Jupyter kernel: **`tfet-ml`** ("Python (TFET-ML)"). JupyterLab itself is in miniconda base: `/home/kpp/miniconda3/bin/jupyter`.
- This is WSL. To start JupyterLab detached from the Claude session and open it in the Windows browser:
  ```bash
  cd /home/kpp/work/projects/Major_project
  setsid nohup /home/kpp/miniconda3/bin/jupyter lab --no-browser --port 8888 > .jupyter.log 2>&1 < /dev/null & disown
  grep -o "http://localhost:8888/lab?token=[a-z0-9]*" .jupyter.log | head -1     # then: cmd.exe /c start "" "<url>"
  ```
  Check whether one is already running first with `/home/kpp/miniconda3/bin/jupyter server list`.
- Machine: 16 CPU cores, ~7 GB RAM, no GPU. Keep training runs to minutes.

## Project layout
```
tfet/                  shared Python package (import with sys.path.append("..") from notebooks/)
  physics.py           analytical simulator: drain_current(), id_vg(params, V_G, V_D, T, noise), NOMINAL, PARAM_RANGES, PARAM_NAMES
  metrics.py           figures_of_merit(V_G, I_D) -> I_on, I_off, on_off, V_th (const. current 1e-10), SS_min, SS_avg
  dataset.py           simulate(devices_df), one_at_a_time, lov_tn_pairs, with_drain_bias, all_parameters, extended (V_D + T)
  evaluate.py          curve_split, regression_report, error_by_decade, fom_per_curve, fom_errors, monotonic_violations, FEATURES
  surrogate.py         LogCurrentSurrogate (MLP on log10 I_D, physics features; penalty OFF by default, mono_weight=0), train_run (fixed step budget, safe for joblib),
docs/PROJECT_LOG.md    running record of every decision, change and result. UPDATE IT AFTER EVERY STEP (the user asked for this)
notebooks/
  01_paper_device_and_data.ipynb      DONE: physics, plots, generates data/ds1..ds5
  02_baseline_rfr_reproduction.ipynb  DONE: reproduces paper RFR, then Part B shows where it breaks
  03_improved_surrogates.ipynb        DONE: 7-model comparison, ablation, extrapolation, learning curve
app/app.py             Streamlit app: data (upload or simulator) -> train -> predict. Run: cd app && ../.venv/bin/streamlit run app.py (needs no notebook outputs)
models/                best_surrogate_ds4.joblib (PI-MLP seed 0, trained with the penalty; from nb03)
data/                  ds1_one_at_a_time (400 curves), ds2_lov_tn (200), ds3_drain_bias (900), ds4_all_params (2500), ds5_extended (3000, V_D + T); learning_curve_rfr.csv
                       long format: curve_id, L_ov, WK, t_n, t_ox, V_D, T, V_G (51 pts, 0-0.5 V), I_D; saved as .pkl
figures/               saved PNGs from the notebooks
tools/build_nb.py      converts a "# %%" / "# %% [markdown]" percent-format .py into .ipynb (kernel tfet-ml)
tools/notebook_src/    nb01.py … nb03.py = editable sources of the notebooks
```
Build and run a notebook:
```bash
cd notebooks
/home/kpp/miniconda3/bin/python ../tools/build_nb.py ../tools/notebook_src/nb03.py 03_improved_surrogates.ipynb
/home/kpp/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=tfet-ml --ExecutePreprocessor.timeout=1200 03_improved_surrogates.ipynb
```
Gotchas:
- In the source files, a markdown cell must start with its own `# %% [markdown]` line, and each code cell must start with `# %%`. Don't nest them.
- If I have a notebook open in JupyterLab, tell me to reload it from disk and not save the old tab.

## Key results so far (notebook 02, dataset 4 test set unless noted)
- Paper RFR on ds1: linear R² 99.5% (paper 99.9%). On ds2/ds3/ds4: R² 98.5 / 76.1 / 93.3%.
- Log-space R² on ds4 is only 63.7%, and the median relative error is 62%. The error per current decade is enormous below 1e-9 A/µm.
- **The RFR never predicts below 1.09e-9 A/µm** (the true minimum is 5e-15), so the off-state is lost. I_OFF is off by 3.34 decades; V_th can't be extracted from any predicted curve (V_th_missing 77.4% = all curves that have a V_th); SS_min error is 20 mV/dec.
- 24.2% of predicted curves are non-monotonic. A staircase shows in I_D and g_m on a fine V_G grid.
- Extrapolation to t_ox < 2 nm gives linear R² 37%.
- Learning curve: 25 to 1600 training curves gives median relative error 451% down to 77%.

## Key results from notebook 03 (ds4 test, MLPs = mean of 3 seeds, noise floor 0.34 % median error)
- Median relative error: RFR-linear 62 %, RFR-log 26 %, XGB-log 17.7 %, XGB-log+feat 11.8 %, MLP-log 1.88 %, MLP-log+feat 1.00 %, PI-MLP 0.99 %.
- PI-MLP: log R² 99.998 %, I_OFF error 0.016 decades, V_th missing 0 %, SS_min error 0.60 mV/dec.
- Extrapolation to t_ox < 2 nm: linear R² 37.3 % (RFR) vs 98.8 % (PI-MLP); median error 74.7 % vs 4.0 %.
- Learning curve: PI-MLP on 25 curves (6.3 %) beats RFR on 1600 (76.8 %) and XGB-log on 1600 (20.1 %).
- **Monotonicity penalty: no measurable benefit** in any test (within seed spread). Report it as a negative result, don't claim it as a gain.
- **Physics features are partly circular:** λ and A_tun are formulas inside the simulator, so their gain may not carry over to TCAD.
- MLP timings in nb03 are rough because runs shared the CPU. The 1.9 % learning-curve plateau is probably the 3000-step budget (untested).

## Git
Repo on branch `main` (`.venv`, checkpoints and `.jupyter.log` are ignored). Commits so far:
- `94604cd` baseline (simulator, datasets, nb01–02)
- `d289901` notebook 03
- `caf2dc8` penalty dropped by default, parallel `DeepEnsemble`, `docs/PROJECT_LOG.md`
- `8a43b19` nb04 drafted (not executed) and this handoff
- (2026-11-10) nb04 removed, Streamlit app added

I said "commit the project so far and move ahead", so committing **finished steps** is fine. Still tell me what you committed.

## Decisions already made (don't re-ask)
- nb03 committed. ✅
- **Monotonicity penalty dropped** from the main model and kept as a negative result. ✅ The main model is now MLP-log+feat (log target and physics features, `mono_weight=0`).
- **Scope cut (2026-11-10):** no notebook 04/05. The novelty is the Streamlit app. Don't bring ensembles/active learning back unless I ask. ✅

## 👉 Where to pick up (session of 2026-11-10 ended here)
The Streamlit app `app/app.py` is built and tested (see the PROJECT_LOG change log). Next, only if I ask: polish the app, a `README.md`, report and slides.

## Technical notes
- Train models side by side with `joblib.Parallel(n_jobs=4)` with `train_run`. Set `os.environ["PYTHONPATH"]` to the project root first so workers can find `tfet`. RAM is ~7 GB, so don't go above 4 jobs.
- Measured speed on ds5 with 4 jobs: ~25 ms per step per network on the full fit set (5000 steps ≈ 2 min), ~12 ms per step on ~100 curves.
- `add_physics_features` works with extra columns such as V_D and T as long as `cols` names them. Dataset 5 inputs are `FEATURES + ["V_D", "T"]`.
- The data has 0.5 % noise, so always show the noise floor (0.34 % median error).
- `monotonic_violations` can't detect staircases, because flat steps never go down. Use fine-grid plots for that.
- `nbformat` is in miniconda, not `.venv`, so validate notebooks with `/home/kpp/miniconda3/bin/python`.
- `nbconvert --inplace` writes the notebook only at the end, so a stopped run leaves it without outputs.

## Next steps
1. Notebooks 01–03: DONE.
2. Streamlit app: DONE.
3. Later, if I ask: `README.md`, report or synopsis, slides.

Start by confirming you've read this and checking that the environment works (`.venv/bin/python -c "import tfet"` from the project root). Then ask me what to do next.
