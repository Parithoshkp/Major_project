# Prompt for the next Claude session

Copy everything below the line into a new Claude Code session started in
`/home/kpp/work/projects/Major_project`.

---

I'm continuing my major project. Please read this whole brief, then look at the files it mentions before doing anything.

## The project
- **Base paper:** `Research_paper.pdf`, C. Akbar, Y. Li, N. Thoti, "Device-Simulation-Based Machine Learning Technique for the Characteristic of Line Tunnel Field-Effect Transistors", IEEE Access 10, 53098 (2022). It's a **scanned PDF with no text layer**. Don't re-read it unless needed; the summary is below.
- **Paper summary:** 3-D TCAD of a scaled line TFET (Si0.6Ge0.4 source, n-epitaxy). Four device parameters: L_ov 0.1–10 nm, WK 4.2–4.5 eV, t_n 1–5 nm, t_ox 1–6 nm. A scikit-learn RandomForestRegressor (50 trees, max_depth None, MSE, random_state 42; 100 trees for the 2500-curve set) maps (V_G, L_ov, t_n, t_ox, WK[, V_D]) to **linear** I_D. It uses z-score normalisation and an 80/20 curve split. Claims: R² > 99%, error < 1%, ~99% less compute than TCAD (5 h per 100 curves). The authors' own future work: material parameters and PVT variation.
- **My project = the paper plus novelty:** log-target and FoM-based evaluation, a physics-informed neural network (physics features plus a monotonicity loss), uncertainty (deep ensembles), active learning (fewer simulations), a temperature/V_D extension, SHAP explainability, multi-objective inverse design, and an interactive explorer.
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
  surrogate.py         LogCurrentSurrogate (MLP on log10 I_D, physics features + monotonicity penalty), train_run (fixed step budget, safe for joblib), DeepEnsemble (ensemble itself not yet run)
notebooks/
  01_paper_device_and_data.ipynb      DONE: physics, plots, generates data/ds1..ds5
  02_baseline_rfr_reproduction.ipynb  DONE: reproduces paper RFR, then Part B shows where it breaks
  03_improved_surrogates.ipynb        DONE: 7-model comparison, ablation, extrapolation, learning curve
models/                best_surrogate_ds4.joblib (PI-MLP seed 0)
data/                  ds1_one_at_a_time (400 curves), ds2_lov_tn (200), ds3_drain_bias (900), ds4_all_params (2500), ds5_extended (3000, V_D + T); learning_curve_rfr.csv
                       long format: curve_id, L_ov, WK, t_n, t_ox, V_D, T, V_G (51 pts, 0-0.5 V), I_D; saved as .pkl
figures/               saved PNGs from the notebooks
tools/build_nb.py      converts a "# %%" / "# %% [markdown]" percent-format .py into .ipynb (kernel tfet-ml)
tools/notebook_src/    nb01.py, nb02.py, nb03.py = editable sources of the notebooks
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
The project is a git repo on branch `main` (`.venv`, checkpoints and `.jupyter.log` are ignored). Commit only when I ask.
Only the baseline commit exists so far. **All notebook 03 work is still uncommitted** (nb03, figures, CSVs, `models/`, the `tfet/surrogate.py` changes and this file).

## Waiting for my answer (ask me these first, before doing anything else)
Notebook 03 is finished and reported to me. I had not answered these when the last session closed:
1. Should notebook 03 be committed?
2. Should the monotonicity penalty be dropped from the main model for notebook 04 and kept in the report as a negative result? (Claude recommended yes: it adds training time for no gain.)
3. Should notebook 04 go ahead?

## Notes for notebook 04
- Train several models side by side with `joblib.Parallel(n_jobs=4)` and `surrogate.train_run` (2 threads each). Set `os.environ["PYTHONPATH"]` to the project root first so workers can find `tfet`. About 8 GB of RAM is tight, so don't go above 4 jobs.
- Rough speed with 4 parallel jobs: 5000 steps takes about 3 min per plain or features MLP and about 4 min with the penalty. Notebook 03 took about 25 min in total, so tell me the expected time before a long run.
- `DeepEnsemble` has never been run, and it trains members one at a time, so it needs parallelising or replacing with `train_run` calls. It has no `n_colloc` or step-budget options of its own.
- `add_physics_features` works with extra columns such as V_D and T as long as `cols` names them. Dataset 5 inputs are `FEATURES + ["V_D", "T"]`.
- The data has 0.5 % noise, so always show the noise floor (0.34 % median error on ds4).
- `monotonic_violations` can't detect staircases, because flat steps never go down. Use fine-grid plots for that.
- `nbformat` is in miniconda, not `.venv`, so validate notebooks with `/home/kpp/miniconda3/bin/python`.

## Next steps (do them in order, one per session step)
1. **DONE: Notebook 03, improved surrogates** (`tools/notebook_src/nb03.py` → `notebooks/03_improved_surrogates.ipynb`)
   - Same ds4 curve split as nb02 (`curve_split(ds4, 0.2)`, seed 42).
   - Compare: RFR-linear (paper), RFR-log, XGBoost-log, XGBoost-log plus physics features (λ, log A_tun via `surrogate.add_physics_features`), plain MLP-log (`physics=False`), and the physics-informed MLP (`physics=True`).
   - Report for each: `regression_report`, `fom_errors`, `monotonic_violations`, train and inference time. Also an error-by-decade comparison, fine-grid SS(V_G) and g_m plots, the t_ox < 2 nm extrapolation test, and a learning curve vs `data/learning_curve_rfr.csv`.
   - Save the best model to `models/`.
   - **First, test `tfet/surrogate.py` quickly** (it's never been run). Time a short training (~200 epochs on ds4) before deciding epoch counts.
2. **Notebook 04:** deep-ensemble uncertainty (confidence bands, ±2σ coverage) on ds5_extended (V_D + T), plus active learning (pick the next simulations by ensemble σ vs random sampling, error vs number of simulated curves).
3. **Notebook 05:** SHAP on FoM models (which knob drives I_ON / I_OFF / SS, checked against the physics); multi-objective inverse design with Optuna NSGA-II (max I_ON, min SS, I_OFF constraint), verifying the best designs with the simulator; an ipywidgets slider explorer.
4. Later, if I ask: a `README.md`, a project report or synopsis, and slides.

Start by confirming you've read this, check that the environment works (`.venv/bin/python -c "import tfet"` from the project root), ask me the three questions under "Waiting for my answer", and tell me the plan for notebook 04 before running anything.
