# TFET Surrogate Builder

Machine-learning surrogates for **line tunnel FETs (TFETs)**, building on
C. Akbar, Y. Li and N. Thoti, *"Device-Simulation-Based Machine Learning Technique for the
Characteristic of Line Tunnel Field-Effect Transistors"*, IEEE Access 10, 53098 (2022).

The paper trains a Random Forest on linear drain current. This project shows that the Random Forest
loses the off-state completely. Predicting **log10(I_D) with a neural network** instead cuts the median
error from 62 % to about 1 % and recovers I_OFF, V_th and the subthreshold swing.

The result is packaged as a **Streamlit web app**. You give it I_D–V_G curves (your own TCAD
data, or curves from our built-in simulator). It trains a neural network on them, then predicts
the curve of any new device in milliseconds.

> **Note:** all results here use data from an analytical line-TFET model (`tfet/physics.py`),
> calibrated to the paper's figures. It is not TCAD data, and accuracy on real TCAD will be lower.

---

## Quick start

Requires Python 3.10+.

```bash
git clone git@github.com:Parithoshkp/Major_project.git
cd Major_project
pip install -r requirements.txt

cd app
streamlit run app.py
```

The app opens in your browser at <http://localhost:8501>.

---

## How to use the app

The app is one page with three steps, top to bottom.

### 1 · Data: choose where the training curves come from

**Option A: you have TCAD data.** Pick **"Upload my TCAD data"** and upload a CSV or TXT file
with one row per point, like this:

```csv
curve_id,L_ov,WK,t_n,t_ox,V_G,I_D
0,5.0,4.40,2.0,3.0,0.00,1.2e-14
0,5.0,4.40,2.0,3.0,0.01,1.3e-14
...
1,2.5,4.30,3.0,1.5,0.00,2.0e-14
...
```

- `V_G` (gate voltage, V) and `I_D` (drain current, A/µm) are required.
- The device-parameter columns are the inputs the model learns from. Use any columns you swept
  (`L_ov`, `WK`, `t_n`, `t_ox`, and also `V_D` or `T` if you varied them).
- `curve_id` is optional. Without it, each unique set of parameters counts as one curve.
- Common names are recognised automatically: `Vg`, `Vgs`, `Ids`, `Id [A/um]`, `tox`, `Lov`, and others.
- Comma, tab or semicolon separators all work. You need at least 5 curves; 50 or more is better.
- Click **"Download a template CSV"** in the app to get an example file.

**Option B: you have no TCAD data.** Pick **"No TCAD data: generate with our Python simulator"**.
Then choose:
- the number of curves to simulate (default 300)
- the simulation noise (default 0.5 %, which mimics TCAD solver noise)
- whether to also vary drain voltage V_D and temperature T

Open **"Preview the data"** to check the table and a few curves.

### 2 · Train: build the model

| Setting | Meaning |
|---|---|
| Training steps | More steps give higher accuracy and slower training. 2,000 takes about 20 s for 300 curves. |
| Held-out test curves | Share of curves hidden from training and used only to score the model (default 20 %). |
| Physics features | Adds the tunnelling length λ and tunnelling area A_tun as inputs. Needs `L_ov`, `t_n`, `t_ox`. |
| Also train the paper's Random Forest | Trains the paper's model on the same data, for comparison. |

Click **🚀 Train model**. You get:
- an **accuracy table** on the held-out curves: median and 95th-percentile error, R², and errors
  in I_ON, I_OFF, V_th and SS_min
- a **test-curve viewer** that plots the true curve against the neural net (dashed) and the Random
  Forest (dotted)

If you change the data or a setting, the app asks you to train again.

### 3 · Predict: get the curve of a new device

- Move the **sliders** to set the device parameters. Each slider covers the range in your training
  data, because predictions are only reliable inside it.
- Optionally pick **"Sweep one parameter"** to draw several curves at once, for example t_ox from 1 to 6 nm.
- Use the four tabs to switch plots: **log I_D–V_G**, **linear I_D–V_G**, **transconductance g_m**
  and **subthreshold swing SS**.
- The table below the plots lists the **figures of merit** of each curve: I_ON, I_OFF, ON/OFF
  ratio, V_th (at 1e-10 A/µm), SS_min and SS_avg.
- Click **Download** to save the predicted curves or the figures of merit as CSV.

In simulator mode, the plot also shows **"simulator (truth)"**, the real answer for that device, so
you can see how close the prediction is. It usually lies right under the neural-net line.

---

## Project structure

```
app/app.py          the Streamlit app
tfet/
  physics.py        analytical line-TFET simulator (Kane BTBT, tunnelling length, TAT floor, temperature)
  dataset.py        sampling devices and simulating I_D–V_G sweeps
  surrogate.py      neural network on log10(I_D), with optional physics features
  metrics.py        figures of merit: I_ON, I_OFF, V_th, SS
  evaluate.py       curve-wise train/test split and error metrics
docs/PROJECT_LOG.md decisions, changes and results
requirements.txt
```

## Key results (analytical data, 2,500-curve set, 20 % held-out curves)

| Model | Median error | I_OFF error | V_th recovered |
|---|---|---|---|
| Random Forest on linear I_D (paper) | 62 % | 3.3 decades | no |
| Neural network on log10 I_D + physics features | 1.0 % | 0.016 decades | yes |

- **Extrapolation** to t_ox < 2 nm, unseen in training: the Random Forest reaches R² 37 % and the neural net 98.8 %.
- **Data efficiency:** the neural net trained on 25 curves beats the Random Forest trained on 1,600.

See `docs/PROJECT_LOG.md` for the full results and limitations.
