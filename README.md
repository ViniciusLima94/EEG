# EEG Trigger Press Prediction

Predicts voluntary button presses from scalp EEG using Motor-Related Cortical Potentials (MRCPs). The goal is to detect the **intention to press** before the press actually occurs — a core requirement for brain-computer interfaces (BCIs).

---

## Live Demo

The animation below replays a trial through the trained LightGBM model at near real-time speed (≈1×). The 4×4 grid shows the 16 filtered EEG channels scrolling in time. The bottom panel shows the smoothed model probability (teal) with detected press events (white lines) and the 500 ms anticipatory horizon (gold shading). Channels and the probability trace turn **coral** whenever the model fires above threshold.

![Live BCI prediction demo](notebooks/report/live_demo_example.gif)

---

## Results — LightGBM LOTO (Subject 9)

Evaluated with **Leave-One-Trial-Out** cross-validation across all 10 trials. Threshold selected on each training fold at ≥80% specificity.

| Metric | Value |
|---|---|
| LOTO AUC | 0.607 |
| LOTO AP | 0.270 |
| Event sensitivity | **70.3 %** (185 / 263 press events detected) |
| False alarms | 31.1 / min |
| Mean detection latency | **397 ms before press onset** |
| Optimal window T | 80 samp = 1.29 s |
| Features | 192 (temporal + frequency + Hjorth, 16 ch × 12) |

The mean latency of 397 ms confirms anticipatory (not reactive) detection: the model fires inside the readiness-potential buildup window, well before the physical press.

---

## Project Structure

```
EEG/
├── notebooks/
│   ├── eeg_lgb_focused.ipynb        ← main model: LightGBM LOTO with full visualisations
│   ├── eeg_lgb_pipeline_opt.ipynb   ← Optuna hyperparameter search
│   ├── eeg_lgbm_multitrial.ipynb    ← per-trial LOTO evaluation
│   ├── LGB_analysis_report.ipynb    ← analysis report and figures
│   ├── eeg_lgb_live_demo.ipynb      ← animated live-inference replay → GIF
│   ├── eeg_pipeline_walkthrough.ipynb  ← step-by-step preprocessing walkthrough
│   ├── eeg_pipeline_v2.ipynb        ← pipeline v2 experiments
│   ├── eeg_erp_evoked.ipynb         ← ERP / evoked-potential analysis
│   ├── eeg_umap.ipynb               ← UMAP feature-space visualisation
│   ├── ica_cleaning.ipynb           ← ICA artefact rejection
│   ├── LabStreamLayer-Test.ipynb    ← LSL stream validation
│   ├── lsl_single_predict.ipynb     ← single-trial LSL inference notebook
│   └── report/                      ← saved plots and animations
│       ├── lgb_timelines.png        ← per-trial probability timelines
│       ├── lgb_erp.png              ← event-related probability (ERP-style)
│       ├── lgb_importance.png       ← feature importance by family
│       ├── lgb_latency.png          ← detection latency distribution
│       └── live_demo_example.gif    ← animated live inference
├── src/
│   ├── preprocessing.py  ← signal processing, windowing, EA, feature helpers
│   ├── postproc.py       ← event-level BCI metrics, artefact removal
│   ├── spectral.py       ← spectral feature extraction
│   ├── io.py             ← data loading utilities
│   └── model/            ← PyTorch model definitions (MLP, Conv1D)
├── scripts/
│   ├── lgb_loto_eval.py        ← standalone LightGBM LOTO evaluation
│   ├── lsl_train_predict.py    ← train + live LSL prediction loop
│   ├── lsl_predict.py          ← live LSL prediction from a saved model
│   ├── run_all_subjects.py     ← batch evaluation across all subjects
│   ├── erp_evoked.py           ← ERP computation script
│   ├── erp_vs_performance.py   ← correlate ERP amplitude with model AUC
│   ├── evaluate_trial.py       ← single-trial evaluation helper
│   ├── inspect_stream.py       ← inspect a live LSL stream
│   ├── convert_way_eeg_gal.py  ← convert WAY-EEG-GAL dataset
│   ├── convert_bnci_001_2014.py    ← convert BNCI 2014-001 dataset
│   └── convert_physionet_eegmmidb.py  ← convert PhysioNet EEG MMIDB
└── data/
    └── subject{0-9}/     ← raw CSV recordings per subject
```

---

## Go/No-Go Task (`scripts/gng_task.py`)

PyGame + LSL experiment for motor intention data collection.

**Run:**
```bash
python scripts/gng_task.py --out-dir data/gng --prefix S01
```

Outputs two CSVs per session: `<prefix>_markers_<datetime>.csv` and `<prefix>_eeg_<datetime>.csv`.

### LSL Marker Codes

| Code | Label | Meaning |
|------|-------|---------|
| 5 | `S1_warning` | S1 warning cue onset |
| 10 | `go_onset` | Go stimulus (S2) onset |
| 20 | `nogo_onset` | No-Go stimulus (S2) onset |
| 30 | `button_press` | Button press detected |
| 11 | `hit` | Go + pressed in time |
| 12 | `miss` | Go + no press |
| 21 | `correct_rejection` | No-Go + no press |
| 22 | `false_alarm` | No-Go + pressed |
| 99 | `end` | Experiment end |

---

## Data Preprocessing

This is the most important thing to understand about the pipeline. Five sequential stages convert raw CSV files into a machine-learning-ready dataset.

### Stage 1 — Load raw CSV

```python
df = pd.read_csv("subject_9_tache_spt_trial_0.csv")
# columns: timestamp, ch1–ch16, button
# ~20 000 samples at 250 Hz ≈ 80 s per trial
```

**Raw CSV columns:**

| Column | Description |
|---|---|
| `timestamp` | Unix timestamp |
| `ch1`–`ch16` | Raw EEG in µV (range ≈ ±100 000 µV before filtering) |
| `button` | 1 while the button is held, 0 otherwise (~380 ms per press) |

The `button` column is converted to **onset-only pulses** immediately after loading — a sustained 1-1-1…1-0 sequence becomes a single 1 at the first sample of each press (`np.diff(b, prepend=0).clip(0)`). This prevents the label from spanning the post-press period.

### Stage 2 — Band-pass filter (0.5–8 Hz)

```python
df[eeg_cols] = filter_data(df[eeg_cols].values.T, fs=250, l_freq=0.5, h_freq=8.0).T
```

The MRCP lives in the **0.5–8 Hz band**:

- **< 0.5 Hz** — electrode drift, sweat artefacts: pure noise
- **0.5–4 Hz (delta)** — Bereitschaftspotential: slow negative readiness ramp starting ~1.5 s before movement
- **4–8 Hz (theta)** — faster motor-preparation oscillations
- **> 8 Hz** — EMG, alpha, and higher-frequency artefacts: removed

Raw amplitudes (±100 kµV) become smooth slow waves (±few µV) after filtering.

### Stage 3 — Spatial filtering and normalisation

Two optional but active steps applied after bandpass filtering:

**Common Average Reference (CAR):** subtracts the instantaneous mean across all 16 channels from each channel sample. Suppresses volume-conducted common-mode noise and highlights spatially focal activity.

```python
arr -= arr.mean(axis=1, keepdims=True)   # (N_samp, C) in-place
```

**Per-trial z-score:** normalises each channel over the full trial duration (mean = 0, std = 1). Removes trial-to-trial amplitude shifts caused by fatigue and electrode drift without touching the temporal structure.

```python
arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8)
```

### Stage 4 — Decimate to 62 Hz

```python
eeg_dec = decimate(arr, q=4, axis=0, zero_phase=True)
# 250 Hz → 62 Hz  (~5 000 samples per trial instead of ~20 000)
```

After the 8 Hz low-pass, no signal content exists above 8 Hz. The Nyquist limit for 62 Hz is 31 Hz — well above the signal band. Decimation reduces memory and windowing cost by 4×.

### Stage 5 — Anticipatory horizon labels

This step makes the problem learnable as a classification task.

The raw `button` column marks *when* a press happened. To be useful for a BCI, the model must predict the press **before** it occurs. `make_horizon_labels` converts the momentary onset signal into an anticipatory label:

```
y_horizon[i] = 1   if any press occurs in y[i : i + HORIZON]
             = 0   otherwise
```

With `HORIZON = 31 samples = 500 ms` at 62 Hz:

```
sample  y_momentary  y_horizon   notes
   960       0           0       (before the 500 ms window)
   961       0           1       ← label flips 500 ms before press
   ...
   991       0           1       (16 ms before press onset)
   992       1           1       ← press ONSET
```

**Effect on class balance (all trials pooled, subject 9):**

| Labels | Class 0 | Class 1 | % positive |
|---|---|---|---|
| `y_momentary` (raw onset) | ~40 000 | ~263 | 0.6 % |
| `y_horizon` (500 ms lookahead) | ~33 000 | ~8 100 | **19.6 %** |

The minority class grows ~30×, giving the model a meaningful training signal.

### Stage 6 — Causal sliding windows

The model receives a **T-sample look-back window** ending at the current time:

```
Window i covers X[i-T : i]  →  predicts label y_horizon[i]
```

Windows are built at `T_MAX = 124 samples (2 s)` once and sliced per Optuna trial:

```python
X_max = build_windows(X_all, y_all, T=T_MAX, per_window_norm=False)
# shape: (41 104, 124, 16)  — M windows × T_MAX timesteps × C channels

# Optuna slices cheaply:
X_t = X_max[:, -T_opt:, :]   # last T_opt timesteps
```

This pre-computation pattern means 25 Optuna trials with different window lengths cost one window build instead of 25.

### Full preprocessing summary

```
Raw CSV  (250 Hz · ~20k samples · button 0/1)
        │
        ▼  band-pass 0.5–8 Hz (MNE filter_data)
        │
        ▼  CAR  — subtract instantaneous channel mean
        │
        ▼  per-trial z-score  — remove amplitude drift
        │
        ▼  decimate ×4  →  62 Hz  (~5k samples)
        │
        ▼  onset-only button  (np.diff.clip(0))
        │
        ▼  make_horizon_labels(HORIZON=31)  →  y[i]=1 if press within 500 ms
        │
        ▼  build_windows(T_MAX=124, stride=1)
        │
  (41 104 windows × 124 timesteps × 16 channels)
```

---

## Feature Engineering

Three complementary feature families are computed per window, all vectorised over the `(N, T, C)` window batch:

### Temporal features — `C × 6 = 96`

| Feature | Description |
|---|---|
| `mean` | Per-channel mean (DC offset / slow drift) |
| `std` | Per-channel standard deviation (signal power) |
| `slope` | Least-squares linear trend (captures the MRCP ramp) |
| `min` | Per-channel minimum value (trough of readiness potential) |
| `argmin` | Normalised position of the minimum (temporal location) |
| `rms` | Root mean square (energy) |

### Frequency features — `C × 3 = 48`

Computed via Welch's method (`nperseg=min(T, 64)`):

| Feature | Description |
|---|---|
| `delta` | Band power 0.5–4 Hz (Bereitschaftspotential band) |
| `theta` | Band power 4–8 Hz (motor-preparation oscillations) |
| `entropy` | Normalised spectral entropy (signal complexity) |

### Hjorth parameters — `C × 3 = 48`

Rotation-invariant temporal descriptors from the MRCP literature:

| Feature | Formula | Meaning |
|---|---|---|
| `activity` | `var(x)` | Signal power |
| `mobility` | `√(var(x') / var(x))` | Approximate mean frequency |
| `complexity` | `mobility(x'') / mobility(x')` | Rate of frequency change |

**Total: 192 features** per window (16 channels × 12 features).

---

## Model — LightGBM with Euclidean Alignment

### Per-fold Euclidean Alignment (EA)

The MRCP amplitude and spatial covariance shift across trials (fatigue, re-attachment). EA whitens each fold's training windows to identity covariance before feature extraction:

```python
# Fit on training windows only
covs  = X_tr.transpose(0,2,1) @ X_tr / T        # (N_tr, C, C)
R_bar = covs.mean(axis=0)                         # (C, C)
W     = inv(sqrtm(R_bar)).real                    # (C, C) whitening matrix

# Apply to train and test
X_tr_aligned = X_tr @ W.T
X_te_aligned = X_te @ W.T    # same W — no test leakage
```

This step is applied **inside every LOTO fold** so the whitening matrix is never estimated on test data.

### Hyperparameter search (Optuna)

25 Optuna trials jointly search the **window length T** and LightGBM hyperparameters, using AUC-ROC as the objective on 5 held-out folds:

| Search space | Range |
|---|---|
| Window T | 12–124 samp (194 ms – 2 s) |
| `n_estimators` | 100–600 |
| `learning_rate` | 0.01–0.30 (log) |
| `max_depth` | 3–9 |
| `num_leaves` | 15–127 |
| `min_child_samples` | 10–120 |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `reg_lambda` | 0.001–10 (log) |

**Why AUC-ROC as the Optuna objective (not AP):** the threshold is calibrated post-hoc at a fixed specificity operating point, so the search should optimise pure ranking quality — which AUC measures directly and independently of class ratio.

### LOTO cross-validation

Full Leave-One-Trial-Out: each of the 10 trials is held out once. For each fold:

1. Apply EA (fit W on training windows only)
2. Compute features from EA-aligned windows
3. Subsample training set to `TRAIN_CAP = 50 000` windows (preserves class ratio via random draw)
4. Fit `LGBMClassifier(class_weight="balanced", ...)`
5. Score training fold → select threshold at `MIN_SPEC` specificity on the training ROC curve
6. Score test fold → report AUC, AP, and event-level metrics

Threshold is the mean of the 10 per-fold training thresholds, applied to the smoothed test scores.

---

## Evaluation Metrics

**Sample-level:**
- **AUC-ROC** — ranking quality, threshold-independent
- **AP** — area under precision-recall curve, sensitive to imbalance

**Event-level** (`event_detection_metrics` in `src/postproc.py`) 
More relevant to real BCI use: was each discrete press event detected before it happened?

| Metric              | Definition                                                                         |
|---------------------|------------------------------------------------------------------------------------|
| `event_sensitivity` | Fraction of press events where `max(P) ≥ threshold` in the 500 ms pre-press window |
| `fa_per_minute`     | Threshold crossings outside any event window, per minute of recording              |
| `mean_latency_ms`   | Mean ms before press onset at which the threshold was first crossed                |

A **500 ms refractory period** is applied to FA counting: after any detection (TP or FA), the next 500 ms cannot open a new FA. The refractory is validated against the 5th-percentile inter-press interval (IPI p5 = 1523 ms >> 500 ms refractory → no real events suppressed).

---

## Configuration Reference

| Parameter | Value | Meaning |
|---|---|---|
| `FS` | 250 Hz | Raw recording rate |
| `DECIM` | 4 | Downsampling factor |
| `FS_EFF` | 62 Hz | Effective rate after decimation |
| `LP` | 0.5 Hz | High-pass cutoff |
| `HP` | 8.0 Hz | Low-pass cutoff |
| `HORIZON` | 31 samp = **500 ms** | Anticipatory label lookahead |
| `T_MAX` | 124 samp = 2.0 s | Maximum window length (Optuna upper bound) |
| `T_OPT` | ~80 samp = 1.29 s | Optimal window (found by Optuna) |
| `SMOOTH_WIN` | 6 samp = 97 ms | Causal rolling-mean on output probabilities |
| `REFRACTORY` | 31 samp = 500 ms | Post-detection FA suppression window |
| `MIN_SPEC` | 0.80–0.90 | Minimum specificity for threshold selection |
| `TRAIN_CAP` | 50 000 | Max windows per LOTO training fold |

---

## Key Signals & Literature

**MRCP components targeted:**

- **Bereitschaftspotential (BP)** — slow negative drift from ~−1.5 s, maximal at Cz
- **Motor Potential (MP)** — steeper negative slope from ~−500 ms
- **Negative Slope (NS')** — peak negativity just before movement onset

**References:**

- Lotte et al. 2007 — *A review of classification algorithms for EEG-based BCIs*  
  SVM with Gaussian/RBF kernel achieves ~96.8 % accuracy for MRCP; Regularized LDA competitive and interpretable.
- Shakeel et al. 2016 — *A Review of Techniques for Detection of Movement Intention Using MRCPs*  
  Matched-filter (template correlation) achieves ~12 ms detection latency; 0.5–8 Hz bandpass and HORIZON ≥ 500 ms recommended.
