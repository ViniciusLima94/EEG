# EEG Trigger Press Prediction

Predicts voluntary button presses from scalp EEG using Motor-Related Cortical Potentials (MRCPs). The goal is to detect the **intention to press** before the press actually occurs — a core requirement for brain-computer interfaces (BCIs).

---

## Project Structure

```
EEG/
├── data/
│   └── subject{0-9}/
│       ├── subject_{N}_tache_spt_trial_{0-9}.csv   ← raw recordings
│       └── model*/                                  ← saved model artefacts
├── notebooks/
│   ├── eeg_lgbm_multitrial.ipynb       ← LightGBM, LOTO CV
│   ├── eeg_rf_pooled_balanced.ipynb    ← Random Forest, pooled balanced
│   ├── eeg_svm_multitrial.ipynb        ← SVM RBF + Regularized LDA, LOTO CV
│   ├── eeg_causal_rf.ipynb             ← Single-trial RF + cross-trial eval
│   ├── eeg_conv1d_causal.ipynb         ← Conv1D single trial
│   └── eeg_conv1d_causal_multitrial.ipynb
└── src/
    ├── preprocessing.py   ← all signal processing + windowing utilities
    └── postproc.py        ← event-level BCI metrics, artifact removal
```

---

## Dataset

| Property | Value |
|---|---|
| Subjects | 10 (subject 0–9) |
| Trials per subject | 7–10 |
| Trial duration | ~60–80 s |
| Recording rate | 250 Hz |
| EEG channels | 16 (ch1–ch16) |
| Task | Voluntary trigger press (self-paced, task = `spt`) |

**Raw CSV columns:** `timestamp`, `ch1`–`ch16`, `button`

- `ch1`–`ch16` are raw EEG voltages in µV (~±100,000 µV before filtering)
- `button` is 1 while the button is held down, 0 otherwise. A single press lasts ~320–500 ms (80–125 samples at 250 Hz).
- Trial 0 for subject 9 has 17 presses in 81 s; later trials reach ~30–35 presses in 60–70 s.

---

## How Training Data Is Constructed

This is the most important thing to understand. The pipeline has five sequential stages.

### Stage 1 — Load raw CSV

```python
df = pd.read_csv("subject_9_tache_spt_trial_0.csv")
# df shape: (20338, 18) — 20338 samples × (timestamp + 16 EEG + button)
# at 250 Hz this is 81 seconds
```

### Stage 2 — Band-pass filter (0.5–8 Hz)

```python
df[eeg_cols] = filter_data(df[eeg_cols].values.T, fs=250, l_freq=0.5, h_freq=8.0).T
```

The MRCP (Motor-Related Cortical Potential) is a slow negative drift in the EEG that begins **~1–1.5 s before voluntary movement**. It lives in the **0.5–8 Hz band**:

- Below 0.5 Hz: electrode drift, sweat artefacts — noise, not signal
- 0.5–4 Hz (delta): the slow readiness ramp (Bereitschaftspotential)
- 4–8 Hz (theta): faster motor-preparation oscillations

Filtering removes everything outside this band. The raw amplitudes (~±100 kµV) become small smooth waves (~±few µV).

### Stage 3 — Decimate to 62 Hz

```python
from scipy.signal import decimate
eeg_dec = decimate(df[eeg_cols].values, q=4, axis=0, zero_phase=True)
# shape: (5085, 16) — every 4th sample kept, anti-aliased
# effective rate: 250 / 4 = 62.5 Hz ≈ FS_EFF = 62 Hz
```

After filtering, the signal has no content above 8 Hz, so 62 Hz is more than sufficient (Nyquist = 31 Hz). Decimation reduces memory by 4× and speeds up windowing.

**After this stage:** one trial = ~5000 samples at 62 Hz instead of ~20000 at 250 Hz.

### Stage 4 — Build anticipatory labels (`make_horizon_labels`)

This is the key step that makes the problem learnable.

**The raw `button` column is NOT used directly as a label.** Here is why:

The button is 1 for ~380 ms after the subject decides to press. That single-sample onset tells the model *when* the press happened — but to be useful for a BCI, the model needs to predict the press *before* it happens. The MRCP provides a detectable neural signature starting 1–1.5 s before the movement onset.

`make_horizon_labels` converts the momentary press signal into an anticipatory label:

```
y_horizon[i] = 1   if any press occurs in y[i : i + HORIZON]
y_horizon[i] = 0   otherwise
```

With `HORIZON = int(1.0 * 62) = 62 samples = 1000 ms`:

```
sample  y_momentary  y_horizon   notes
  1430       0           0      (before the 1 s window)
  1431       0           1      (984 ms before press onset)  ← label flips to 1
  1432       0           1      (968 ms before press onset)
    ...
  1491       0           1      (16 ms before press onset)
  1492       1           1      <-- press ONSET (button goes 0→1)
  1493       1           1
    ...
```

**Effect on class balance (trial 0, single trial):**

| Labels | Class 0 | Class 1 | % positive |
|---|---|---|---|
| `y_momentary` (raw button) | 4,663 | 406 | 8.0% |
| `y_horizon` (1000 ms lookahead) | 3,626 | 1,443 | **28.5%** |

The minority class grows ~3.5×, giving the model a much stronger training signal.

### Stage 5 — Build causal sliding windows (`build_windows`)

The model does not see individual EEG samples — it sees a **window** of T consecutive samples ending at the current time:

```
Window i covers X[i-T : i]  →  predicts label y[i]
```

```python
X_win, y_win = build_windows(X, y_horizon, T=62, stride=1)
# X shape before:  (5085, 16)   — N samples × C channels
# X_win shape:     (5023, 992)  — M windows × (C×T features), M = N - T
# y_win shape:     (5023,)      — one label per window endpoint
```

**Inside each window, per-window z-score normalisation is applied:**

```python
w = X[i-T : i]          # (T, C) — raw window
mu = w.mean(axis=0)      # (C,)   — per-channel mean
sd = w.std(axis=0) + ε   # (C,)   — per-channel std
w = (w - mu) / sd        # (T, C) — normalised
```

This removes trial-level and session-level amplitude drift independently in every window, so the model sees *relative* EEG shape rather than absolute amplitude. This is critical for cross-trial generalisation.

The final feature vector is `w.flatten()` → shape `(C×T,)`. For C=16 channels and T=62 samples (1 s), that is **992 features per window**.

---

## Full Pipeline Summary

```
Raw CSV (250 Hz, ~20k samples, button 0/1)
        │
        ▼ filter_data(lp=0.5, hp=8.0)    — keep 0.5–8 Hz MRCP band
        │
        ▼ decimate(q=4)                   — 250 Hz → 62 Hz  (~5k samples)
        │
        ▼ make_horizon_labels(HORIZON=62) — y[i]=1 if press within next 1 s
        │
        ▼ build_windows(T=T, stride=1)   — causal windows, per-window z-score
        │
 (M windows × C×T features,  M labels)
        │
        ▼ Classifier (RF / LightGBM / SVM / LDA)
        │
 P(press in next 1 s) per sample
        │
        ▼ smooth_probs (rolling mean, 500 ms)
        │
        ▼ threshold (max sensitivity s.t. specificity ≥ 0.80)
        │
 Binary prediction: "press imminent" / "idle"
```

---

## Cross-Trial Generalisation

The main challenge is that the MRCP amplitude and shape vary across trials (fatigue, attention, electrode drift). The multi-trial notebooks address this differently:

| Notebook | Strategy | CV Method |
|---|---|---|
| `eeg_lgbm_multitrial` | Train on all 9 other trials, test on 1 | LOTO (Leave-One-Trial-Out) |
| `eeg_svm_multitrial` | Same | LOTO |
| `eeg_rf_pooled_balanced` | Pool all train trials, K-balanced sample | GroupKFold(3) by trial ID in CV |
| `eeg_causal_rf` | Single training trial, evaluate on all others | N/A (no CV — single split) |

**GroupKFold is essential:** adjacent EEG windows share T-1 overlapping samples. If the same-trial windows end up in both train and val, AUC will be inflated because the model has seen near-identical inputs. GroupKFold ensures all windows from a given trial stay in the same fold.

---

## Window Boundary Safety

`build_windows` skips any window whose T-sample lookback would cross a trial boundary:

```python
for i in range(T, N, stride):
    if groups[i] != groups[i - T]:
        continue   # window would span two trials — skip
```

Without this, the first T windows of each trial would include EEG from the previous trial — corrupting the temporal context.

---

## Pre-computation Pattern (multi-trial notebooks)

Building windows is the expensive step. The notebooks pre-compute at `T_MAX` once:

```python
_X_max = build_windows(X_all, y_all, T=T_MAX)  # (M, T_MAX, C)

# Optuna then slices cheaply:
def get_windows_for_T(T):
    X = _X_max[:, T_MAX - T:, :]  # last T timesteps
    # z-score per window
    return X.reshape(M, T * C), y_all
```

This means 100 Optuna trials with different T values cost ~1 window build instead of 100.

---

## Configuration Reference

| Parameter | Value | Meaning |
|---|---|---|
| `FS` | 250 Hz | Raw recording rate |
| `DECIM` | 4 | Downsampling factor |
| `FS_EFF` | 62 Hz | Effective rate after decimation |
| `LP` | 0.5 Hz | High-pass cutoff (removes electrode drift) |
| `HP` | 8.0 Hz | Low-pass cutoff (removes EMG / alpha) |
| `HORIZON` | 62 samples = 1000 ms | Anticipatory label lookahead |
| `T_MAX` | 62–93 samples | Maximum window length searched by Optuna |
| `STRIDE` | 1 | Window step size (1 = one window per sample) |
| `SMOOTH_WINDOW` | 31 samples = 500 ms | Causal rolling-mean smoothing of output probabilities |

---

## Key Signals & Literature

**MRCP components targeted:**
- **Bereitschaftspotential (BP)** — slow negative drift starting ~1.5 s before movement, max at Cz
- **Motor Potential (MP)** — steeper negative slope ~500 ms before movement
- **Negative Slope (NS')** — peak negativity immediately before movement

**References:**
- Lotte et al. 2007 — *A review of classification algorithms for EEG-based BCIs*
  - Best performing: SVM with Gaussian/RBF kernel (~96.8% accuracy for MRCP)
  - Regularized LDA (Ledoit-Wolf) competitive and interpretable
- Shakeel et al. 2016 — *A Review of Techniques for Detection of Movement Intention Using MRCPs*
  - Matched filter (template correlation) achieves ~12 ms detection latency
  - 0.5–8 Hz bandpass recommended; HORIZON ≥ 1000 ms to capture full BP
