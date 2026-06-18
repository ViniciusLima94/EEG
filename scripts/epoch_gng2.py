"""
Response-locked epoching for subjectV gng2 task.

Positive (label=1): [-EPOCH_S, 0] before actual button press (GO hits)
Negative (label=0): [-EPOCH_S, 0] before virtual press = nogo_cue + median_GO_RT

Improvements over v1:
  1. EPOCH_S = 1.5 s  — captures the full MRCP ramp onset (~1.5 s before press)
  2. Bandpass 0.5–4 Hz — avoids sub-Hz DC drift while keeping the MRCP band
  3. Late-slope feature  — slope over last 500 ms only (NS'/BP component)
  4. Class balancing     — undersample majority class per fold before fitting
  5. Channel selection   — central/frontocentral/parietal only (MRCP maximal there)

Pipeline:  bandpass → CAR → baseline → features → StandardScaler → shrinkage LDA
  Leave-one-session-out cross-validation
"""

import os
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT     = os.path.join(os.path.dirname(__file__), "..", "data", "subjectV")
DATE     = "20260617"
FS       = 250.0
EPOCH_S  = 1.5                   # 1. longer epoch
N_EPOCH  = int(EPOCH_S * FS)
LATE_MS  = 200                   # ms for late-epoch amplitude window
N_LATE   = int(LATE_MS / 1000 * FS)
LATE_SLOPE_MS = 500              # 3. late-slope window
N_LATE_SLOPE  = int(LATE_SLOPE_MS / 1000 * FS)
N_SEGS   = 6                     # segments (1 per 250 ms at 1.5 s)

# 5. MRCP is maximal at central/frontocentral/parietal electrodes
# 10-20 layout: FP1=ch0 F3=ch1 F7=ch2 C3=ch3 T3=ch4 P3=ch5 T5=ch6 O1=ch7
#               FP2=ch8 F4=ch9 F8=ch10 C4=ch11 T4=ch12 P4=ch13 T6=ch14 O2=ch15
CENTRAL_COLS = ["ch1", "ch3", "ch5", "ch9", "ch11", "ch13"]  # F3 C3 P3 F4 C4 P4
ALL_COLS     = [f"ch{i}" for i in range(16)]


def load_session(stamp):
    eeg = pd.read_csv(os.path.join(ROOT, f"V_eeg_{DATE}_{stamp}.csv"))
    mrk = pd.read_csv(os.path.join(ROOT, f"V_markers_{DATE}_{stamp}.csv"))
    return eeg, mrk


def preprocess(eeg_array):
    """(N, C_all) → bandpass 0.5–4 Hz + CAR, returns (N, C_all) float32."""
    x = filter_data(eeg_array.T.astype(np.float64), FS, 0.5, 4.0,   # 2. 0.5–4 Hz
                    verbose=False).T
    x -= x.mean(axis=1, keepdims=True)  # CAR across all channels
    return x.astype(np.float32)


def epoch_features(epoch):
    """
    epoch: (N_EPOCH, C)  →  1-D feature vector

    Features per channel:
      mean, slope (full), 6 segment means, late mean (200 ms),
      late slope (500 ms), delta power  =  11 × C
    """
    T = epoch.shape[0]
    baseline = epoch[:int(0.1 * FS)].mean(axis=0)
    ep = epoch - baseline

    t_c = np.arange(T, dtype=np.float64) - T / 2

    feat_mean  = ep.mean(axis=0)
    feat_slope = (ep * t_c[:, None]).sum(axis=0) / (t_c ** 2).sum()

    seg_size  = T // N_SEGS
    feat_segs = np.stack(
        [ep[i * seg_size:(i + 1) * seg_size].mean(axis=0) for i in range(N_SEGS)]
    ).T.flatten()

    feat_late = ep[-N_LATE:].mean(axis=0)

    # 3. late slope — trend in the final 500 ms only
    ep_late   = ep[-N_LATE_SLOPE:]
    t_late    = np.arange(N_LATE_SLOPE, dtype=np.float64) - N_LATE_SLOPE / 2
    feat_late_slope = (ep_late * t_late[:, None]).sum(axis=0) / (t_late ** 2).sum()

    _, psd     = welch(ep.T, fs=FS, nperseg=min(T, 128))
    freqs      = np.fft.rfftfreq(min(T, 128), 1.0 / FS)
    feat_delta = psd[:, (freqs >= 0.5) & (freqs <= 4.0)].mean(axis=1)

    return np.concatenate([feat_mean, feat_slope, feat_segs,
                           feat_late, feat_late_slope, feat_delta]).astype(np.float32)


def balance_classes(X, y, rng):
    """Undersample majority class to match minority class size."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n    = min(len(idx0), len(idx1))
    idx  = np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])
    idx  = np.sort(idx)
    return X[idx], y[idx]


def extract_epochs(eeg, mrk, use_cols):
    """Returns raw epochs (n, N_EPOCH, C) and labels (n,)."""
    ts   = eeg["lsl_timestamp"].values
    data = preprocess(eeg[ALL_COLS].values)
    # 5. select central channels after CAR (CAR uses all channels)
    ch_idx = [ALL_COLS.index(c) for c in use_cols]
    data   = data[:, ch_idx]

    go_ts    = mrk.loc[mrk.marker_label == "go_onset",     "lsl_timestamp"].values
    hit_ts   = mrk.loc[mrk.marker_label == "hit",          "lsl_timestamp"].values
    press_ts = mrk.loc[mrk.marker_label == "button_press", "lsl_timestamp"].values
    nogo_ts  = mrk.loc[mrk.marker_label == "nogo_onset",   "lsl_timestamp"].values

    rts = [h - go_ts[go_ts < h][-1] for h in hit_ts if len(go_ts[go_ts < h])]
    median_rt = float(np.median(rts)) if rts else 0.8
    print(f"    {len(rts)} GO hits  median RT = {median_rt*1000:.0f} ms")

    virtual_press_ts = nogo_ts + median_rt

    epochs, labels = [], []
    for ts_arr, label in [(press_ts, 1), (virtual_press_ts, 0)]:
        for ts0 in ts_arr:
            idx   = np.searchsorted(ts, ts0)
            start = idx - N_EPOCH
            if start >= 0 and idx <= len(data):
                epochs.append(data[start:idx])
                labels.append(label)

    return np.array(epochs, dtype=np.float32), np.array(labels, dtype=np.int8)


# ── Load all sessions ─────────────────────────────────────────────────────────
stamps = sorted(
    f.split("_")[-1][:-4]
    for f in os.listdir(ROOT)
    if f.startswith("V_eeg_") and f.endswith(".csv")
)

all_raw, all_y, all_session = [], [], []
for sid, stamp in enumerate(stamps):
    print(f"Session {sid}  ({stamp})")
    eeg, mrk = load_session(stamp)
    raw, y   = extract_epochs(eeg, mrk, CENTRAL_COLS)
    print(f"    epochs: {(y==1).sum()} pos  {(y==0).sum()} neg  shape {raw.shape}")
    all_raw.append(raw)
    all_y.append(y)
    all_session.append(np.full(len(y), sid, dtype=np.int32))

raw_all = np.concatenate(all_raw)   # (M, N_EPOCH, C)
y_all   = np.concatenate(all_y)
s_all   = np.concatenate(all_session)

n_features = epoch_features(raw_all[0]).shape[0]
print(f"\nTotal: {len(y_all)} epochs  {y_all.sum()} pos  "
      f"features={n_features}  channels={len(CENTRAL_COLS)}")

# ── Leave-one-session-out CV ──────────────────────────────────────────────────
aucs = []
rng  = np.random.default_rng(42)

for test_sid in range(len(stamps)):
    tr = s_all != test_sid
    te = s_all == test_sid

    X_tr = np.array([epoch_features(e) for e in raw_all[tr]])
    X_te = np.array([epoch_features(e) for e in raw_all[te]])
    y_tr, y_te = y_all[tr], y_all[te]

    # 4. balance classes in training set
    X_tr_b, y_tr_b = balance_classes(X_tr, y_tr, rng)

    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    clf.fit(X_tr_b, y_tr_b)
    prob = clf.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_te, prob)
    aucs.append(auc)
    print(f"  Session {test_sid}: AUC = {auc:.3f}  "
          f"(train {len(y_tr_b)} balanced  test {te.sum()})")

print(f"\nMean AUC = {np.mean(aucs):.3f}  ±{np.std(aucs):.3f}")
