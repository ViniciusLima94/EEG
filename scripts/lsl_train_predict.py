#!/usr/bin/env python3
"""
Record L seconds of EEG via LSL, train a LightGBM press detector,
then switch to live prediction on the same stream.

The 17th LSL channel is assumed to be the button signal (same layout as CSVs).

Usage:
    python scripts/lsl_train_predict.py --duration 120
    python scripts/lsl_train_predict.py --duration 60 --t-win 0.5 --save models/session.pkl
"""
import argparse
import pickle
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from pylsl import StreamInlet, resolve_byprop

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import make_horizon_labels, euclidean_align, extract_features  # type: ignore[import]


# ── Running z-score (online) ───────────────────────────────────────────────────


class RunningZScore:
    def __init__(self, capacity: int, n_ch: int):
        self._buf = np.zeros((capacity, n_ch), dtype=np.float32)
        self._ptr = 0
        self._n = 0

    def push(self, x: np.ndarray) -> np.ndarray:
        cap = len(self._buf)
        self._buf[self._ptr % cap] = x
        self._ptr += 1
        self._n = min(self._n + 1, cap)
        if self._n < 2:
            return x.copy()
        a = self._buf[: self._n]
        return ((x - a.mean(0)) / (a.std(0) + 1e-8)).astype(np.float32)


# ── Threshold calibration ──────────────────────────────────────────────────────


def thresh_at_spec(fpr, tpr, thresholds, min_spec: float = 0.95) -> float:
    for spec in [min_spec, 0.90, 0.85, 0.80, 0.70, 0.60]:
        valid = (1 - fpr) >= spec
        if valid.any() and tpr[valid].max() > 0:
            if spec < min_spec:
                print(f"  [thresh] relaxed specificity {min_spec:.0%} → {spec:.0%}")
            return float(thresholds[np.argmax(tpr[valid])])
    return float(thresholds[np.argmax(tpr - fpr)])


# ── Offline training ───────────────────────────────────────────────────────────


def train_model(
    X_rec: np.ndarray, btn_rec: np.ndarray, ts_rec: np.ndarray, args
) -> dict:
    from mne.filter import filter_data
    from scipy.signal import decimate as sp_decimate
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_curve, roc_auc_score

    # Auto-detect sampling rate from recorded timestamps
    FS = round(len(ts_rec) / (ts_rec[-1] - ts_rec[0]))
    DECIM = 4 if FS >= 100 else 1
    FS_EFF = FS // DECIM
    LP, HP = args.lp, args.hp
    HORIZON = max(1, int(args.horizon * FS_EFF))
    T_WIN = max(4, int(args.t_win * FS_EFF))
    SMOOTH_WIN = max(1, int(0.1 * FS_EFF))
    REFRACTORY = int(0.5 * FS_EFF)
    N_CH = X_rec.shape[1]

    print(f"  FS={FS}  DECIM={DECIM}  FS_EFF={FS_EFF}", flush=True)
    print(
        f"  T_WIN={T_WIN} ({T_WIN/FS_EFF*1000:.0f} ms)  HORIZON={HORIZON} ({HORIZON/FS_EFF*1000:.0f} ms)",
        flush=True,
    )

    # Offline bandpass (non-causal, better quality than causal IIR)
    if args.no_filter:
        X_filt = X_rec.copy()
    else:
        X_filt = filter_data(X_rec.T, FS, LP, HP, verbose=False).T

    # Common Average Reference
    X_filt -= X_filt.mean(axis=1, keepdims=True)

    # Decimate
    if DECIM > 1:
        X_dec = sp_decimate(X_filt, DECIM, axis=0, zero_phase=True)
        btn_dec = btn_rec[::DECIM][: len(X_dec)]
    else:
        X_dec = X_filt
        btn_dec = btn_rec

    # Trial z-score
    X_dec = (X_dec - X_dec.mean(0)) / (X_dec.std(0) + 1e-8)

    # Percentile clipping (winsorize p1/p99) — save bounds for streaming inference
    clip_lo = np.percentile(X_dec, 1, axis=0)
    clip_hi = np.percentile(X_dec, 99, axis=0)
    X_dec = np.clip(X_dec, clip_lo, clip_hi)

    # Rising edge → horizon labels
    btn_edges = np.diff(btn_dec.astype(float), prepend=0).clip(0).astype(np.int64)
    n_presses = int(btn_edges.sum())
    print(f"  {n_presses} button presses in recording", flush=True)
    if n_presses < 3:
        print("  [warn] very few presses — model quality will be limited", flush=True)
    if n_presses == 0:
        raise ValueError(
            "No button presses detected. Press the button during recording."
        )

    y_h = make_horizon_labels(btn_edges, HORIZON)

    # Sliding windows (no per-window z-score — trial z-score already applied above)
    wins, labels, mom_labels = [], [], []
    for i in range(T_WIN, len(X_dec)):
        wins.append(X_dec[i - T_WIN : i].astype(np.float32))
        labels.append(int(y_h[i]))
        mom_labels.append(int(btn_edges[i]))

    X_win = np.array(wins)  # (M, T, C)
    y_win = np.array(labels)  # (M,)
    y_mom_win = np.array(mom_labels, dtype=np.int64)  # (M,) momentary press labels

    # Euclidean Alignment — fit on all windows, keep W for online use
    X_ea, W = euclidean_align(X_win)

    # MF template: mean of EA-aligned windows at press onset
    press_idx = np.where(y_mom_win > 0)[0]
    if len(press_idx) > 0:
        mf_template = X_ea[press_idx].mean(axis=0).astype(np.float32)  # (T_WIN, N_CH)
    else:
        mf_template = np.zeros((T_WIN, N_CH), dtype=np.float32)

    # Feature extraction
    print(f"  Extracting features from {len(X_ea)} windows…", flush=True)
    feats = np.array([extract_features(w, FS_EFF) for w in X_ea])

    # Matched-filter features (one correlation score per channel)
    _t_norm = (mf_template - mf_template.mean(0)) / (mf_template.std(0) + 1e-8)
    _mu = X_ea.mean(axis=1, keepdims=True)
    _sd = X_ea.std(axis=1, keepdims=True) + 1e-8
    mf_feats = ((X_ea - _mu) / _sd * _t_norm[None]).mean(axis=1).astype(np.float32)
    feats = np.concatenate([feats, mf_feats], axis=1)

    # Temporal 80/20 train/test split
    split = int(0.8 * len(feats))
    X_tr, X_te = feats[:split], feats[split:]
    y_tr, y_te = y_win[:split], y_win[split:]
    print(
        f"  Train {len(X_tr)} / Test {len(X_te)}  (pos: {y_tr.sum()} / {y_te.sum()})",
        flush=True,
    )

    if y_tr.sum() == 0:
        raise ValueError(
            "No positive examples in training split. Record longer or press more often."
        )

    # Train LightGBM
    model = LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)

    # Threshold from test set ROC
    if y_te.sum() > 0:
        probs = model.predict_proba(X_te)[:, 1]  # type: ignore[index]
        auc = roc_auc_score(y_te, probs)
        fpr, tpr, thr = roc_curve(y_te, probs)
        thresh = thresh_at_spec(fpr, tpr, thr, args.min_spec)
        print(f"  AUC={auc:.4f}  thresh={thresh:.4f}", flush=True)
    else:
        print("  [warn] no positives in test split — using threshold 0.5", flush=True)
        thresh = 0.5

    return dict(
        model=model,
        W=W,
        mf_template=mf_template,
        T_OPT=T_WIN,
        FS=FS,
        DECIM=DECIM,
        FS_EFF=FS_EFF,
        LP=LP,
        HP=HP,
        HORIZON=HORIZON,
        SMOOTH_WIN=SMOOTH_WIN,
        REFRACTORY=REFRACTORY,
        thresh=thresh,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        USE_FILTER=not args.no_filter,
        USE_EA=True,
        EEG_COLS=[f"ch{i+1}" for i in range(N_CH)],
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="LSL record → train → live predict")
    ap.add_argument(
        "--duration", type=float, default=120, help="Recording duration (s)"
    )
    ap.add_argument("--stream", default="PiEEG", help="LSL stream name")
    ap.add_argument(
        "--t-win", type=float, default=0.5, help="Feature window length (s)"
    )
    ap.add_argument("--horizon", type=float, default=2.5, help="Label lookahead (s)")
    ap.add_argument("--lp", type=float, default=0.5, help="Bandpass low cut (Hz)")
    ap.add_argument("--hp", type=float, default=8.0, help="Bandpass high cut (Hz)")
    ap.add_argument(
        "--min-spec", type=float, default=0.95, help="Min specificity for threshold"
    )
    ap.add_argument(
        "--warmup", type=int, default=5, help="Warmup seconds before detection"
    )
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable bandpass filter; use only CAR (train and predict)")
    ap.add_argument("--hop", type=int, default=0,
                    help="New decimated samples between predictions (0 = T_OPT, non-overlapping)")
    ap.add_argument("--save", default=None, help="Path to save trained model pkl")
    ap.add_argument(
        "--log",
        default=None,
        help="Path to save prediction log CSV (default: auto-named)",
    )
    args = ap.parse_args()

    # ── Connect ────────────────────────────────────────────────────────────────
    print(f"Searching for LSL stream '{args.stream}' …", flush=True)
    streams = resolve_byprop("name", args.stream, timeout=10)
    if not streams:
        sys.exit(f"No stream '{args.stream}' found.")
    inlet = StreamInlet(streams[0])
    N_CH_STREAM = streams[0].channel_count()
    N_CH = N_CH_STREAM - 1  # last channel = button
    print(f"Connected  ({N_CH_STREAM} channels: {N_CH} EEG + 1 button)\n", flush=True)

    # ── Phase 1: Record ────────────────────────────────────────────────────────
    print(
        f"── Recording {args.duration:.0f}s — press the button during the task ──\n",
        flush=True,
    )
    raw_eeg: list = []
    raw_btn: list = []
    raw_ts: list = []
    t_start: float = 0.0
    n_presses = 0
    btn_prev = 0.0
    recording = False

    while True:
        sample, ts = inlet.pull_sample()
        if sample is None:
            continue
        if not recording:
            t_start = float(ts)  # type: ignore[arg-type]
            recording = True

        eeg = sample[:N_CH]
        btn = float(sample[N_CH_STREAM - 1])
        raw_eeg.append(eeg)
        raw_btn.append(btn)
        raw_ts.append(ts)

        if btn > 0.5 and btn_prev <= 0.5:
            n_presses += 1
        btn_prev = btn

        elapsed = float(ts) - t_start  # type: ignore[arg-type]
        if len(raw_ts) % 500 == 0:
            pct = min(elapsed / args.duration, 1.0)
            bar = int(pct * 30) * "█" + int((1 - pct) * 30) * "░"
            print(
                f"\r  {bar}  {elapsed:.0f}/{args.duration:.0f}s  {n_presses} presses",
                end="",
                flush=True,
            )

        if elapsed >= args.duration:
            break

    pct_bar = "█" * 30
    print(
        f"\r  {pct_bar}  {args.duration:.0f}/{args.duration:.0f}s  {n_presses} presses — done\n",
        flush=True,
    )

    X_rec = np.array(raw_eeg, dtype=np.float64)
    btn_rec = np.array(raw_btn, dtype=np.float64)
    ts_rec = np.array(raw_ts, dtype=np.float64)

    # ── Phase 2: Train ─────────────────────────────────────────────────────────
    print("── Training ──\n", flush=True)
    m = train_model(X_rec, btn_rec, ts_rec, args)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(m, fh)
        print(f"  Saved → {save_path}", flush=True)

    print("\nTraining complete.\n", flush=True)

    # ── Phase 3: Live prediction ───────────────────────────────────────────────
    model = m["model"]
    W = m["W"].astype(np.float64)
    T_OPT = int(m["T_OPT"])
    FS = int(m["FS"])
    DECIM = int(m["DECIM"])
    FS_EFF = int(m["FS_EFF"])
    LP, HP = float(m["LP"]), float(m["HP"])
    SMOOTH_WIN = int(m["SMOOTH_WIN"])
    thresh = float(m["thresh"])
    REFRACTORY = int(m["REFRACTORY"])
    clip_lo = m["clip_lo"].astype(np.float32)
    clip_hi = m["clip_hi"].astype(np.float32)
    USE_FILTER  = bool(m.get("USE_FILTER", True))
    USE_EA      = bool(m.get("USE_EA", True))
    mf_template = m.get("mf_template", None)
    if mf_template is not None:
        mf_template = mf_template.astype(np.float32)

    if not USE_FILTER:
        print("  [info] bandpass filter disabled (training setting) — applying CAR only")
    if not USE_EA:
        print("  [info] Euclidean Alignment disabled (training setting)")

    # Causal bandpass for online inference
    nyq = FS / 2.0
    sos: np.ndarray = butter(4, [LP / nyq, HP / nyq], btype="band", output="sos")  # type: ignore[assignment]
    fz = [np.zeros((sos.shape[0], 2)) for _ in range(N_CH)]

    def bandpass_car(raw: np.ndarray) -> np.ndarray:
        if not USE_FILTER:
            out = np.array([float(v) if np.isfinite(v) else 0.0 for v in raw[:N_CH]], dtype=np.float32)
            out -= out.mean()
            return out
        out = np.empty(N_CH, dtype=np.float32)
        for c in range(N_CH):
            sample_val = float(raw[c]) if np.isfinite(raw[c]) else 0.0
            y, fz[c] = sosfilt(sos, [sample_val], zi=fz[c])
            if not np.isfinite(fz[c]).all():
                fz[c] = np.zeros_like(fz[c])
            out[c] = y[0] if np.isfinite(y[0]) else 0.0
        out -= out.mean()
        return out

    # Pre-warm filter state and running z-score from the tail of the recording
    # so inference starts with a stable filter (no IIR transient) and a
    # pre-filled z-score buffer — same condition as the offline training pipeline.
    print("Pre-warming filter and z-score from recording tail …", end="", flush=True)
    rz = RunningZScore(30 * FS_EFF, N_CH)
    for raw_sample in X_rec[-FS:]:   # last 1 second of raw recording
        x_warm = np.nan_to_num(bandpass_car(raw_sample.astype(np.float32)), nan=0.0, posinf=0.0, neginf=0.0)
        rz.push(np.clip(x_warm, clip_lo, clip_hi))
    print(" done.", flush=True)

    HOP       = args.hop if args.hop > 0 else T_OPT
    win_buf:   list = []
    score_buf = deque([0.0] * SMOOTH_WIN, maxlen=SMOOTH_WIN)
    n_raw = 0
    n_dec = 0
    last_det = -REFRACTORY

    print(f"  hop={HOP} samp ({HOP/FS_EFF*1000:.0f} ms per prediction)", flush=True)

    # ── Prediction log ─────────────────────────────────────────────────────────
    log_rows: list = []
    btn_live = 0.0

    def infer() -> None:
        nonlocal last_det
        win_arr = np.array(win_buf[:T_OPT], dtype=np.float64)
        if not np.isfinite(win_arr).all():
            return
        if USE_EA:
            win = np.nan_to_num(win_arr @ W.T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            win = win_arr.astype(np.float32)
        feats = extract_features(win, FS_EFF)
        if mf_template is not None:
            t_norm = (mf_template - mf_template.mean(0)) / (mf_template.std(0) + 1e-8)
            mu_mf = win.mean(0); sd_mf = win.std(0) + 1e-8
            mf = ((win - mu_mf) / sd_mf * t_norm).mean(0).astype(np.float32)
            feats = np.concatenate([feats, mf])
        score = float(model.predict_proba(feats[None, :])[0, 1])
        score_buf.append(score)
        score_s = float(np.mean(score_buf))

        det = int(score_s >= thresh and (n_dec - last_det) > REFRACTORY)
        if det:
            last_det = n_dec
            print(f"[DETECTION]  t={n_dec/FS_EFF:8.2f}s  score={score_s:.4f}", flush=True)
        elif n_dec % (FS_EFF * 5) == 0:
            print(f"[running]    t={n_dec/FS_EFF:8.2f}s  score={score_s:.4f}", flush=True)

        log_rows.append(
            (round(n_dec / FS_EFF, 4), round(btn_live, 0), round(score_s, 5), det)
        )

    # Warmup
    warmup_dec = args.warmup * FS_EFF
    print(f"Warming up for {args.warmup}s …", end="", flush=True)
    while n_dec < warmup_dec:
        sample, _ = inlet.pull_sample()
        if sample is None:
            continue
        x = np.clip(
            np.nan_to_num(bandpass_car(np.array(sample[:N_CH], dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0), clip_lo, clip_hi
        )
        n_raw += 1
        if n_raw % DECIM == 0:
            rz.push(x)   # settle z-score only; don't fill win_buf
            n_dec += 1
    win_buf.clear()
    print(" done.", flush=True)

    print(
        f"\n── Detection running  (thresh={thresh:.4f}, Ctrl-C to stop) ──\n",
        flush=True,
    )
    try:
        while True:
            sample, _ = inlet.pull_sample()
            if sample is None:
                continue
            btn_live = float(sample[N_CH_STREAM - 1])
            x = np.clip(
                bandpass_car(np.array(sample[:N_CH], dtype=np.float32)),
                clip_lo,
                clip_hi,
            )
            n_raw += 1
            if n_raw % DECIM != 0:
                continue
            win_buf.append(rz.push(x))
            n_dec += 1
            if len(win_buf) >= T_OPT:
                infer()
                win_buf = win_buf[HOP:]
    except KeyboardInterrupt:
        print("\nStopped.")

    # ── Save prediction log ────────────────────────────────────────────────────
    if log_rows:
        import csv, datetime

        log_path = args.log
        if log_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"predict_log_{ts}.csv"
        with open(log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "button", "score", "detection"])
            w.writerows(log_rows)
        print(f"Prediction log saved → {log_path}  ({len(log_rows)} rows)")


if __name__ == "__main__":
    main()
