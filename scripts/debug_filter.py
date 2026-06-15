#!/usr/bin/env python3
"""
Offline filter diagnostics — no LSL stream required.

Feeds real EEG data sample-by-sample through the same pipeline used in
lsl_predict.py and reports exactly where NaN first appears, comparing
filter ON vs filter OFF.

Usage:
    python scripts/debug_filter.py
    python scripts/debug_filter.py --csv data/subject0/subject_0_tache_gng_trial_0.csv
    python scripts/debug_filter.py --model notebooks/models/lgb_demo.pkl
"""
import argparse
import pickle
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import compute_hjorth  # type: ignore[import]

LP, HP, FS = 0.5, 8.0, 500
DECIM = 4
FS_EFF = FS // DECIM
ORDER = 4

EEG_COLS = [f"ch{i+1}" for i in range(16)]


# ── Minimal feature extractor (mirrors lsl_predict.py) ────────────────────────

def extract_features(w: np.ndarray) -> np.ndarray:
    from scipy.signal import welch
    T = w.shape[0]
    t_c   = np.arange(T, dtype=np.float64) - (T - 1) / 2
    t_var = (t_c ** 2).sum() + 1e-12
    w64   = w.astype(np.float64)
    mean_ = w64.mean(0); std_  = w64.std(0)
    slope = (w64 * t_c[:, None]).sum(0) / t_var
    min_  = w64.min(0); argm_ = w64.argmin(0) / T
    rms_  = np.sqrt((w64 ** 2).mean(0))
    temporal = np.stack([mean_, std_, slope, min_, argm_, rms_], axis=1).flatten()
    f, p = welch(w64.T, fs=FS_EFF, nperseg=min(T, 64), axis=-1)
    df   = f[1] - f[0]
    dp   = p[:, (f >= 0.5) & (f <= 4.0)].sum(-1) * df
    tp   = p[:, (f >= 4.0) & (f <= 8.0)].sum(-1) * df
    pn   = p[:, f > 0] / (p[:, f > 0].sum(-1, keepdims=True) + 1e-12)
    se   = -(pn * np.log(pn + 1e-12)).sum(-1)
    freq = np.stack([dp, tp, se], axis=1).flatten()
    return np.concatenate([temporal, freq, compute_hjorth(w).flatten()]).astype(np.float32)


class RunningZScore:
    def __init__(self, capacity, n_ch):
        self._buf = np.zeros((capacity, n_ch), dtype=np.float32)
        self._ptr = 0; self._n = 0

    def push(self, x):
        cap = len(self._buf)
        self._buf[self._ptr % cap] = x
        self._ptr += 1
        self._n = min(self._n + 1, cap)
        if self._n < 2:
            return x.copy()
        a = self._buf[:self._n]
        return ((x - a.mean(0)) / (a.std(0) + 1e-8)).astype(np.float32)


# ── Pipeline simulation ────────────────────────────────────────────────────────

def run_pipeline(
    samples: np.ndarray,   # (N, C) raw EEG samples
    W: np.ndarray,         # (C, C) EA matrix
    T_OPT: int,
    use_filter: bool,
    label: str,
    verbose: bool = True,
) -> dict:
    N, N_CH = samples.shape
    nyq = FS / 2.0
    sos = butter(ORDER, [LP / nyq, HP / nyq], btype="band", output="sos")
    fz  = [np.zeros((sos.shape[0], 2)) for _ in range(N_CH)]
    rz  = RunningZScore(30 * FS_EFF, N_CH)
    win_buf = deque(maxlen=T_OPT)

    stats = dict(
        label=label,
        use_filter=use_filter,
        n_nan_raw=0,
        n_nan_after_filter=0,
        n_nan_filter_state=0,
        n_nan_after_zscore=0,
        n_nan_in_win=0,
        n_nan_after_ea=0,
        n_infer_attempted=0,
        n_infer_skipped_nan=0,
        first_nan_sample=None,
        first_nan_source=None,
    )

    def record_nan(source, idx):
        if stats["first_nan_sample"] is None:
            stats["first_nan_sample"] = idx
            stats["first_nan_source"] = source

    for i, raw in enumerate(samples):
        # ── 1. Raw input ──────────────────────────────────────────────────────
        if not np.isfinite(raw).all():
            stats["n_nan_raw"] += 1
            record_nan("raw_input", i)

        # ── 2. Filter / CAR ───────────────────────────────────────────────────
        out = np.empty(N_CH, dtype=np.float32)
        if use_filter:
            for c in range(N_CH):
                sv = float(raw[c]) if np.isfinite(raw[c]) else 0.0
                y, fz[c] = sosfilt(sos, [sv], zi=fz[c])
                if not np.isfinite(fz[c]).all():
                    stats["n_nan_filter_state"] += 1
                    record_nan("filter_state_ch%d" % c, i)
                    fz[c] = np.zeros_like(fz[c])
                out[c] = y[0] if np.isfinite(y[0]) else 0.0
        else:
            for c in range(N_CH):
                out[c] = float(raw[c]) if np.isfinite(raw[c]) else 0.0

        out -= out.mean()
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.isfinite(out).all():
            stats["n_nan_after_filter"] += 1
            record_nan("after_filter_car", i)

        # ── 3. Decimate ───────────────────────────────────────────────────────
        if i % DECIM != 0:
            continue

        # ── 4. Running z-score ────────────────────────────────────────────────
        z = rz.push(out)
        if not np.isfinite(z).all():
            stats["n_nan_after_zscore"] += 1
            record_nan("running_zscore", i)

        win_buf.append(z)

        if len(win_buf) < T_OPT:
            continue

        # ── 5. Window array ───────────────────────────────────────────────────
        win_arr = np.array(win_buf, dtype=np.float64)
        if not np.isfinite(win_arr).all():
            stats["n_nan_in_win"] += 1
            record_nan("win_buf", i)

        # ── 6. EA matmul ──────────────────────────────────────────────────────
        stats["n_infer_attempted"] += 1
        if not np.isfinite(win_arr).all():
            stats["n_infer_skipped_nan"] += 1
            continue

        ea = (win_arr @ W.T)
        if not np.isfinite(ea).all():
            stats["n_nan_after_ea"] += 1
            record_nan("after_ea_matmul", i)

    if verbose:
        tag = f"[{label}]"
        print(f"\n{tag} use_filter={use_filter}")
        print(f"  raw NaN inputs      : {stats['n_nan_raw']}")
        print(f"  filter state resets : {stats['n_nan_filter_state']}")
        print(f"  NaN after filter/CAR: {stats['n_nan_after_filter']}")
        print(f"  NaN after z-score   : {stats['n_nan_after_zscore']}")
        print(f"  NaN in win_buf      : {stats['n_nan_in_win']}")
        print(f"  NaN after EA matmul : {stats['n_nan_after_ea']}")
        print(f"  infer attempted     : {stats['n_infer_attempted']}")
        print(f"  infer skipped (NaN) : {stats['n_infer_skipped_nan']}")
        if stats["first_nan_sample"] is not None:
            print(f"  >>> First NaN at sample {stats['first_nan_sample']} — source: {stats['first_nan_source']}")
        else:
            print(f"  >>> No NaN detected anywhere in the pipeline")

    return stats


# ── Raw signal stats ───────────────────────────────────────────────────────────

def report_raw_stats(samples: np.ndarray):
    print("\n── Raw signal statistics ──")
    print(f"  shape       : {samples.shape}")
    print(f"  dtype       : {samples.dtype}")
    print(f"  NaN count   : {np.isnan(samples).sum()}")
    print(f"  Inf count   : {np.isinf(samples).sum()}")
    print(f"  min / max   : {samples.min():.4g} / {samples.max():.4g}")
    print(f"  mean / std  : {samples.mean():.4g} / {samples.std():.4g}")
    per_ch_std = samples.std(axis=0)
    flat = per_ch_std < 0.01
    if flat.any():
        print(f"  flat channels (std<0.01): {np.where(flat)[0].tolist()}")
    else:
        print(f"  no flat channels")


# ── W matrix stats ─────────────────────────────────────────────────────────────

def report_W_stats(W: np.ndarray):
    print("\n── EA matrix W ──")
    print(f"  shape    : {W.shape}")
    print(f"  NaN      : {np.isnan(W).sum()}")
    print(f"  Inf      : {np.isinf(W).sum()}")
    print(f"  min/max  : {W.min():.4g} / {W.max():.4g}")
    cond = np.linalg.cond(W)
    print(f"  cond(W)  : {cond:.4g}{'  [ILL-CONDITIONED — likely source of NaN]' if cond > 1e10 else ''}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Offline filter NaN diagnostics")
    ap.add_argument(
        "--csv",
        default="data/subject0/subject_0_tache_gng_trial_0.csv",
        help="EEG CSV to feed through the pipeline",
    )
    ap.add_argument(
        "--model",
        default="notebooks/models/lgb_demo.pkl",
        help="Trained model pkl (for W and T_OPT)",
    )
    ap.add_argument(
        "--n-samples", type=int, default=0,
        help="Limit to first N samples (0 = all)",
    )
    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    csv_path = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    cols = [c for c in EEG_COLS if c in df.columns]
    if not cols:
        sys.exit("No EEG columns found in CSV. Check column names.")
    samples = df[cols].values.astype(np.float64)
    if args.n_samples > 0:
        samples = samples[: args.n_samples]
    report_raw_stats(samples)

    # ── Load model ─────────────────────────────────────────────────────────────
    pkl_path = ROOT / args.model if not Path(args.model).is_absolute() else Path(args.model)
    print(f"\nLoading model: {pkl_path}")
    with open(pkl_path, "rb") as fh:
        m = pickle.load(fh)

    W     = m["W"].astype(np.float64)
    T_OPT = int(m["T_OPT"])
    report_W_stats(W)
    print(f"\n  T_OPT={T_OPT} samples ({T_OPT/FS_EFF*1000:.0f} ms)")

    # ── Run both pipelines ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running pipeline — FILTER ON")
    print("=" * 60)
    run_pipeline(samples, W, T_OPT, use_filter=True,  label="FILTER_ON")

    print("\n" + "=" * 60)
    print("Running pipeline — FILTER OFF (CAR only)")
    print("=" * 60)
    run_pipeline(samples, W, T_OPT, use_filter=False, label="FILTER_OFF")

    print()


if __name__ == "__main__":
    main()
