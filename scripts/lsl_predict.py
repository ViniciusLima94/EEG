#!/usr/bin/env python3
"""
Online EEG press detection via LSL.

Connects to a PiEEG stream, warms up the filter and z-score buffer,
then streams continuous predictions using the trained LightGBM model.

Usage:
    python scripts/lsl_predict.py
    python scripts/lsl_predict.py --model notebooks/models/lgb_demo.pkl
    python scripts/lsl_predict.py --stream PiEEG --hop 20 --warmup 5
"""
import argparse
import csv
import datetime
import pickle
import sys
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
from scipy.signal import butter, sosfilt
from pylsl import StreamInlet, resolve_byprop

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import extract_features  # type: ignore[import]


# ── Running z-score (online approximation of trial-level normalisation) ────────

class RunningZScore:
    """Maintains a rolling window to approximate trial-level z-score online."""

    def __init__(self, capacity: int, n_ch: int):
        self._buf = np.zeros((capacity, n_ch), dtype=np.float32)
        self._ptr = 0
        self._n   = 0

    def push(self, x: np.ndarray) -> np.ndarray:
        cap = len(self._buf)
        self._buf[self._ptr % cap] = x
        self._ptr += 1
        self._n = min(self._n + 1, cap)
        if self._n < 2:
            return x.copy()
        a = self._buf[:self._n]
        return ((x - a.mean(0)) / (a.std(0) + 1e-8)).astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Online EEG press detection via LSL")
    ap.add_argument("--model",  default="notebooks/models/lgb_demo.pkl",
                    help="Path to trained model pkl (relative to repo root or absolute)")
    ap.add_argument("--stream", default="PiEEG", help="LSL stream name")
    ap.add_argument("--hop", type=int, default=0,
                    help="New decimated samples between predictions (0 = T_OPT, non-overlapping)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Seconds to collect before starting detection (warms up filter + z-score)")
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable bandpass filter (only CAR); must match training setting")
    ap.add_argument("--debug", action="store_true",
                    help="Print per-stage NaN diagnostics and raw stream stats")
    ap.add_argument("--log", default="",
                    help="CSV log path (default: logs/lsl_predict_<datetime>.csv; empty = auto)")
    args = ap.parse_args()

    if args.debug:
        # Convert ALL numpy invalid-value warnings to errors so we get a traceback
        # showing exactly which operation produces NaN.
        warnings.filterwarnings("error", category=RuntimeWarning,
                                message="invalid value encountered")
    else:
        # In normal use, suppress the matmul NaN warning — the guard in infer()
        # already skips the prediction for that window.
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="invalid value encountered in matmul")

    # ── Load model ─────────────────────────────────────────────────────────────
    pkl = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    print(f"Loading model: {pkl}")
    with open(pkl, "rb") as fh:
        m = pickle.load(fh)

    model          = m["model"]
    W              = m["W"].astype(np.float64)
    USE_EA         = bool(m.get("USE_EA", True))
    if USE_EA and not np.isfinite(W).all():
        n_bad = int(~np.isfinite(W).sum())
        print(f"[warn] W has {n_bad} non-finite entries — disabling EA")
        USE_EA = False
    T_OPT          = int(m["T_OPT"])
    MODEL_FS_EFF   = int(m["FS_EFF"])   # effective rate the model was trained at
    LP, HP         = float(m["LP"]), float(m["HP"])
    SMOOTH_WIN_M   = int(m["SMOOTH_WIN"])
    thresh         = float(m["thresh"])
    EEG_COLS       = m["EEG_COLS"]
    N_CH           = len(EEG_COLS)
    REFRACTORY_M   = int(m.get("REFRACTORY", MODEL_FS_EFF // 2))
    PERSIST_K      = int(m.get("PERSIST_K", 1))
    USE_FILTER     = not args.no_filter
    MF_TEMPLATE    = m.get("mf_template", None)   # (T_OPT, N_CH) or None

    # ── Connect to LSL and infer actual sample rate ─────────────────────────────
    print(f"\nSearching for LSL stream '{args.stream}' …", flush=True)
    streams = resolve_byprop("name", args.stream, timeout=10)
    if not streams:
        sys.exit(f"No stream '{args.stream}' found.")
    inlet      = StreamInlet(streams[0])
    stream_info = streams[0]
    N_CH_STREAM = stream_info.channel_count()

    nominal_fs = stream_info.nominal_srate()
    if nominal_fs > 0:
        STREAM_FS = int(round(nominal_fs))
        print(f"Connected  ({N_CH_STREAM} channels, nominal {STREAM_FS} Hz)", flush=True)
    else:
        # Irregular stream — measure rate from first 200 samples
        print(f"Connected  ({N_CH_STREAM} channels, measuring rate …)", end="", flush=True)
        ts_buf = []
        while len(ts_buf) < 200:
            _, ts = inlet.pull_sample()
            if ts is not None:
                ts_buf.append(float(ts))
        STREAM_FS = int(round((len(ts_buf) - 1) / (ts_buf[-1] - ts_buf[0])))
        print(f" {STREAM_FS} Hz", flush=True)

    # ── Adapt decimation and effective rate to the actual stream ────────────────
    DECIM   = max(1, round(STREAM_FS / MODEL_FS_EFF))
    FS_EFF  = STREAM_FS // DECIM

    if STREAM_FS != int(m["FS"]) or FS_EFF != MODEL_FS_EFF:
        print(
            f"  [rate] stream={STREAM_FS} Hz, model trained at {m['FS']} Hz  →"
            f"  DECIM={DECIM}, FS_EFF={FS_EFF} Hz (model expected {MODEL_FS_EFF} Hz)"
        )

    # Rescale sample-count params from model's FS_EFF to actual FS_EFF
    REFRACTORY = max(1, round(REFRACTORY_M * FS_EFF / MODEL_FS_EFF))
    SMOOTH_WIN = max(1, round(SMOOTH_WIN_M * FS_EFF / MODEL_FS_EFF))
    HOP        = args.hop if args.hop > 0 else T_OPT

    if not USE_FILTER:
        print("  [info] bandpass filter disabled — applying CAR only")

    mf_str = f"yes (template {MF_TEMPLATE.shape})" if MF_TEMPLATE is not None else "no"
    print(f"  T_OPT={T_OPT} ({T_OPT/FS_EFF*1000:.0f} ms)  thresh={thresh:.4f}  MF={mf_str}")
    print(f"  hop={HOP} samp ({HOP/FS_EFF*1000:.0f} ms)  refractory={REFRACTORY/FS_EFF*1000:.0f} ms  persist={PERSIST_K} samp")
    print(f"  {N_CH} channels: {EEG_COLS}")

    # ── CSV log setup ──────────────────────────────────────────────────────────
    has_btn = N_CH_STREAM > N_CH   # stream sends extra channel(s) after EEG
    if args.log:
        log_path = Path(args.log)
    else:
        ts_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = ROOT / "logs" / f"lsl_predict_{ts_str}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    header = ["lsl_timestamp", "t_dec_s", "score_raw", "score_smooth", "score_gated", "detection"]
    if has_btn:
        header.append("button")
    log_writer.writerow(header)
    print(f"  Logging → {log_path}")

    # ── Causal bandpass filter (coefficients use actual stream rate) ────────────
    nyq = STREAM_FS / 2.0
    sos: np.ndarray = butter(4, [LP / nyq, HP / nyq], btype="band", output="sos")  # type: ignore[assignment]
    fz = [np.zeros((sos.shape[0], 2)) for _ in range(N_CH)]

    # ── Debug counters ─────────────────────────────────────────────────────────
    dbg = dict(
        n_raw_nan=0, n_filter_state_reset=0, n_filter_out_nan=0,
        n_zscore_nan=0, n_guard_fired=0, n_matmul_nan=0,
        n_feats_nan=0, raw_min=np.inf, raw_max=-np.inf,
    )

    def _dbg_raw(raw: np.ndarray, idx: int):
        if not args.debug:
            return
        dbg["raw_min"] = min(dbg["raw_min"], float(np.nanmin(raw)))
        dbg["raw_max"] = max(dbg["raw_max"], float(np.nanmax(raw)))
        if not np.isfinite(raw).all():
            dbg["n_raw_nan"] += 1
            if dbg["n_raw_nan"] <= 3:
                bad = np.where(~np.isfinite(raw))[0]
                print(f"[dbg] sample {idx}: NaN/inf in raw — ch {bad.tolist()} val {raw[bad].tolist()}", flush=True)

    def _dbg_stats(every: int = 500):
        if not args.debug or n_raw % every != 0:
            return
        print(
            f"[dbg] t={n_raw/STREAM_FS:.1f}s  raw=[{dbg['raw_min']:.3g},{dbg['raw_max']:.3g}]"
            f"  raw_nan={dbg['n_raw_nan']}  filt_reset={dbg['n_filter_state_reset']}"
            f"  filt_nan={dbg['n_filter_out_nan']}  zscore_nan={dbg['n_zscore_nan']}"
            f"  guard={dbg['n_guard_fired']}  matmul_nan={dbg['n_matmul_nan']}"
            f"  feats_nan={dbg['n_feats_nan']}",
            flush=True,
        )

    def bandpass_car(raw: np.ndarray) -> np.ndarray:
        """raw (N_CH,) → (bandpass filtered +) CAR (N_CH,)"""
        if not USE_FILTER:
            out = np.array([float(v) if np.isfinite(v) else 0.0 for v in raw[:N_CH]], dtype=np.float32)
            out -= out.mean()
            return out
        out = np.empty(N_CH, dtype=np.float32)
        for c in range(N_CH):
            sample_val = float(raw[c]) if np.isfinite(raw[c]) else 0.0
            y, fz[c] = sosfilt(sos, [sample_val], zi=fz[c])
            if not np.isfinite(fz[c]).all():
                dbg["n_filter_state_reset"] += 1
                fz[c] = np.zeros_like(fz[c])
            out[c] = y[0] if np.isfinite(y[0]) else 0.0
        out -= out.mean()
        if args.debug and not np.isfinite(out).all():
            dbg["n_filter_out_nan"] += 1
        return out

    # ── Shared state ───────────────────────────────────────────────────────────
    rz        = RunningZScore(30 * FS_EFF, N_CH)
    win_buf:   list = []
    score_buf = deque([0.0] * SMOOTH_WIN, maxlen=SMOOTH_WIN)
    gate_buf  = deque([0.0] * PERSIST_K,  maxlen=PERSIST_K)
    n_raw     = 0
    n_dec     = 0
    last_det  = -REFRACTORY
    last_ts   = 0.0   # LSL timestamp of the most recent raw sample
    last_btn  = 0     # button state from extra stream channel

    def infer() -> None:
        nonlocal last_det
        win_arr = np.array(win_buf[:T_OPT], dtype=np.float64)

        if not np.isfinite(win_arr).all():
            dbg["n_guard_fired"] += 1
            if args.debug and dbg["n_guard_fired"] <= 3:
                n_bad = int((~np.isfinite(win_arr)).sum())
                print(
                    f"[dbg] guard fired (#{dbg['n_guard_fired']}) t={n_dec/FS_EFF:.2f}s"
                    f"  {n_bad} non-finite  min={np.nanmin(win_arr):.3g} max={np.nanmax(win_arr):.3g}",
                    flush=True,
                )
            return

        if USE_EA:
            win = np.nan_to_num(win_arr @ W.T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            win = win_arr.astype(np.float32)

        if args.debug and not np.isfinite(win).all():
            dbg["n_matmul_nan"] += 1
            if dbg["n_matmul_nan"] <= 3:
                print(
                    f"[dbg] NaN after matmul (#{dbg['n_matmul_nan']}) — inputs finite!"
                    f"  win_arr=[{win_arr.min():.3g},{win_arr.max():.3g}]  W=[{W.min():.3g},{W.max():.3g}]",
                    flush=True,
                )

        feats = extract_features(win, FS_EFF)
        if MF_TEMPLATE is not None:
            t_norm = (MF_TEMPLATE - MF_TEMPLATE.mean(0)) / (MF_TEMPLATE.std(0) + 1e-8)
            mu = win.mean(0); sd = win.std(0) + 1e-8
            mf = ((win - mu) / sd * t_norm).mean(0).astype(np.float32)
            feats = np.concatenate([feats, mf])

        if args.debug and not np.isfinite(feats).all():
            dbg["n_feats_nan"] += 1
            if dbg["n_feats_nan"] <= 3:
                print(f"[dbg] NaN in features (#{dbg['n_feats_nan']})", flush=True)

        score   = float(model.predict_proba(feats[None, :])[0, 1])
        score_buf.append(score)
        score_s = float(np.mean(score_buf))
        gate_buf.append(score_s)
        score_g = float(min(gate_buf))

        detection = 0
        if score_g >= thresh and (n_dec - last_det) > REFRACTORY:
            last_det  = n_dec
            detection = 1
            print(f"[DETECTION]  t={n_dec/FS_EFF:8.2f}s  score={score_g:.4f}", flush=True)
        elif n_dec % (FS_EFF * 5) == 0:
            print(f"[running]    t={n_dec/FS_EFF:8.2f}s  score={score_g:.4f}", flush=True)

        row = [f"{last_ts:.6f}", f"{n_dec/FS_EFF:.4f}",
               f"{score:.4f}", f"{score_s:.4f}", f"{score_g:.4f}", detection]
        if has_btn:
            row.append(last_btn)
        log_writer.writerow(row)

    # ── Warmup phase (fill filter transients + z-score buffer) ─────────────────
    warmup_dec = args.warmup * FS_EFF
    if warmup_dec > 0:
        print(f"Warming up for {args.warmup}s …", end="", flush=True)
        while n_dec < warmup_dec:
            sample, ts = inlet.pull_sample()
            if sample is None:
                continue
            last_ts = float(ts) if ts is not None else last_ts
            raw = np.array(sample[:N_CH], dtype=np.float32)
            _dbg_raw(raw, n_raw)
            x = np.nan_to_num(bandpass_car(raw), nan=0.0, posinf=0.0, neginf=0.0)
            n_raw += 1
            if n_raw % DECIM == 0:
                z = rz.push(x)
                if args.debug and not np.isfinite(z).all():
                    dbg["n_zscore_nan"] += 1
                # don't fill win_buf during warmup; first prediction uses only fresh data
                n_dec += 1
        print(" done.", flush=True)

    # ── Detection loop ─────────────────────────────────────────────────────────
    print(f"\n── Detection running  (thresh={thresh:.4f}, Ctrl-C to stop) ──\n", flush=True)
    try:
        while True:
            sample, ts = inlet.pull_sample()
            if sample is None:
                continue
            last_ts = float(ts) if ts is not None else last_ts
            if has_btn:
                last_btn = int(sample[N_CH]) if len(sample) > N_CH else 0
            raw = np.array(sample[:N_CH], dtype=np.float32)
            _dbg_raw(raw, n_raw)
            x = np.nan_to_num(bandpass_car(raw), nan=0.0, posinf=0.0, neginf=0.0)
            n_raw += 1
            _dbg_stats()

            if n_raw % DECIM != 0:
                continue

            z = rz.push(x)
            if args.debug and not np.isfinite(z).all():
                dbg["n_zscore_nan"] += 1
            win_buf.append(z)
            n_dec += 1

            if len(win_buf) >= T_OPT:
                infer()
                win_buf = win_buf[HOP:]

    except KeyboardInterrupt:
        if args.debug:
            print(f"\n[dbg] final counters: {dbg}", flush=True)
        print("\nStopped.")
    finally:
        log_file.flush()
        log_file.close()
        print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
