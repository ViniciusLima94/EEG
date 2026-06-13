#!/usr/bin/env python3
"""
Online EEG press detection via LSL — DSCausalConv1D model.

Connects to a PiEEG stream, warms up the filter and z-score buffer, then
streams continuous predictions using the trained depthwise-separable Conv1D.

No feature extraction — raw decimated windows go directly into the model.

Usage:
    python scripts/lsl_predict_conv1d.py
    python scripts/lsl_predict_conv1d.py --model notebooks/models/conv1d_demo.pkl
    python scripts/lsl_predict_conv1d.py --stream PiEEG --hop 20 --warmup 5
    python scripts/lsl_predict_conv1d.py --device mps
"""
import argparse
import csv
import datetime
import pickle
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt
from pylsl import StreamInlet, resolve_byprop

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model.conv1d import DSCausalConv1D


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
    ap = argparse.ArgumentParser(description="Online EEG press detection — Conv1D")
    ap.add_argument("--model",  default="notebooks/models/conv1d_demo.pkl",
                    help="Path to trained Conv1D pkl (relative to repo root or absolute)")
    ap.add_argument("--stream", default="PiEEG", help="LSL stream name")
    ap.add_argument("--hop",    type=int, default=0,
                    help="New decimated samples between predictions (0 = T_OPT, non-overlapping)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Seconds to collect before starting detection")
    ap.add_argument("--device", default="cpu",
                    help="Torch device for inference: cpu | cuda | mps (default: cpu)")
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable bandpass filter (CAR only)")
    ap.add_argument("--debug",  action="store_true",
                    help="Print NaN diagnostics and raw stream stats")
    ap.add_argument("--log",    default="",
                    help="CSV log path (auto-named if empty)")
    args = ap.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    pkl = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    print(f"Loading model: {pkl}")
    with open(pkl, "rb") as fh:
        m = pickle.load(fh)

    arch           = m["arch"]
    T_OPT          = int(m["T_OPT"])
    MODEL_FS_EFF   = int(m["FS_EFF"])
    LP, HP         = float(m["LP"]), float(m["HP"])
    SMOOTH_WIN_M   = int(m["SMOOTH_WIN"])
    thresh         = float(m["thresh"])
    EEG_COLS       = m["EEG_COLS"]
    N_CH           = len(EEG_COLS)
    REFRACTORY_M   = int(m.get("REFRACTORY", MODEL_FS_EFF))
    PERSIST_K      = int(m.get("PERSIST_K", 1))
    USE_FILTER     = not args.no_filter

    # Reconstruct model
    DEVICE = torch.device(args.device)
    net = DSCausalConv1D(
        in_channels=arch["n_channels"],
        n_filters=arch["n_filters"],
        n_layers=arch["n_layers"],
        kernel_size=arch["kernel_size"],
        depth_mult=arch["depth_mult"],
        grow_filters=arch["grow_filters"],
        dropout_rate=arch["dropout"],
        activation=arch["activation"],
        use_bn=arch["use_bn"],
    )
    net.load_state_dict(m["state_dict"])
    net.to(DEVICE)
    net.eval()
    total_params = sum(p.numel() for p in net.parameters())

    # ── Connect to LSL ─────────────────────────────────────────────────────────
    print(f"\nSearching for LSL stream '{args.stream}' …", flush=True)
    streams = resolve_byprop("name", args.stream, timeout=10)
    if not streams:
        sys.exit(f"No stream '{args.stream}' found.")
    inlet       = StreamInlet(streams[0])
    stream_info = streams[0]
    N_CH_STREAM = stream_info.channel_count()

    nominal_fs = stream_info.nominal_srate()
    if nominal_fs > 0:
        STREAM_FS = int(round(nominal_fs))
        print(f"Connected  ({N_CH_STREAM} channels, nominal {STREAM_FS} Hz)", flush=True)
    else:
        print(f"Connected  ({N_CH_STREAM} channels, measuring rate …)", end="", flush=True)
        ts_buf = []
        while len(ts_buf) < 200:
            _, ts = inlet.pull_sample()
            if ts is not None:
                ts_buf.append(float(ts))
        STREAM_FS = int(round((len(ts_buf) - 1) / (ts_buf[-1] - ts_buf[0])))
        print(f" {STREAM_FS} Hz", flush=True)

    DECIM      = max(1, round(STREAM_FS / MODEL_FS_EFF))
    FS_EFF     = STREAM_FS // DECIM
    REFRACTORY = max(1, round(REFRACTORY_M * FS_EFF / MODEL_FS_EFF))
    SMOOTH_WIN = max(1, round(SMOOTH_WIN_M * FS_EFF / MODEL_FS_EFF))
    HOP        = args.hop if args.hop > 0 else T_OPT

    if STREAM_FS != int(m["FS"]) or FS_EFF != MODEL_FS_EFF:
        print(f"  [rate] stream={STREAM_FS} Hz → DECIM={DECIM}, FS_EFF={FS_EFF} Hz "
              f"(model trained at {MODEL_FS_EFF} Hz)")

    print(f"  Model      : DSCausalConv1D  {arch['n_layers']} blocks  "
          f"D={arch['depth_mult']}  k={arch['kernel_size']}  {total_params:,} params")
    print(f"  T_OPT      : {T_OPT} ({T_OPT/FS_EFF*1000:.0f} ms)  "
          f"thresh={thresh:.4f}  device={DEVICE}")
    print(f"  hop        : {HOP} samp ({HOP/FS_EFF*1000:.0f} ms)  "
          f"refractory={REFRACTORY/FS_EFF*1000:.0f} ms  persist={PERSIST_K} samp")
    print(f"  {N_CH} channels: {EEG_COLS}")

    # ── CSV log ────────────────────────────────────────────────────────────────
    has_btn = N_CH_STREAM > N_CH
    if args.log:
        log_path = Path(args.log)
    else:
        ts_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = ROOT / "logs" / f"lsl_predict_conv1d_{ts_str}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    header = ["lsl_timestamp", "t_dec_s", "score_raw", "score_smooth", "score_gated", "detection"]
    if has_btn:
        header.append("button")
    log_writer.writerow(header)
    print(f"  Logging → {log_path}")

    # ── Causal bandpass filter ─────────────────────────────────────────────────
    nyq = STREAM_FS / 2.0
    sos = butter(4, [LP / nyq, HP / nyq], btype="band", output="sos")
    fz  = [np.zeros((sos.shape[0], 2)) for _ in range(N_CH)]

    def bandpass_car(raw: np.ndarray) -> np.ndarray:
        if not USE_FILTER:
            out = np.array([float(v) if np.isfinite(v) else 0.0 for v in raw[:N_CH]],
                           dtype=np.float32)
            out -= out.mean()
            return out
        out = np.empty(N_CH, dtype=np.float32)
        for c in range(N_CH):
            val = float(raw[c]) if np.isfinite(raw[c]) else 0.0
            y, fz[c] = sosfilt(sos, [val], zi=fz[c])
            if not np.isfinite(fz[c]).all():
                fz[c] = np.zeros_like(fz[c])
            out[c] = y[0] if np.isfinite(y[0]) else 0.0
        out -= out.mean()
        return out

    # ── Shared state ───────────────────────────────────────────────────────────
    rz        = RunningZScore(30 * FS_EFF, N_CH)
    win_buf:   list = []
    score_buf = deque([0.0] * SMOOTH_WIN, maxlen=SMOOTH_WIN)
    gate_buf  = deque([0.0] * PERSIST_K,  maxlen=PERSIST_K)
    n_raw     = 0
    n_dec     = 0
    last_det  = -REFRACTORY
    last_ts   = 0.0
    last_btn  = 0
    n_nan     = 0

    def infer() -> None:
        nonlocal last_det, n_nan
        win_arr = np.array(win_buf[:T_OPT], dtype=np.float32)   # (T, C)

        if not np.isfinite(win_arr).all():
            n_nan += 1
            if args.debug and n_nan <= 3:
                print(f"[dbg] non-finite window (#{n_nan})", flush=True)
            return

        # (T, C) → (1, C, T)
        x_t = torch.from_numpy(win_arr.T[None]).to(DEVICE)
        with torch.no_grad():
            score = float(torch.sigmoid(net(x_t)).cpu().item())

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

    # ── Warmup ─────────────────────────────────────────────────────────────────
    warmup_dec = args.warmup * FS_EFF
    if warmup_dec > 0:
        print(f"\nWarming up for {args.warmup}s …", end="", flush=True)
        while n_dec < warmup_dec:
            sample, ts = inlet.pull_sample()
            if sample is None:
                continue
            last_ts = float(ts) if ts is not None else last_ts
            raw = np.array(sample[:N_CH], dtype=np.float32)
            x   = np.nan_to_num(bandpass_car(raw), nan=0.0, posinf=0.0, neginf=0.0)
            n_raw += 1
            if n_raw % DECIM == 0:
                rz.push(x)
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
            x   = np.nan_to_num(bandpass_car(raw), nan=0.0, posinf=0.0, neginf=0.0)
            n_raw += 1

            if n_raw % DECIM != 0:
                continue

            z = rz.push(x)
            win_buf.append(z)
            n_dec += 1

            if len(win_buf) >= T_OPT:
                infer()
                win_buf = win_buf[HOP:]

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        log_file.flush()
        log_file.close()
        print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
