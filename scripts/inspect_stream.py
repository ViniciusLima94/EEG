#!/usr/bin/env python3
"""
Print raw LSL stream stats — no processing, no model.
Reveals channel count, actual sample rate, value range, and NaN presence.

Usage:
    python scripts/inspect_stream.py
    python scripts/inspect_stream.py --stream PiEEG --duration 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, resolve_byprop

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="LSL stream inspector")
    ap.add_argument("--stream",   default="PiEEG", help="LSL stream name")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="Seconds to collect (default: 5)")
    args = ap.parse_args()

    print(f"Searching for '{args.stream}' …", flush=True)
    streams = resolve_byprop("name", args.stream, timeout=10)
    if not streams:
        sys.exit(f"No stream '{args.stream}' found.")

    info = streams[0]
    print(f"\nStream info:")
    print(f"  name          : {info.name()}")
    print(f"  channel count : {info.channel_count()}")
    print(f"  nominal rate  : {info.nominal_srate()} Hz")
    print(f"  type          : {info.type()}")

    inlet = StreamInlet(info)
    n_ch  = info.channel_count()

    print(f"\nCollecting {args.duration}s of samples …", flush=True)
    samples = []
    t_start = None

    while True:
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue
        if t_start is None:
            t_start = float(ts)
        samples.append(sample)
        if float(ts) - t_start >= args.duration:
            break

    t_elapsed = float(ts) - t_start  # type: ignore[possibly-undefined]
    arr = np.array(samples, dtype=np.float64)   # (N, n_ch)
    N   = len(arr)
    actual_rate = N / t_elapsed

    print(f"\nResults ({N} samples over {t_elapsed:.2f}s):")
    print(f"  actual sample rate : {actual_rate:.1f} Hz")
    print(f"  shape              : {arr.shape}")

    print(f"\nPer-channel stats (all {n_ch} channels):")
    print(f"  {'ch':>4}  {'min':>12}  {'max':>12}  {'mean':>12}  {'std':>12}  {'NaN':>6}  {'Inf':>6}")
    for c in range(n_ch):
        col = arr[:, c]
        print(f"  {c+1:>4}  {col.min():>12.4g}  {col.max():>12.4g}  "
              f"{col.mean():>12.4g}  {col.std():>12.4g}  "
              f"{np.isnan(col).sum():>6}  {np.isinf(col).sum():>6}")

    print(f"\nGlobal:")
    print(f"  NaN count : {np.isnan(arr).sum()}")
    print(f"  Inf count : {np.isinf(arr).sum()}")
    print(f"  global min: {np.nanmin(arr):.4g}")
    print(f"  global max: {np.nanmax(arr):.4g}")

    # Check consecutive identical samples (stuck channel)
    diffs = np.diff(arr[:, :n_ch-1], axis=0)   # exclude last ch (button)
    stuck = (np.abs(diffs) < 1e-12).all(axis=0)
    if stuck.any():
        print(f"\n  [warn] channels with zero variance (stuck): {np.where(stuck)[0].tolist()}")
    else:
        print(f"\n  no stuck channels")


if __name__ == "__main__":
    main()
