#!/usr/bin/env python3
"""
Convert WAY-EEG-GAL (PhysioNet) .mat files to the project CSV format.

Dataset: self-paced grasp-lift, 12 subjects, 32ch EEG, 500 Hz.
Each subject has 9 sessions (HS_P{N}_S{1..9}.mat) and one AllLifts table
(P{N}_AllLifts.mat) that records trial-relative event timestamps.

Event chosen: tHandStart — voluntary onset of the reaching movement.
This is the closest equivalent to a self-paced button press and is where
the MRCP slow ramp is expected to begin building up (~1–2 s beforehand).

Absolute onset = AllLifts['StartTime'] + AllLifts['tHandStart']

Output layout
─────────────
  data/subject{14 + (N-1)}/subject_{14+N-1}_tache_way_trial_{S-1}.csv

Columns: timestamp, <32 EEG channels>, button
  button: 6-sample pulse at each tHandStart onset (robust to decimate-4)

Usage
─────
  python scripts/convert_way_eeg_gal.py
  python scripts/convert_way_eeg_gal.py --src "data/WAY-EEG-GAL (PhysioNet)" --out data --subject_offset 14
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

PULSE_W = 6   # samples at 500 Hz — guarantees ≥1 sample survives decimate-4


def load_allifts(lifts_path: Path) -> dict:
    """Return AllLifts as a dict of column→array."""
    mat = sio.loadmat(str(lifts_path), simplify_cells=True)["P"]
    cols = mat["ColNames"].tolist()
    data = mat["AllLifts"]
    return dict(zip(cols, data.T))


def convert_subject(subj_dir: Path, subj_n: int, out_root: Path) -> int:
    lifts_path = next(subj_dir.glob("P*_AllLifts.mat"), None)
    if lifts_path is None:
        print(f"  [skip] no AllLifts file in {subj_dir}")
        return 0

    al = load_allifts(lifts_path)
    runs_col  = al["Run"].astype(int)
    start_col = al["StartTime"]
    thand_col = al["tHandStart"]

    out_dir = out_root / f"subject{subj_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for run in sorted(np.unique(runs_col)):
        # Find matching HS (EEG + sensors) file for this session
        hs_path = next(subj_dir.glob(f"HS_*_S{run}.mat"), None)
        if hs_path is None:
            print(f"  [skip] run {run}: HS file not found")
            continue

        mat   = sio.loadmat(str(hs_path), simplify_cells=True)["hs"]
        eeg   = mat["eeg"]
        X     = np.array(eeg["sig"], dtype=np.float32)   # (N, 32)
        fs    = int(eeg["samplingrate"])
        ch_names = list(eeg["names"])
        N     = X.shape[0]

        # Absolute tHandStart times for this run
        mask    = runs_col == run
        abs_t   = start_col[mask] + thand_col[mask]       # seconds
        onsets  = np.round(abs_t * fs).astype(int)
        onsets  = onsets[(onsets >= 0) & (onsets < N)]    # clip to valid range

        timestamps = np.arange(N, dtype=np.float64) / fs

        # Button: 6-sample pulse at each hand-onset (survives decimate-4 + rising-edge)
        button = np.zeros(N, dtype=np.int8)
        for t in onsets:
            button[t : min(N, t + PULSE_W)] = 1

        # LED: ParticipantLED rising edge — LED lights up 2364 ms ± 70 ms before
        # tHandStart (std=70 ms, by far the most temporally consistent misc event).
        # Late VEP components (P300+ at 300–600 ms post-LED) enter the first ~200 ms
        # of the 2 s press epoch; storing this event lets downstream code subtract them.
        misc       = mat["misc"]
        misc_names = list(misc["names"])
        led_idx    = misc_names.index("ParticipantLED")
        led_sig    = np.array(misc["sig"], dtype=np.float32)[:N, led_idx]
        led_thresh = (led_sig.min() + led_sig.max()) / 2
        led_binary  = (led_sig > led_thresh).astype(np.int8)
        led_onsets  = np.where(np.diff(led_binary, prepend=0).clip(0) > 0)[0]  # rising edges
        led = np.zeros(N, dtype=np.int8)
        for t in led_onsets:
            led[t : min(N, t + PULSE_W)] = 1

        df = pd.DataFrame(X, columns=pd.Index(ch_names))
        df.insert(0, "timestamp", timestamps)
        df["button"] = button
        df["led"]    = led

        trial_idx = int(run) - 1
        fname = out_dir / f"subject_{subj_n}_tache_way_trial_{trial_idx}.csv"
        df.to_csv(fname, index=False)
        n_saved += 1

        ipis = np.diff(np.sort(abs_t))
        print(f"  saved {fname.name}  "
              f"({N} samp, {len(onsets)} events, "
              f"IPI med={np.median(ipis):.1f}s, fs={fs})")

    return n_saved


def main():
    ap = argparse.ArgumentParser(description="Convert WAY-EEG-GAL .mat → project CSV")
    ap.add_argument("--src", default="data/WAY-EEG-GAL (PhysioNet)",
                    help="Root directory containing P1/, P2/, … sub-folders")
    ap.add_argument("--out", default="data",
                    help="Output root (subject folders created here)")
    ap.add_argument("--subject_offset", type=int, default=14,
                    help="First subject number (P1 → subject_offset, P2 → offset+1, …)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    total = 0

    for p_dir in sorted(src.glob("P*")):
        if not p_dir.is_dir():
            continue
        p_num = int(p_dir.name[1:])
        subj_n = args.subject_offset + (p_num - 1)
        print(f"\n{p_dir.name}  →  subject {subj_n}")
        total += convert_subject(p_dir, subj_n, out)

    print(f"\nDone. {total} CSV files written.")


if __name__ == "__main__":
    main()
