#!/usr/bin/env python3
"""
Convert BNCI Horizon 2020 dataset 001-2014 (.mat) to the project CSV format.

Each .mat file contains 9 runs; runs 0–2 are calibration (no cue events),
runs 3–8 are the main session (48 cued MI trials each, 4 classes balanced).

Output layout
─────────────
  data/subject{N}/subject_{N}_tache_mi_trial_{T}.csv

Where:
  N  = 11, 12, 13  for subjects A01, A02, A03
  T  = 0–5   from the training file  (A0XT.mat, runs 3–8)
  T  = 6–11  from the evaluation file (A0XE.mat, runs 3–8)

Column layout (matches load_trial expectations):
  timestamp  — seconds from start of run
  Fz … POz   — 22 EEG channels (µV), EOG channels dropped
  button     — 1-sample pulse at each cue onset, 0 elsewhere
  label      — MI class (1=left hand, 2=right hand, 3=feet, 4=tongue), 0 elsewhere

Usage
─────
  python scripts/convert_bnci_001_2014.py
  python scripts/convert_bnci_001_2014.py --src data/motor_imagery --out data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

EEG_CH = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1",  "Pz",  "P2",  "POz",
]   # channels 0–21; channels 22–24 are EOG → dropped

SUBJECT_MAP = {"A01": 11, "A02": 12, "A03": 13}
SESSION_MAP = {"T": (range(3, 9), range(0, 6)),   # runs 3–8 → trial indices 0–5
               "E": (range(3, 9), range(6, 12))}   # runs 3–8 → trial indices 6–11
MAIN_RUNS   = range(3, 9)


def convert_file(mat_path: Path, subject_n: int, trial_offset: int, out_root: Path) -> int:
    mat    = sio.loadmat(str(mat_path), simplify_cells=True)
    runs   = mat["data"]
    out_dir = out_root / f"subject{subject_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for run_idx, trial_idx in zip(MAIN_RUNS, range(trial_offset, trial_offset + 6)):
        run   = runs[run_idx]
        X     = np.array(run["X"], dtype=np.float32)       # (N, 25)
        trial = np.array(run["trial"]).flatten().astype(int)
        fs    = int(run["fs"])
        N     = X.shape[0]

        if len(trial) == 0:
            print(f"  [skip] run {run_idx}: no trials")
            continue

        # 22 EEG channels only
        X_eeg = X[:, :22]

        # Timestamp
        timestamps = np.arange(N, dtype=np.float64) / fs

        # Button column: pulse wide enough to survive decimate-by-4 (load_trial
        # takes every 4th sample, so a 1-sample pulse can disappear; 6 samples
        # = 24 ms at 250 Hz guarantees the decimated grid hits at least one '1').
        # load_trial then reduces to a single rising edge via np.diff + clip.
        PULSE_W = 6
        button = np.zeros(N, dtype=np.int8)
        for t in trial:
            button[t : min(N, t + PULSE_W)] = 1

        # Column order must be [timestamp, eeg..., button] — load_trial filters
        # df.iloc[:, 1:-1] (everything between timestamp and the last column).
        df = pd.DataFrame(X_eeg, columns=EEG_CH)
        df.insert(0, "timestamp", timestamps)
        df["button"] = button

        fname = out_dir / f"subject_{subject_n}_tache_mi_trial_{trial_idx}.csv"
        df.to_csv(fname, index=False)
        n_saved += 1
        print(f"  saved {fname.name}  ({N} samples, {len(trial)} events, fs={fs})")

    return n_saved


def main():
    ap = argparse.ArgumentParser(description="Convert BNCI 001-2014 .mat → project CSV")
    ap.add_argument("--src", default="data/motor_imagery",
                    help="Directory containing A01T.mat etc.")
    ap.add_argument("--out", default="data",
                    help="Output root (subject folders created here)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    total = 0

    for subj_key, subj_n in SUBJECT_MAP.items():
        for session, offset in [("T", 0), ("E", 6)]:
            mat_path = src / f"{subj_key}{session}.mat"
            if not mat_path.exists():
                print(f"[skip] {mat_path} not found")
                continue
            print(f"\n{mat_path.name}  →  subject {subj_n}  (trial offset {offset})")
            total += convert_file(mat_path, subj_n, offset, out)

    print(f"\nDone. {total} CSV files written.")


if __name__ == "__main__":
    main()
