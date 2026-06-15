#!/usr/bin/env python3
"""
Convert PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB) EDF+ files
to the project CSV format.

Dataset: 109 subjects, 64ch EEG, 160 Hz, 14 runs each.
Files: S{NNN}R{RR}.edf  (e.g. S001R03.edf)

Run structure
─────────────
  Runs 1–2  : baseline (no T1/T2 events)
  Runs 3,7,11 : motor EXECUTION  — left fist (T1) / right fist (T2)
  Runs 4,8,12 : motor IMAGERY    — left fist (T1) / right fist (T2)
  Runs 5,9,13 : motor EXECUTION  — both fists (T1) / both feet (T2)
  Runs 6,10,14: motor IMAGERY    — both fists (T1) / both feet (T2)

Only motor-EXECUTION runs are exported (imagery has no MRCP).
Each run becomes one trial CSV.

Output layout
─────────────
  data/subject{N}/subject_{N}_tache_physio_trial_{T}.csv

Columns: timestamp, <64 EEG channels>, button
  button: 6-sample pulse at every T1/T2 onset (robust to decimate-4 → 40 Hz)

Trial index mapping (T = run-index within selected runs, 0-based):
  Runs 3,7,11  → trial 0, 1, 2   (unilateral execution)
  Runs 5,9,13  → trial 3, 4, 5   (bilateral execution)

Usage
─────
  python scripts/convert_physionet_eegmmidb.py
  python scripts/convert_physionet_eegmmidb.py --src data/physionet-eegmmidb --out data
  python scripts/convert_physionet_eegmmidb.py --subjects 1 2 3 --subject_offset 20
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")

PULSE_W = 6   # samples at 160 Hz — guarantees ≥1 sample survives decimate-4

# (run_number, trial_index) for motor execution only
EXEC_RUNS = {
    3: 0, 7: 1, 11: 2,   # unilateral: left fist / right fist
    5: 3, 9: 4, 13: 5,   # bilateral:  both fists / both feet
}


def convert_subject(subj_dir: Path, subj_n: int, out_root: Path) -> int:
    out_dir = out_root / f"subject{subj_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    # Infer PhysioNet subject number from folder name (S001 → 1, or plain int)
    folder = subj_dir.name
    phys_n = int("".join(c for c in folder if c.isdigit()))

    for run_num, trial_idx in sorted(EXEC_RUNS.items()):
        edf_path = subj_dir / f"S{phys_n:03d}R{run_num:02d}.edf"
        if not edf_path.exists():
            print(f"  [skip] {edf_path.name} not found")
            continue

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        fs  = int(raw.info["sfreq"])            # 160 Hz
        N   = len(raw.times)
        ch_names = raw.ch_names                 # 64 EEG channel names

        X = np.array(raw.get_data(), dtype=np.float32).T * 1e6  # V → µV, shape (N, 64)

        # Annotation → button pulse (use array attributes to avoid dict-value ambiguity)
        button = np.zeros(N, dtype=np.int8)
        annots = raw.annotations
        move_mask = np.isin(annots.description, ["T1", "T2"])
        onsets_s: np.ndarray = annots.onset[move_mask]  # float64 array, seconds
        n_events = int(move_mask.sum())
        for t_s in onsets_s:
            onset_samp = int(round(float(t_s) * fs))
            button[onset_samp : min(N, onset_samp + PULSE_W)] = 1

        if n_events == 0:
            print(f"  [skip] {edf_path.name}: no T1/T2 annotations found")
            continue

        timestamps = np.arange(N, dtype=np.float64) / fs

        df = pd.DataFrame(X, columns=ch_names)
        df.insert(0, "timestamp", timestamps)
        df["button"] = button

        fname = out_dir / f"subject_{subj_n}_tache_physio_trial_{trial_idx}.csv"
        df.to_csv(fname, index=False)
        n_saved += 1

        ipis = np.diff(np.where(np.diff(button.astype(np.int8), prepend=0).clip(0))[0]) / fs
        print(f"  saved {fname.name}  "
              f"({N} samp, {n_events} events, "
              f"IPI med={np.median(ipis):.1f}s, fs={fs})")

    return n_saved


def main():
    ap = argparse.ArgumentParser(
        description="Convert PhysioNet EEGMMIDB EDF+ files → project CSV"
    )
    ap.add_argument("--src", default="data/physionet-eegmmidb",
                    help="Root directory containing S001/, S002/, … sub-folders (or .edf files directly)")
    ap.add_argument("--out", default="data",
                    help="Output root (subject folders created here)")
    ap.add_argument("--subject_offset", type=int, default=20,
                    help="First output subject number (PhysioNet S001 → offset, S002 → offset+1, …)")
    ap.add_argument("--subjects", type=int, nargs="+", default=None,
                    help="PhysioNet subject numbers to convert (e.g. --subjects 1 2 3). Default: all found.")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    total = 0

    # Discover subject folders: look for S001/, S002/, … or flat layout
    if any(src.glob("S*.edf")):
        # Flat layout: all EDF files in one directory
        phys_nums = sorted({
            int("".join(c for c in p.stem[:4] if c.isdigit()))
            for p in src.glob("S*.edf")
        })
        subj_dirs = {n: src for n in phys_nums}
    else:
        # Sub-folder layout: S001/, S002/, …
        subj_dirs = {}
        for d in sorted(src.glob("S*")):
            if d.is_dir():
                n = int("".join(c for c in d.name if c.isdigit()))
                subj_dirs[n] = d

    if args.subjects:
        subj_dirs = {n: subj_dirs[n] for n in args.subjects if n in subj_dirs}

    for phys_n, subj_dir in sorted(subj_dirs.items()):
        subj_n = args.subject_offset + (phys_n - 1)
        print(f"\nS{phys_n:03d}  →  subject {subj_n}  ({subj_dir})")
        total += convert_subject(subj_dir, subj_n, out)

    print(f"\nDone. {total} CSV files written.")


if __name__ == "__main__":
    main()
