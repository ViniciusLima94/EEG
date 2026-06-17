#!/usr/bin/env python3
"""
Convert WAY-EEG-GAL (PhysioNet) .mat files to the project CSV format.

Dataset: self-paced grasp-lift, 12 subjects, 32ch EEG, 500 Hz.
Each subject has 9 sessions (HS_P{N}_S{1..9}.mat) and one AllLifts table
(P{N}_AllLifts.mat) that records trial-relative event timestamps.

Two event modes (--event):
  handstart (default)
      tHandStart — voluntary onset of the reaching movement.
      Closest equivalent to a self-paced button press; MRCP ramp starts ~1-2 s prior.

  grip
      First crossing of GF_THRESH_FRAC × GF_Max in the continuous grip-force signal
      (channel 'GF' from WS_*.mat, co-sampled with EEG at 500 Hz).
      Analogous to pressing a force-sensitive trigger past a mechanical threshold.
      Falls ~200-400 ms after tHandStart (after finger contact, before peak force).

Absolute onset = AllLifts['StartTime'] + AllLifts['t<event>']   (handstart mode)
               = eeg_t[first GF threshold crossing]              (grip mode)

Output layout
─────────────
  data/subject{14 + (N-1)}/subject_{14+N-1}_tache_way_trial_{S-1}.csv

Columns: timestamp, <32 EEG channels>, button, led
  button: 6-sample pulse at each event onset (robust to decimate-4)

Usage
─────
  python scripts/convert_way_eeg_gal.py
  python scripts/convert_way_eeg_gal.py --event grip
  python scripts/convert_way_eeg_gal.py --src data/WAY-EEG-GAL --out data --subject_offset 14
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

PULSE_W        = 6    # samples at 500 Hz — guarantees ≥1 sample survives decimate-4
GF_THRESH_FRAC = 0.3  # fraction of trial GF_Max used as grip-force threshold
GF_PERSIST     = 5   # consecutive samples above threshold required (~10 ms at 500 Hz)
GF_KIN_IDX     = 41  # index of 'GF' channel in kin array (WS file)

# PiEEG-16 montage — maps WAY-EEG-GAL channel names → ch1…ch16 output names.
# Note: T3/T5/T4/T6 (old 10-20) = T7/P7/T8/P8 (new 10-20). Cz and Fz are
# absent from PiEEG-16, so MRCP sensitivity will be reduced in this mode.
PIEEG16_MAP = {
    "Fp1": "ch1",  "F3":  "ch2",  "F7":  "ch3",  "C3":  "ch4",
    "T7":  "ch5",  "P3":  "ch6",  "P7":  "ch7",  "O1":  "ch8",
    "Fp2": "ch9",  "F4":  "ch10", "F8":  "ch11", "C4":  "ch12",
    "T8":  "ch13", "P4":  "ch14", "P8":  "ch15", "O2":  "ch16",
}


def load_allifts(lifts_path: Path) -> dict:
    """Return AllLifts as a dict of column→array."""
    mat = sio.loadmat(str(lifts_path), simplify_cells=True)["P"]
    cols = mat["ColNames"].tolist()
    data = mat["AllLifts"]
    return dict(zip(cols, data.T))


def gf_onset_times(ws_path: Path, al: dict, run: int) -> np.ndarray:
    """
    Return absolute GF threshold-crossing times (seconds, session-relative)
    for all trials in `run`, using WS continuous grip-force signal.

    Strategy per trial:
      threshold = GF_THRESH_FRAC × GF_Max  (from AllLifts, per trial)
      onset     = eeg_t[first sample where GF > threshold]
    Falls back to tHandStart absolute time if no crossing is found.
    """
    ws   = sio.loadmat(str(ws_path), simplify_cells=True)["ws"]
    wins = ws["win"]  # list of trial dicts

    mask      = (al["Run"].astype(int) == run)
    starts    = al["StartTime"][mask]
    thand     = al["tHandStart"][mask]
    n_trials  = mask.sum()

    # wins may have more entries than lifts in this run — align by StartTime
    # win[i]['trial_start_time'] ≈ StartTime for that lift
    win_starts = np.array([w["trial_start_time"] for w in wins])

    onsets = []
    for i in range(n_trials):
        abs_hand = starts[i] + thand[i]     # fallback
        # Use signal max (not AllLifts GF_Max which is in Newtons, different unit)
        j_pre = int(np.argmin(np.abs(win_starts - starts[i])))
        sig_min = wins[j_pre]["kin"][:, GF_KIN_IDX].min()  # GF goes negative during grip
        threshold = GF_THRESH_FRAC * sig_min               # negative threshold
        if threshold >= 0:
            onsets.append(abs_hand)
            continue

        # Match this lift to the closest WS trial window
        j = j_pre  # already matched above
        gf  = wins[j]["kin"][:, GF_KIN_IDX]
        t   = wins[j]["eeg_t"]   # trial-relative (starts at 0)
        t_hand_rel = thand[i]   # tHandStart is also trial-relative

        # GF goes negative during grip — detect first sustained downward crossing
        crossing = np.where(gf < threshold)[0]
        if len(crossing) == 0:
            onsets.append(abs_hand)
            continue

        t_cross_rel = None
        for idx in crossing:
            if idx + GF_PERSIST <= len(gf) and np.all(gf[idx:idx + GF_PERSIST] < threshold):
                t_cross_rel = float(t[idx])
                break
        if t_cross_rel is None:
            onsets.append(abs_hand)
            continue

        # Only accept crossings after tHandStart (both trial-relative)
        if t_cross_rel < t_hand_rel:
            hand_samp = np.searchsorted(t, t_hand_rel)
            after = crossing[crossing >= hand_samp]
            t_cross_rel = t_hand_rel  # fallback
            for idx in after:
                if idx + GF_PERSIST <= len(gf) and np.all(gf[idx:idx + GF_PERSIST] < threshold):
                    t_cross_rel = float(t[idx])
                    break

        # Convert trial-relative crossing time to session-absolute for HS indexing
        t_cross = starts[i] + t_cross_rel

        onsets.append(float(t_cross))

    return np.array(onsets)


def convert_subject(subj_dir: Path, subj_n: int, out_root: Path, event: str, channels: str) -> int:
    lifts_path = next(subj_dir.glob("P*_AllLifts.mat"), None)
    if lifts_path is None:
        print(f"  [skip] no AllLifts file in {subj_dir}")
        return 0

    al        = load_allifts(lifts_path)
    runs_col  = al["Run"].astype(int)
    start_col = al["StartTime"]
    thand_col = al["tHandStart"]

    out_dir = out_root / f"subject{subj_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for run in sorted(np.unique(runs_col)):
        hs_path = next(subj_dir.glob(f"HS_*_S{run}.mat"), None)
        if hs_path is None:
            print(f"  [skip] run {run}: HS file not found")
            continue

        mat      = sio.loadmat(str(hs_path), simplify_cells=True)["hs"]
        eeg      = mat["eeg"]
        X        = np.array(eeg["sig"], dtype=np.float32)   # (N, 32)
        fs       = int(eeg["samplingrate"])
        ch_names = list(eeg["names"])
        N        = X.shape[0]
        timestamps = np.arange(N, dtype=np.float64) / fs

        mask = runs_col == run

        if event == "grip":
            ws_path = next(subj_dir.glob(f"WS_*_S{run}.mat"), None)
            if ws_path is None:
                print(f"  [warn] run {run}: WS file not found, falling back to tHandStart")
                abs_t = start_col[mask] + thand_col[mask]
            else:
                abs_t = gf_onset_times(ws_path, al, run)
        else:
            abs_t = start_col[mask] + thand_col[mask]

        onsets = np.round(abs_t * fs).astype(int)
        onsets = onsets[(onsets >= 0) & (onsets < N)]

        button = np.zeros(N, dtype=np.int8)
        for t in onsets:
            button[t : min(N, t + PULSE_W)] = 1

        # LED: ParticipantLED rising edge
        misc       = mat["misc"]
        misc_names = list(misc["names"])
        led_idx    = misc_names.index("ParticipantLED")
        led_sig    = np.array(misc["sig"], dtype=np.float32)[:N, led_idx]
        led_thresh = (led_sig.min() + led_sig.max()) / 2
        led_binary = (led_sig > led_thresh).astype(np.int8)
        led_onsets = np.where(np.diff(led_binary, prepend=0).clip(0) > 0)[0]
        led = np.zeros(N, dtype=np.int8)
        for t in led_onsets:
            led[t : min(N, t + PULSE_W)] = 1

        df = pd.DataFrame(X, columns=pd.Index(ch_names))
        df.insert(0, "timestamp", timestamps)
        df["button"] = button
        df["led"]    = led

        if channels == "pieeg16":
            missing = [c for c in PIEEG16_MAP if c not in df.columns]
            if missing:
                print(f"  [warn] run {run}: missing channels {missing}, skipping pieeg16 filter")
            else:
                df = df[["timestamp"] + list(PIEEG16_MAP.keys()) + ["button", "led"]]
                df = df.rename(columns=PIEEG16_MAP)

        trial_idx = int(run) - 1
        fname = out_dir / f"subject_{subj_n}_tache_way_trial_{trial_idx}.csv"
        df.to_csv(fname, index=False)
        n_saved += 1

        ipis = np.diff(np.sort(abs_t))
        print(f"  saved {fname.name}  "
              f"({N} samp, {len(onsets)} events, "
              f"IPI med={np.median(ipis):.1f}s  event={event})")

    return n_saved


def main():
    ap = argparse.ArgumentParser(description="Convert WAY-EEG-GAL .mat → project CSV")
    ap.add_argument("--src", default="data/WAY-EEG-GAL",
                    help="Root directory containing P1/, P2/, … sub-folders")
    ap.add_argument("--out", default="data",
                    help="Output root (subject folders created here)")
    ap.add_argument("--subject_offset", type=int, default=14,
                    help="First subject number (P1 → subject_offset, P2 → offset+1, …)")
    ap.add_argument("--event", choices=["handstart", "grip"], default="handstart",
                    help="Event to use as button label: 'handstart' (default) or 'grip' "
                         "(first GF threshold crossing, analogous to force-trigger press)")
    ap.add_argument("--channels", choices=["all", "pieeg16"], default="all",
                    help="Channel subset: 'all' keeps all 32 ch (default); "
                         "'pieeg16' selects the 16 electrodes matching the PiEEG-16 montage "
                         "and renames them ch1…ch16 to match GNG CSV format")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    total = 0

    for p_dir in sorted(src.glob("P*")):
        if not p_dir.is_dir():
            continue
        p_num  = int(p_dir.name[1:])
        subj_n = args.subject_offset + (p_num - 1)
        print(f"\n{p_dir.name}  →  subject {subj_n}  (event={args.event})")
        total += convert_subject(p_dir, subj_n, out, args.event, args.channels)

    print(f"\nDone. {total} CSV files written.")


if __name__ == "__main__":
    main()
