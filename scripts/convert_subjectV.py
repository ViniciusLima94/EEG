"""
Convert subjectV raw session files to the LightGBM pipeline format.

Input  : data/subjectV/V_eeg_{DATE}_{STAMP}.csv
         data/subjectV/V_markers_{DATE}_{STAMP}.csv
Output : data/subjectV/subject_V_tache_gng2_trial_{N}.csv  (N = 0 … n_sessions-1)

Column mapping
  lsl_timestamp → timestamp
  ch0 … ch15   → ch1 … ch16  (EEG channels)
  ch16 (always 0) → dropped; replaced by reconstructed 'button'
  nogo_onset added: 1-sample pulse at each nogo stimulus onset (marker code 20)

Button reconstruction:
  For every 'button_press' marker, set button=1 for PRESS_SAMPLES consecutive
  samples starting at the nearest EEG timestamp.  This matches the sustained
  button-hold format used by the existing subject data.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "subjectV")
DATE = "20260617"
PRESS_SAMPLES = 25  # 100 ms at 250 Hz

files = os.listdir(ROOT)
stamps = sorted(set(
    f.split("_")[-1][:-4]
    for f in files
    if f.startswith("V_eeg_") and f.endswith(".csv")
))

print(f"Found {len(stamps)} sessions: {stamps}")

for trial_idx, stamp in enumerate(stamps):
    eeg_path = os.path.join(ROOT, f"V_eeg_{DATE}_{stamp}.csv")
    mrk_path = os.path.join(ROOT, f"V_markers_{DATE}_{stamp}.csv")

    if not os.path.exists(eeg_path) or not os.path.exists(mrk_path):
        print(f"  [skip] stamp {stamp}: missing eeg or markers file")
        continue

    df_eeg = pd.read_csv(eeg_path)
    df_mrk = pd.read_csv(mrk_path)

    timestamps = df_eeg["lsl_timestamp"].values

    # Rename timestamp
    df_out = df_eeg.rename(columns={"lsl_timestamp": "timestamp"})

    # Drop ch16 (always-zero digital channel) before renaming to avoid collision
    df_out = df_out.drop(columns=["ch16"], errors="ignore")

    # Rename EEG channels ch0-ch15 → ch1-ch16
    rename_map = {f"ch{i}": f"ch{i+1}" for i in range(16)}
    df_out = df_out.rename(columns=rename_map)

    # Reconstruct button column from button_press markers
    button = np.zeros(len(df_out), dtype=np.int8)
    press_ts = df_mrk.loc[df_mrk["marker_label"] == "button_press", "lsl_timestamp"].values

    for ts in press_ts:
        idx = np.searchsorted(timestamps, ts)
        i0 = max(0, idx)
        i1 = min(len(button), idx + PRESS_SAMPLES)
        button[i0:i1] = 1

    df_out["button"] = button

    # Reconstruct nogo_onset column (1-sample pulse at each nogo stimulus onset)
    nogo_onset = np.zeros(len(df_out), dtype=np.int8)
    nogo_ts = df_mrk.loc[df_mrk["marker_label"] == "nogo_onset", "lsl_timestamp"].values

    for ts in nogo_ts:
        idx = np.searchsorted(timestamps, ts)
        if 0 <= idx < len(nogo_onset):
            nogo_onset[idx] = 1

    df_out["nogo_onset"] = nogo_onset

    out_path = os.path.join(ROOT, f"subject_V_tache_gng2_trial_{trial_idx}.csv")
    df_out.to_csv(out_path, index=False)
    n_presses = int((np.diff(np.concatenate([[0], button, [0]])) == 1).sum())
    n_nogo = int(nogo_onset.sum())
    print(f"  trial {trial_idx}  stamp={stamp}  rows={len(df_out):,}  presses={n_presses}  nogo_onsets={n_nogo}  → {os.path.basename(out_path)}")

print("Done.")
