"""
Shared preprocessing utilities for causal sliding-window EEG models.

Usage
-----
from src.preprocessing import load_trial, build_windows
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from mne.filter import filter_data


def load_trial(
    subject: int,
    tache: str,
    trial: int,
    data_path: str,
    eeg_cols: Optional[list] = None,
    fs: int = 250,
    lp: float = 0.1,
    hp: float = 60.0,
    apply_filter: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Load one CSV trial, optionally band-pass filter it.

    Parameters
    ----------
    subject    : subject index (0-based)
    tache      : task name, e.g. "spt"
    trial      : trial index
    data_path  : directory containing the CSV files
    eeg_cols   : expected channel list; asserted if provided
    fs         : sampling frequency (Hz)
    lp, hp     : band-pass cut-offs (Hz)
    apply_filter: whether to run the MNE band-pass filter

    Returns
    -------
    df       : filtered DataFrame (columns: timestamp, ch1..ch16, button)
    eeg_cols : list of EEG channel names (excludes timestamp & button)
    """
    path = os.path.join(
        data_path, f"subject_{subject}_tache_{tache}_trial_{trial}.csv"
    )
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ("button", "timestamp")]
    if eeg_cols is not None:
        assert cols == eeg_cols, (
            f"Trial {trial} channel mismatch: expected {eeg_cols}, got {cols}"
        )
    if apply_filter:
        df.iloc[:, 1:-1] = filter_data(
            df.iloc[:, 1:-1].values.T, fs, lp, hp
        ).T
    return df, cols


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    T: int,
    groups: Optional[np.ndarray] = None,
    per_window_norm: bool = True,
) -> tuple:
    """
    Build causal sliding windows of length T.

    Window i covers X[i-T : i] and predicts label y[i] (one step ahead).
    Windows that span two different groups (trials / subjects) are skipped
    to prevent cross-boundary leakage.

    Parameters
    ----------
    X      : (N, C) raw EEG array
    y      : (N,)   integer labels
    T      : look-back window length in samples
    groups : (N,)   group id per sample (trial id, subject id, …).
             If None, a single uniform group is assumed (no boundary skipping
             beyond the first T samples).
    per_window_norm : z-score each window per channel independently.
             Removes session-level amplitude drift without fitting a scaler.

    Returns
    -------
    X_win : (M, C*T)  flattened — RF / MLP format       [if groups is None]
            (M, C, T) channels-first — Conv1D format     [if groups is not None]
    y_win : (M,)
    g_win : (M,)  only returned when groups is not None
    """
    N = X.shape[0]
    if groups is None:
        groups = np.zeros(N, dtype=np.int64)
        return_groups = False
    else:
        groups = np.asarray(groups)
        return_groups = True

    wins, labels, grps = [], [], []

    for i in range(T, N):
        if groups[i] != groups[i - T]:
            continue
        w = X[i - T : i].copy()  # (T, C)
        if per_window_norm:
            mu = w.mean(axis=0, keepdims=True)
            sd = w.std(axis=0, keepdims=True) + 1e-8
            w = (w - mu) / sd
        if return_groups:
            wins.append(w.T)      # (C, T) — Conv1D channels-first
        else:
            wins.append(w.flatten())  # (C*T,) — flat for RF/MLP
        labels.append(y[i])
        grps.append(groups[i])

    X_win = np.array(wins, dtype=np.float32)
    y_win = np.array(labels, dtype=np.int64)
    g_win = np.array(grps, dtype=np.int64)

    if return_groups:
        return X_win, y_win, g_win
    return X_win, y_win
