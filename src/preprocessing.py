"""
Shared preprocessing utilities for causal sliding-window EEG models.

Usage
-----
from src.preprocessing import load_trial, build_windows, make_horizon_labels
from src.preprocessing import make_soft_horizon_labels, compute_hjorth
from src.preprocessing import window_band_power, make_mrcp_template, matched_filter_score
from src.preprocessing import euclidean_align
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
    lp: float = 0.5,
    hp: float = 8.0,
    apply_filter: bool = True,
    decimate: int = 1,
    car: bool = False,
) -> tuple[pd.DataFrame, list]:
    """
    Load one CSV trial, optionally band-pass filter it and/or decimate.

    Parameters
    ----------
    subject      : subject index (0-based)
    tache        : task name, e.g. "spt"
    trial        : trial index
    data_path    : directory containing the CSV files
    eeg_cols     : expected channel list; asserted if provided
    fs           : sampling frequency (Hz) of the raw recording
    lp, hp       : band-pass cut-offs (Hz). Defaults 0.5–8 Hz target the MRCP
                   delta/theta band while rejecting sub-Hz electrode drift.
    apply_filter : whether to run the MNE band-pass filter
    decimate     : integer downsampling factor applied after filtering.
                   Uses scipy.signal.decimate (zero-phase, anti-aliased).
                   Effective fs after loading = fs // decimate. Default 1 = off.
    car          : if True, apply Common Average Reference after filtering:
                   subtract the instantaneous mean across all channels from each
                   channel. Reduces volume-conducted common-mode noise.

    Returns
    -------
    df       : filtered (and optionally decimated) DataFrame
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
    if car:
        arr = df[cols].values.astype(np.float64)
        arr -= arr.mean(axis=1, keepdims=True)
        df[cols] = arr
    if decimate > 1:
        from scipy.signal import decimate as sp_decimate
        eeg_decimated = sp_decimate(df[cols].values, decimate, axis=0, zero_phase=True)
        n_out = eeg_decimated.shape[0]
        df = df.iloc[::decimate].iloc[:n_out].reset_index(drop=True)
        df[cols] = eeg_decimated
    return df, cols


def make_horizon_labels(y: np.ndarray, horizon: int) -> np.ndarray:
    """
    y_h[i] = 1 if any press occurs in y[i : i+horizon], else 0.

    Converts a momentary onset label into an anticipatory label: positive
    whenever a button press will occur within the next `horizon` samples.
    The last (horizon-1) positions cannot have full lookahead and are set to 0.
    """
    if horizon <= 1:
        return y.copy()
    N = len(y)
    cs = np.concatenate([[0], np.cumsum(y.astype(np.int64))])
    y_h = np.zeros(N, dtype=y.dtype)
    end = N - horizon + 1
    if end > 0:
        y_h[:end] = (cs[horizon : horizon + end] - cs[:end] > 0).astype(y.dtype)
    return y_h


def make_soft_horizon_labels(y: np.ndarray, horizon: int) -> np.ndarray:
    """
    Soft (graduated) anticipatory labels — linearly ramped version of make_horizon_labels.

    y_s[i] ramps from 0 to 1 as the next button press approaches:
        y_s[i] = (horizon - dist_to_next_press) / horizon,  clamped to [0, 1]
        y_s[i] = 0 if no press within the next `horizon` samples.

    Use with MSE / Huber regression loss or as a soft target for LightGBM.
    The last (horizon-1) samples are zeroed (no full lookahead).

    Parameters
    ----------
    y       : (N,) momentary press labels (0/1)
    horizon : look-ahead window in samples (same as used in make_horizon_labels)

    Returns
    -------
    y_s : (N,) float32 soft labels in [0, 1]
    """
    if horizon <= 1:
        return y.astype(np.float32).copy()
    N = len(y)
    y_s = np.zeros(N, dtype=np.float32)
    press_indices = np.where(y > 0)[0]
    for k in press_indices:
        start = max(0, k - horizon + 1)
        end = min(N - horizon + 1, k + 1)
        for i in range(start, end):
            weight = float(horizon - (k - i)) / horizon
            if weight > y_s[i]:
                y_s[i] = weight
    return y_s


def compute_hjorth(window: np.ndarray) -> np.ndarray:
    """
    Compute Hjorth parameters (activity, mobility, complexity) per channel.

    From the MRCP literature: these three parameters efficiently characterise
    the morphology and temporal frequency of slow cortical potentials without
    requiring an explicit FFT.

    Parameters
    ----------
    window : (T, C) array — a single EEG window (any normalization)

    Returns
    -------
    params : (C, 3) float32 array
        [:, 0] = Activity   — variance of the signal (amplitude power)
        [:, 1] = Mobility   — sqrt(var(d1) / var(signal)) ~ mean frequency
        [:, 2] = Complexity — Mob(d2)/Mob(d1) ~ rate of frequency change
    """
    d1 = np.diff(window, axis=0)       # (T-1, C)
    d2 = np.diff(d1,     axis=0)       # (T-2, C)

    var0 = np.var(window, axis=0) + 1e-12   # (C,)
    var1 = np.var(d1,     axis=0) + 1e-12
    var2 = np.var(d2,     axis=0) + 1e-12

    activity   = var0
    mobility   = np.sqrt(var1 / var0)
    mob_d2     = np.sqrt(var2 / var1)
    complexity = mob_d2 / (mobility + 1e-12)

    return np.stack([activity, mobility, complexity], axis=1).astype(np.float32)


def window_band_power(
    window: np.ndarray,
    fs: float,
    bands: Optional[dict] = None,
) -> np.ndarray:
    """
    Per-channel band power for one EEG window using Welch's method.

    Papers recommend delta (0.5–4 Hz) and theta (4–8 Hz) as the primary MRCP
    frequency bands. Delta captures the slow negative readiness ramp; theta
    captures the faster motor-preparation oscillations.

    Parameters
    ----------
    window : (T, C) EEG window (normalised or raw)
    fs     : effective sampling rate in Hz
    bands  : dict {name: (fmin, fmax)}. Default: delta + theta MRCP bands.

    Returns
    -------
    features : (C * n_bands,) float32 flat feature vector
    """
    from scipy.signal import welch

    if bands is None:
        bands = {"delta": (0.5, 4.0), "theta": (4.0, 8.0)}

    T = window.shape[0]
    nperseg = min(T, 32)
    freqs, psd = welch(window.T, fs=fs, nperseg=nperseg, axis=-1)  # (C, n_freqs)

    df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    feats = []
    for fmin, fmax in bands.values():
        mask = (freqs >= fmin) & (freqs <= fmax)
        power = psd[:, mask].sum(axis=-1) * df_freq   # (C,)
        feats.append(power)

    return np.concatenate(feats).astype(np.float32)   # (C * n_bands,)


def make_mrcp_template(
    X: np.ndarray,
    y_momentary: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Estimate the MRCP waveform template by averaging EEG epochs time-locked to
    button press onsets.

    This is the reference signal used by the matched-filter classifier: if the
    current EEG window resembles the template, a press is imminent.

    Parameters
    ----------
    X           : (N, C) EEG array (filtered, at FS_EFF)
    y_momentary : (N,) momentary press labels (0/1) — use ORIGINAL y, not horizon-expanded
    T           : look-back window length in samples (same as used in build_windows)

    Returns
    -------
    template : (T, C) float32 — average pre-press EEG epoch
    """
    press_idx = np.where(y_momentary > 0)[0]
    epochs = [X[k - T : k] for k in press_idx if k >= T]
    if not epochs:
        return np.zeros((T, X.shape[1]), dtype=np.float32)
    return np.mean(epochs, axis=0).astype(np.float32)


def matched_filter_score(
    window: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    Per-channel normalised cross-correlation between one EEG window and the MRCP template.

    A score near +1 means the current window closely matches the pre-movement
    pattern; near 0 means no match. This is the matched-filter feature used by
    Shakeel et al. (2016), which achieves ~12 ms detection latency.

    Parameters
    ----------
    window   : (T, C) current EEG window (per-window normalised)
    template : (T, C) MRCP template from make_mrcp_template

    Returns
    -------
    scores : (C,) float32 correlation coefficient per channel, in [-1, 1]
    """
    w_norm = (window   - window.mean(axis=0))   / (window.std(axis=0)   + 1e-8)
    t_norm = (template - template.mean(axis=0)) / (template.std(axis=0) + 1e-8)
    return (w_norm * t_norm).mean(axis=0).astype(np.float32)


def euclidean_align(X3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean Alignment (EA) — per-trial spatial whitening.

    Computes a whitening matrix W from the mean spatial covariance of the
    input windows, then applies it so the aligned set has identity mean
    covariance. Use once per trial (or per subject), computed from training
    windows only, then apply the same W to test windows.

    Parameters
    ----------
    X3d : (N, T, C) array of EEG windows

    Returns
    -------
    X_aligned : (N, T, C) whitened windows
    W         : (C, C) whitening matrix  —  store and reuse on held-out data
                via  X_test_aligned = X_test @ W.T
    """
    from scipy.linalg import sqrtm, inv

    T = X3d.shape[1]
    X64 = X3d.astype(np.float64)
    covs = np.einsum('ntc,ntd->ncd', X64, X64) / T   # (N, C, C)
    R_mean = covs.mean(axis=0)                         # (C, C)
    W = inv(sqrtm(R_mean)).real                        # (C, C)
    return (X3d @ W.T).astype(np.float32), W


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    T: int,
    groups: Optional[np.ndarray] = None,
    per_window_norm: bool = True,
    stride: int = 1,
    add_derivative: bool = False,
) -> tuple:
    """
    Build causal sliding windows of length T.

    Window i covers X[i-T : i] and predicts label y[i] (one step ahead).
    Windows that span two different groups (trials / subjects) are skipped
    to prevent cross-boundary leakage.

    Parameters
    ----------
    X              : (N, C) raw EEG array
    y              : (N,)   integer labels
    T              : look-back window length in samples
    groups         : (N,)   group id per sample (trial id, subject id, …).
                     If None, a single uniform group is assumed (no boundary skipping
                     beyond the first T samples).
    per_window_norm: z-score each window per channel independently.
                     Removes session-level amplitude drift without fitting a scaler.
    stride         : step size between consecutive window end-points.
                     stride=1 (default) gives one window per sample.
                     stride=4 gives one window every 4 samples, reducing autocorrelation
                     between adjacent windows by ~4x and speeding up training.
    add_derivative : if True, append the first temporal difference (np.diff) of the
                     window (C*(T-1) values) to the flat feature vector. Captures the
                     MRCP ramp slope directly — helpful for tree and linear models that
                     cannot detect trends from raw sample values alone.
                     Only applied in flat (RF/MLP) mode; ignored when groups is provided.

    Returns
    -------
    X_win : (M, C*T)         flat — RF / MLP format          [groups=None, no derivative]
            (M, C*(2T-1))    flat with derivative             [groups=None, add_derivative=True]
            (M, C, T)        channels-first — Conv1D format   [groups provided]
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

    for i in range(T, N, stride):
        if groups[i] != groups[i - T]:
            continue
        w = X[i - T : i].copy()  # (T, C)
        if per_window_norm:
            mu = w.mean(axis=0, keepdims=True)
            sd = w.std(axis=0, keepdims=True) + 1e-8
            w = (w - mu) / sd
        if return_groups:
            wins.append(w.T)          # (C, T) — Conv1D channels-first
        else:
            flat = w.flatten()        # (C*T,)
            if add_derivative:
                dw = np.diff(w, axis=0)   # (T-1, C)
                flat = np.concatenate([flat, dw.flatten()])
            wins.append(flat)
        labels.append(y[i])
        grps.append(groups[i])

    X_win = np.array(wins, dtype=np.float32)
    y_win = np.array(labels, dtype=np.int64)
    g_win = np.array(grps, dtype=np.int64)

    if return_groups:
        return X_win, y_win, g_win
    return X_win, y_win
