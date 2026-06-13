"""
Shared preprocessing utilities for causal sliding-window EEG models.

Usage
-----
from src.preprocessing import load_trial, build_windows, make_horizon_labels
from src.preprocessing import make_soft_horizon_labels, compute_hjorth
from src.preprocessing import window_band_power, make_mrcp_template, matched_filter_score
from src.preprocessing import euclidean_align, trial_quality, extract_features
# load_trial flags: apply_filter, car, trial_zscore, decimate, clip_percentile
"""

from __future__ import annotations

import os
import csv
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
    trial_zscore: bool = False,
    clip_percentile: float = 0.0,
    channels: Optional[list] = None,
    despike_sigma: float = 0.0,
    skip_seconds: float = 0.0,
) -> tuple[pd.DataFrame, list]:
    """
    Load one CSV trial, optionally band-pass filter it and/or decimate.

    Parameters
    ----------
    subject          : subject index (0-based)
    tache            : task name, e.g. "spt"
    trial            : trial index
    data_path        : directory containing the CSV files
    eeg_cols         : expected channel list; asserted if provided
    fs               : sampling frequency (Hz) of the raw recording
    lp, hp           : band-pass cut-offs (Hz). Defaults 0.5–8 Hz target the MRCP
                       delta/theta band while rejecting sub-Hz electrode drift.
    apply_filter     : whether to run the MNE band-pass filter
    decimate         : integer downsampling factor applied after filtering.
                       Uses scipy.signal.decimate (zero-phase, anti-aliased).
                       Effective fs after loading = fs // decimate. Default 1 = off.
    car              : if True, apply Common Average Reference after filtering:
                       subtract the instantaneous mean across all channels from each
                       channel. CAR is computed on the full montage before any
                       channel subsetting, preserving the correct common average.
    trial_zscore     : if True, z-score each channel over the full trial duration
                       after CAR (mean=0, std=1 per channel). Removes session-level
                       amplitude differences that cause trial-to-trial distribution
                       shift — complementary to EA which removes covariance drift.
    clip_percentile  : if > 0, replace per-channel samples outside
                       [clip_percentile, 100-clip_percentile] with the channel median.
                       Applied after trial_zscore. Typical value: 1.0 (i.e. p1/p99).
    channels         : if provided, subset the returned DataFrame and channel list
                       to this ordered list of channel names after all processing.
                       All channels must exist in the file. CAR (if enabled) is
                       computed on the full montage before subsetting.
                       Example: channels=["ch1", "ch2", "ch9"]
    despike_sigma    : if > 0, run a per-channel Hampel filter on raw ADC values
                       before bandpass filtering. Uses a 101-sample sliding window
                       (≈400 ms at 250 Hz): samples where
                       |x − local_median| > despike_sigma × local_MAD/0.6745
                       are replaced by the local median. Unlike a global-MAD
                       approach, this adapts to slow baseline drift and catches
                       multi-sample bursts as well as isolated single-sample spikes.
                       Typical value: 5.0. Default 0.0 = off.
    skip_seconds     : seconds to discard from the start of the trial after all
                       processing (filter, CAR, z-score, decimate). The full
                       signal is filtered first so the bandpass can settle, then
                       the initial portion is trimmed. Use this to remove
                       electrode-settling and filter-transient artefacts.
                       Default 0.0 = off.

    Returns
    -------
    df       : filtered (and optionally decimated) DataFrame
    eeg_cols : list of EEG channel names (excludes timestamp & button)
    """
    path = os.path.join(
        data_path, f"subject_{subject}_tache_{tache}_trial_{trial}.csv"
    )
    
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(open(path, "r").read(4096)).delimiter
    df = pd.read_csv(path, delimiter=delimiter)

    cols = [c for c in df.columns if c not in ("button", "timestamp", "led")]
    if eeg_cols is not None:
        validate_against = list(channels) if channels is not None else cols
        assert list(eeg_cols) == validate_against, (
            f"Trial {trial} channel mismatch: expected {eeg_cols}, got {validate_against}"
        )
    if despike_sigma > 0.0:
        from scipy.ndimage import median_filter as _mf
        arr = df[cols].values.astype(np.float64)
        # Hampel filter: local median/MAD in a sliding window.
        # Window = 2 * hampel_half_win + 1 samples (default 101 samples ≈ 400 ms at 250 Hz).
        _hw = 50  # half-window in raw samples
        _wlen = 2 * _hw + 1
        for c in range(arr.shape[1]):
            ch = arr[:, c]
            local_med = _mf(ch, size=_wlen, mode='nearest')
            local_mad = _mf(np.abs(ch - local_med), size=_wlen, mode='nearest') / 0.6745
            bad = (local_mad > 1e-12) & (np.abs(ch - local_med) > despike_sigma * local_mad)
            if bad.any():
                ch[bad] = local_med[bad]
                arr[:, c] = ch
        df[cols] = arr
    if apply_filter:
        df[cols] = filter_data(
            df[cols].values.T, fs, lp, hp, verbose=False
        ).T
    if car:
        arr = df[cols].values.astype(np.float64)
        arr -= arr.mean(axis=1, keepdims=True)
        df[cols] = arr
    if trial_zscore:
        arr = df[cols].values.astype(np.float64)
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8)
        df[cols] = arr
    if clip_percentile > 0.0:
        arr = df[cols].values.astype(np.float64)
        lo  = np.percentile(arr, clip_percentile, axis=0)
        hi  = np.percentile(arr, 100.0 - clip_percentile, axis=0)
        arr = np.clip(arr, lo, hi)
        df[cols] = arr
    if decimate > 1:
        from scipy.signal import decimate as sp_decimate
        eeg_decimated = sp_decimate(df[cols].values, decimate, axis=0, zero_phase=True)
        n_out = eeg_decimated.shape[0]
        # OR-pool button (and led) over each K-sample window so a brief press
        # that falls between decimated samples is never silently dropped.
        btn_raw = df["button"].to_numpy(dtype=np.int8)
        btn_dec = np.array(
            [bool(btn_raw[i * decimate : (i + 1) * decimate].any()) for i in range(n_out)],
            dtype=np.int8,
        )
        led_raw = df["led"].to_numpy(dtype=np.int8) if "led" in df.columns else None
        led_dec = (
            np.array(
                [bool(led_raw[i * decimate : (i + 1) * decimate].any()) for i in range(n_out)],
                dtype=np.int8,
            )
            if led_raw is not None
            else None
        )
        df = df.iloc[::decimate].iloc[:n_out].reset_index(drop=True)
        df[cols] = eeg_decimated
        df["button"] = btn_dec
        if led_dec is not None:
            df["led"] = led_dec
    if channels is not None:
        missing = [c for c in channels if c not in cols]
        if missing:
            raise ValueError(f"Trial {trial}: requested channels not found: {missing}")
        cols = list(channels)
        df = df[["timestamp"] + cols + ["button"]].copy()
    if skip_seconds > 0.0:
        fs_eff = fs // max(decimate, 1)
        skip_n = int(skip_seconds * fs_eff)
        df = df.iloc[skip_n:].reset_index(drop=True)
    # Keep only the rising edge of each button press (011110 → 010000)
    b = df["button"].to_numpy()
    df["button"] = np.diff(b, prepend=0).clip(0).astype(b.dtype)
    if "led" in df.columns:
        l = df["led"].to_numpy()
        df["led"] = np.diff(l, prepend=0).clip(0).astype(l.dtype)
    return df, cols


def make_post_press_mask(y_momentary: np.ndarray, blank_samples: int) -> np.ndarray:
    """
    Boolean mask of samples to EXCLUDE from the negative class during training.

    Samples in [onset, onset + blank_samples) after each press onset are marked
    True. This prevents post-movement EEG rebound and muscle artifacts from being
    trained as "rest" negatives, which otherwise inflate the false-alarm rate.

    Only apply to the TRAINING fold; evaluate on all windows in the test fold.

    Parameters
    ----------
    y_momentary   : (N,) momentary press labels (rising-edge only)
    blank_samples : samples to blank after each press (e.g. int(1.5 * FS_EFF))

    Returns
    -------
    mask : (N,) bool — True where samples should be dropped from negatives
    """
    mask = np.zeros(len(y_momentary), dtype=bool)
    for onset in np.where(y_momentary > 0)[0]:
        mask[onset : min(len(mask), onset + blank_samples)] = True
    return mask


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


def _temporal(w: np.ndarray, n_segs: int = 4) -> np.ndarray:
    """(T, C) → (C*(6 + n_segs),)

    Base features (×C): mean, std, slope, min, argmin, rms
    Ramp profile (×C): mean of each of n_segs equal temporal segments.
    The ramp profile encodes the CNV trajectory (early baseline → late negative
    ramp) that a single slope feature cannot capture for non-linear ramps.
    """
    T = w.shape[0]
    t_c   = np.arange(T, dtype=np.float64) - (T - 1) / 2
    t_var = (t_c ** 2).sum() + 1e-12
    w64   = w.astype(np.float64)
    base = np.stack([
        w64.mean(0), w64.std(0),
        (w64 * t_c[:, None]).sum(0) / t_var,
        w64.min(0), w64.argmin(0) / T,
        np.sqrt((w64 ** 2).mean(0)),
    ], axis=1)  # (C, 6)
    seg_size = max(1, T // n_segs)
    segs = np.stack([
        w64[i * seg_size : (i + 1) * seg_size].mean(0)
        for i in range(n_segs)
    ], axis=1)  # (C, n_segs)
    return np.concatenate([base, segs], axis=1).flatten().astype(np.float32)


def _freq(w: np.ndarray, fs: float) -> np.ndarray:
    """(T, C) → (C*3,)  [delta_power, theta_power, spectral_entropy] × channel"""
    from scipy.signal import welch
    f, p = welch(w.T, fs=fs, nperseg=min(w.shape[0], 64), axis=-1)  # (C, F)
    df   = f[1] - f[0]
    dp   = p[:, (f >= 0.5) & (f <= 4.0)].sum(-1) * df
    tp   = p[:, (f >= 4.0) & (f <= 8.0)].sum(-1) * df
    pn   = p[:, f > 0] / (p[:, f > 0].sum(-1, keepdims=True) + 1e-12)
    se   = -(pn * np.log(pn + 1e-12)).sum(-1)
    return np.stack([dp, tp, se], axis=1).flatten().astype(np.float32)


def extract_features(w: np.ndarray, fs: float) -> np.ndarray:
    """EA-aligned (T, C) → flat feature vector: temporal + freq + Hjorth per channel."""
    return np.concatenate([_temporal(w), _freq(w, fs), compute_hjorth(w).flatten()])


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


def trial_quality(
    X: np.ndarray,
    y_mom: np.ndarray,
    fs_eff: float,
    kurtosis_warn: float = 6.0,
    kurtosis_bad: float = 10.0,
    artifact_warn: float = 0.10,
    artifact_bad: float = 0.20,
    flat_std: float = 0.05,
    min_presses: int = 2,
) -> dict:
    """
    Compute quality metrics for one loaded trial and return a flag.

    Parameters
    ----------
    X             : (N, C) z-scored EEG array (output of load_trial)
    y_mom         : (N,) momentary press labels (rising edges)
    fs_eff        : effective sampling rate after decimation
    kurtosis_warn : per-channel kurtosis above which flag → 'warn'
    kurtosis_bad  : per-channel kurtosis above which flag stays 'warn'
                    (high kurtosis alone does not auto-exclude a trial)
    artifact_warn : fraction of (sample × channel) pairs with |z| > 2.5
                    above which flag → 'warn'
    artifact_bad  : same fraction above which flag → 'bad'
    flat_std      : per-channel std below which the channel is "flat"
                    (disconnected electrode); any flat channel → 'bad'
    min_presses   : fewer presses than this → 'bad'

    Returns
    -------
    dict with keys: n_presses, duration_s, press_rate_per_min, mean_ipi_s,
                    flat_channels, max_kurtosis, artifact_frac,
                    flag ('ok'|'warn'|'bad'), reasons (list[str])
    """
    from scipy.stats import kurtosis as _kurtosis

    N, C = X.shape
    press_idx = np.where(y_mom > 0)[0]
    n_presses = len(press_idx)
    duration_s = N / fs_eff
    press_rate = n_presses / duration_s * 60
    mean_ipi = float(np.diff(press_idx).mean() / fs_eff) if n_presses > 1 else float("nan")

    ch_std = X.std(axis=0)
    flat_channels = [int(i) for i in np.where(ch_std < flat_std)[0]]

    ch_kurt = _kurtosis(X, axis=0, fisher=False)  # fisher=False → normal distribution = 3
    max_kurtosis = float(ch_kurt.max())

    artifact_frac = float(np.mean(np.abs(X) > 2.5))

    reasons: list = []
    flag = "ok"

    if n_presses < min_presses:
        reasons.append(f"n_presses={n_presses} < {min_presses}")
        flag = "bad"
    if flat_channels:
        reasons.append(f"flat_ch={flat_channels}")
        if flag != "bad":
            flag = "warn"  # handled by zeroing the channel, not by dropping the trial
    if artifact_frac > artifact_bad:
        reasons.append(f"artifact_frac={artifact_frac:.2f} > {artifact_bad}")
        flag = "bad"
    elif artifact_frac > artifact_warn and flag == "ok":
        reasons.append(f"artifact_frac={artifact_frac:.2f} > {artifact_warn}")
        flag = "warn"
    if max_kurtosis > kurtosis_bad:
        reasons.append(f"max_kurt={max_kurtosis:.1f} > {kurtosis_bad}")
        if flag not in ("bad",):
            flag = "warn"
    elif max_kurtosis > kurtosis_warn and flag == "ok":
        reasons.append(f"max_kurt={max_kurtosis:.1f} > {kurtosis_warn}")
        flag = "warn"

    return dict(
        n_presses=n_presses,
        duration_s=round(duration_s, 1),
        press_rate_per_min=round(press_rate, 1),
        mean_ipi_s=round(mean_ipi, 2) if not np.isnan(mean_ipi) else None,
        flat_channels=flat_channels,
        max_kurtosis=round(max_kurtosis, 2),
        artifact_frac=round(artifact_frac, 4),
        flag=flag,
        reasons=reasons,
    )


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
