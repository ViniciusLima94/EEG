from __future__ import annotations

import numpy as np
import pandas as pd
from .spectral import band_power


def remove_artifacts_interpolate(
    df: pd.DataFrame,
    eeg_cols: list,
    q_upper: float = 0.99,
    q_lower: float = 0.01,
    margin: int = 5,
) -> pd.DataFrame:
    """
    Detect artifact samples (beyond quantile thresholds) per channel,
    expand the bad window by `margin` samples on each side,
    then linearly interpolate across the gap.

    Parameters
    ----------
    df       : EEG dataframe
    eeg_cols : channel column names
    q_upper  : upper quantile threshold
    q_lower  : lower quantile threshold
    margin   : extra samples to null on each side of the artifact
               (catches the rising/falling edge of the spike)
    """
    df_clean = df.copy()

    for ch in eeg_cols:
        col = df_clean[ch].copy()
        upper = col.quantile(q_upper)
        lower = col.quantile(q_lower)

        # 1. flag bad samples
        bad = (col > upper) | (col < lower)

        # 2. expand the mask by `margin` samples each side
        bad_expanded = bad.copy()
        for shift in range(1, margin + 1):
            bad_expanded |= bad.shift(shift, fill_value=False)
            bad_expanded |= bad.shift(-shift, fill_value=False)

        # 3. set bad samples to NaN, then interpolate linearly
        col[bad_expanded] = np.nan
        col = col.interpolate(method="linear", limit_direction="both")

        df_clean[ch] = col
        print(
            f"{ch:6s}  artifacts={bad.sum():4d}  "
            f"expanded={bad_expanded.sum():4d}  "
            f"[{lower:.1f}, {upper:.1f}]"
        )

    return df_clean


def event_detection_metrics(
    probs: np.ndarray,
    y_momentary: np.ndarray,
    threshold: float,
    horizon: int,
    fs_eff: float,
) -> dict:
    """
    Event-level detection metrics for button-press prediction.

    Instead of sample-level AUC, evaluates whether each discrete press *event*
    was predicted at least once within the `horizon` samples before its onset.
    This matches the real BCI use-case: "did the system fire before the press?"

    Parameters
    ----------
    probs        : (N,) per-sample predicted probabilities
    y_momentary  : (N,) original momentary press labels (0/1) — NOT horizon-expanded
    threshold    : decision threshold
    horizon      : anticipatory window in samples (same as HORIZON used for labelling)
    fs_eff       : effective sampling rate (Hz) — used for latency and FA rate

    Returns
    -------
    dict with:
        event_sensitivity  — fraction of press events with at least one true detection
        fa_per_minute      — false alarm events per minute of recording time
        mean_latency_ms    — mean detection advance (ms before press onset), detected events only
        n_events           — total press events in recording
        n_detected         — events detected (true positives)
        n_false_alarms     — false alarm events (runs of preds=1 outside any event window)
    """
    N = len(probs)
    preds = (probs >= threshold).astype(np.int8)

    # 1. Find press event onsets (first sample of each contiguous run of 1s)
    y = y_momentary.astype(np.int8)
    onsets = list(np.where((y[1:] == 1) & (y[:-1] == 0))[0] + 1)
    if y[0] == 1:
        onsets = [0] + onsets

    # 2. Mark every sample inside a pre-press window as "protected"
    #    (true-positive zone; predictions here are NOT false alarms)
    protected = np.zeros(N, dtype=bool)
    for onset in onsets:
        protected[max(0, onset - horizon) : onset + 1] = True

    # 3. Event sensitivity + mean detection latency
    n_detected = 0
    latencies_ms: list[float] = []
    for onset in onsets:
        win_start = max(0, onset - horizon)
        win_probs = probs[win_start:onset]
        if len(win_probs) > 0 and win_probs.max() >= threshold:
            n_detected += 1
            first_det = win_start + int(np.argmax(win_probs >= threshold))
            latencies_ms.append((onset - first_det) / fs_eff * 1000.0)

    # 4. False alarms: runs of preds=1 entirely outside protected windows
    fa_signal = preds & (~protected).astype(np.int8)
    n_fa = int(np.sum((fa_signal[1:] == 1) & (fa_signal[:-1] == 0)))
    if fa_signal[0] == 1:
        n_fa += 1

    non_event_seconds = float((~protected).sum()) / fs_eff
    fa_per_min = n_fa / max(non_event_seconds / 60.0, 1e-9)

    return {
        "event_sensitivity": n_detected / max(len(onsets), 1),
        "fa_per_minute": fa_per_min,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "n_events": len(onsets),
        "n_detected": n_detected,
        "n_false_alarms": n_fa,
    }


def extract_features(window: np.ndarray, fs: float, BANDS: dict) -> np.ndarray:
    """window shape: (n_samples, n_channels) → 1-D feature vector."""
    feats = []
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        for fmin, fmax in BANDS.values():
            feats.append(band_power(sig, fs, fmin, fmax))
        mean = np.mean(sig)
        feats.append(np.std(sig) / mean if mean != 0 else 0.0)
    return np.array(feats, dtype=np.float32)
