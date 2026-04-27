import numpy as np
from scipy.signal import welch


def band_power(sig: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    """Compute average power in [fmin, fmax] Hz using Welch's method."""
    nperseg = min(len(sig), 64)
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return float(float(psd[idx].sum() * (freqs[1] - freqs[0]) if len(freqs) > 1 else 0))
