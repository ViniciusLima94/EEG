import numpy as np
import pandas as pd


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
