"""EEG src package

Expose core modules and commonly used helpers for convenience.
"""

__version__ = "0.1.0"

from . import io
from .postproc import remove_artifacts_interpolate, extract_features
from .spectral import band_power
from .preprocessing import load_trial, build_windows
from . import model
from .model import MLP, CausalConv1D, make_model

__all__ = [
    "io",
    "postproc",
    "spectral",
    "model",
    "preprocessing",
    "MLP",
    "CausalConv1D",
    "make_model",
    "load_trial",
    "build_windows",
    "remove_artifacts_interpolate",
    "extract_features",
    "band_power",
    "__version__",
]
