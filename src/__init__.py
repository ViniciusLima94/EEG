"""EEG src package

Expose core modules and commonly used helpers for convenience.
"""

__version__ = "0.1.0"

from . import io  # keep module import for consumers that want the submodule
from .postproc import remove_artifacts_interpolate, extract_features
from .spectral import band_power
from . import model  # expose model subpackage
from .model import MLP

__all__ = [
    "io",
    "postproc",
    "spectral",
    "model",
    "MLP",
    "remove_artifacts_interpolate",
    "extract_features",
    "band_power",
    "__version__",
]
