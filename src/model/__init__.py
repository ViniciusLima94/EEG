"""
src.model package — PyTorch model definitions.

Convenient imports:
    from src.model import MLP
    from src.model import CausalConv1D, make_model
    from src import MLP, CausalConv1D, make_model
"""

from .mlp import MLP
from .conv1d import CausalConv1D, make_model

__all__ = ["MLP", "CausalConv1D", "make_model"]
