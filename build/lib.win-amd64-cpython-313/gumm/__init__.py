"""
GUMM — Gaussian Uniform Mixture Model
======================================
Unsupervised membership classifier combining EM with spatial point-pattern
analysis (Ripley's K) for automatic threshold selection.
"""

from ._normalize import normalize_features
from ._spatial   import rotate_and_find_elbow, robust_adaptive_ripley_k
from ._model     import GUMM

__version__ = '0.1.0'
__all__ = [
    'GUMM',
    'normalize_features',
    'robust_adaptive_ripley_k',
    'rotate_and_find_elbow',
]
