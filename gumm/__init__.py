"""
GUMM — Gaussian Uniform Mixture Model
======================================
Unsupervised membership classifier.  Fits K Gaussian cluster components
embedded in a uniform background via EM, with automatic threshold selection
combining a rotation-elbow method and Ripley's K spatial point-pattern test.

Computationally expensive regions (Monte Carlo Ripley K simulations and
weighted covariance computation) are parallelised with OpenMP.
"""

from ._normalize import normalize_features
from ._spatial   import rotate_and_find_elbow, robust_adaptive_ripley_k
from ._model     import GUMM

__version__ = '0.3.0'
__all__ = [
    'GUMM',
    'normalize_features',
    'robust_adaptive_ripley_k',
    'rotate_and_find_elbow',
]
