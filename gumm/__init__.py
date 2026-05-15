"""
GUMM — Gaussian Uniform Mixture Model
======================================
Unsupervised membership classifier.  Fits K Gaussian cluster components
embedded in a uniform background via EM, then automatically selects a
membership threshold using a rotation-elbow method refined by Ripley's K
spatial point-pattern test.
"""

from ._normalize import normalize_features
from ._spatial   import rotate_and_find_elbow, robust_adaptive_ripley_k
from ._model     import GUMM

__version__ = '0.2.0'
__all__ = [
    'GUMM',
    'normalize_features',
    'robust_adaptive_ripley_k',
    'rotate_and_find_elbow',
]
