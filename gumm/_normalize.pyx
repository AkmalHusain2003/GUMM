# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cnp.import_array()


def normalize_features(object X, double padding=0.1):
    """
    Normalize features to [0, 1] via robust percentile-based min-max scaling.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    padding : float
        Fractional padding added beyond [q1, q99] bounds.

    Returns
    -------
    X_norm : ndarray of shape (n_samples, n_features)
    scaling_params : dict[int, dict]
        Keys: feature index.
        Values: {'min', 'max', 'range'} used for scaling.
    """
    cdef:
        cnp.ndarray[cnp.double_t, ndim=2] Xd = np.asarray(X, dtype=np.float64)
        int    n_features = Xd.shape[1]
        int    i
        double q1, q99, data_range, pad_amt, min_val, max_val

    cdef cnp.ndarray[cnp.double_t, ndim=2] X_norm = np.empty_like(Xd)
    scaling_params = {}

    for i in range(n_features):
        col        = Xd[:, i]
        q1         = np.percentile(col, 1.0)
        q99        = np.percentile(col, 99.0)
        data_range = q99 - q1
        pad_amt    = data_range * padding
        min_val    = q1  - pad_amt
        max_val    = q99 + pad_amt

        scaling_params[i] = {
            'min':   min_val,
            'max':   max_val,
            'range': max_val - min_val,
        }

        X_norm[:, i] = np.clip(
            (col - min_val) / (max_val - min_val), 0.0, 1.0
        )

    return X_norm, scaling_params
