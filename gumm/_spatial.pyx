# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport acos, sqrt

cnp.import_array()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
cdef double PI = 3.141592653589793

# Integer codes for edge-correction — avoids Python string comparison in the
# O(n²) hot path.
cdef int EDGE_ISOTROPIC = 0
cdef int EDGE_BORDER    = 1
cdef int EDGE_NONE      = 2


# ---------------------------------------------------------------------------
# C-level helpers (not exposed to Python)
# ---------------------------------------------------------------------------

cdef double _isotropic_weight(
    double[:] point,
    double    r,
    double[:, :] bounds,
) except -1.0:
    """
    Isotropic edge-correction weight at radius *r*.

    Approximates the proportion of the circle's circumference that lies
    inside the study area (one dimension at a time) and returns its
    reciprocal.  Based on the ripley(r) estimator from Ripley (1976).
    """
    cdef:
        double w = 1.0
        double d1, d2
        int    i

    for i in range(2):
        d1 = point[i] - bounds[i, 0]
        d2 = bounds[i, 1] - point[i]
        if d1 < r:
            w *= 1.0 - acos(d1 / r) / PI
        if d2 < r:
            w *= 1.0 - acos(d2 / r) / PI

    return 1.0 / w if w > 0.0 else 0.0


cdef double _border_weight(
    double[:] point,
    double    r,
    double[:, :] bounds,
):
    """
    Border edge-correction: returns 0 when the r-disc crosses the boundary
    (i.e. excludes the point from the estimator at this radius).
    """
    cdef int i
    for i in range(2):
        if point[i] - r < bounds[i, 0] or point[i] + r > bounds[i, 1]:
            return 0.0
    return 1.0


cdef double _estimate_k_function(
    double[:, :] pts,
    double       r,
    double[:, :] bounds,
    double       area,
    int          correction_type,
) except -1.0:
    """
    Weighted Ripley's K estimate at a single radius *r*.

    Computes  K(r) = (A / n(n-1)) * sum_i sum_{j≠i} w_ij * 1[d_ij <= r]
    using a pure C double-loop, eliminating all Python/numpy overhead.
    """
    cdef:
        int    n = pts.shape[0]
        int    i, j
        double k_r = 0.0
        double r_sq = r * r
        double dx, dy, w, counts

    if n < 2:
        return 0.0

    for i in range(n):
        counts = 0.0
        for j in range(n):
            if i != j:
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                if dx * dx + dy * dy <= r_sq:
                    counts += 1.0

        if correction_type == EDGE_ISOTROPIC:
            w = _isotropic_weight(pts[i], r, bounds)
        elif correction_type == EDGE_BORDER:
            w = _border_weight(pts[i], r, bounds)
        else:
            w = 1.0

        k_r += counts * w

    return (area / (<double>n * (<double>n - 1.0))) * k_r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rotate_and_find_elbow(object data):
    """
    Elbow detection via chord rotation.

    Rotates the 2-D curve so its chord is horizontal, then returns the
    original y-value at the point of maximum perpendicular deviation
    (i.e. the minimum rotated-y coordinate).

    Parameters
    ----------
    data : array-like of shape (n_points, 2)

    Returns
    -------
    prob_cut : float
    """
    cdef:
        cnp.ndarray[cnp.double_t, ndim=2] d = np.asarray(data, dtype=np.float64)
        double theta, co, si
        int    elbow_idx, n_pts

    n_pts = d.shape[0]
    theta = np.arctan2(d[n_pts - 1, 1] - d[0, 1], d[n_pts - 1, 0] - d[0, 0])
    co    = np.cos(theta)
    si    = np.sin(theta)

    rot_matrix = np.array([[co, -si], [si, co]], dtype=np.float64)
    rot_data   = d.dot(rot_matrix)

    elbow_idx = int(np.argmin(rot_data[:, 1]))
    return d[elbow_idx, 1] + 0.05


def robust_adaptive_ripley_k(
    object points,
    object probabilities,
    radii                  = None,
    int    max_nsim        = 100,
    int    min_points      = 30,
    str    edge_correction = 'isotropic',
    double confidence_level = 0.99,
):
    """
    Adaptive Ripley's K with Monte Carlo CSR envelope testing.

    Scans a grid of probability thresholds and identifies the one at which
    the observed spatial pattern deviates most significantly from Complete
    Spatial Randomness (CSR).  Significance is assessed via Monte Carlo
    simulation of CSR point patterns (Ripley, 1976; Diggle, 2003).

    Parameters
    ----------
    points : array-like of shape (n, 2)
    probabilities : array-like of shape (n,)
    radii : array-like, optional
        Evaluation radii.  Defaults to 30 linearly-spaced values in
        [0, sqrt(area) / 4].
    max_nsim : int
        Number of Monte Carlo CSR replications per threshold.
    min_points : int
        Minimum points required after thresholding.
    edge_correction : {'isotropic', 'border', 'none'}
    confidence_level : float

    Returns
    -------
    optimal_threshold : float or None
    diagnostics : dict
    """
    cdef:
        cnp.ndarray[cnp.double_t, ndim=2] pts  = np.ascontiguousarray(
            points, dtype=np.float64
        )
        cnp.ndarray[cnp.double_t, ndim=1] prbs = np.ascontiguousarray(
            probabilities, dtype=np.float64
        )

    if pts.shape[1] != 2:
        raise ValueError("points must be 2-dimensional")
    if pts.shape[0] != prbs.shape[0]:
        raise ValueError("points and probabilities must have the same length")

    cdef:
        int    correction_type
        double area, max_radius
        double max_deviation = -1e308
        double deviation, p_value, thresh
        int    i, ri, n, n_radii

    # ---- edge-correction type code ----
    if edge_correction == 'isotropic':
        correction_type = EDGE_ISOTROPIC
    elif edge_correction == 'border':
        correction_type = EDGE_BORDER
    else:
        correction_type = EDGE_NONE

    # ---- study area with 5 % buffer ----
    bounds_arr = np.array([
        [pts[:, 0].min(), pts[:, 0].max()],
        [pts[:, 1].min(), pts[:, 1].max()],
    ], dtype=np.float64)
    buf = (bounds_arr[:, 1] - bounds_arr[:, 0]) * 0.05
    bounds_arr[:, 0] -= buf
    bounds_arr[:, 1] += buf
    area = float(np.prod(bounds_arr[:, 1] - bounds_arr[:, 0]))

    cdef double[:, :] bounds = bounds_arr

    # ---- radii grid ----
    if radii is None:
        max_radius = sqrt(area) / 4.0
        radii = np.linspace(0.0, max_radius, 30)
    cdef cnp.ndarray[cnp.double_t, ndim=1] radii_arr = np.asarray(
        radii, dtype=np.float64
    )
    n_radii = radii_arr.shape[0]

    # Theoretical K under CSR:  K(r) = π r²
    k_theo = np.pi * radii_arr ** 2

    thresholds = np.percentile(prbs, np.linspace(5.0, 95.0, 30))
    optimal_threshold = None
    diagnostics = {'thresholds': [], 'deviations': [], 'p_values': []}

    cdef double[:, :] sel_view, csr_view
    cdef cnp.ndarray[cnp.double_t, ndim=1] k_obs_arr
    cdef cnp.ndarray[cnp.double_t, ndim=2] k_sims_arr
    cdef cnp.ndarray[cnp.double_t, ndim=2] sel_pts, csr_pts

    for thresh in thresholds:
        mask = prbs > thresh
        if int(np.sum(mask)) < min_points:
            continue

        sel_pts  = np.ascontiguousarray(pts[mask], dtype=np.float64)
        sel_view = sel_pts
        n        = sel_pts.shape[0]

        # ---- observed K at each radius ----
        k_obs_arr = np.empty(n_radii, dtype=np.float64)
        for ri in range(n_radii):
            k_obs_arr[ri] = _estimate_k_function(
                sel_view, radii_arr[ri], bounds, area, correction_type
            )

        # ---- Monte Carlo CSR simulations ----
        k_sims_arr = np.empty((max_nsim, n_radii), dtype=np.float64)
        for i in range(max_nsim):
            csr_pts = np.ascontiguousarray(
                np.column_stack([
                    np.random.uniform(bounds_arr[0, 0], bounds_arr[0, 1], n),
                    np.random.uniform(bounds_arr[1, 0], bounds_arr[1, 1], n),
                ]),
                dtype=np.float64,
            )
            csr_view = csr_pts
            for ri in range(n_radii):
                k_sims_arr[i, ri] = _estimate_k_function(
                    csr_view, radii_arr[ri], bounds, area, correction_type
                )

        # ---- supremum test statistic & Monte Carlo p-value ----
        # T_obs = sup_r |K_obs(r) - K_theo(r)|
        deviation = float(np.max(np.abs(k_obs_arr - k_theo)))
        p_value   = float(np.mean(
            np.max(np.abs(k_sims_arr - k_theo[np.newaxis, :]), axis=1) >= deviation
        ))

        diagnostics['thresholds'].append(float(thresh))
        diagnostics['deviations'].append(deviation)
        diagnostics['p_values'].append(p_value)

        if deviation > max_deviation and p_value < 0.05:
            max_deviation     = deviation
            optimal_threshold = float(thresh)

    diagnostics.update({
        'optimal_threshold': optimal_threshold,
        'max_deviation':     float(max_deviation) if max_deviation > -1e307 else None,
        'radii':             radii_arr,
        'area':              area,
        'n_points':          pts.shape[0],
        'confidence_level':  confidence_level,
    })

    return optimal_threshold, diagnostics
