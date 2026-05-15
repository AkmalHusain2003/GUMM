# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math   cimport acos, sqrt
from cython.parallel cimport prange
cimport openmp

cnp.import_array()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
cdef double PI = 3.141592653589793

# Integer codes for edge-correction — avoids Python string comparison in the
# O(n²) per-simulation hot path.
cdef int EDGE_ISOTROPIC = 0
cdef int EDGE_BORDER    = 1
cdef int EDGE_NONE      = 2


# ---------------------------------------------------------------------------
# nogil C-level helpers
#
# All functions are declared `nogil` so they can be called freely from
# OpenMP parallel regions (prange blocks).  They operate exclusively on
# typed memoryviews and C scalars — no Python objects are touched.
# ---------------------------------------------------------------------------

cdef double _isotropic_weight_2d(
    double      px,
    double      py,
    double      r,
    double[:, :] bounds,
) nogil:
    """
    Isotropic edge-correction weight (Ripley 1976) at radius *r*.

    Approximates the fraction of the circle's circumference inside the
    study area, one dimension at a time, and returns the reciprocal.
    Uses scalar x/y arguments to avoid any memoryview-slice overhead
    inside the OpenMP parallel region.
    """
    cdef double w = 1.0, d1, d2

    # x dimension
    d1 = px - bounds[0, 0]
    d2 = bounds[0, 1] - px
    if d1 < r:
        w *= 1.0 - acos(d1 / r) / PI
    if d2 < r:
        w *= 1.0 - acos(d2 / r) / PI

    # y dimension
    d1 = py - bounds[1, 0]
    d2 = bounds[1, 1] - py
    if d1 < r:
        w *= 1.0 - acos(d1 / r) / PI
    if d2 < r:
        w *= 1.0 - acos(d2 / r) / PI

    return 1.0 / w if w > 0.0 else 0.0


cdef double _border_weight_2d(
    double      px,
    double      py,
    double      r,
    double[:, :] bounds,
) nogil:
    """
    Border edge-correction: excludes points whose r-disc crosses the boundary.
    """
    if px - r < bounds[0, 0] or px + r > bounds[0, 1]:
        return 0.0
    if py - r < bounds[1, 0] or py + r > bounds[1, 1]:
        return 0.0
    return 1.0


cdef double _estimate_k_pts(
    double[:]    pts_x,
    double[:]    pts_y,
    int          n,
    double       r,
    double[:, :] bounds,
    double       area,
    int          correction_type,
) nogil:
    """
    Weighted Ripley's K estimate at radius *r* for *n* points given as
    separate x/y 1-D arrays.

    K(r) = (A / n(n-1)) * Σ_i Σ_{j≠i} w_ij * 1[d_ij ≤ r]

    All arithmetic is pure C — safe inside an OpenMP parallel region.
    """
    cdef:
        int    i, j
        double k_r  = 0.0
        double r_sq = r * r
        double dx, dy, w, counts

    if n < 2:
        return 0.0

    for i in range(n):
        counts = 0.0
        for j in range(n):
            if i != j:
                dx = pts_x[i] - pts_x[j]
                dy = pts_y[i] - pts_y[j]
                if dx * dx + dy * dy <= r_sq:
                    counts += 1.0

        if correction_type == EDGE_ISOTROPIC:
            w = _isotropic_weight_2d(pts_x[i], pts_y[i], r, bounds)
        elif correction_type == EDGE_BORDER:
            w = _border_weight_2d(pts_x[i], pts_y[i], r, bounds)
        else:
            w = 1.0

        k_r += counts * w

    return (area / (<double>n * (<double>n - 1.0))) * k_r


cdef double _estimate_k_sim(
    double[:, :] sim_x,
    double[:, :] sim_y,
    int          sim_idx,
    int          n,
    double       r,
    double[:, :] bounds,
    double       area,
    int          correction_type,
) nogil:
    """
    Ripley's K for the *sim_idx*-th pre-generated CSR simulation.

    Takes the full (max_nsim, n) arrays and indexes by *sim_idx* to avoid
    creating any Python objects (memoryview sub-slices) inside the
    OpenMP parallel region.
    """
    cdef:
        int    i, j
        double k_r  = 0.0
        double r_sq = r * r
        double dx, dy, w, counts
        double px, py

    if n < 2:
        return 0.0

    for i in range(n):
        px     = sim_x[sim_idx, i]
        py     = sim_y[sim_idx, i]
        counts = 0.0

        for j in range(n):
            if i != j:
                dx = px - sim_x[sim_idx, j]
                dy = py - sim_y[sim_idx, j]
                if dx * dx + dy * dy <= r_sq:
                    counts += 1.0

        if correction_type == EDGE_ISOTROPIC:
            w = _isotropic_weight_2d(px, py, r, bounds)
        elif correction_type == EDGE_BORDER:
            w = _border_weight_2d(px, py, r, bounds)
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
    (minimum rotated-y coordinate).

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
    # Negative indexing (d[-1, :]) is unsafe with wraparound=False;
    # use explicit last-index arithmetic instead.
    theta = np.arctan2(
        d[n_pts - 1, 1] - d[0, 1],
        d[n_pts - 1, 0] - d[0, 0],
    )
    co = np.cos(theta)
    si = np.sin(theta)

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
    int    n_threads       = 1,
):
    """
    Adaptive Ripley's K with parallelised Monte Carlo CSR envelope testing.

    Scans a grid of probability thresholds and identifies the one at which
    the observed spatial pattern deviates most significantly from Complete
    Spatial Randomness (CSR).

    Monte Carlo simulations are parallelised with OpenMP across *n_threads*
    threads.  Random CSR coordinates are generated in bulk with NumPy
    before releasing the GIL, so the parallel region is entirely nogil.

    Parameters
    ----------
    points : array-like of shape (n, 2)
    probabilities : array-like of shape (n,)
    radii : array-like, optional
    max_nsim : int
        Number of Monte Carlo CSR replications.
    min_points : int
        Minimum points required after thresholding.
    edge_correction : {'isotropic', 'border', 'none'}
    confidence_level : float
    n_threads : int
        OpenMP threads for the MC parallel loop.

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

    # Theoretical K under CSR: K(r) = π r²
    k_theo = np.pi * radii_arr ** 2

    thresholds = np.percentile(prbs, np.linspace(5.0, 95.0, 30))
    optimal_threshold = None
    diagnostics = {'thresholds': [], 'deviations': [], 'p_values': []}

    cdef:
        cnp.ndarray[cnp.double_t, ndim=2] sel_pts
        cnp.ndarray[cnp.double_t, ndim=1] k_obs_arr
        cnp.ndarray[cnp.double_t, ndim=2] k_sims_arr
        cnp.ndarray[cnp.double_t, ndim=2] csr_x_arr
        cnp.ndarray[cnp.double_t, ndim=2] csr_y_arr
        cnp.ndarray[cnp.double_t, ndim=1] sel_x_1d
        cnp.ndarray[cnp.double_t, ndim=1] sel_y_1d
        double[:, :] csr_x_view, csr_y_view
        double[:, :] k_sims_view
        double[:]    sel_x, sel_y, radii_view

    openmp.omp_set_num_threads(n_threads)

    for thresh in thresholds:
        mask = prbs > thresh
        if int(np.sum(mask)) < min_points:
            continue

        sel_pts  = np.ascontiguousarray(pts[mask], dtype=np.float64)
        n        = sel_pts.shape[0]

        # Separate x/y 1-D views for the observed points
        sel_x_1d = np.ascontiguousarray(sel_pts[:, 0], dtype=np.float64)
        sel_y_1d = np.ascontiguousarray(sel_pts[:, 1], dtype=np.float64)
        sel_x    = sel_x_1d
        sel_y    = sel_y_1d

        # ---- observed K at each radius (sequential — called once) ----
        k_obs_arr  = np.empty(n_radii, dtype=np.float64)
        radii_view = radii_arr
        for ri in range(n_radii):
            k_obs_arr[ri] = _estimate_k_pts(
                sel_x, sel_y, n, radii_view[ri], bounds, area, correction_type
            )

        # ---- Pre-generate ALL CSR coordinates (numpy, requires GIL) ----
        # shape: (max_nsim, n) — bulk generation before releasing the GIL
        csr_x_arr  = np.ascontiguousarray(
            np.random.uniform(bounds_arr[0, 0], bounds_arr[0, 1], (max_nsim, n)),
            dtype=np.float64,
        )
        csr_y_arr  = np.ascontiguousarray(
            np.random.uniform(bounds_arr[1, 0], bounds_arr[1, 1], (max_nsim, n)),
            dtype=np.float64,
        )
        csr_x_view = csr_x_arr
        csr_y_view = csr_y_arr

        k_sims_arr  = np.empty((max_nsim, n_radii), dtype=np.float64)
        k_sims_view = k_sims_arr

        # ---- Parallelised MC loop — GIL released ----
        # Each iteration i owns row k_sims_view[i, :] exclusively.
        # _estimate_k_sim and _isotropic_weight_2d/_border_weight_2d are
        # all declared nogil, so no Python call enters the parallel region.
        for i in prange(max_nsim, nogil=True, schedule='dynamic'):
            for ri in range(n_radii):
                k_sims_view[i, ri] = _estimate_k_sim(
                    csr_x_view, csr_y_view, i, n,
                    radii_view[ri], bounds, area, correction_type,
                )

        # ---- supremum test statistic & Monte Carlo p-value ----
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
