# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from scipy.stats      import multivariate_normal
from tqdm             import tqdm
from cython.parallel  cimport prange
cimport openmp

from ._normalize import normalize_features
from ._spatial   import rotate_and_find_elbow, robust_adaptive_ripley_k

cnp.import_array()


# ---------------------------------------------------------------------------
# Module-level C function
#
# Typed memoryview locals are only valid at module scope in Cython — not
# inside Python-class `def` methods.
#
# Parallelisation strategy for _weighted_cov:
#   Sigma_{p,q} = (1/N_k) Σ_j  γ_j (x_{j,p} - μ_p)(x_{j,q} - μ_q)
#
#   We parallelise over the OUTPUT row index p.  Thread `t` handles row
#   p = t and writes only to cov[p, :].  Because different threads write to
#   different rows there are no write conflicts, so no atomic/reduction
#   directives are needed.
#
#   The diff matrix diff[j, p] = X[j, p] - mu[p] is pre-computed in a
#   sequential pass (one write per cell, O(n·D)) before the parallel block
#   so all threads can read it concurrently without conflict.
# ---------------------------------------------------------------------------

cdef cnp.ndarray _weighted_cov(
    double[:, :] X,
    double[:]    gamma_k,
    double[:]    mu,
    double       N_k,
    int          n_threads,
):
    """
    Weighted sample covariance matrix with OpenMP row-parallel accumulation.

    Computes  Σ = (1/N_k) Σ_j γ_j (x_j - μ)(x_j - μ)^T
    """
    cdef:
        int    n_samples  = X.shape[0]
        int    n_features = X.shape[1]
        int    j, p, q

    # ---- Pre-compute diff[j, p] = X[j, p] - mu[p]  (sequential) ----
    cdef cnp.ndarray diff_buf = np.empty((n_samples, n_features), dtype=np.float64)
    cdef double[:, :] diff   = diff_buf

    for j in range(n_samples):
        for p in range(n_features):
            diff[j, p] = X[j, p] - mu[p]

    # ---- Parallel accumulation over output rows ----
    cdef cnp.ndarray cov_out = np.zeros((n_features, n_features), dtype=np.float64)
    cdef double[:, :] cov   = cov_out

    openmp.omp_set_num_threads(n_threads)

    # Each thread p writes exclusively to cov[p, :] → no write conflicts.
    for p in prange(n_features, nogil=True, schedule='static'):
        for j in range(n_samples):
            for q in range(n_features):
                cov[p, q] = cov[p, q] + gamma_k[j] * diff[j, p] * diff[j, q]
        for q in range(n_features):
            cov[p, q] = cov[p, q] / N_k

    return cov_out


# ---------------------------------------------------------------------------
# GUMM
# ---------------------------------------------------------------------------

class GUMM:
    """
    Gaussian Uniform Mixture Model (GUMM).

    Models data as K Gaussian cluster components embedded in a uniform
    background noise component:

        p(x) = π_U · U(x) + Σ_{k=1}^{K} π_k · N(x; μ_k, Σ_k)

    Parameters are estimated via Expectation-Maximization.  After fitting,
    points whose aggregate Gaussian posterior exceeds an automatically
    determined threshold are assigned to their most likely Gaussian
    component; remaining points are labelled 0 (background).

    Parameters
    ----------
    n_components : int
        Number of Gaussian cluster components K.
    n_epochs : int
        Maximum EM iterations.
    stable_percentage : float
        Convergence criterion: stop when the log-likelihood has not improved
        for ``stable_percentage * n_epochs`` consecutive steps.
    padding : float
        Fractional padding for feature normalisation bounds.
    max_nsim : int
        Monte Carlo replications used inside the Ripley's K spatial test
        when computing the automatic probability threshold.
    n_threads : int
        OpenMP threads used by (a) the covariance M-step and (b) the Ripley K
        Monte Carlo loop.  Set to -1 to use all available cores.
    random_state : int, optional
    """

    def __init__(
        self,
        n_components      = 1,
        n_epochs          = 1000,
        stable_percentage = 0.1,
        padding           = 0.1,
        max_nsim          = 100,
        n_threads         = 1,
        random_state      = None,
    ):
        self.n_components      = int(n_components)
        self.n_epochs          = int(n_epochs)
        self.stable_percentage = float(stable_percentage)
        self.padding           = float(padding)
        self.max_nsim          = int(max_nsim)

        # Resolve -1 to the physical core count
        if int(n_threads) == -1:
            import multiprocessing
            self.n_threads = multiprocessing.cpu_count()
        else:
            self.n_threads = int(n_threads)

        if random_state is not None:
            np.random.seed(int(random_state))

        # Set by fit()
        self.weights_         = None   # (K+1,)    [π_u, π_1 .. π_K]
        self.means_           = None   # (K, D)
        self.covariances_     = None   # (K, D, D)
        self.probabilities_   = None   # (N, K+1)  training responsibilities
        self.probability_cut_ = None   # scalar threshold on aggregate Gaussian posterior
        self.scaling_params_  = None   # from normalize_features

    # ------------------------------------------------------------------
    # Private — initialisation
    # ------------------------------------------------------------------

    def _initialize_params(self, int n_dimensions):
        """Randomly initialise mixture parameters for K components."""
        cdef:
            int    K         = self.n_components
            double pi_k_init = 0.5 / K   # K Gaussians share 50 %; uniform gets 50 %

        components = [
            {
                'pi_k': pi_k_init,
                'mu':   np.random.uniform(0.1, 0.9, n_dimensions),
                'cov':  np.eye(n_dimensions) * np.random.uniform(0.1, 0.9, n_dimensions),
            }
            for _ in range(K)
        ]
        return {'pi_u': 0.5, 'components': components}

    # ------------------------------------------------------------------
    # Private — E-step
    # ------------------------------------------------------------------

    def _expectation_step(self, X, params):
        """
        E-step: compute the (N, K+1) responsibility matrix.

        Column layout:
            0     → γ_U   (uniform background)
            1..K  → γ_k   (k-th Gaussian component)

        Returns False on numerical failure (e.g. singular covariance).

        Note: multivariate_normal.pdf is SciPy/LAPACK — cannot be called
        inside an OpenMP nogil region; this step remains sequential.
        """
        cdef int k

        try:
            n = X.shape[0]
            K = self.n_components

            # Uniform density: π_U / vol(bounding_box)
            feature_range   = X.max(axis=0) - X.min(axis=0)
            uniform_density = params['pi_u'] / np.prod(feature_range)

            # raw[:, 0]   = π_U · U(x)               (constant across samples)
            # raw[:, k+1] = π_k · N(x; μ_k, Σ_k)
            raw       = np.empty((n, K + 1), dtype=np.float64)
            raw[:, 0] = uniform_density

            for k, comp in enumerate(params['components']):
                raw[:, k + 1] = comp['pi_k'] * multivariate_normal(
                    mean=comp['mu'],
                    cov=comp['cov'],
                ).pdf(X)

            row_sums       = raw.sum(axis=1)                    # (N,)  p(x_i)
            log_likelihood = float(np.sum(np.log(row_sums)))

            gamma = raw / row_sums[:, np.newaxis]               # (N, K+1)

            params['gamma']      = gamma
            params['likelihood'] = log_likelihood
            return True

        except np.linalg.LinAlgError:
            return False

    # ------------------------------------------------------------------
    # Private — M-step
    # ------------------------------------------------------------------

    def _maximization_step(self, X, params):
        """
        M-step: update π_U, and per Gaussian component (π_k, μ_k, Σ_k).

        Weighted covariance computation uses OpenMP via the module-level
        C function _weighted_cov (parallelised over feature rows).
        K-component updates are independent; the per-component loop calls
        _weighted_cov which is internally parallel, so no outer parallelism
        over K is added (K is typically small).
        """
        cdef:
            int    k
            double N_k, N = float(X.shape[0])

        gamma = params['gamma']   # (N, K+1)
        params['pi_u'] = float(gamma[:, 0].sum()) / N

        X_c = np.ascontiguousarray(X, dtype=np.float64)

        for k, comp in enumerate(params['components']):
            gamma_k_arr = np.ascontiguousarray(gamma[:, k + 1], dtype=np.float64)
            N_k         = float(gamma_k_arr.sum())

            if N_k < 1e-10:
                # Collapsed component — preserve current parameters
                continue

            mu_k = (gamma_k_arr[:, np.newaxis] * X_c).sum(axis=0) / N_k

            # _weighted_cov is a module-level cdef function using prange
            cov_k = _weighted_cov(
                X_c,
                gamma_k_arr,
                np.ascontiguousarray(mu_k, dtype=np.float64),
                N_k,
                self.n_threads,
            )

            comp['pi_k'] = N_k / N
            comp['mu']   = mu_k
            comp['cov']  = cov_k

    # ------------------------------------------------------------------
    # Private — store sklearn-style public attributes
    # ------------------------------------------------------------------

    def _store_fitted_attributes(self, params):
        """Copy fitted params into sklearn-style attributes."""
        comps = params['components']

        self.weights_ = np.array(
            [params['pi_u']] + [c['pi_k'] for c in comps],
            dtype=np.float64,
        )   # (K+1,)  index 0 = uniform

        self.means_ = np.array(
            [c['mu'] for c in comps], dtype=np.float64
        )   # (K, D)

        self.covariances_ = np.array(
            [c['cov'] for c in comps], dtype=np.float64
        )   # (K, D, D)

    # ------------------------------------------------------------------
    # Private — threshold selection
    # ------------------------------------------------------------------

    def _find_probability_cut(self, total_gauss_prob, X_norm):
        """
        Find the threshold on the aggregate Gaussian posterior
        ( = Σ_k γ_k  =  1 - γ_U ).

        Step 1 — rotation-elbow method on the posterior CDF.
        Step 2 — Ripley's K spatial test (only when n_features >= 2);
                 if the test is significant (p < 0.05), the final
                 threshold is the 50/50 average of the spatial and
                 elbow thresholds.

        n_threads is forwarded to robust_adaptive_ripley_k so its
        Monte Carlo loop also runs in parallel.
        """
        probs = total_gauss_prob   # (N,) in [0, 1]

        # ---- Step 1: rotation-elbow ----
        percentiles = np.arange(0.01, 0.99, 0.01)
        perc_probs  = np.column_stack([
            percentiles,
            np.percentile(probs, percentiles * 100.0),
        ])
        rot_cut = float(rotate_and_find_elbow(perc_probs))

        # ---- Step 2: Ripley's K spatial refinement ----
        if X_norm.shape[1] >= 2:
            ripley_cut, diagnostics = robust_adaptive_ripley_k(
                X_norm[:, :2],
                probs,
                max_nsim=self.max_nsim,
                edge_correction='isotropic',
                confidence_level=0.99,
                n_threads=self.n_threads,
            )

            if ripley_cut is not None and len(diagnostics['thresholds']) > 0:
                # argmin avoids float-equality comparison
                thresholds_arr = np.array(diagnostics['thresholds'])
                idx            = int(np.argmin(np.abs(thresholds_arr - ripley_cut)))
                if diagnostics['p_values'][idx] < 0.05:
                    return 0.5 * float(ripley_cut) + 0.5 * rot_cut

        return rot_cut

    # ------------------------------------------------------------------
    # Private — apply stored normalisation to new data
    # ------------------------------------------------------------------

    def _transform(self, X):
        """Apply stored scaling_params_ to new data (no re-fitting)."""
        X_arr      = np.ascontiguousarray(X, dtype=np.float64)
        n_features = X_arr.shape[1]
        X_norm     = np.empty_like(X_arr)

        for i in range(n_features):
            sp           = self.scaling_params_[i]
            X_norm[:, i] = np.clip(
                (X_arr[:, i] - sp['min']) / sp['range'], 0.0, 1.0
            )
        return X_norm

    # ------------------------------------------------------------------
    # Private — responsibilities for arbitrary (normalised) data
    # ------------------------------------------------------------------

    def _responsibilities(self, X_norm):
        """
        Compute (N, K+1) responsibility matrix from stored fitted params.

        Returns
        -------
        gamma    : ndarray (N, K+1)
        row_sums : ndarray (N,)   — p(x) under the full mixture
        """
        cdef int k
        n = X_norm.shape[0]
        K = self.n_components

        feature_range   = X_norm.max(axis=0) - X_norm.min(axis=0)
        uniform_density = self.weights_[0] / np.prod(feature_range)

        raw       = np.empty((n, K + 1), dtype=np.float64)
        raw[:, 0] = uniform_density

        for k in range(K):
            raw[:, k + 1] = self.weights_[k + 1] * multivariate_normal(
                mean=self.means_[k],
                cov=self.covariances_[k],
            ).pdf(X_norm)

        row_sums = raw.sum(axis=1)
        gamma    = raw / row_sums[:, np.newaxis]
        return gamma, row_sums

    # ------------------------------------------------------------------
    # Private — shared label assignment
    # ------------------------------------------------------------------

    def _assign_labels(self, gamma):
        """
        Convert (N, K+1) responsibility matrix to integer cluster labels.

        label = 0 : background (aggregate Gaussian posterior ≤ threshold)
        label = k : member of Gaussian component k  (1-based, k = 1..K)
        """
        total_gauss = 1.0 - gamma[:, 0]          # (N,)
        is_member   = total_gauss > self.probability_cut_

        labels = np.zeros(gamma.shape[0], dtype=int)

        if self.n_components == 1:
            labels[is_member] = 1
        else:
            # Assign each member to the Gaussian with highest responsibility
            gauss_gamma = gamma[:, 1:]                     # (N, K)
            best_k      = np.argmax(gauss_gamma, axis=1)   # (N,) 0-based
            labels[is_member] = best_k[is_member] + 1      # shift to 1-based

        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit GUMM to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        cdef:
            double likelihood, likelihood_old = -1e308
            int    epoch, n_stable = 0, stable_limit

        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        X_norm, self.scaling_params_ = normalize_features(X_arr, self.padding)
        n_dimensions = X_norm.shape[1]

        params       = self._initialize_params(n_dimensions)
        stable_limit = int(self.stable_percentage * self.n_epochs)

        with tqdm(total=self.n_epochs, desc="Training GUMM") as pbar:
            for epoch in range(self.n_epochs):
                if not self._expectation_step(X_norm, params):
                    params = self._initialize_params(n_dimensions)
                    continue

                self._maximization_step(X_norm, params)
                likelihood = params['likelihood']

                if likelihood > likelihood_old:
                    likelihood_old = likelihood
                    n_stable       = 0
                else:
                    n_stable += 1

                pbar.update(1)
                pbar.set_postfix({'ll': f'{likelihood:.4f}', 'stable': n_stable})

                if n_stable >= stable_limit:
                    print(f"\nConverged at epoch {epoch + 1}")
                    break

        self._store_fitted_attributes(params)

        self.probabilities_ = params['gamma']            # (N, K+1)
        total_gauss         = 1.0 - self.probabilities_[:, 0]
        self.probability_cut_ = self._find_probability_cut(total_gauss, X_norm)

        return self

    def predict_proba(self, X):
        """
        Posterior responsibility matrix for *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_components + 1)
            Column 0 = uniform background posterior.
            Columns 1..K = per-Gaussian-component posteriors.
            Each row sums to 1.
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X_norm   = self._transform(X)
        gamma, _ = self._responsibilities(X_norm)
        return gamma

    def score_samples(self, X):
        """
        Per-sample log-likelihood  log p(x)  under the fitted mixture.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_p : ndarray of shape (n_samples,)
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before score_samples().")

        X_norm      = self._transform(X)
        _, row_sums = self._responsibilities(X_norm)
        return np.log(row_sums)

    def predict(self, X):
        """
        Cluster label for each sample in *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
            0 = background.
            k = Gaussian component k  (k = 1..n_components).
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")

        gamma = self.predict_proba(X)
        return self._assign_labels(gamma)

    def fit_predict(self, X):
        """
        Fit GUMM to *X* and return cluster labels.

        Reuses training responsibilities from fit() — no redundant E-step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
            0 = background.
            k = Gaussian component k  (k = 1..n_components).
        """
        self.fit(X)
        return self._assign_labels(self.probabilities_)
