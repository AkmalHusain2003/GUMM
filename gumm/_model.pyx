# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from scipy.stats import multivariate_normal
from tqdm import tqdm

from ._normalize import normalize_features
from ._spatial   import rotate_and_find_elbow, robust_adaptive_ripley_k

cnp.import_array()


# ---------------------------------------------------------------------------
# Module-level C function — typed memoryviews live safely here
# ---------------------------------------------------------------------------

cdef cnp.ndarray _weighted_cov(
    double[:, :] X,
    double[:]    gamma_g,
    double[:]    mu,
    double       N_k,
):
    """
    Weighted sample covariance via a pure C triple-loop.

    Cython eliminates all bounds checks and Python overhead for
    the innermost features × features accumulation.
    """
    cdef:
        int    n_samples  = X.shape[0]
        int    n_features = X.shape[1]
        int    j, k, l
        double gj

    cdef cnp.ndarray cov_out  = np.zeros((n_features, n_features), dtype=np.float64)
    cdef cnp.ndarray diff_buf = np.empty(n_features,               dtype=np.float64)

    cdef double[:, :] cov    = cov_out
    cdef double[:]    diff_j = diff_buf

    for j in range(n_samples):
        gj = gamma_g[j]
        for k in range(n_features):
            diff_j[k] = X[j, k] - mu[k]
        for k in range(n_features):
            for l in range(n_features):
                cov[k, l] += gj * diff_j[k] * diff_j[l]

    for k in range(n_features):
        for l in range(n_features):
            cov[k, l] /= N_k

    return cov_out


# ---------------------------------------------------------------------------
# GUMM
# ---------------------------------------------------------------------------

class GUMM:
    """
    Gaussian Uniform Mixture Model (GUMM).

    Fits a two-component mixture (multivariate Gaussian + uniform) via EM
    and assigns binary membership using an automatic probability threshold.

    Parameters
    ----------
    n_epochs : int
        Maximum EM iterations.
    stable_percentage : float
        Stop when the likelihood has not improved for
        ``stable_percentage * n_epochs`` consecutive steps.
    padding : float
        Fractional padding for feature normalisation bounds.
    random_state : int, optional
    """

    def __init__(
        self,
        n_epochs          = 1000,
        stable_percentage = 0.1,
        padding           = 0.1,
        random_state      = None,
    ):
        self.n_epochs          = int(n_epochs)
        self.stable_percentage = float(stable_percentage)
        self.padding           = float(padding)

        if random_state is not None:
            np.random.seed(int(random_state))

        self.cluster_params   = None
        self.probabilities_   = None
        self.probability_cut_ = None
        self.scaling_params_  = None

    # ------------------------------------------------------------------

    def _initialize_cluster(self, n_dimensions):
        n_dimensions = int(n_dimensions)
        return {
            'pi_u': 0.5,
            'pi_g': 0.5,
            'mu':   np.random.uniform(0.1, 0.9, n_dimensions),
            'cov':  np.eye(n_dimensions) * np.random.uniform(0.1, 0.9, n_dimensions),
        }

    def _expectation_step(self, X, cluster):
        """E-step: compute posterior responsibilities. Returns False on numerical failure."""
        try:
            gamma_g = cluster['pi_g'] * multivariate_normal(
                mean=cluster['mu'],
                cov=cluster['cov'],
            ).pdf(X)

            # Uniform density = 1 / volume of the feature-space bounding box
            feature_range = X.max(axis=0) - X.min(axis=0)
            gamma_u       = cluster['pi_u'] / np.prod(feature_range)
            gammas_sum    = gamma_g + gamma_u

            cluster['gamma_g']    = gamma_g / gammas_sum
            cluster['gamma_u']    = gamma_u / gammas_sum
            cluster['likelihood'] = float(np.sum(np.log(gammas_sum)))
            return True
        except np.linalg.LinAlgError:
            return False

    def _maximization_step(self, X, cluster):
        """M-step: update mixture parameters from responsibilities."""
        gamma_g_arr = np.ascontiguousarray(cluster['gamma_g'], dtype=np.float64)
        N_k         = float(gamma_g_arr.sum())
        N           = float(X.shape[0])

        mu_arr = (gamma_g_arr[:, np.newaxis] * X).sum(axis=0) / N_k

        # _weighted_cov is a module-level cdef function — runs at C speed
        cov_arr = _weighted_cov(
            np.ascontiguousarray(X,      dtype=np.float64),
            gamma_g_arr,
            np.ascontiguousarray(mu_arr, dtype=np.float64),
            N_k,
        )

        cluster.update({
            'pi_u': float(np.sum(cluster['gamma_u'])) / N,
            'pi_g': N_k / N,
            'mu':   mu_arr,
            'cov':  cov_arr,
        })

    def _find_probability_cut(self, probs, X, probability_cut):
        """
        Derive the membership probability threshold.

        A float is treated as a quantile in [0, 1].
        'auto' combines the rotation-elbow method with Ripley's K spatial
        test when X has >= 2 features.
        """
        if isinstance(probability_cut, (int, float)):
            return float(np.percentile(probs, float(probability_cut) * 100.0))

        # ---- rotation-elbow threshold ----
        percentiles = np.arange(0.01, 0.99, 0.01)
        perc_probs  = np.column_stack([
            percentiles,
            np.percentile(probs, percentiles * 100.0),
        ])
        rot_cut = float(rotate_and_find_elbow(perc_probs))

        # ---- spatial refinement (requires >= 2 features) ----
        if X.shape[1] >= 2:
            ripley_cut, diagnostics = robust_adaptive_ripley_k(
                X[:, :2],
                probs,
                edge_correction='isotropic',
                confidence_level=0.99,
            )

            if ripley_cut is not None and len(diagnostics['thresholds']) > 0:
                # Locate the matching threshold safely via argmin (avoids float equality)
                thresholds_arr = np.array(diagnostics['thresholds'])
                idx            = int(np.argmin(np.abs(thresholds_arr - ripley_cut)))
                if diagnostics['p_values'][idx] < 0.05:
                    return 0.5 * float(ripley_cut) + 0.5 * rot_cut

        return rot_cut

    # ------------------------------------------------------------------

    def fit_predict(self, X, probability_cut='auto'):
        """
        Fit GUMM to *X* and return binary cluster membership.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        probability_cut : 'auto' or float
            'auto' selects the threshold automatically.
            A float in [0, 1] is treated as a quantile of the posterior.

        Returns
        -------
        membership : ndarray of shape (n_samples,), dtype int
            1 = Gaussian cluster member, 0 = background.
        """
        cdef:
            double likelihood, likelihood_old = -1e308
            int    epoch, n_stable = 0, stable_limit

        X_arr = np.ascontiguousarray(X, dtype=np.float64)

        X_norm, self.scaling_params_ = normalize_features(X_arr, self.padding)
        n_dimensions                 = X_norm.shape[1]
        self.cluster_params          = self._initialize_cluster(n_dimensions)
        stable_limit                 = int(self.stable_percentage * self.n_epochs)

        with tqdm(total=self.n_epochs, desc="Training GUMM") as pbar:
            for epoch in range(self.n_epochs):
                if not self._expectation_step(X_norm, self.cluster_params):
                    self.cluster_params = self._initialize_cluster(n_dimensions)
                    continue

                self._maximization_step(X_norm, self.cluster_params)
                likelihood = self.cluster_params['likelihood']

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

        self.probabilities_   = self.cluster_params['gamma_g']
        self.probability_cut_ = self._find_probability_cut(
            self.probabilities_, X_arr, probability_cut
        )

        return (self.probabilities_ > self.probability_cut_).astype(int)
