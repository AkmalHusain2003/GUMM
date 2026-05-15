# GUMM

**Gaussian Uniform Mixture Model** — an unsupervised membership classifier that separates K structured cluster components from a uniform background via Expectation-Maximization, with an automatic threshold driven by spatial point-pattern analysis.

---

## Algorithm

The model fits a mixture of K Gaussians plus one uniform background:

$$p(\mathbf{x}) = \pi_U \cdot \mathcal{U}(\mathbf{x}) \;+\; \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x};\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$$

**E-step** — compute posterior responsibilities $\gamma_U^{(i)},\, \gamma_1^{(i)},\ldots,\gamma_K^{(i)}$ for each point.  
**M-step** — update $(\pi_U, \pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ via weighted MLE.

Once training converges, the membership threshold on the aggregate Gaussian posterior is set automatically by two methods applied in order:

| Step | Method | Description |
|---|---|---|
| 1 | **Rotation elbow** | Rotates the posterior CDF curve; threshold = minimum curvature point + 0.05 |
| 2 | **Ripley's K** | Monte Carlo supremum test on the 2-D spatial pattern (Ripley, 1976); if significant (p < 0.05), the final threshold is the 50/50 average with the elbow threshold |

---

## Installation

Requires a C compiler and Cython ≥ 3.0.

```bash
pip install .
```

Development install:

```bash
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from gumm import GUMM

rng = np.random.default_rng(0)

# Two clusters embedded in uniform noise
c1 = rng.multivariate_normal([2, 2], np.eye(2) * 0.3, 150)
c2 = rng.multivariate_normal([8, 8], np.eye(2) * 0.3, 120)
bg = rng.uniform(0, 10, (600, 2))
X  = np.vstack([c1, c2, bg])

model  = GUMM(n_components=2, n_epochs=500, random_state=42)
labels = model.fit_predict(X)
# labels: 0 = background, 1 = component 1, 2 = component 2

# Probabilities for new data
proba = model.predict_proba(X_new)   # shape (n, 3)  — [uniform, c1, c2]

# Per-sample log-likelihood
log_p = model.score_samples(X_new)  # shape (n,)
```

---

## API

### `GUMM(n_components, n_epochs, stable_percentage, padding, max_nsim, random_state)`

| Parameter | Default | Description |
|---|---|---|
| `n_components` | `1` | Number of Gaussian cluster components K |
| `n_epochs` | `1000` | Maximum EM iterations |
| `stable_percentage` | `0.1` | Stop after `n_epochs × stable_percentage` non-improving steps |
| `padding` | `0.1` | Fractional padding for feature normalisation bounds |
| `max_nsim` | `100` | Monte Carlo replications in the Ripley's K spatial test |
| `random_state` | `None` | Integer seed for reproducibility |

### Methods

| Method | Description |
|---|---|
| `fit(X)` | Fit the model; returns `self` |
| `predict(X)` | Integer cluster labels (0 = background, 1..K = clusters) |
| `predict_proba(X)` | Responsibility matrix of shape `(n_samples, K+1)` |
| `score_samples(X)` | Per-sample log-likelihood log p(x) |
| `fit_predict(X)` | `fit(X)` + `predict(X)` in one call, reusing training responsibilities |

### Attributes after `fit()`

| Attribute | Shape | Description |
|---|---|---|
| `weights_` | `(K+1,)` | Mixture weights — index 0 = uniform, 1..K = Gaussians |
| `means_` | `(K, D)` | Gaussian means in normalised feature space |
| `covariances_` | `(K, D, D)` | Gaussian covariance matrices |
| `probabilities_` | `(N, K+1)` | Training responsibilities |
| `probability_cut_` | scalar | Threshold on aggregate Gaussian posterior |
| `scaling_params_` | dict | Per-feature normalisation bounds |

---

## Utility functions

```python
from gumm import normalize_features, robust_adaptive_ripley_k, rotate_and_find_elbow
```

| Function | Description |
|---|---|
| `normalize_features(X, padding)` | Robust [0, 1] normalisation via percentile bounds |
| `rotate_and_find_elbow(data)` | Elbow detection via chord rotation |
| `robust_adaptive_ripley_k(points, probs, ..., max_nsim, ...)` | Adaptive Ripley's K with CSR envelope |

---

## License

MIT
