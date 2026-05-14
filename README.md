# GUMM

**Gaussian Uniform Mixture Model** — an unsupervised membership classifier that separates a structured cluster from a uniform background via Expectation-Maximization, with automatic threshold selection driven by spatial point-pattern analysis.

---

## Algorithm

The model fits a two-component mixture:

$$p(\mathbf{x}) = \pi_G \cdot \mathcal{N}(\mathbf{x};\,\boldsymbol{\mu},\boldsymbol{\Sigma}) \;+\; \pi_U \cdot \mathcal{U}(\mathbf{x})$$

**E-step** — compute posterior responsibilities $\gamma_G^{(i)}$, $\gamma_U^{(i)}$ for each point.  
**M-step** — update $(\pi_G,\pi_U,\boldsymbol{\mu},\boldsymbol{\Sigma})$ via weighted MLE.

Once training converges, the membership threshold is set automatically by combining two methods:

| Method | Description |
|---|---|
| **Rotation elbow** | Rotates the posterior CDF curve and finds the maximum curvature point |
| **Ripley's K** | Monte Carlo envelope test on the 2-D spatial pattern; significant clustering shifts the threshold |

---

## Installation

A C compiler and Cython ≥ 3.0 are required.

```bash
pip install .
```

For an editable / development install:

```bash
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from gumm import GUMM

rng = np.random.default_rng(0)

# Cluster embedded in uniform noise
cluster     = rng.multivariate_normal([5, 5, 5], np.eye(3), size=200)
background  = rng.uniform(0, 10, size=(800, 3))
X           = np.vstack([cluster, background])

model      = GUMM(n_epochs=500, random_state=42)
membership = model.fit_predict(X)          # ndarray of 0 / 1

print(f"Members found : {membership.sum()}")
print(f"Posterior cut : {model.probability_cut_:.4f}")
```

---

## API

### `GUMM(n_epochs, stable_percentage, padding, random_state)`

| Parameter | Default | Description |
|---|---|---|
| `n_epochs` | `1000` | Maximum EM iterations |
| 'n_sims' | '100' | Maximum Monte Carlo Simulation of Ripley's K |
| `stable_percentage` | `0.1` | Stop after `n_epochs × stable_percentage` non-improving steps |
| `padding` | `0.1` | Fractional padding added to feature normalization bounds |
| `random_state` | `None` | Integer seed for reproducibility |

#### `fit_predict(X, probability_cut='auto') → ndarray`

Fit the model and return a binary membership array (`1` = cluster, `0` = background).  
Pass a float in `[0, 1]` as `probability_cut` to override the automatic threshold with a fixed quantile.

#### Attributes after fitting

| Attribute | Description |
|---|---|
| `probabilities_` | Posterior Gaussian responsibility $\gamma_G^{(i)}$ for each point |
| `probability_cut_` | Threshold used to binarise membership |
| `scaling_params_` | Min/max bounds used for feature normalisation |

---

## Utility functions

```python
from gumm import normalize_features, robust_adaptive_ripley_k, rotate_and_find_elbow
```

| Function | Description |
|---|---|
| `normalize_features(X, padding)` | Robust percentile-based [0, 1] normalisation |
| `rotate_and_find_elbow(data)` | Elbow detection via chord rotation |
| `robust_adaptive_ripley_k(points, probs, ...)` | Adaptive Ripley's K with CSR envelope |

---

## License

MIT
