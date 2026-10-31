# rpca-kit

**Separate shared structure from sparse disruptions in your matrix data.**

`rpca-kit` is a lightweight Robust PCA toolkit for Python. Start with NumPy on CPU,
choose among five decomposition methods, and inspect the fitted components and
convergence diagnostics. Optional PyTorch support gives you the same solver API
on CPU or CUDA.

[PyPI](https://pypi.org/project/rpca-kit/) ·
[Source](https://github.com/giovannyTorres/rpca) ·
[Algorithm guide](https://github.com/giovannyTorres/rpca/blob/main/docs/algorithms.md) ·
[Issues](https://github.com/giovannyTorres/rpca/issues)

## What you get

- **A small starting point:** Python 3.10+ and NumPy are enough for the core package.
- **Separate outputs you can inspect:** low-rank structure, sparse deviations, and
  the remaining residual, all with the same shape as your input.
- **A method for your corruption pattern:** isolated bad readings, whole unusual
  samples, or a model that allows dense noise.
- **Visible numerical behavior:** convergence status, iteration count, objective,
  and per-iteration history. No silent fallback when a requested GPU is unavailable.

```text
observed = low_rank + sparse + residual
```

This is an **alpha research package**, not an automatic anomaly-labeling system.
A useful decomposition depends on whether your data fit the selected model.

## Where it can fit

These are application scenarios, not claims of deployed customer results:

| Scenario | Arrange the matrix | What to investigate |
| --- | --- | --- |
| Sensor monitoring | Sensors × time samples | Shared operating patterns versus isolated faulty readings with `pcp`. |
| Batch or sample quality review | Measurements × samples | Whole unusual samples with `outlier_pursuit`; each candidate sample must be a column. |
| Fixed-camera video experiments | Flattened pixels × frames | A low-rank background and sparse moving regions, if the camera is stable and motion occupies a small portion of the scene. |

You supply the numeric matrix and domain interpretation. The package does not
load videos, select alert thresholds, or decide whether a deviation is a fault.

## Install

```bash
python -m pip install rpca-kit
```

For the optional PyTorch backend:

```bash
python -m pip install 'rpca-kit[torch]'
```

The `gpu` extra is an alias for the same PyTorch dependency. A CUDA-capable GPU,
compatible driver, and appropriate PyTorch build are still required; follow the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).
Installing an extra does not install a working CUDA driver.

## Quickstart: recover sensor patterns beneath spikes

This fully reproducible **synthetic** example models 40 sensors sampled at 120
times. Their clean readings share three smooth patterns. We then inject 100
large, isolated reading errors and ask PCP to separate the two components.

```python
import numpy as np
from rpca_kit import pcp

rng = np.random.default_rng(7)
time = np.linspace(0, 4 * np.pi, 120)
patterns = np.vstack([np.sin(time), np.cos(time), np.sin(0.5 * time)])
clean = rng.normal(size=(40, 3)) @ patterns

spikes = np.zeros_like(clean)
positions = rng.choice(clean.size, size=100, replace=False)
spikes.flat[positions] = rng.choice([-12.0, 12.0], size=100)
observed = clean + spikes

result = pcp(observed, tol=1e-7, max_iter=2000)
recovery_error = np.linalg.norm(result.low_rank - clean) / np.linalg.norm(clean)

print(f"converged: {result.converged}")
print(f"component shapes: {result.low_rank.shape}, {result.sparse.shape}")
print(f"relative recovery error: {recovery_error:.2e}")
assert result.converged
assert recovery_error < 1e-4
np.testing.assert_allclose(
    observed, result.low_rank + result.sparse + result.residual
)
```

The input and each output are `(40, 120)` arrays. `result.low_rank` estimates the
shared patterns; `result.sparse` contains the estimated spikes. A NumPy CPU run
produces a relative recovery error of approximately `7.3e-08` for this seeded
example; the final digits can vary across numerical libraries.

We can measure recovery here because we created `clean`. For real measurements,
that ground truth is usually unknown. A small reconstruction residual alone does
not establish that the components have the interpretation you want.

## Example: prioritize unusual samples for review

If an entire sample may be corrupted, use **column-sparse** decomposition instead
of treating each bad entry independently. Here, each column contains 32
measurements for one of 70 synthetic samples. Three columns receive large
perturbations.

```python
import numpy as np
from rpca_kit import outlier_pursuit

rng = np.random.default_rng(57)
basis, _ = np.linalg.qr(rng.normal(size=(32, 2)))
normal_samples = basis @ rng.normal(size=(2, 70))
observed = normal_samples.copy()
observed[:, [2, 31, 59]] += rng.normal(scale=8.0, size=(32, 3))

result = outlier_pursuit(
    observed, lambda_=0.5, tol=1e-7, max_iter=3000
)
scores = np.linalg.norm(result.sparse, axis=0)
review_first = np.sort(np.argsort(scores)[-3:])

print(f"converged: {result.converged}")
print(f"samples to inspect first: {review_first.tolist()}")
assert result.converged
np.testing.assert_array_equal(review_first, [2, 31, 59])
```

Expected review indices: `[2, 31, 59]`. The column norms are **review scores, not
probabilities**. We select three because this example contains three perturbed
samples; a real application needs its own threshold or review budget. Tune
`lambda_` to your data and validate the results. Original clean values at
arbitrarily corrupted columns are not generally identifiable.

Both examples run independently with the basic NumPy installation. Their source
files are also available as
[readme_quickstart.py](https://github.com/giovannyTorres/rpca/blob/main/examples/readme_quickstart.py)
and [readme_sensors.py](https://github.com/giovannyTorres/rpca/blob/main/examples/readme_sensors.py).

## Choose a solver

| Function | Model | Choose it when |
| --- | --- | --- |
| `pcp` | Principal Component Pursuit, solved by IALM | You expect low-rank structure plus sparse entrywise corruption. |
| `admm` | The same PCP objective, with fixed penalty | You want explicit control over the numerical penalty `mu`. |
| `stable_pcp` | Penalized stable PCP, solved by FISTA | You need a nonzero dense residual as well as low-rank and sparse terms. |
| `outlier_pursuit` | Low rank plus column-sparse corruption | Entire samples, arranged as columns, may be unusual. |
| `godec` | Deterministic rank/cardinality projections | You can specify the rank and maximum number of sparse entries. |

The function API returns an `RPCAResult`. For a stateful interface,
`RPCA(method="ialm").fit(data)` stores `low_rank_`, `sparse_`, `residual_`, and
`result_`. Each fit decomposes its input matrix; there is no learned
out-of-sample `transform` or forecasting step.

### Parameters worth understanding

- **`lambda_`:** controls the penalty on the sparse component. The default,
  `1 / sqrt(max(data.shape))`, is a starting point, not automatic tuning.
  A larger value penalizes sparse deviations more strongly.
- **`tau` in `stable_pcp`:** controls overall regularization. This is a penalized
  objective, not a user-specified hard bound on the noise norm.
- **`tol` and `max_iter`:** control numerical stopping. Inspect `converged`;
  exhausting the iteration budget returns the last result with a warning.
- **`rank` and `cardinality` in `godec`:** are required bounds supplied by you.
  `cardinality` is a count of entries, not a percentage.

`result.error` is the chosen solver's stopping criterion, **not a universal
accuracy score**. The [algorithm guide](https://github.com/giovannyTorres/rpca/blob/main/docs/algorithms.md)
explains objectives, stopping rules, and assumptions.

## CPU, PyTorch, and CUDA

NumPy input uses CPU by default. Select PyTorch CPU explicitly with
`backend="torch", device="cpu"`, or CUDA with `backend="torch", device="cuda"`
when calling a solver. With `backend="auto"`, a PyTorch tensor keeps its backend
and device. `result.to_numpy()` makes an explicit, independent CPU copy.

CI tests NumPy on Python 3.10–3.13 and PyTorch on CPU. **CUDA execution is
implemented but has not been validated on GPU hardware for this release.**
Skipped GPU tests are not GPU validation. Requesting unavailable CUDA raises an
error rather than silently switching to CPU. GPU transfers and synchronization
can outweigh any benefit for small matrices; no universal speedup is claimed.

## Know the limits

- Input must be a dense, finite, real, nonempty 2D matrix. Missing values,
  sparse-matrix inputs, and streaming data are not supported.
- No automatic centering or scaling is applied. Decide how different units and
  measurement scales should be handled before fitting.
- `float32` and `float64` are preserved; integers become `float64`. For `float32`,
  start with `tol=1e-5` rather than assuming the default `1e-7` is attainable.
- The solvers use exact SVD. Large dense matrices can be expensive in time and
  memory. GoDec here does not include the original randomized acceleration.
- Low-rank and sparse components need identifiability assumptions. GoDec is
  nonconvex; convergence does not certify global optimality or correct recovery.
- PyTorch operations do not construct autograd graphs. This is a decomposition
  toolkit, not a differentiable training layer.
- Unrepresentable norms or objectives raise `FloatingPointError`; rescale the
  data and regularization parameters before retrying.

## Develop and reproduce

From a checkout of the [repository](https://github.com/giovannyTorres/rpca):

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests examples scripts
python scripts/release.py
```

The release script tests, builds, and checks the distributions without uploading
unless explicitly requested. See the
[publishing guide](https://github.com/giovannyTorres/rpca/blob/main/docs/publishing.md)
and [CI runs](https://github.com/giovannyTorres/rpca/actions).

## References and license

These are established algorithms, implemented here in a compact shared API:

- Candès, Li, Ma, and Wright. *Robust Principal Component Analysis?* JACM, 2011.
  [Paper](https://doi.org/10.1145/1970392.1970395).
- Lin, Chen, and Ma. *The Augmented Lagrange Multiplier Method for Exact Recovery
  of Corrupted Low-Rank Matrices*, 2010. [Paper](https://arxiv.org/abs/1009.5055).
- Zhou, Li, Wright, Candès, and Ma. *Stable Principal Component Pursuit*, 2010.
  [Paper](https://arxiv.org/abs/1001.2363). This package uses the penalized
  formulation, not a directly constrained noise-ball formulation.
- Xu, Caramanis, and Sanghavi. *Robust PCA via Outlier Pursuit*, 2010.
  [Paper](https://arxiv.org/abs/1010.4237).
- Zhou and Tao. *GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy
  Case*, ICML, 2011. This implementation uses deterministic alternating
  projections with exact SVD, not the randomized acceleration.

Created by Giovanny Torres. Released under the
[MIT license](https://github.com/giovannyTorres/rpca/blob/main/LICENSE).
