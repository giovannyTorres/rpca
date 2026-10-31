# Changes

## 0.1.2 (2026-10-30)

- Scale Frobenius and column norms to prevent false zero solutions and NaN
  diagnostics on very small or very large matrices.
- Normalize stable PCP's stopping measure by the data norm, so rescaling data
  and `tau` together preserves the requested relative accuracy.
- Raise `FloatingPointError` for unrepresentable diagnostics instead of
  returning invalid results; support lazy negative PyTorch views in NumPy conversion.
- Validate wheel and source installations independently with examples, exact
  version checks, dependency checks, and SHA-256 hashes.
- Require a matching release tag before the manual PyPI publishing workflow.

## 0.1.1

Documentation-focused release. Algorithms and public APIs are unchanged.

- Clearer English package overview, installation and workflow guidance.
- Two reproducible examples with explicit input/output shapes and synthetic-data caveats.
- Practical application ideas, solver selection and CPU/CUDA validation limits.
- Absolute documentation links for PyPI and regression tests for README examples.

## 0.1.0

- Principal Component Pursuit with IALM and fixed-penalty ADMM.
- Penalized stable PCP with accelerated proximal gradient updates.
- Column-sparse Outlier Pursuit.
- Deterministic GoDec alternating projections with exact truncated SVD.
- NumPy CPU and optional PyTorch CPU/CUDA backends.
- Shared result object, convergence diagnostics and estimator interface.
- Numerical recovery, input validation and optional backend parity tests.
