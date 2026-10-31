# 0.1.2 local validation validation

Local Windows checks on 2026-10-30:

| Environment | Result |
| --- | --- |
| Python 3.13.5, NumPy 2.5.4, PyTorch 2.15.0+cpu | 115 passed, 5 CUDA tests skipped |
| Python 3.10.18, NumPy 1.24.0, no PyTorch | 93 passed, 27 optional tests skipped |

The full CPU suite passes with warnings treated as errors. Source coverage is
94.1% (354/376 statements); Ruff also passes. Tests include recovery against
known components, independent stationarity checks, dtype/backend parity,
readme examples, and the new scale regressions.

`tests/test_numerical_range.py` verifies nonzero rank-one recovery at scales
`1e-200` and `1e200` with PCP, ADMM and Outlier Pursuit on both CPU backends.
Previously, the very small matrix was mistaken for zero and the very large
matrix produced NaN stopping diagnostics. Stable PCP is checked against an
analytic rank-one shrinkage solution at scales `1e-12`, `1` and `1e12`.
An unrepresentable squared objective raises `FloatingPointError`.

Run `python scripts/check_artifacts.py --output artifact-validation.json` after
building. It compares shipped code and required source files against the
checkout, installs wheel and source separately without PyTorch, verifies
versions and dependencies, and runs all three examples outside the repository.
The JSON records hashes of the exact artifacts checked.

These are synthetic correctness checks. CUDA execution, GPU performance,
remote CI and real-world recovery have not been established for this candidate.
