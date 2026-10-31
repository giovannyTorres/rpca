# Models, parameters, and convergence

For an input `D` with shape `m × n`, every solver returns `L`, `S`, and
`E = D - L - S`. Norms and stopping criteria operate on the supplied matrix.
The package does not automatically center or standardize your data. Norm
calculations scale intermediate values before squaring to reduce numerical
underflow and overflow; returned components remain in the original units.

## PCP: IALM and ADMM

The convex Principal Component Pursuit problem is:

```text
min ||L||_* + lambda ||S||_1    subject to D = L + S
```

`||L||_*` is the sum of singular values; `||S||_1` sums absolute values over all
entries. `pcp(D, lambda_=None, ...)` uses IALM with an adaptive numerical penalty.
`admm(D, lambda_=None, mu=None, ...)` keeps the penalty `mu` fixed. Both alternate
singular-value thresholding, entrywise soft thresholding, and a dual update.

The default `lambda_=None` means `1/sqrt(max(m,n))`. A larger `lambda_` penalizes
sparse entries more strongly. `mu` controls the numerical optimization procedure,
not the relative weight of the sparse component in the model. `tol` must be
positive and `max_iter` a positive integer. Check `converged` after changing the
iteration budget.

The default initial penalty is `1.25 / ||D||_2`. IALM multiplies it by `rho=1.5`
only when the primal error exceeds the dual error, up to a cap of
`initial_mu * 1e7`; `rho` must be at least 1. Pausing growth when dual change
dominates allows further progress on the objective. The history records
`penalty`. The final stopping criterion is the maximum of:

```text
primal = ||D-L-S||_F / ||D||_F
dual   = mu * ||S_current-S_previous||_F / max(1, ||Y_current||_F)
```

`Y` is the updated Lagrange multiplier. Both errors must be at most `tol`.
The history retains each error separately.

## Penalized stable PCP

`stable_pcp(D, tau=1.0, lambda_=None, ...)` minimizes:

```text
0.5 ||D - L - S||_F² + tau ||L||_* + tau lambda ||S||_1
```

The solver uses accelerated proximal gradient (FISTA). The gradient of the
quadratic term with respect to the pair `(L,S)` has Lipschitz constant 2.
A larger `tau` increases regularization; a very small `tau` may require more
iterations. The model allows a nonzero dense residual. Its tolerance applies
to a stationarity measure, not a requirement for zero reconstruction residual.

The step size is `1/2`. Extrapolation restarts if it increases the objective.
For the extrapolated pair `Z` and next proximal pair `P(Z)`, the stopping error is
`2*||P(Z)-Z||_F / ||D||_F` for nonzero data. The joint norm combines the squared Frobenius
norms of both components. This is a proximal-gradient mapping measure.
Rescaling `D` and `tau` by the same positive factor preserves this stopping
criterion. Tiny input values do not bypass the requested relative tolerance.

For an arbitrary fixed `tau`, this penalized objective is not equivalent to
specifying a hard noise bound `||E||_F <= delta`. The package does not estimate
`tau` or the noise level automatically.

## Outlier Pursuit

`outlier_pursuit(D, lambda_=None, mu=None, ...)` solves:

```text
min ||L||_* + lambda sum_j ||S[:,j]||_2    subject to D = L + S
```

The proximal operation on `S` shrinks entire column norms. The fitted corruption
is column-sparse, not necessarily entrywise sparse. Arrange each potentially
unusual sample as a **column**. The theoretical choice of `lambda` depends on
assumptions about the fraction of outlier columns. The default shared with PCP
is only an initial tuning value.

The primal/dual stopping criterion matches ADMM above. At arbitrarily corrupted
columns, the original clean entries of `L` are generally not identifiable.
The main objectives are recovering the clean subspace and locating unusual
columns, not reconstructing the unknown original values of every corrupted
sample.

## Deterministic GoDec-style projections

`godec(D, rank=r, cardinality=k, ...)` alternates an exact-SVD rank-`r`
approximation with retention of the `k` largest-magnitude residual entries.
It implements the alternating-projection core of:

```text
min 0.5 ||D - L - S||_F²
subject to rank(L) <= r, number of nonzero entries in S <= k
```

You supply `r` and `k` as complexity bounds; the package does not infer them.
This implementation does not include the randomized bilateral projections of
the original accelerated algorithm. The problem is nonconvex and may stagnate
or converge to a local solution.

The stopping error is the joint norm of the changes in `(L,S)` between two
iterations, divided by `||D||_F`; it does not require a zero residual.
`rank` must be an integer from 0 to `min(m,n)`, and `cardinality` an integer
from 0 to `m*n`. Cardinality is a count, not a fraction.

## Diagnostics and precision

`objective` is the selected method's objective value. Values from different
methods are not directly comparable without accounting for their definitions.
`history` stores per-iteration diagnostics and `error` the final stopping
criterion. Solver docstrings document the implemented formulas.

Zero input returns zero matrices, `n_iter=0`, `converged=True`, and an empty
history. Reaching `max_iter` without satisfying the stopping criterion returns
the last iterate and emits `ConvergenceWarning`.

Diagnostics must be finite. If an input norm or objective exceeds the available
numerical range, the solver raises `FloatingPointError` and asks for rescaling.
For example, a squared residual of order `1e400` cannot be represented in
float64 even though each input entry may be finite. This is separate from
ordinary nonconvergence and is not reported as a usable decomposition.

`float32` operations may stagnate before reaching very small tolerances. Start
with `tol=1e-5` for that precision, including on PyTorch. The package never relaxes
your requested tolerance automatically. For recovery tests, use `float64`,
reproducible data, and relative errors against both the observations and the
known synthetic components. When timing CUDA, synchronize the device before
and after the measured work.

## Result and estimator contracts

Every solver returns `RPCAResult`. Its `low_rank`, `sparse`, and `residual` arrays
have the input shape and retain the computation backend. Unpacking a result
returns only `low_rank, sparse`; the residual remains available as an attribute.
`result.to_numpy()` returns an independent CPU copy. `result.rank` computes the
numerical rank of the fitted low-rank component with a dtype-aware threshold.

`RPCA(...).fit(data)` returns the estimator and stores fitted arrays and
`result_`. `fit_transform(data)` returns the full `RPCAResult`, not an array.
There is no out-of-sample `transform`: every fit solves a fresh decomposition.

See the [package README](https://github.com/giovannyTorres/rpca/blob/main/README.md)
for runnable examples and the CPU/CUDA validation limits.
