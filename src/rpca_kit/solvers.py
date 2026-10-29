"""Robust matrix decompositions with NumPy and optional PyTorch backends.

All methods accept dense, finite, real matrices. No solver modifies its input.
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral, Real
from typing import Any

from ._backend import get_backend
from .result import ConvergenceWarning, RPCAResult


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return value


def _integer(name: str, value: Any, lower: int, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be between {lower} and {upper or 'infinity'}")
    return value


def _setup(data, backend, device, max_iter, tol):
    max_iter = _integer("max_iter", max_iter, 1)
    tol = _positive("tol", tol)
    ops, x = get_backend(data, backend=backend, device=device)
    return ops, x, max_iter, tol


def _finish(ops, x, low, sparse, converged, n_iter, error, objective, history, method):
    if not converged:
        warnings.warn(
            f"{method} did not converge in {n_iter} iterations (error={error:.3g}). "
            "Inspect result.history or increase max_iter.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return RPCAResult(
        low_rank=low, sparse=sparse, residual=x - low - sparse,
        converged=converged, n_iter=n_iter, error=error, objective=objective,
        history=history, method=method, backend=ops.name, device=ops.device,
    )


def _zero_result(ops, x, method):
    return _finish(ops, x, ops.zeros_like(x), ops.zeros_like(x), True, 0, 0.0, 0.0, [], method)


def _equality(
    data, *, method, lambda_, mu, rho, max_iter, tol, backend, device, column_sparse=False,
):
    ops, x, max_iter, tol = _setup(data, backend, device, max_iter, tol)
    lam = 1 / math.sqrt(max(x.shape)) if lambda_ is None else _positive("lambda_", lambda_)
    if mu is not None:
        mu = _positive("mu", mu)
    rho = _positive("rho", rho)
    if rho < 1:
        raise ValueError("rho must be >= 1")
    scale = ops.norm(x)
    if scale == 0:
        return _zero_result(ops, x, method)
    spectral = ops.spectral_norm(x)
    penalty = 1.25 / spectral if mu is None else mu
    penalty_cap = penalty * 1e7
    low, sparse = ops.zeros_like(x), ops.zeros_like(x)
    dual = ops.zeros_like(x)
    if method == "ialm":
        dual = x / max(spectral, ops.max_abs(x) / lam)
    history = []
    shrink = ops.group_shrink if column_sparse else ops.shrink
    sparse_norm = ops.group_norm if column_sparse else ops.sum_abs
    converged = False
    for iteration in range(1, max_iter + 1):
        old_sparse = sparse
        low, rank, nuclear = ops.svt(x - sparse + dual / penalty, 1 / penalty)
        sparse = shrink(x - low + dual / penalty, lam / penalty)
        residual = x - low - sparse
        dual = dual + penalty * residual
        primal_error = ops.norm(residual) / scale
        # The primal residual alone can vanish before the optimum is reached.
        dual_error = penalty * ops.norm(sparse - old_sparse) / max(1.0, ops.norm(dual))
        error = max(primal_error, dual_error)
        objective = nuclear + lam * sparse_norm(sparse)
        history.append({
            "iteration": iteration, "error": error, "residual_error": primal_error,
            "dual_error": dual_error, "objective": objective, "rank": rank,
            "penalty": penalty,
        })
        if error <= tol:
            converged = True
            break
        # Blind geometric growth can make the iterate appear feasible while
        # freezing progress on the objective. Pause continuation while dual
        # change dominates; dual is unscaled and needs no rescaling here.
        if primal_error > dual_error:
            penalty = min(penalty * rho, penalty_cap)
    return _finish(ops, x, low, sparse, converged, iteration, error, objective, history, method)


def pcp(
    data, *, lambda_=None, mu=None, rho=1.5, max_iter=1000, tol=1e-7,
    backend="auto", device=None,
) -> RPCAResult:
    """Principal Component Pursuit using inexact augmented Lagrange multipliers.

    Minimize ``||L||_* + lambda_ * ||S||_1`` subject to ``data = L + S``.
    ``lambda_`` defaults to ``1/sqrt(max(data.shape))``. ``mu`` is the initial
    penalty, defaulting to ``1.25 / ||data||_2``; ``rho >= 1`` grows it to a cap
    when primal error exceeds dual error, pausing growth otherwise.
    ``error`` is the maximum of relative feasibility and normalized dual change.
    """
    return _equality(
        data, method="ialm", lambda_=lambda_, mu=mu, rho=rho,
        max_iter=max_iter, tol=tol, backend=backend, device=device,
    )


def admm(
    data, *, lambda_=None, mu=None, max_iter=1000, tol=1e-7,
    backend="auto", device=None,
) -> RPCAResult:
    """Solve PCP by two-block ADMM with a fixed positive penalty ``mu``.

    The objective and defaults match :func:`pcp`, but the penalty stays fixed.
    Convergence requires both primal feasibility and normalized dual change.
    """
    return _equality(
        data, method="admm", lambda_=lambda_, mu=mu, rho=1.0,
        max_iter=max_iter, tol=tol, backend=backend, device=device,
    )


def outlier_pursuit(
    data, *, lambda_=None, mu=None, max_iter=1000, tol=1e-7,
    backend="auto", device=None,
) -> RPCAResult:
    """Separate a low-rank matrix and entire corrupted columns using ADMM.

    Minimize ``||L||_* + lambda_ * sum_j ||S[:,j]||_2`` subject to ``data=L+S``.
    The default lambda is ``1/sqrt(max(data.shape))``; tune it for the expected
    fraction of outlier columns. This identifies the clean subspace; individual
    low-rank values at arbitrary outlier columns are not identifiable in general.
    """
    return _equality(
        data, method="outlier_pursuit", lambda_=lambda_, mu=mu, rho=1.0,
        max_iter=max_iter, tol=tol, backend=backend, device=device, column_sparse=True,
    )


def stable_pcp(
    data, *, lambda_=None, tau=1.0, max_iter=1000, tol=1e-7,
    backend="auto", device=None,
) -> RPCAResult:
    """Penalized stable PCP using monotonically restarted FISTA.

    Minimize ``0.5*||data-L-S||_F**2 + tau*(||L||_* + lambda_*||S||_1)``.
    ``tau`` controls shrinkage/noise; this is the *penalized*, not the constrained
    noise-ball, formulation. The step is 1/2, the inverse joint gradient Lipschitz
    constant. ``error`` is a normalized proximal-gradient mapping, not data fit.
    """
    ops, x, max_iter, tol = _setup(data, backend, device, max_iter, tol)
    lam = 1 / math.sqrt(max(x.shape)) if lambda_ is None else _positive("lambda_", lambda_)
    tau = _positive("tau", tau)
    scale = ops.norm(x)
    if scale == 0:
        return _zero_result(ops, x, "stable")
    low, sparse = ops.zeros_like(x), ops.zeros_like(x)
    extrap_low, extrap_sparse = low, sparse
    momentum = 1.0
    previous_objective = math.inf
    history = []
    converged = False
    for iteration in range(1, max_iter + 1):
        gradient = extrap_low + extrap_sparse - x
        new_low, rank, nuclear = ops.svt(extrap_low - 0.5 * gradient, 0.5 * tau)
        new_sparse = ops.shrink(extrap_sparse - 0.5 * gradient, 0.5 * tau * lam)
        residual_norm = ops.norm(x - new_low - new_sparse)
        objective = 0.5 * residual_norm**2 + tau * (nuclear + lam * ops.sum_abs(new_sparse))
        if objective > previous_objective:
            # A monotone restart prevents oscillatory extrapolation from being accepted.
            extrap_low, extrap_sparse = low, sparse
            momentum = 1.0
            gradient = low + sparse - x
            new_low, rank, nuclear = ops.svt(low - 0.5 * gradient, 0.5 * tau)
            new_sparse = ops.shrink(sparse - 0.5 * gradient, 0.5 * tau * lam)
            residual_norm = ops.norm(x - new_low - new_sparse)
            objective = 0.5 * residual_norm**2 + tau * (nuclear + lam * ops.sum_abs(new_sparse))
        error = 2 * math.hypot(
            ops.norm(new_low - extrap_low), ops.norm(new_sparse - extrap_sparse),
        ) / max(1.0, scale)
        history.append({
            "iteration": iteration, "error": error, "residual_error": residual_norm / scale,
            "objective": objective, "rank": rank,
        })
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
        fraction = (momentum - 1) / next_momentum
        extrap_low = new_low + fraction * (new_low - low)
        extrap_sparse = new_sparse + fraction * (new_sparse - sparse)
        low, sparse = new_low, new_sparse
        momentum, previous_objective = next_momentum, objective
        if error <= tol:
            converged = True
            break
    return _finish(ops, x, low, sparse, converged, iteration, error, objective, history, "stable")


def godec(
    data, *, rank, cardinality, max_iter=1000, tol=1e-7, backend="auto", device=None,
) -> RPCAResult:
    """Deterministic GoDec-style alternating rank/cardinality projections.

    Minimize ``0.5*||data-L-S||_F**2`` subject to rank(L)<=rank and
    nnz(S)<=cardinality. ``cardinality`` is an integer count, not a fraction.
    Uses exact truncated SVD, not randomized bilateral projections. This is a
    nonconvex local solver. ``error`` measures relative iterate change; convergence
    does not certify global optimality or a zero reconstruction residual.
    """
    ops, x, max_iter, tol = _setup(data, backend, device, max_iter, tol)
    rank = _integer("rank", rank, 0, min(x.shape))
    cardinality = _integer("cardinality", cardinality, 0, math.prod(x.shape))
    scale = ops.norm(x)
    if scale == 0:
        return _zero_result(ops, x, "godec")
    low, sparse = ops.zeros_like(x), ops.zeros_like(x)
    converged = False
    history = []
    for iteration in range(1, max_iter + 1):
        new_low = ops.truncated_svd(x - sparse, rank)
        new_sparse = ops.hard_threshold(x - new_low, cardinality)
        error = math.hypot(ops.norm(new_low - low), ops.norm(new_sparse - sparse)) / scale
        low, sparse = new_low, new_sparse
        residual_norm = ops.norm(x - low - sparse)
        objective = 0.5 * residual_norm**2
        history.append({
            "iteration": iteration, "error": error, "residual_error": residual_norm / scale,
            "objective": objective,
        })
        if error <= tol:
            converged = True
            break
    return _finish(ops, x, low, sparse, converged, iteration, error, objective, history, "godec")


_METHODS = {
    "ialm": pcp, "admm": admm, "stable": stable_pcp,
    "outlier_pursuit": outlier_pursuit, "godec": godec,
}


def decompose(data, *, method="ialm", **kwargs) -> RPCAResult:
    """Dispatch to one of ialm, admm, stable, outlier_pursuit, or godec."""
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"method must be one of {tuple(_METHODS)}")
    return _METHODS[method](data, **kwargs)
