"""Independent recovery, optimality, and device-contract checks."""

import numpy as np
import pytest

from rpca_kit import (
    RPCA,
    ConvergenceWarning,
    admm,
    decompose,
    godec,
    outlier_pursuit,
    pcp,
    stable_pcp,
)


def recovery_problem(seed=941, rows=48, columns=60):
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((rows, 2)))
    right, _ = np.linalg.qr(rng.standard_normal((columns, 2)))
    low_rank = (left * np.array([45.0, 30.0])) @ right.T
    sparse = np.zeros_like(low_rank)
    support = rng.choice(low_rank.size, int(0.035 * low_rank.size), replace=False)
    sparse.flat[support] = rng.choice([-1.0, 1.0], len(support)) * 8.0
    return low_rank + sparse, low_rank, sparse


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / max(1.0, np.linalg.norm(expected))


@pytest.mark.parametrize("solver", [pcp, admm])
def test_pcp_recovers_independent_low_rank_and_sparse_components(solver):
    data, low_rank, sparse = recovery_problem()
    original = data.copy()
    result = solver(data, tol=1e-7, max_iter=2000)
    assert result.converged
    assert result.n_iter <= 2000
    assert relative_error(result.low_rank, low_rank) < 3e-4
    assert relative_error(result.sparse, sparse) < 3e-4
    assert np.linalg.norm(result.residual) / np.linalg.norm(data) < 2e-7
    np.testing.assert_array_equal(data, original)
    np.testing.assert_allclose(
        data, result.low_rank + result.sparse + result.residual, atol=1e-12
    )


def test_pcp_is_equivariant_under_positive_rescaling():
    data, _, _ = recovery_problem(rows=32, columns=40)
    baseline = pcp(data, tol=1e-8)
    scaled = pcp(7.0 * data, tol=1e-8)
    assert baseline.converged and scaled.converged
    np.testing.assert_allclose(scaled.low_rank / 7.0, baseline.low_rank, atol=3e-6)
    np.testing.assert_allclose(scaled.sparse / 7.0, baseline.sparse, atol=3e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_float32_recovers_components_at_precision_appropriate_tolerance(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    data, low_rank, sparse = recovery_problem(rows=24, columns=30)
    result = pcp(data.astype(np.float32), backend=backend, tol=1e-5)
    assert result.converged
    numpy_result = result.to_numpy()
    assert numpy_result.low_rank.dtype == np.float32
    assert relative_error(numpy_result.low_rank, low_rank) < 2e-3
    assert relative_error(numpy_result.sparse, sparse) < 2e-3


def test_stable_pcp_satisfies_independent_proximal_kkt_conditions():
    rng = np.random.default_rng(16)
    low_rank = rng.normal(size=(20, 2)) @ rng.normal(size=(2, 26))
    data = low_rank + rng.normal(scale=0.07, size=low_rank.shape)
    data[1, 5] += 12.0
    data[15, 12] -= 9.0
    tau, lambda_ = 0.45, 0.25
    result = stable_pcp(data, tau=tau, lambda_=lambda_, tol=1e-7, max_iter=5000)
    assert result.converged
    gradient = result.low_rank + result.sparse - data
    u, values, vt = np.linalg.svd(result.low_rank - gradient, full_matrices=False)
    proximal_low = (u * np.maximum(values - tau, 0.0)) @ vt
    argument = result.sparse - gradient
    proximal_sparse = np.sign(argument) * np.maximum(np.abs(argument) - tau * lambda_, 0.0)
    assert relative_error(proximal_low, result.low_rank) < 2e-4
    assert relative_error(proximal_sparse, result.sparse) < 2e-4
    nuclear_norm = np.linalg.svd(result.low_rank, compute_uv=False).sum()
    objective = (
        0.5 * np.linalg.norm(gradient) ** 2
        + tau * nuclear_norm
        + tau * lambda_ * np.abs(result.sparse).sum()
    )
    np.testing.assert_allclose(result.objective, objective, rtol=1e-6)
    assert objective < 0.5 * np.linalg.norm(data) ** 2
    assert np.linalg.norm(result.residual) > 0


def test_outlier_pursuit_finds_column_outliers_and_clean_subspace():
    rng = np.random.default_rng(57)
    basis, _ = np.linalg.qr(rng.normal(size=(32, 2)))
    low_rank = basis @ rng.normal(size=(2, 70))
    outlier_columns = np.array([2, 31, 59])
    sparse = np.zeros_like(low_rank)
    sparse[:, outlier_columns] = rng.normal(scale=8.0, size=(32, 3))
    data = low_rank + sparse
    result = outlier_pursuit(data, lambda_=0.5, tol=1e-7, max_iter=3000)
    assert result.converged
    assert np.linalg.norm(result.residual) / np.linalg.norm(data) < 2e-7
    detected = np.where(np.linalg.norm(result.sparse, axis=0) > 1e-3)[0]
    np.testing.assert_array_equal(detected, outlier_columns)
    clean = np.ones(data.shape[1], dtype=bool)
    clean[outlier_columns] = False
    assert relative_error(result.low_rank[:, clean], low_rank[:, clean]) < 1e-3
    # At corrupted columns the low-rank coordinates need not be identifiable.
    orthogonal_part = result.low_rank - basis @ (basis.T @ result.low_rank)
    assert np.linalg.norm(orthogonal_part) / np.linalg.norm(result.low_rank) < 1e-3


def test_godec_obeys_constraints_and_improves_over_plain_truncated_svd():
    data, _, sparse = recovery_problem(rows=40, columns=50)
    cardinality = int(np.count_nonzero(sparse))
    result = godec(data, rank=2, cardinality=cardinality, tol=1e-7, max_iter=1500)
    assert np.linalg.matrix_rank(result.low_rank, tol=1e-7) <= 2
    assert np.count_nonzero(result.sparse) <= cardinality
    u, values, vt = np.linalg.svd(data, full_matrices=False)
    plain_low_rank = (u[:, :2] * values[:2]) @ vt[:2]
    assert np.linalg.norm(result.residual) < 0.1 * np.linalg.norm(data - plain_low_rank)
    np.testing.assert_allclose(data, result.low_rank + result.sparse + result.residual)


@pytest.mark.parametrize("solver", [pcp, admm, stable_pcp, outlier_pursuit])
def test_zero_matrix_returns_finite_converged_decomposition(solver):
    result = solver(np.zeros((7, 9)))
    assert result.converged
    assert np.isfinite(result.objective)
    assert np.isfinite(result.error)
    np.testing.assert_array_equal(result.low_rank, np.zeros((7, 9)))
    np.testing.assert_array_equal(result.sparse, np.zeros((7, 9)))
    np.testing.assert_array_equal(result.residual, np.zeros((7, 9)))


def test_iteration_limit_reports_nonconvergence():
    data, _, _ = recovery_problem()
    with pytest.warns(ConvergenceWarning):
        result = pcp(data, max_iter=1, tol=1e-12)
    assert not result.converged
    assert result.n_iter == 1
    assert result.error > 1e-12


@pytest.mark.parametrize("solver", [pcp, admm, stable_pcp, outlier_pursuit])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": -1},
        {"max_iter": 1.5},
        {"tol": 0},
        {"tol": -1.0},
        {"tol": np.nan},
        {"lambda_": 0.0},
        {"lambda_": -0.5},
        {"lambda_": np.inf},
    ],
)
def test_invalid_shared_parameters_fail_before_solving(solver, kwargs):
    with pytest.raises((ValueError, TypeError)):
        solver(np.eye(4), **kwargs)


@pytest.mark.parametrize("tau", [0.0, -1.0, np.nan, np.inf])
def test_stable_rejects_invalid_regularization(tau):
    with pytest.raises((ValueError, TypeError)):
        stable_pcp(np.eye(3), tau=tau)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": -1, "cardinality": 1},
        {"rank": 5, "cardinality": 1},
        {"rank": 1.5, "cardinality": 1},
        {"rank": 1, "cardinality": -1},
        {"rank": 1, "cardinality": 17},
        {"rank": 1, "cardinality": 0.5},
    ],
)
def test_godec_rejects_invalid_constraints(kwargs):
    with pytest.raises((ValueError, TypeError)):
        godec(np.eye(4), **kwargs)


def test_result_unpacking_estimator_and_dispatch():
    data, _, _ = recovery_problem(rows=24, columns=30)
    model = RPCA(method="ialm", tol=1e-7)
    assert model.fit(data) is model
    assert model.result_.converged
    low_rank, sparse = model.result_
    np.testing.assert_array_equal(low_rank, model.low_rank_)
    np.testing.assert_array_equal(sparse, model.sparse_)
    transformed = model.fit_transform(data)
    assert transformed is model.result_
    assert transformed.converged
    direct = decompose(data, method="ialm", tol=1e-7)
    assert direct.converged
    np.testing.assert_allclose(transformed.low_rank, direct.low_rank)
    with pytest.raises(ValueError):
        decompose(data, method="missing")
    result_numpy = direct.to_numpy()
    assert isinstance(result_numpy.low_rank, np.ndarray)


@pytest.mark.parametrize(
    "solver, solver_kwargs",
    [
        (pcp, {}),
        (admm, {}),
        (stable_pcp, {}),
        (outlier_pursuit, {}),
        (godec, {"rank": 2, "cardinality": 16}),
    ],
)
def test_torch_cpu_matches_numpy(solver, solver_kwargs):
    torch = pytest.importorskip("torch")
    data, _, _ = recovery_problem(rows=20, columns=24)
    kwargs = {"max_iter": 3000, "tol": 1e-7, **solver_kwargs}
    numpy_result = solver(data, backend="numpy", **kwargs)
    torch_result = solver(torch.tensor(data, dtype=torch.float64), **kwargs)
    assert numpy_result.converged
    assert torch_result.converged
    assert torch_result.backend == "torch"
    assert torch_result.low_rank.device.type == "cpu"
    assert not torch_result.low_rank.requires_grad
    np.testing.assert_allclose(
        numpy_result.low_rank, torch_result.to_numpy().low_rank, atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(
        numpy_result.sparse, torch_result.to_numpy().sparse, atol=2e-4, rtol=2e-4
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "solver, solver_kwargs",
    [
        (pcp, {}),
        (admm, {}),
        (stable_pcp, {}),
        (outlier_pursuit, {}),
        (godec, {"rank": 2, "cardinality": 23}),
    ],
)
def test_cuda_executes_on_device_and_matches_numpy(solver, solver_kwargs):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    data, _, _ = recovery_problem(rows=24, columns=28)
    kwargs = {"tol": 1e-7, "max_iter": 3000, **solver_kwargs}
    expected = solver(data, **kwargs)
    result = solver(data, device="cuda", **kwargs)
    assert expected.converged
    assert result.converged
    assert result.low_rank.is_cuda
    assert result.sparse.is_cuda
    np.testing.assert_allclose(
        result.to_numpy().low_rank, expected.low_rank, atol=2e-4, rtol=2e-4
    )
