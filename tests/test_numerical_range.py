"""Check scale-independent recovery and stationarity against known solutions."""
import numpy as np
import pytest

from rpca_kit import admm, outlier_pursuit, pcp, stable_pcp, to_numpy


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("solver", [pcp, admm, outlier_pursuit])
@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_equality_solvers_recover_nonzero_data_at_extreme_scales(backend, solver, scale):
    if backend == "torch":
        pytest.importorskip("torch")
    data = np.ones((4, 5)) * scale
    result = solver(data, backend=backend, lambda_=1.0)
    assert result.converged
    assert result.rank == 1
    assert np.isfinite(result.error) and np.isfinite(result.objective)
    np.testing.assert_allclose(to_numpy(result.low_rank) / scale, 1, atol=1e-6)
    np.testing.assert_allclose(to_numpy(result.sparse) / scale, 0, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_stable_pcp_matches_analytic_rank_one_solution_at_each_scale(backend, scale):
    if backend == "torch":
        pytest.importorskip("torch")
    # With lambda=1, nuclear shrinkage is cheaper than entrywise shrinkage.
    # SVT of this rank-one constant matrix gives the analytic optimum.
    data = np.ones((4, 5)) * scale
    tau = 0.1 * scale
    expected = 1 - 0.1 / np.sqrt(20)
    result = stable_pcp(data, tau=tau, lambda_=1.0, backend=backend, tol=1e-8)
    assert result.converged
    np.testing.assert_allclose(to_numpy(result.low_rank) / scale, expected, atol=1e-7)
    np.testing.assert_allclose(to_numpy(result.sparse) / scale, 0, atol=1e-7)


def test_unrepresentable_objective_fails_instead_of_returning_nan():
    with pytest.raises(FloatingPointError, match="rescale"):
        stable_pcp(np.ones((4, 5)) * 1e200, tau=1e199)


def test_torch_lazy_negative_views_convert_without_storage_aliasing():
    torch = pytest.importorskip("torch")
    data = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    negative = data._neg_view()
    converted = to_numpy(negative)
    np.testing.assert_array_equal(converted, -data.numpy())
    converted[:] = 0
    np.testing.assert_array_equal(data.numpy(), [[1, 2], [3, 4]])
    result = pcp(negative, backend="numpy")
    assert result.converged
