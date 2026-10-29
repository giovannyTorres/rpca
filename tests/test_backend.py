"""Backend validation and proximal operator regression tests."""

import numpy as np
import pytest

from rpca_kit._backend import get_backend, to_numpy
from rpca_kit.result import RPCAResult


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.float16])
def test_numpy_copy_and_dtype(dtype):
    original = np.arange(6).reshape(2, 3).astype(dtype)
    backend, x = get_backend(original)
    assert backend.name == "numpy"
    assert backend.device == "cpu"
    assert x.dtype == (dtype if dtype in (np.float32, np.float64) else np.float64)
    x[0, 0] = 20
    assert original[0, 0] == 0


@pytest.mark.parametrize(
    "x", [[], [1, 2], np.empty((0, 2)), [[np.nan]], [[np.inf]], [[1j]], [[True]], [["1"]], [[object()]]]
)
def test_reject_invalid_data(x):
    with pytest.raises(ValueError):
        get_backend(x)


@pytest.mark.parametrize("backend,device", [("bad", None), ("numpy", "cuda"), ("auto", "gpu"), ("auto", "cuda:-1")])
def test_reject_invalid_backend_device(backend, device):
    with pytest.raises(ValueError):
        get_backend([[1]], backend=backend, device=device)


@pytest.mark.parametrize("name", ["numpy", "torch"])
def test_proximal_operators(name):
    if name == "torch":
        pytest.importorskip("torch")
    b, x = get_backend(np.diag([5.0, 2.0, 0.0]), backend=name)
    value, rank, nuclear = b.svt(x, 2)
    np.testing.assert_allclose(to_numpy(value), np.diag([3, 0, 0]))
    assert rank == 1
    assert nuclear == pytest.approx(3)
    np.testing.assert_allclose(to_numpy(b.shrink(x, 2)), np.diag([3, 0, 0]))
    np.testing.assert_allclose(to_numpy(b.group_shrink(x, 2)), np.diag([3, 0, 0]))
    np.testing.assert_allclose(to_numpy(b.truncated_svd(x, 1)), np.diag([5, 0, 0]))
    assert b.group_norm(x) == pytest.approx(7)
    assert b.norm(x) == pytest.approx(np.sqrt(29))
    assert b.spectral_norm(x) == pytest.approx(5)


@pytest.mark.parametrize("name", ["numpy", "torch"])
def test_hard_threshold_deterministic_ties(name):
    if name == "torch":
        pytest.importorskip("torch")
    b, x = get_backend([[2.0, -2.0], [2.0, -1.0]], backend=name)
    np.testing.assert_array_equal(to_numpy(b.hard_threshold(x, 2)), [[2, -2], [0, 0]])
    np.testing.assert_array_equal(to_numpy(b.hard_threshold(x, 0)), np.zeros((2, 2)))
    np.testing.assert_array_equal(to_numpy(b.hard_threshold(x, 4)), to_numpy(x))
    for k in (-1, 5, 1.5, True):
        with pytest.raises(ValueError):
            b.hard_threshold(x, k)


@pytest.mark.parametrize("name", ["numpy", "torch"])
def test_hard_threshold_noncontiguous_input(name):
    if name == "torch":
        pytest.importorskip("torch")
    b, x = get_backend([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], backend=name)
    transposed = x.T
    result = b.hard_threshold(transposed, 2)
    np.testing.assert_array_equal(to_numpy(result), [[0, 0], [5, 0], [0, 6]])
    if name == "numpy":
        assert result.flags.c_contiguous
    else:
        assert result.is_contiguous()
    np.testing.assert_array_equal(to_numpy(transposed), [[1, 4], [5, 3], [2, 6]])


def test_torch_detached_copy_and_explicit_numpy():
    torch = pytest.importorskip("torch")
    source = torch.ones((3, 3), dtype=torch.float32, requires_grad=True)
    b, copied = get_backend(source)
    assert b.name == "torch"
    assert not copied.requires_grad
    assert copied.dtype == torch.float32
    copied[0, 0] = 10
    assert source[0, 0].item() == 1
    _, numpy = get_backend(source, backend="numpy")
    assert numpy.dtype == np.float32
    assert numpy[0, 0] == 1


def test_torch_sparse_rejected():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="dense"):
        get_backend(torch.eye(2).to_sparse())


def test_cuda_unavailable_is_explicit():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CUDA"):
        get_backend([[1.0]], device="cuda")


def test_result_unpack_and_numpy_copy():
    x = np.eye(2)
    result = RPCAResult(x, x * 0, x * 0, True, 3, 0.0, 2.0, [{"error": 0.0}], "pcp", "numpy", "cpu")
    low_rank, sparse = result
    assert low_rank is result.low_rank and sparse is result.sparse
    assert result.rank == 2
    copied = result.to_numpy()
    copied.low_rank[0, 0] = 5
    copied.history[0]["error"] = 10
    assert result.low_rank[0, 0] == 1
    assert result.history[0]["error"] == 0
