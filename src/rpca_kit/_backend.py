"""Small NumPy/PyTorch adapter used by the RPCA solvers.

PyTorch is imported only when tensors or the torch backend are requested.
Solvers are numerical routines, rather than differentiable torch operators:
all inputs are copied and tensors are detached from their autograd graphs.
"""

from __future__ import annotations

import importlib
import numbers
import sys
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "The torch backend requires PyTorch. Install rpca-kit[gpu], "
            "or install the appropriate CUDA build from https://pytorch.org/."
        ) from exc


def _is_tensor(data: Any) -> bool:
    module = sys.modules.get("torch")
    if module is None:
        return False
    return isinstance(data, module.Tensor)


def to_numpy(data: Any, *, copy: bool = True) -> np.ndarray:
    """Return a NumPy array; tensors are detached and moved to CPU.

    Copies by default so changes to the returned array cannot mutate results.
    Setting ``copy=False`` may share CPU storage with the input.
    """
    if _is_tensor(data):
        tensor = data.detach().cpu()
        if tensor.dtype == _torch().bfloat16:
            tensor = tensor.to(dtype=_torch().float64)
        array = tensor.numpy()
    else:
        array = np.asarray(data)
    return array.copy() if copy else array


class Backend:
    """Internal arithmetic adapter for dense real matrices."""

    def __init__(self, name: str, device: str = "cpu") -> None:
        if name not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'")
        self.name = name
        self.device = device
        self.xp = np if name == "numpy" else _torch()

    def zeros_like(self, x: Any) -> Any:
        return self.xp.zeros_like(x)

    def norm(self, x: Any) -> float:
        """Frobenius norm of a matrix."""
        if self.name == "numpy":
            return float(np.linalg.norm(x, ord="fro"))
        return float(self.xp.linalg.vector_norm(x).item())

    def spectral_norm(self, x: Any) -> float:
        if self.name == "numpy":
            return float(np.linalg.norm(x, ord=2))
        return float(self.xp.linalg.matrix_norm(x, ord=2).item())

    def max_abs(self, x: Any) -> float:
        if self.name == "numpy":
            return float(np.max(np.abs(x)))
        return float(self.xp.max(self.xp.abs(x)).item())

    def sum_abs(self, x: Any) -> float:
        if self.name == "numpy":
            return float(np.sum(np.abs(x)))
        return float(self.xp.sum(self.xp.abs(x)).item())

    def svt(self, x: Any, tau: float) -> tuple[Any, int, float]:
        """Singular value thresholding, retained rank and nuclear norm."""
        u, s, vh = self.xp.linalg.svd(x, full_matrices=False)
        if self.name == "numpy":
            s = np.maximum(s - tau, 0)
            rank = int(np.count_nonzero(s))
            nuclear_norm = float(s.sum())
        else:
            s = self.xp.clamp(s - tau, min=0)
            rank = int(self.xp.count_nonzero(s).item())
            nuclear_norm = float(s.sum().item())
        return (u * s) @ vh, rank, nuclear_norm

    def shrink(self, x: Any, tau: float) -> Any:
        """Elementwise soft thresholding."""
        if self.name == "numpy":
            return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
        return self.xp.sign(x) * self.xp.clamp(self.xp.abs(x) - tau, min=0)

    def group_shrink(self, x: Any, tau: float) -> Any:
        """Soft threshold the Euclidean norm of each column."""
        if self.name == "numpy":
            norms = np.linalg.norm(x, axis=0)
            scale = np.maximum(norms - tau, 0) / np.where(norms > 0, norms, 1)
        else:
            norms = self.xp.linalg.vector_norm(x, dim=0)
            denominator = self.xp.where(norms > 0, norms, self.xp.ones_like(norms))
            scale = self.xp.clamp(norms - tau, min=0) / denominator
        return x * scale

    def group_norm(self, x: Any) -> float:
        """Sum of column Euclidean norms (the l2,1 norm)."""
        if self.name == "numpy":
            return float(np.linalg.norm(x, axis=0).sum())
        return float(self.xp.linalg.vector_norm(x, dim=0).sum().item())

    def truncated_svd(self, x: Any, rank: int) -> Any:
        """Best rank-at-most-``rank`` approximation using compact SVD."""
        if isinstance(rank, bool) or not isinstance(rank, numbers.Integral):
            raise ValueError("rank must be an integer")
        if rank < 0 or rank > min(x.shape):
            raise ValueError("rank must lie between 0 and min(x.shape)")
        if rank == 0:
            return self.zeros_like(x)
        u, s, vh = self.xp.linalg.svd(x, full_matrices=False)
        return (u[:, :rank] * s[:rank]) @ vh[:rank, :]

    def hard_threshold(self, x: Any, k: int) -> Any:
        """Keep at most k entries, with ties broken by flattened index."""
        size = int(x.size) if self.name == "numpy" else int(x.numel())
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ValueError("k must be an integer")
        if k < 0 or k > size:
            raise ValueError("k must lie between 0 and the number of entries")
        if self.name == "numpy":
            result = np.zeros_like(x, order="C")
        else:
            result = self.xp.zeros_like(x, memory_format=self.xp.contiguous_format)
        if k == 0:
            return result
        flat = x.reshape(-1)
        if self.name == "numpy":
            indices = np.argsort(-np.abs(flat), kind="stable")[:k]
        else:
            indices = self.xp.argsort(self.xp.abs(flat), descending=True, stable=True)[:k]
        result.reshape(-1)[indices] = flat[indices]
        return result

    def rank(self, x: Any) -> int:
        if self.name == "numpy":
            return int(np.linalg.matrix_rank(x))
        return int(self.xp.linalg.matrix_rank(x).item())


def get_backend(
    data: Any, backend: str = "auto", device: str | None = None
) -> tuple[Backend, Any]:
    """Validate and copy a dense, finite, two-dimensional real input.

    ``auto`` preserves torch tensors, otherwise choosing NumPy, except that
    ``device='cuda'`` explicitly requests torch. CUDA requests never fall back
    to CPU. Only float32 and float64 are retained; other real numeric dtypes
    are promoted to float64.
    """
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if device is not None and not isinstance(device, str):
        raise ValueError("device must be None, 'cpu', 'cuda', or 'cuda:<index>'")
    if device is not None:
        valid_cuda = device == "cuda" or (
            device.startswith("cuda:") and device[5:].isdigit()
        )
        if device != "cpu" and not valid_cuda:
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    tensor_input = _is_tensor(data)
    if backend == "auto":
        backend = "torch" if tensor_input or (device or "").startswith("cuda") else "numpy"
    if backend == "numpy" and (device or "cpu") != "cpu":
        raise ValueError("the numpy backend supports device='cpu' only")

    if tensor_input:
        torch = _torch()
        if data.layout != torch.strided:
            raise ValueError("input must be a dense strided tensor")
        if data.is_complex() or data.dtype == torch.bool or data.is_quantized:
            raise ValueError("input must contain real numeric values, excluding booleans")
        if data.ndim != 2 or any(size == 0 for size in data.shape):
            raise ValueError("input must be a nonempty two-dimensional matrix")
    else:
        try:
            data = np.asarray(data)
        except (TypeError, ValueError) as exc:
            raise ValueError("input must be a rectangular real numeric matrix") from exc
        if data.ndim != 2 or any(size == 0 for size in data.shape):
            raise ValueError("input must be a nonempty two-dimensional matrix")
        if data.dtype.kind not in "iuf":
            raise ValueError("input must contain real numeric values, excluding booleans")

    if backend == "numpy":
        if tensor_input:
            dtype = data.dtype if data.dtype in (torch.float32, torch.float64) else torch.float64
            array = data.detach().to(device="cpu", dtype=dtype).numpy().copy()
        else:
            dtype = data.dtype if data.dtype in (np.dtype("float32"), np.dtype("float64")) else np.float64
            array = np.array(data, dtype=dtype, copy=True, order="C")
        if not np.isfinite(array).all():
            raise ValueError("input must contain only finite values")
        return Backend("numpy", "cpu"), array

    torch = _torch()
    target = torch.device(device or (str(data.device) if tensor_input else "cpu"))
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("the torch backend supports CPU and CUDA devices")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
        if target.index is not None and target.index >= torch.cuda.device_count():
            raise ValueError("the requested CUDA device index is not available")
    if tensor_input:
        dtype = data.dtype if data.dtype in (torch.float32, torch.float64) else torch.float64
        array = data.detach().to(device=target, dtype=dtype).clone().contiguous()
    else:
        dtype = torch.float32 if data.dtype == np.dtype("float32") else torch.float64
        numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
        array = torch.tensor(
            np.array(data, dtype=numpy_dtype, copy=True, order="C"), dtype=dtype, device=target
        )
    if not bool(torch.isfinite(array).all().item()):
        raise ValueError("input must contain only finite values")
    return Backend("torch", str(array.device)), array
