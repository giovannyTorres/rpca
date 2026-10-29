"""Decomposition result and solver warning types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator

from ._backend import Backend, to_numpy


class ConvergenceWarning(RuntimeWarning):
    """A solver exhausted its iteration budget before reaching tolerance."""


@dataclass
class RPCAResult:
    """Low-rank, sparse and residual terms with convergence diagnostics.

    ``residual`` is always ``input - low_rank - sparse``. Unpacking returns
    the two fitted components: ``low_rank, sparse = result``. For noisy
    variants the residual is the dense noise estimate.
    """

    low_rank: Any
    sparse: Any
    residual: Any
    converged: bool
    n_iter: int
    error: float
    objective: float
    history: list[dict[str, float | int]]
    method: str
    backend: str
    device: str

    @property
    def rank(self) -> int:
        """Numerical matrix rank using the backend's dtype-aware tolerance."""
        return Backend(self.backend, self.device).rank(self.low_rank)

    def __iter__(self) -> Iterator[Any]:
        yield self.low_rank
        yield self.sparse

    def to_numpy(self) -> RPCAResult:
        """Return an independent result whose arrays live in CPU memory."""
        return replace(
            self,
            low_rank=to_numpy(self.low_rank),
            sparse=to_numpy(self.sparse),
            residual=to_numpy(self.residual),
            history=[item.copy() for item in self.history],
            backend="numpy",
            device="cpu",
        )
