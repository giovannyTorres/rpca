"""Small stateful interface for repeated robust decompositions."""

from .solvers import _METHODS, decompose


class RPCA:
    """Estimator-style wrapper for :func:`decompose`.

    ``fit_transform(data)`` returns the full RPCAResult. ``fit(data)`` returns
    self and stores ``result_``, ``low_rank_``, ``sparse_`` and ``residual_``.
    There is no out-of-sample transform: each fit solves a matrix decomposition.
    """

    def __init__(self, method="ialm", *, backend="auto", device=None, **kwargs):
        if not isinstance(method, str) or method not in _METHODS:
            raise ValueError(f"method must be one of {tuple(_METHODS)}")
        self.method = method
        self.backend = backend
        self.device = device
        self.kwargs = dict(kwargs)

    def fit(self, data):
        """Decompose data and store fitted attributes; return self."""
        result = decompose(
            data, method=self.method, backend=self.backend, device=self.device, **self.kwargs,
        )
        self.result_ = result
        self.low_rank_ = result.low_rank
        self.sparse_ = result.sparse
        self.residual_ = result.residual
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        return self

    def fit_transform(self, data):
        """Fit and return the full result, including diagnostics."""
        return self.fit(data).result_
