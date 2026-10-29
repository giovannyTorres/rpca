"""RPCA on CPU and GPU: convex and nonconvex robust matrix decompositions."""

from ._backend import to_numpy
from .estimator import RPCA
from .result import ConvergenceWarning, RPCAResult
from .solvers import admm, decompose, godec, outlier_pursuit, pcp, stable_pcp

__version__ = "0.1.1"

__all__ = [
    "RPCA", "RPCAResult", "ConvergenceWarning", "pcp", "admm", "stable_pcp",
    "outlier_pursuit", "godec", "decompose", "to_numpy", "__version__",
]
