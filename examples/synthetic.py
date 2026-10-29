"""Recover a synthetic low-rank matrix with sparse gross errors."""

import argparse

import numpy as np

from rpca_kit import pcp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    low_rank = rng.standard_normal((80, 3)) @ rng.standard_normal((3, 60))
    sparse = np.zeros_like(low_rank)
    indices = rng.choice(sparse.size, size=100, replace=False)
    sparse.flat[indices] = rng.choice([-20.0, 20.0], size=len(indices))
    data = low_rank + sparse
    result = pcp(data, device=args.device, max_iter=1000, tol=1e-7)
    cpu = result.to_numpy()
    error = np.linalg.norm(cpu.low_rank - low_rank) / np.linalg.norm(low_rank)
    print(f"backend={result.backend}, device={result.device}")
    print(f"converged={result.converged}, iterations={result.n_iter}")
    print(f"low-rank relative recovery error={error:.3e}")
    print(f"stopping criterion={result.error:.3e}")


if __name__ == "__main__":
    main()
