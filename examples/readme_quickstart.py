"""Reproducible synthetic example from the English package README."""

import numpy as np
from rpca_kit import pcp

rng = np.random.default_rng(7)
time = np.linspace(0, 4 * np.pi, 120)
patterns = np.vstack([np.sin(time), np.cos(time), np.sin(0.5 * time)])
clean = rng.normal(size=(40, 3)) @ patterns

spikes = np.zeros_like(clean)
positions = rng.choice(clean.size, size=100, replace=False)
spikes.flat[positions] = rng.choice([-12.0, 12.0], size=100)
observed = clean + spikes

result = pcp(observed, tol=1e-7, max_iter=2000)
recovery_error = np.linalg.norm(result.low_rank - clean) / np.linalg.norm(clean)

print(f"converged: {result.converged}")
print(f"component shapes: {result.low_rank.shape}, {result.sparse.shape}")
print(f"relative recovery error: {recovery_error:.2e}")
assert result.converged
assert recovery_error < 1e-4
np.testing.assert_allclose(
    observed, result.low_rank + result.sparse + result.residual
)
