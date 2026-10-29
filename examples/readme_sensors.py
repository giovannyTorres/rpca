"""Reproducible synthetic example from the English package README."""

import numpy as np
from rpca_kit import outlier_pursuit

rng = np.random.default_rng(57)
basis, _ = np.linalg.qr(rng.normal(size=(32, 2)))
normal_samples = basis @ rng.normal(size=(2, 70))
observed = normal_samples.copy()
observed[:, [2, 31, 59]] += rng.normal(scale=8.0, size=(32, 3))

result = outlier_pursuit(
    observed, lambda_=0.5, tol=1e-7, max_iter=3000
)
scores = np.linalg.norm(result.sparse, axis=0)
review_first = np.sort(np.argsort(scores)[-3:])

print(f"converged: {result.converged}")
print(f"samples to inspect first: {review_first.tolist()}")
assert result.converged
np.testing.assert_array_equal(review_first, [2, 31, 59])
