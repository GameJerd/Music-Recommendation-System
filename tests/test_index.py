import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from recsys.engines.index import BruteForceIndex


def test_bruteforce_matches_exhaustive_sort():
    rng = np.random.default_rng(0)
    X = normalize(rng.normal(size=(200, 16))).astype(np.float32)
    q = X[3]
    idx, scores = BruteForceIndex(sparse.csr_matrix(X)).search(q, 10)
    expected = np.argsort(-(X @ q), kind="stable")[:10]
    assert idx.tolist() == expected.tolist()
    assert idx[0] == 3 and np.all(np.diff(scores) <= 0)


def test_exclude_and_k_larger_than_catalog():
    X = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    idx, _ = BruteForceIndex(X).search(np.array([1, 0, 0, 0], dtype=np.float32), 10, exclude=np.array([0, 2]))
    assert sorted(idx.tolist()) == [1, 3]
