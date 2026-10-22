"""Nearest-neighbour search over L2-normalised row vectors (inner product == cosine).

Engines depend only on NeighborIndex, so moving scale tier swaps this class, not the engines.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


class NeighborIndex(ABC):
    @abstractmethod
    def search(self, q: np.ndarray, k: int, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-k rows by cosine to dense query q, best first: (row indices, scores)."""

    @abstractmethod
    def score(self, q: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Cosine of q to the given rows (used to score another engine's candidates)."""


class BruteForceIndex(NeighborIndex):
    """Exact search: one sparse matrix-vector product per query, O(nnz). Tier S.

    ponytail: linear scan; at tier M swap for a FaissIndex implementing the same two methods.
    """

    def __init__(self, X: sparse.csr_matrix):
        self.X = X.tocsr().astype(np.float32)

    def search(self, q, k, exclude=None):
        s = np.asarray(self.X @ q).ravel()
        if exclude is not None and len(exclude):
            s[exclude] = -np.inf
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        top = top[np.isfinite(s[top])]
        return top, s[top]

    def score(self, q, rows):
        return np.asarray(self.X[rows] @ q).ravel()
