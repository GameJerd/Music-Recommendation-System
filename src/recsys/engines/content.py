import numpy as np
from scipy import sparse

from recsys.engines.index import BruteForceIndex, NeighborIndex


class ContentEngine:
    """Content recommendations keyed by track_id, over a NeighborIndex of content vectors."""

    def __init__(self, X: sparse.csr_matrix, track_ids: list[str], index: NeighborIndex | None = None):
        self.X = X.tocsr()
        self.track_ids = np.asarray(track_ids)
        self.row = {tid: i for i, tid in enumerate(track_ids)}
        self.index = index or BruteForceIndex(self.X)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self.row

    def rows(self, track_ids) -> np.ndarray:
        return np.array([self.row[t] for t in track_ids if t in self.row], dtype=np.int64)

    def profile(self, track_ids) -> np.ndarray | None:
        """Normalised mean vector of the given tracks; None if none are known."""
        rows = self.rows(track_ids)
        if not len(rows):
            return None
        v = np.asarray(self.X[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(v)
        return v / norm if norm else None

    def recommend(self, q: np.ndarray, k: int, exclude_ids=()) -> list[tuple[str, float]]:
        idx, scores = self.index.search(q, k, exclude=self.rows(exclude_ids))
        return list(zip(self.track_ids[idx].tolist(), scores.tolist()))

    def similar(self, track_id: str, k: int) -> list[tuple[str, float]]:
        """Nearest tracks to a seed, excluding the seed. KeyError if unknown."""
        q = self.X[self.row[track_id]].toarray().ravel()
        return self.recommend(q, k, exclude_ids=[track_id])

    def score(self, q: np.ndarray, track_ids) -> np.ndarray:
        return self.index.score(q, self.rows(track_ids))
