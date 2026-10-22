from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.engines.index import BruteForceIndex, NeighborIndex


def dup_keys(tracks: pd.DataFrame) -> np.ndarray:
    """Normalised (title, artists) key; tracks sharing it are copies of one recording (re-releases)."""
    return (tracks["name"].str.strip().str.lower() + "\x1f"
            + tracks["artists"].map(lambda a: "\x1f".join(a).lower())).to_numpy()


def dedupe(track_ids, scores, key_of: dict[str, str], k: int) -> list[tuple[str, float]]:
    """Keep the first (best-scoring) copy of each key, up to k results."""
    seen, out = set(), []
    for tid, s in zip(track_ids, scores):
        key = key_of.get(tid, tid)
        if key not in seen:
            seen.add(key)
            out.append((tid, float(s)))
            if len(out) == k:
                break
    return out


class ContentEngine:
    """Content recommendations keyed by track_id, over a NeighborIndex of content vectors.

    Results are deduplicated by `keys` (see dup_keys); excluding a track also excludes its copies.
    """

    def __init__(self, X: sparse.csr_matrix, track_ids: list[str], keys=None, index: NeighborIndex | None = None):
        self.X = X.tocsr()
        self.track_ids = np.asarray(track_ids)
        self.row = {tid: i for i, tid in enumerate(track_ids)}
        keys = self.track_ids if keys is None else np.asarray(keys)
        self.key_of = dict(zip(track_ids, keys))
        self.rows_of_key = defaultdict(list)
        for i, key in enumerate(keys):
            self.rows_of_key[key].append(i)
        self.index = index or BruteForceIndex(self.X)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self.row

    def rows(self, track_ids) -> np.ndarray:
        return np.array([self.row[t] for t in track_ids if t in self.row], dtype=np.int64)

    def rows_with_copies(self, track_ids) -> np.ndarray:
        keys = {self.key_of[t] for t in track_ids if t in self.row}
        return np.array([r for key in keys for r in self.rows_of_key[key]], dtype=np.int64)

    def profile(self, track_ids) -> np.ndarray | None:
        """Normalised mean vector of the given tracks; None if none are known."""
        rows = self.rows(track_ids)
        if not len(rows):
            return None
        v = np.asarray(self.X[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(v)
        return v / norm if norm else None

    def recommend(self, q: np.ndarray, k: int, exclude_ids=()) -> list[tuple[str, float]]:
        exclude = self.rows_with_copies(exclude_ids)
        n = 2 * k + 10
        while True:  # over-fetch until k distinct recordings survive dedupe
            idx, scores = self.index.search(q, n, exclude=exclude)
            out = dedupe(self.track_ids[idx], scores, self.key_of, k)
            if len(out) == k or len(idx) < n:
                return out
            n *= 4

    def similar(self, track_id: str, k: int) -> list[tuple[str, float]]:
        """Nearest tracks to a seed, excluding the seed and its copies. KeyError if unknown."""
        q = self.X[self.row[track_id]].toarray().ravel()
        return self.recommend(q, k, exclude_ids=[track_id])

    def score(self, q: np.ndarray, track_ids) -> np.ndarray:
        return self.index.score(q, self.rows(track_ids))
