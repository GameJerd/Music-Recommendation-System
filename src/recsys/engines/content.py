"""Content-based recommendations keyed by track_id.

Ranking: nearest neighbours by cosine on the content vectors, optionally nudged by the dataset's popularity
(score = cosine + popularity_boost * popularity/100), then re-ranked so that copies of one recording appear
once (dup_keys) and no primary artist fills more than `artist_cap` slots.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.engines.index import BruteForceIndex, NeighborIndex


def dup_keys(tracks: pd.DataFrame) -> np.ndarray:
    """Normalised (title, artists) key; tracks sharing it are copies of one recording (re-releases)."""
    return (tracks["name"].str.strip().str.lower() + "\x1f"
            + tracks["artists"].map(lambda a: "\x1f".join(a).lower())).to_numpy()


@dataclass
class Rec:
    track_id: str
    score: float
    similarity: float | None  # cosine to the query; None for popularity-only results


@dataclass
class UserResult:
    mode: str           # mode actually served: "content", or "popular" for a cold start
    cold_start: bool
    n_history: int
    items: list[Rec]


class ContentEngine:
    def __init__(self, X: sparse.csr_matrix, tracks: pd.DataFrame, candidates: int = 200, artist_cap: int = 0,
                 popularity_boost: float = 0.0, index: NeighborIndex | None = None):
        self.X = X.tocsr()
        self.track_ids = tracks["track_id"].to_numpy()
        self.row = {tid: i for i, tid in enumerate(self.track_ids)}
        self.keys = dup_keys(tracks)
        self.rows_of_key = defaultdict(list)
        for i, key in enumerate(self.keys):
            self.rows_of_key[key].append(i)
        self.artist = tracks["artists"].map(lambda a: a[0] if len(a) else "").to_numpy()
        self.popularity = tracks["popularity"].to_numpy(np.float32) / 100
        self.by_popularity = np.argsort(-self.popularity, kind="stable")
        self.candidates, self.artist_cap, self.popularity_boost = candidates, artist_cap, popularity_boost
        self.index = index or BruteForceIndex(self.X)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self.row

    def rows(self, track_ids) -> np.ndarray:
        return np.array([self.row[t] for t in track_ids if t in self.row], dtype=np.int64)

    def rows_with_copies(self, track_ids) -> np.ndarray:
        keys = {self.keys[self.row[t]] for t in track_ids if t in self.row}
        return np.array([r for key in keys for r in self.rows_of_key[key]], dtype=np.int64)

    def profile(self, track_ids) -> np.ndarray | None:
        """Normalised mean vector of the given tracks; None if none are known."""
        rows = self.rows(track_ids)
        if not len(rows):
            return None
        v = np.asarray(self.X[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(v)
        return v / norm if norm else None

    def _select(self, rows, scores, sims, k) -> list[Rec]:
        """Best-first; one copy per recording; at most artist_cap tracks per primary artist."""
        seen, per_artist, out = set(), defaultdict(int), []
        for r, s, c in zip(rows, scores, sims):
            if self.keys[r] in seen or (self.artist_cap and per_artist[self.artist[r]] >= self.artist_cap):
                continue
            seen.add(self.keys[r])
            per_artist[self.artist[r]] += 1
            out.append(Rec(self.track_ids[r], float(s), None if c is None else float(c)))
            if len(out) == k:
                break
        return out

    def recommend(self, q: np.ndarray, k: int, exclude_ids=()) -> list[Rec]:
        exclude = self.rows_with_copies(exclude_ids)
        n = max(self.candidates, 2 * k + 10)
        while True:  # over-fetch until k results survive dedupe and the artist cap
            idx, sims = self.index.search(q, n, exclude=exclude)
            scores = sims + self.popularity_boost * self.popularity[idx]
            order = np.argsort(-scores, kind="stable")
            out = self._select(idx[order], scores[order], sims[order], k)
            if len(out) == k or len(idx) < n:
                return out
            n *= 4

    def similar(self, track_id: str, k: int) -> list[Rec]:
        """Nearest tracks to a seed, excluding the seed and its copies. KeyError if unknown."""
        return self.recommend(self.X[self.row[track_id]].toarray().ravel(), k, exclude_ids=[track_id])

    def popular(self, k: int, exclude_ids=()) -> list[Rec]:
        """Most popular tracks by the dataset's popularity column (cold-start fallback)."""
        excluded = set(self.rows_with_copies(exclude_ids).tolist())
        n = max(self.candidates, 2 * k + 10 + len(excluded))
        while True:
            rows = [r for r in self.by_popularity[:n] if r not in excluded]
            out = self._select(rows, self.popularity[rows], [None] * len(rows), k)
            if len(out) == k or n >= len(self.by_popularity):
                return out
            n *= 4

    def for_user(self, history, k: int, seeds=(), avoid=(), mode: str = "content",
                 exclude_history: bool = True) -> UserResult:
        """history: tracks the user played or liked; seeds: tracks a new user picked; avoid: skipped tracks.
        The profile is the mean of history + seeds. With neither, it is a cold start served by popularity."""
        liked = list(dict.fromkeys(list(history) + list(seeds)))
        exclude = list(avoid) + (liked if exclude_history else list(seeds))
        q = self.profile(liked)
        if mode == "popular" or q is None:
            return UserResult("popular", q is None, len(history), self.popular(k, exclude))
        return UserResult("content", False, len(history), self.recommend(q, k, exclude))
