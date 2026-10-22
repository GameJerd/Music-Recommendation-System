"""Collaborative filtering on implicit feedback: ALS (main) and item-item cosine KNN (baseline).

Confidence follows the PRD, c_ui = 1 + alpha * r_ui, where r_ui = play_weight * plays + like_weight * likes.
`implicit` uses the stored matrix value itself as the confidence (scaled by its own `alpha`), so we store
1 + alpha * r and pass alpha=1 to implicit. See docs/decisions.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from scipy import sparse
from threadpoolctl import threadpool_limits


def build_matrix(inter: pd.DataFrame, weights: dict[str, float],
                 user_ids: list[str] | None = None, item_ids: list[str] | None = None):
    """users x items CSR of r_ui, plus the id lists defining its rows and columns."""
    df = inter[inter["event_type"].isin(list(weights))]
    user_ids = user_ids if user_ids is not None else sorted(df["user_id"].unique())
    item_ids = item_ids if item_ids is not None else sorted(df["track_id"].unique())
    u = pd.Index(user_ids).get_indexer(df["user_id"])
    i = pd.Index(item_ids).get_indexer(df["track_id"])
    keep = (u >= 0) & (i >= 0)
    w = df["event_type"].map(weights).to_numpy(np.float32) * df["weight"].to_numpy(np.float32)
    R = sparse.csr_matrix((w[keep], (u[keep], i[keep])), shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    R.sum_duplicates()
    return R, list(user_ids), list(item_ids)


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray) -> np.ndarray:
    scores = scores.copy()
    scores[exclude] = -np.inf
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return top[np.isfinite(scores[top])]


class ALSModel:
    """User and item factors keyed by user_id / track_id. Serving needs only numpy (no implicit)."""

    def __init__(self, user_ids, item_ids, user_factors, item_factors, params: dict):
        self.user_ids, self.item_ids = list(user_ids), np.asarray(item_ids)
        self.user_factors = np.asarray(user_factors, dtype=np.float32)
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        self.params = params
        self.user_row = {u: r for r, u in enumerate(self.user_ids)}
        self.item_row = {t: r for r, t in enumerate(self.item_ids)}

    @classmethod
    def fit(cls, inter: pd.DataFrame, params: dict, seed: int) -> "ALSModel":
        R, users, items = build_matrix(inter, params["event_weights"])
        C = R.copy()
        C.data = 1 + params["alpha"] * C.data
        with threadpool_limits(1, "blas"):  # implicit warns that multithreaded BLAS slows ALS
            model = AlternatingLeastSquares(factors=params["factors"], regularization=params["regularization"],
                                            iterations=params["iterations"], alpha=1.0, random_state=seed)
            model.fit(C, show_progress=False)
        return cls(users, items, model.user_factors, model.item_factors, params)

    def recommend(self, user_id: str, k: int, exclude_ids=()) -> list[tuple[str, float]]:
        """Top-k tracks for a known user, excluding the given tracks. [] for an unknown user."""
        if user_id not in self.user_row:
            return []
        scores = self.item_factors @ self.user_factors[self.user_row[user_id]]
        excl = np.array([self.item_row[t] for t in exclude_ids if t in self.item_row], dtype=np.int64)
        top = top_k(scores, k, excl)
        return list(zip(self.item_ids[top].tolist(), scores[top].tolist()))

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "als.npz", user_factors=self.user_factors, item_factors=self.item_factors,
                 user_ids=np.array(self.user_ids), item_ids=self.item_ids)
        (out_dir / "als_params.json").write_text(json.dumps(self.params, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "ALSModel":
        z = np.load(out_dir / "als.npz")
        params = json.loads((out_dir / "als_params.json").read_text(encoding="utf-8"))
        return cls(z["user_ids"].tolist(), z["item_ids"], z["user_factors"], z["item_factors"], params)


class ItemKNN:
    """Item-item cosine KNN on the interaction matrix (baseline)."""

    def __init__(self, inter: pd.DataFrame, weights: dict[str, float], k_neighbours: int = 100):
        self.R, self.user_ids, self.item_ids = build_matrix(inter, weights)
        self.item_ids = np.asarray(self.item_ids)
        self.user_row = {u: r for r, u in enumerate(self.user_ids)}
        self.model = CosineRecommender(K=k_neighbours)
        self.model.fit(self.R, show_progress=False)

    def recommend_batch(self, user_ids: list[str], k: int) -> dict[str, list[str]]:
        """Top-k per known user, filtering the user's own training items."""
        known = [u for u in user_ids if u in self.user_row]
        rows = np.array([self.user_row[u] for u in known])
        ids, _ = self.model.recommend(rows, self.R[rows], N=k, filter_already_liked_items=True)
        return {u: [self.item_ids[i] for i in r if i >= 0] for u, r in zip(known, ids)}
