"""Hybrid blender (PRD: Recommendation engine).

    s(u,i) = (1 - w_cf(u)) * s_content(u,i) + w_cf(u) * s_cf(u,i),   w_cf(u) = w_max * n_u / (n_u + k)

n_u is the user's count of play events (training history + live). Both engines' top candidates are
pooled, *both* engines score the whole pool, and each score is min-max normalised over the pool.
A candidate with no CF vector (cold item) gets its content score only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recsys.engines.collaborative import ALSModel
from recsys.engines.content import ContentEngine, dedupe
from recsys.engines.popularity import PopularityEngine

MODES = ("hybrid", "content", "cf", "popular")


def cf_weight(n: int, w_max: float, k: float) -> float:
    return w_max * n / (n + k) if n > 0 else 0.0


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to 0-1 over the non-NaN entries; all-equal values map to 1."""
    out = np.full(len(x), np.nan, dtype=np.float64)
    ok = ~np.isnan(x)
    if ok.any():
        lo, hi = x[ok].min(), x[ok].max()
        out[ok] = (x[ok] - lo) / (hi - lo) if hi - lo > 1e-12 else 1.0
    return out


def aggregate_history(inter: pd.DataFrame) -> pd.DataFrame:
    """Per (user, track): play and like counts. This is what the API serves user histories from."""
    pos = inter[inter["event_type"].isin(["play", "like"])]
    h = pos.pivot_table(index=["user_id", "track_id"], columns="event_type", values="weight",
                        aggfunc="sum", fill_value=0).reset_index()
    for col in ("play", "like"):
        if col not in h:
            h[col] = 0
    return h.rename(columns={"play": "plays", "like": "likes"})[["user_id", "track_id", "plays", "likes"]]


@dataclass
class Rec:
    track_id: str
    score: float
    content_score: float | None = None
    cf_score: float | None = None


@dataclass
class Result:
    mode: str
    cold_start: bool
    n_plays: int
    weights: dict = field(default_factory=lambda: {"content": 0.0, "cf": 0.0})
    items: list[Rec] = field(default_factory=list)


class HybridEngine:
    def __init__(self, content: ContentEngine, als: ALSModel, history: pd.DataFrame, cfg: dict):
        self.content, self.als = content, als
        self.blend, self.event_weights = cfg["blend"], cfg["cf"]["event_weights"]
        self.history = {u: dict(zip(g["track_id"], zip(g["plays"], g["likes"])))
                        for u, g in history.groupby("user_id")}
        self.popularity = PopularityEngine(history.groupby("track_id")["plays"].sum())

    def user_history(self, user_id: str, live_events=()) -> tuple[dict[str, float], int, bool]:
        """({track_id: r_ui}, n plays, has live events) from training history plus live (track_id, event_type)."""
        w = self.event_weights
        r = {t: w["play"] * p + w["like"] * lk for t, (p, lk) in self.history.get(user_id, {}).items()}
        n = sum(p for p, _ in self.history.get(user_id, {}).values())
        live = False
        for tid, etype in live_events:
            if etype in w:
                r[tid] = r.get(tid, 0.0) + w[etype]
                n += etype == "play"
                live = True
        return r, int(n), live

    def _cf_vector(self, user_id, r, live) -> np.ndarray | None:
        vec = None if live else self.als.user_vector(user_id)
        return vec if vec is not None else self.als.fold_in(r)

    def _not_copies(self, track_ids, played) -> list[str]:
        played_keys = {self.content.key_of.get(t, t) for t in played}
        return [t for t in track_ids if self.content.key_of.get(t, t) not in played_keys]

    def recommend(self, user_id: str, k: int, mode: str = "hybrid", live_events=(), seeds=(),
                  exclude_played: bool = True) -> Result:
        r, n, live = self.user_history(user_id, live_events)
        played = list(r)
        exclude = played if exclude_played else []
        key_of = self.content.key_of

        if mode == "popular":
            recs = self.popularity.recommend(k * 3 + len(exclude), set(exclude))
            ids = self._not_copies([t for t, _ in recs], exclude)
            score = dict(recs)
            return Result("popular", n == 0, n, items=[Rec(t, s) for t, s in dedupe(ids, [score[t] for t in ids],
                                                                                    key_of, k)])
        if n == 0:  # cold start: content from picked seed tracks, else popularity
            q = self.content.profile(seeds)
            if q is None:
                res = self.recommend(user_id, k, "popular", live_events, exclude_played=exclude_played)
                res.cold_start = True
                return res
            recs = self.content.recommend(q, k, exclude_ids=list(seeds) + exclude)
            return Result("content", True, 0, {"content": 1.0, "cf": 0.0},
                          [Rec(t, s, content_score=s) for t, s in recs])

        q = self.content.profile(played)
        vec = self._cf_vector(user_id, r, live)
        if mode == "content" or vec is None:
            recs = self.content.recommend(q, k, exclude_ids=exclude)
            return Result("content", False, n, {"content": 1.0, "cf": 0.0},
                          [Rec(t, s, content_score=s) for t, s in recs])
        if mode == "cf":
            recs = self.als.recommend_vector(vec, k * 3 + 10, exclude)
            ids = self._not_copies([t for t, _ in recs], exclude)
            score = dict(recs)
            return Result("cf", False, n, {"content": 0.0, "cf": 1.0},
                          [Rec(t, s, cf_score=s) for t, s in dedupe(ids, [score[t] for t in ids], key_of, k)])

        # hybrid: pool both engines' candidates, score the pool with both, normalise, blend
        c = self.blend["candidates"]
        pool = [t for t, _ in self.content.recommend(q, c, exclude_ids=exclude)]
        pool += [t for t, _ in self.als.recommend_vector(vec, c, exclude)]
        pool = self._not_copies(list(dict.fromkeys(pool)), exclude)
        w = cf_weight(n, self.blend["w_max"], self.blend["k"])
        items = self._blend(pool, self.content.score(q, pool), self.als.score(vec, pool), w, k)
        return Result("hybrid", False, n, {"content": 1 - w, "cf": w}, items)

    def similar(self, track_id: str, k: int) -> tuple[list[Rec], float]:
        """Content neighbours of a seed, with item-item CF (ALS item cosine) blended in when the seed has
        a CF vector. Returns (items, cf weight used). KeyError if the track is unknown."""
        c = self.blend["candidates"]
        content = self.content.similar(track_id, c)
        cf = self.als.similar_items(track_id, c)
        w = self.blend["similar_cf_weight"] if cf else 0.0
        pool = self._not_copies(list(dict.fromkeys([t for t, _ in content] + [t for t, _ in cf])), [track_id])
        q = self.content.X[self.content.row[track_id]].toarray().ravel()
        return self._blend(pool, self.content.score(q, pool), self.als.item_cosine(track_id, pool), w, k), w

    def _blend(self, pool, content_raw, cf_raw, w, k) -> list[Rec]:
        cs, fs = minmax(np.asarray(content_raw, dtype=np.float64)), minmax(np.asarray(cf_raw, dtype=np.float64))
        if w == 0:
            fs = np.full(len(pool), np.nan)
        s = np.where(np.isnan(fs), cs, (1 - w) * cs + w * np.nan_to_num(fs))
        order = np.argsort(-s, kind="stable")
        kept = dedupe(np.asarray(pool)[order], s[order], self.content.key_of, k)
        pos = {t: i for i, t in enumerate(pool)}
        return [Rec(t, sc, float(cs[pos[t]]), None if np.isnan(fs[pos[t]]) else float(fs[pos[t]]))
                for t, sc in kept]
