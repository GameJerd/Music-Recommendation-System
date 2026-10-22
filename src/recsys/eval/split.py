"""Per-user temporal split (PRD Evaluation, amended in docs/decisions.md).

Positive events (play, like) per user, ordered by time: test = the last max(1, round(0.1 n)) when n >= 2,
validation = the max(1, round(0.1 n)) before that when n >= 3, the rest is train.
Cold items: a seeded share of the tracks with interactions has *all* its events moved to test.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITIVE = ["play", "like"]


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cold_items: set


def holdout_sizes(n: int) -> tuple[int, int]:
    """(n_val, n_test) for a user with n positive events."""
    h = max(1, round(0.1 * n))
    return (h if n >= 3 else 0), (h if n >= 2 else 0)


def temporal_split(inter: pd.DataFrame, cold_share: float = 0.02, seed: int = 42) -> Split:
    pos = inter[inter["event_type"].isin(POSITIVE)].sort_values(["user_id", "ts"], kind="stable")
    rng = np.random.default_rng(seed)
    tracks = np.sort(pos["track_id"].unique())
    cold = set(rng.choice(tracks, size=int(round(cold_share * len(tracks))), replace=False).tolist())
    is_cold = pos["track_id"].isin(cold)
    warm = pos[~is_cold]

    n = warm.groupby("user_id")["user_id"].transform("size").to_numpy()
    rank_from_end = warm.groupby("user_id").cumcount(ascending=False).to_numpy()
    sizes = np.array([holdout_sizes(int(k)) for k in range(int(n.max()) + 1)])
    n_val, n_test = sizes[n, 0], sizes[n, 1]
    part = np.where(rank_from_end < n_test, "test", np.where(rank_from_end < n_test + n_val, "val", "train"))
    test = pd.concat([warm[part == "test"], pos[is_cold]])
    return Split(warm[part == "train"], warm[part == "val"], test, cold)


def targets(held_out: pd.DataFrame, train: pd.DataFrame) -> dict[str, set]:
    """Held-out tracks per user that the user did not already have in train (those are filtered from recs)."""
    seen = train.groupby("user_id")["track_id"].apply(set)
    out = {}
    for uid, tids in held_out.groupby("user_id")["track_id"]:
        new = set(tids) - seen.get(uid, set())
        if new:
            out[uid] = new
    return out
