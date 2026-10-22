import numpy as np
import pandas as pd
import pytest

from recsys.engines.collaborative import ALSModel, ItemKNN, build_matrix
from recsys.engines.popularity import PopularityEngine

# 2 factors = one per planted group; more capacity lets ALS memorise the single missing play as a zero
PARAMS = {"factors": 2, "regularization": 0.1, "iterations": 20, "alpha": 10,
          "event_weights": {"play": 1.0, "like": 2.0}}


def _planted():
    """Two taste groups that co-play disjoint item sets; the last user of each group misses one item."""
    rows = []
    for g, items in enumerate([[f"a{i}" for i in range(6)], [f"b{i}" for i in range(6)]]):
        for u in range(10):
            for t in (items[:-1] if u == 9 else items):
                rows.append({"user_id": f"g{g}u{u}", "track_id": t, "event_type": "play", "weight": 1})
    return pd.DataFrame(rows)


def test_matrix_sums_weighted_events_and_ignores_skips():
    inter = pd.DataFrame({"user_id": ["u"] * 4, "track_id": ["x", "x", "x", "y"],
                          "event_type": ["play", "play", "like", "skip"], "weight": 1})
    R, users, items = build_matrix(inter, PARAMS["event_weights"])
    assert items == ["x"] and R.toarray().tolist() == [[4.0]]


def test_als_and_knn_recover_the_missing_item_of_the_group():
    inter = _planted()
    als = ALSModel.fit(inter, PARAMS, seed=0)
    seen = set(inter.loc[inter["user_id"] == "g0u9", "track_id"])
    assert als.recommend("g0u9", 1, seen)[0][0] == "a5"
    assert als.recommend("unknown", 5) == []
    assert ItemKNN(inter, PARAMS["event_weights"], 10).recommend_batch(["g1u9"], 1) == {"g1u9": ["b5"]}


def test_als_save_load_roundtrip(tmp_path):
    als = ALSModel.fit(_planted(), PARAMS, seed=0)
    als.save(tmp_path)
    again = ALSModel.load(tmp_path)
    assert again.recommend("g1u9", 3) == pytest.approx(als.recommend("g1u9", 3))


def test_popularity_excludes():
    inter = pd.DataFrame({"user_id": list("abcab"), "track_id": list("xxxyz"), "event_type": "play", "weight": 1})
    assert [t for t, _ in PopularityEngine.from_interactions(inter).recommend(2, exclude={"x"})] == ["y", "z"]


def test_fold_in_reproduces_trained_user_vector():
    inter = _planted()
    als = ALSModel.fit(inter, PARAMS, seed=0)
    r = inter[inter["user_id"] == "g0u3"].groupby("track_id").size().astype(float).to_dict()
    a, b = als.fold_in(r), als.user_vector("g0u3")
    assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.99
    assert als.fold_in({"not-in-model": 1.0}) is None
