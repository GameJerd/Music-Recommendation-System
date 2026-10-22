import numpy as np
import pandas as pd
import pytest

from recsys.config import load_config
from recsys.data.schema import AUDIO_COLUMNS
from recsys.data.simulate import self_checks, simulate


@pytest.fixture(scope="module")
def catalog():
    """FIXTURE: a random 3,000-track catalog for testing the generator's mechanics, not the real dataset."""
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame(rng.normal(size=(n, len(AUDIO_COLUMNS))), columns=AUDIO_COLUMNS)
    df["track_id"] = [f"t{i}" for i in range(n)]
    df["artists"] = [[f"artist{i % 150}"] for i in range(n)]
    df["year"] = rng.integers(1921, 2021, n)
    df["popularity"] = rng.integers(0, 101, n)
    return df


@pytest.fixture(scope="module")
def params():
    return {**load_config()["simulation"], "n_users": 400}


def test_same_seed_same_data(catalog, params):
    a, ua, _ = simulate(catalog, params, seed=7)
    b, ub, _ = simulate(catalog, params, seed=7)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ua, ub)
    c, _, _ = simulate(catalog, params, seed=8)
    assert not a.equals(c)


def test_schema_buckets_and_no_persona_leak(catalog, params):
    inter, users, _ = simulate(catalog, params, seed=7)
    assert list(inter.columns) == ["user_id", "track_id", "event_type", "weight", "ts", "source"]
    assert set(inter["event_type"]) == {"play", "like", "skip"} and set(inter["source"]) == {"simulated"}
    plays = inter[inter["event_type"] == "play"].groupby("user_id").size()
    assert (plays.reindex(users["user_id"]).to_numpy() == users["n_plays"].to_numpy()).all()
    assert inter["ts"].between(pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-30", tz="UTC")).all()
    buckets = self_checks(inter, users, len(catalog), params, seed=7)["activity_buckets"]
    assert max(abs(o - t) for o, t in zip(buckets["observed"], buckets["target"])) < 0.07  # 400 users: noisy


def test_visible_share_controls_how_much_plays_follow_visible_taste(catalog, params):
    """Mean visible (persona) match of played tracks rises with visible_share."""
    match = {}
    for vs in (0.0, 0.5, 1.0):
        inter, users, world = simulate(catalog, {**params, "visible_share": vs}, seed=3)
        plays = inter[inter["event_type"] == "play"].merge(users[["user_id", "primary_persona"]], on="user_id")
        row = pd.Index(world.track_ids).get_indexer(plays["track_id"])
        col = pd.Index(list(world.favourites)).get_indexer(plays["primary_persona"])
        match[vs] = world.vis[row, col].mean()
    assert match[0.0] < match[0.5] < match[1.0]
    assert abs(match[0.0]) < 0.2 and match[1.0] > 1.0
