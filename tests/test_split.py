import pandas as pd

from recsys.eval.split import holdout_sizes, targets, temporal_split


def test_holdout_sizes_keep_light_users_evaluable():
    assert [holdout_sizes(n) for n in (1, 2, 3, 4, 14, 30)] == [(0, 0), (0, 1), (1, 1), (1, 1), (1, 1), (3, 3)]


def _events(n_users=20, n_each=10):
    rows = [{"user_id": f"u{u}", "track_id": f"t{(u + j) % 40}", "event_type": "play", "weight": 1,
             "ts": pd.Timestamp("2026-07-01", tz="UTC") + pd.Timedelta(minutes=j), "source": "simulated"}
            for u in range(n_users) for j in range(n_each)]
    return pd.DataFrame(rows)


def test_split_is_temporal_and_cold_items_fully_held_out():
    inter = _events()
    s = temporal_split(inter, cold_share=0.1, seed=0)
    assert len(s.cold_items) == round(0.1 * inter["track_id"].nunique())  # 29 tracks -> 3
    assert not s.train["track_id"].isin(s.cold_items).any() and not s.val["track_id"].isin(s.cold_items).any()
    for uid, g in s.train.groupby("user_id"):
        later = pd.concat([s.val, s.test])
        later = later[(later["user_id"] == uid) & ~later["track_id"].isin(s.cold_items)]
        assert g["ts"].max() < later["ts"].min()
    assert len(s.train) + len(s.val) + len(s.test) == len(inter)


def test_targets_drop_tracks_already_in_train():
    train = pd.DataFrame({"user_id": ["a", "a"], "track_id": ["x", "y"]})
    held = pd.DataFrame({"user_id": ["a", "a", "b"], "track_id": ["y", "z", "y"]})
    assert targets(held, train) == {"a": {"z"}, "b": {"y"}}
