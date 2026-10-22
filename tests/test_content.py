import pytest

from recsys.engines.content import ContentEngine
from recsys.features.build import build_content

W = {"audio": 0.5, "genre": 0.35, "decade": 0.15}


@pytest.fixture
def engine_factory(tracks):
    X, _ = build_content(tracks, W)
    return lambda **kw: ContentEngine(X, tracks, candidates=10, **kw)


def ids(recs):
    return [r.track_id for r in recs]


def test_artist_cap_limits_tracks_per_primary_artist(engine_factory):
    def artists(eng):
        return [eng.artist[eng.row[t]] for t in ids(eng.recommend(eng.profile(["t3"]), 6, exclude_ids=["t3"]))]

    assert artists(engine_factory(artist_cap=0)).count("Synth A") == 2  # t2 and one copy of t1/t7
    capped = artists(engine_factory(artist_cap=1))
    assert len(capped) == len(set(capped)) == 4  # Folk C, Synth A, Beyoncé, Nobody Known


def test_popularity_boost_reorders_toward_popular_tracks(engine_factory):
    q = engine_factory().profile(["t6"])
    plain = engine_factory().recommend(q, 5, ["t6"])
    boosted = engine_factory(popularity_boost=5.0).recommend(q, 5, ["t6"])
    assert boosted[0].track_id == "t5"  # popularity 90
    assert all(r.score == pytest.approx(r.similarity) for r in plain)
    assert all(r.score > r.similarity for r in boosted)


def test_popular_excludes_copies_and_dedupes(engine_factory):
    eng = engine_factory()
    assert ids(eng.popular(3)) == ["t5", "t1", "t2"]
    assert ids(eng.popular(3, exclude_ids=["t7"])) == ["t5", "t2", "t3"]  # t7 is a copy of t1
    assert all(r.similarity is None for r in eng.popular(3))


def test_for_user_paths(engine_factory):
    eng = engine_factory()
    cold = eng.for_user([], 3)
    assert cold.cold_start and cold.mode == "popular" and ids(cold.items)[0] == "t5"
    seeded = eng.for_user([], 3, seeds=["t3"])
    assert not seeded.cold_start and seeded.mode == "content" and ids(seeded.items)[0] == "t4"
    warm = eng.for_user(["t1"], 6, avoid=["t4"])
    assert not {"t1", "t7", "t4"} & set(ids(warm.items))  # played, its copy, skipped
    assert "t1" in ids(eng.for_user(["t1"], 6, exclude_history=False).items)
    assert eng.for_user(["t1"], 3, mode="popular").mode == "popular"
