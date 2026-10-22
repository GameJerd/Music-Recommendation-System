import pytest

from recsys.engines.hybrid import cf_weight


def rec(client, user, **params):
    r = client.get(f"/recommend/user/{user}", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_known_user_hybrid_has_breakdown_and_excludes_played_and_copies(client):
    r = rec(client, "u_a", k=5)
    assert r["mode"] == "hybrid" and not r["cold_start"] and r["n_plays"] == 6
    assert r["weights"]["cf"] == pytest.approx(cf_weight(6, 0.75, 7.5))
    ids = [i["track_id"] for i in r["items"]]
    assert not {"t1", "t2", "t5", "t7"} & set(ids)  # played, plus t7 = copy of played t1
    for i in r["items"]:
        assert i["content_score"] is not None
        if i["cf_score"] is None:  # cold item: content only
            assert i["score"] == pytest.approx(i["content_score"])
        else:
            w = r["weights"]["cf"]
            assert i["score"] == pytest.approx((1 - w) * i["content_score"] + w * i["cf_score"])
    assert [i["score"] for i in r["items"]] == sorted((i["score"] for i in r["items"]), reverse=True)


def test_exclude_played_can_be_turned_off(client):
    ids = [i["track_id"] for i in rec(client, "u_c", k=7, exclude_played="false")["items"]]
    assert {"t3", "t4"} <= set(ids)


@pytest.mark.parametrize("mode", ["content", "cf", "popular"])
def test_single_engine_modes(client, mode):
    r = rec(client, "u_b", k=3, mode=mode)
    assert r["mode"] == mode and len(r["items"]) >= 1
    assert not {"t1", "t2", "t4", "t7"} & {i["track_id"] for i in r["items"]}


def test_unknown_user_is_cold_start(client):
    r = rec(client, "nobody")
    assert r["cold_start"] and r["n_plays"] == 0 and r["mode"] == "popular"
    assert {i["track_id"] for i in r["items"][:2]} == {"t1", "t2"}  # most played: 4 plays each
    r = rec(client, "nobody", seed=["t3"], k=3)
    assert r["cold_start"] and r["mode"] == "content" and r["weights"] == {"content": 1.0, "cf": 0.0}
    assert r["items"][0]["track_id"] == "t4" and "t3" not in [i["track_id"] for i in r["items"]]


def test_events_move_a_new_user_onto_the_blend(client):
    r = client.post("/events", json={"user_id": "newbie", "track_id": "t3", "event_type": "play"})
    assert r.status_code == 201 and r.json()["source"] == "live"
    r = rec(client, "newbie", k=3)
    assert not r["cold_start"] and r["n_plays"] == 1 and r["mode"] == "hybrid"
    assert r["weights"]["cf"] == pytest.approx(0.75 / 8.5)  # 9% CF after one play (folded in, no retrain)
    assert "t3" not in [i["track_id"] for i in r["items"]]
    client.post("/events", json={"user_id": "newbie", "track_id": "t4", "event_type": "skip"})
    assert rec(client, "newbie")["n_plays"] == 1  # skips are logged but not used


def test_event_and_param_validation(client):
    assert client.post("/events", json={"user_id": "x", "track_id": "nope", "event_type": "play"}).status_code == 404
    assert client.post("/events", json={"user_id": "x", "track_id": "t1", "event_type": "dance"}).status_code == 422
    assert client.post("/events", json={"user_id": "", "track_id": "t1", "event_type": "play"}).status_code == 422
    assert client.get("/recommend/user/u_a", params={"mode": "magic"}).status_code == 422
    assert client.get("/recommend/user/u_a", params={"k": 0}).status_code == 422
    assert client.get("/recommend/user/nobody", params={"seed": "nope"}).status_code == 404


def test_admin_reload(client):
    r = client.post("/admin/reload")
    assert r.status_code == 200 and r.json()["model_version"] == "content-test+cf-test"
