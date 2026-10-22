def rec(client, user, **params):
    r = client.get(f"/recommend/user/{user}", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def event(client, user, track, kind="play"):
    r = client.post("/events", json={"user_id": user, "track_id": track, "event_type": kind})
    assert r.status_code == 201, r.text
    return r.json()


def test_new_user_without_seeds_gets_popular(client):
    r = rec(client, "nobody", k=3)
    assert r["cold_start"] and r["mode"] == "popular" and r["n_history"] == 0
    assert [i["track_id"] for i in r["items"]] == ["t5", "t1", "t2"]  # dataset popularity 90, 60, 40
    assert all(i["similarity"] is None for i in r["items"])


def test_new_user_with_seed_gets_content(client):
    r = rec(client, "nobody", seed=["t3"], k=3)
    assert not r["cold_start"] and r["mode"] == "content"
    ids = [i["track_id"] for i in r["items"]]
    assert ids[0] == "t4" and "t3" not in ids


def test_logged_events_build_the_profile(client):
    assert event(client, "ana", "t3")["source"] == "live"
    event(client, "ana", "t3", "like")
    event(client, "ana", "t4", "skip")
    r = rec(client, "ana", k=6)
    assert r["mode"] == "content" and r["n_history"] == 1 and not r["cold_start"]
    ids = [i["track_id"] for i in r["items"]]
    assert "t3" not in ids and "t4" not in ids  # played and skipped
    assert "t3" in [i["track_id"] for i in rec(client, "ana", k=6, exclude_played="false")["items"]]
    assert rec(client, "ana", k=3, mode="popular")["mode"] == "popular"


def test_validation(client):
    assert client.post("/events", json={"user_id": "x", "track_id": "nope", "event_type": "play"}).status_code == 404
    assert client.post("/events", json={"user_id": "x", "track_id": "t1", "event_type": "dance"}).status_code == 422
    assert client.post("/events", json={"user_id": "", "track_id": "t1", "event_type": "play"}).status_code == 422
    assert client.get("/recommend/user/x", params={"mode": "hybrid"}).status_code == 422
    assert client.get("/recommend/user/x", params={"k": 0}).status_code == 422
    assert client.get("/recommend/user/x", params={"seed": "nope"}).status_code == 404


def test_history_newest_first_with_track_and_audio(client):
    event(client, "bo", "t1")
    event(client, "bo", "t5", "like")
    h = client.get("/users/bo/history").json()
    assert [(x["event_type"], x["track"]["track_id"]) for x in h] == [("like", "t5"), ("play", "t1")]
    assert h[0]["track"]["audio"]["danceability"] == 0.6
    assert client.get("/users/nobody/history").json() == []


def test_recommendations_carry_audio_for_covers(client):
    item = rec(client, "nobody", seed=["t3"], k=1)["items"][0]
    assert set(item["audio"]) >= {"energy", "valence", "tempo", "acousticness"}


def test_web_app_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Undertone</title>" in r.text
    assert client.get("/app.js").status_code == 200 and client.get("/docs").status_code == 200


def test_admin_reload(client):
    r = client.post("/admin/reload")
    assert r.status_code == 200 and r.json()["model_version"] == "content-test"
