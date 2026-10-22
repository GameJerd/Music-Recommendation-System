def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "content-test+cf-test", "scale_tier": "S", "n_tracks": 7}


def test_search_by_title_and_non_ascii_artist(client):
    ids = [t["track_id"] for t in client.get("/tracks/search", params={"q": "night drive"}).json()]
    assert ids == ["t1", "t3", "t7"]  # duplicate titles, most popular first
    assert [t["track_id"] for t in client.get("/tracks/search", params={"q": "beyoncé"}).json()] == ["t5"]


def test_track_lookup_and_404(client):
    assert client.get("/tracks/t2").json()["artists"] == ["Synth A", "Synth B"]
    assert client.get("/tracks/nope").status_code == 404


def test_similar(client):
    items = client.get("/recommend/similar/t1", params={"k": 3}).json()["items"]
    ids = [i["track_id"] for i in items]
    assert len(ids) == 3 and "t1" not in ids
    assert ids[0] == "t2"  # same artist, genre, decade, close audio, and co-played
    assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)


def test_similar_blends_item_cf_only_when_seed_has_cf_vector(client):
    r = client.get("/recommend/similar/t1", params={"k": 5}).json()
    assert r["weights"] == {"content": 0.7, "cf": 0.3}
    by_id = {i["track_id"]: i for i in r["items"]}
    assert by_id["t2"]["cf_score"] is not None and by_id["t6"]["cf_score"] is None  # t6 never played
    r = client.get("/recommend/similar/t6", params={"k": 5}).json()
    assert r["weights"] == {"content": 1.0, "cf": 0.0}
    assert all(i["cf_score"] is None and i["score"] == i["content_score"] for i in r["items"])


def test_similar_drops_copies_of_seed_and_collapses_duplicates(client):
    ids = [i["track_id"] for i in client.get("/recommend/similar/t1", params={"k": 6}).json()["items"]]
    assert "t7" not in ids and len(ids) == 5  # t7 is a copy of the seed; 7 tracks - seed - copy
    ids = [i["track_id"] for i in client.get("/recommend/similar/t2", params={"k": 6}).json()["items"]]
    assert ids.count("t1") + ids.count("t7") == 1


def test_similar_errors(client):
    assert client.get("/recommend/similar/nope").status_code == 404
    assert client.get("/recommend/similar/t1", params={"k": 0}).status_code == 422
    assert client.get("/tracks/search", params={"q": ""}).status_code == 422
