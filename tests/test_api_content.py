def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "content-test", "scale_tier": "S", "n_tracks": 7}


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
    assert ids[0] == "t2"  # same artist, genre, decade and close audio
    assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)
    assert all(i["score"] == i["similarity"] for i in items)  # default popularity_boost is 0
    assert {"year", "popularity", "genre"} <= set(items[0])


def test_similar_drops_copies_of_seed_and_collapses_duplicates(client):
    ids = [i["track_id"] for i in client.get("/recommend/similar/t1", params={"k": 6}).json()["items"]]
    assert "t7" not in ids and len(ids) == 5  # t7 is a copy of the seed; 7 tracks - seed - copy
    ids = [i["track_id"] for i in client.get("/recommend/similar/t2", params={"k": 6}).json()["items"]]
    assert ids.count("t1") + ids.count("t7") == 1


def test_similar_errors(client):
    assert client.get("/recommend/similar/nope").status_code == 404
    assert client.get("/recommend/similar/t1", params={"k": 0}).status_code == 422
    assert client.get("/tracks/search", params={"q": ""}).status_code == 422
