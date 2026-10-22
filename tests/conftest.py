import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recsys.config import load_config
from recsys.data.loaders import attach_genres
from recsys.features.build import build_content, save_content
from recsys.store.db import make_engine, write_tracks

# FIXTURE: a tiny hand-written catalog for unit tests only. It is NOT the Spotify dataset;
# gate checks and scripts always run on the real CSVs in data/raw/.
AUDIO = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
         "liveness", "speechiness", "loudness", "tempo"]
ROWS = [
    # id,   name,         artists,                 year, pop, audio
    ("t1", "Night Drive", ["Synth A"],             1985, 60, [0.8, 0.9, 0.6, 0.1, 0.5, 0.1, 0.05, -6, 120]),
    ("t2", "Neon Road",   ["Synth A", "Synth B"],  1987, 40, [0.75, 0.85, 0.55, 0.15, 0.4, 0.2, 0.04, -7, 118]),
    ("t3", "Night Drive", ["Folk C"],              1972, 30, [0.3, 0.2, 0.4, 0.9, 0.1, 0.1, 0.03, -15, 80]),
    ("t4", "River Song",  ["Folk C"],              1974, 20, [0.35, 0.25, 0.45, 0.85, 0.05, 0.15, 0.04, -14, 85]),
    ("t5", "Halo",        ["Beyoncé"],             2008, 90, [0.6, 0.7, 0.5, 0.3, 0.0, 0.2, 0.06, -5, 80]),
    ("t6", "Untagged",    ["Nobody Known"],        2010, 5,  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -10, 100]),
    ("t7", "night drive ", ["Synth A"],            1985, 10, [0.8, 0.9, 0.6, 0.1, 0.5, 0.1, 0.05, -6, 120]),  # copy of t1
]
GENRES = {"Synth A": ["synthpop", "new wave"], "Synth B": ["synthwave"], "Folk C": ["folk"],
          "Beyoncé": ["pop", "r&b"]}


@pytest.fixture
def tracks() -> pd.DataFrame:
    df = pd.DataFrame(
        [dict(track_id=i, name=n, artists=a, year=y, popularity=p, **dict(zip(AUDIO, f)))
         for i, n, a, y, p, f in ROWS]
    )
    return attach_genres(df, GENRES)


@pytest.fixture
def client(tracks, tmp_path):
    from recsys.api.main import create_app

    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "content": str(tmp_path / "content"), "db": str(tmp_path / "test.db")}
    X, info = build_content(tracks, cfg["features"]["block_weights"])
    save_content(tmp_path / "content", X, tracks, {"version": "content-test", **info})
    write_tracks(make_engine(tmp_path / "test.db"), tracks)
    with TestClient(create_app(cfg)) as c:
        yield c
