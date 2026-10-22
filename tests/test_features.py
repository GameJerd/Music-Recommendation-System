import numpy as np
import pytest
from scipy.sparse.linalg import norm
from sklearn.preprocessing import normalize

from recsys.features.audio import audio_block
from recsys.features.build import build_content
from recsys.features.text import decade_block, genre_block

W = {"audio": 0.5, "genre": 0.35, "decade": 0.15}


def test_genres_are_union_of_all_artists_primary_first(tracks):
    assert tracks.set_index("track_id").loc["t2", "genres"] == ["synthpop", "new wave", "synthwave"]
    assert tracks.set_index("track_id").loc["t6", "genres"] == []


def test_rows_are_unit_norm_even_without_genres(tracks):
    X, _ = build_content(tracks, W)
    assert np.allclose(norm(X, axis=1), 1.0, atol=1e-6)


def test_cosine_is_weighted_sum_of_block_cosines(tracks):
    X, _ = build_content(tracks, W)
    blocks = {"audio": normalize(audio_block(tracks)), "genre": normalize(genre_block(tracks)[0]).toarray(),
              "decade": normalize(decade_block(tracks)[0]).toarray()}
    i, j = 0, 1  # t1, t2: both have every block
    expected = sum(w * blocks[b][i] @ blocks[b][j] for b, w in W.items())
    assert (X[i] @ X[j].T).toarray()[0, 0] == pytest.approx(expected, abs=1e-5)


def test_zero_weight_drops_block(tracks):
    X, info = build_content(tracks, {"audio": 1.0, "genre": 0.0, "decade": 0.0})
    assert info["blocks"] == ["audio"] and X.shape[1] == 9
