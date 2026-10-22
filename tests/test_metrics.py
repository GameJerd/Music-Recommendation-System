import numpy as np
import pytest

from recsys.eval.metrics import Catalog, list_metrics


def test_list_metrics():
    cat = Catalog(genres=[{"rock"}, {"rock", "pop"}, {"jazz"}, set()],
                  decade=np.array([1970, 1970, 1950, 1970]), artist=np.array(["A", "A", "B", "C"]),
                  popularity=np.array([50, 60, 20, 10]),
                  audio_unit=np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=float))
    m = list_metrics(0, [1, 2], cat)
    assert m["genre_match"] == 0.5 and m["decade_match"] == 0.5 and m["same_artist"] == 0.5
    assert m["distinct_artists"] == 2 and m["mean_popularity"] == 40
    assert m["audio_diversity"] == pytest.approx(1.0)  # orthogonal pair
    assert list_metrics(0, [1, 3], cat)["audio_diversity"] == pytest.approx(0.0)  # identical pair
