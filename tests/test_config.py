import pytest

from recsys.config import anchor_k, load_config


def test_default_config_keeps_70_30_anchor():
    blend = load_config()["blend"]
    assert blend["w_max"] == 0.75
    assert blend["k"] == pytest.approx(anchor_k(blend["w_max"]))  # 7.5


def test_anchor_k_formula():
    assert anchor_k(0.75) == pytest.approx(7.5)
    assert anchor_k(0.6) == pytest.approx(5.0)
