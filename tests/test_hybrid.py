import numpy as np
import pytest

from recsys.config import anchor_k, load_config
from recsys.engines.hybrid import cf_weight, minmax

W_MAX, K = 0.75, 7.5


def test_blend_anchor_five_plays_is_70_content_30_cf():
    assert cf_weight(5, W_MAX, K) == pytest.approx(0.30)
    blend = load_config()["blend"]
    assert cf_weight(5, blend["w_max"], blend["k"]) == pytest.approx(0.30)


# PRD table "How much of each, at the starting values": n -> CF share (rounded to whole percent)
PRD_TABLE = {0: 0.00, 1: 0.09, 3: 0.21, 5: 0.30, 9: 0.41, 30: 0.60, 100: 0.70}


@pytest.mark.parametrize("n,cf", PRD_TABLE.items())
def test_blend_matches_prd_table(n, cf):
    w = cf_weight(n, W_MAX, K)
    assert w == pytest.approx(cf, abs=0.005)
    assert (1 - w) == pytest.approx(1 - cf, abs=0.005)


def test_heavy_users_never_reach_w_max_and_anchor_holds_for_other_w_max():
    assert cf_weight(10**6, W_MAX, K) < W_MAX
    for w_max in (0.5, 0.6, 0.9):
        assert cf_weight(5, w_max, anchor_k(w_max)) == pytest.approx(0.30)


def test_minmax_ignores_nan_and_handles_constant():
    out = minmax(np.array([2.0, np.nan, 4.0, 3.0]))
    assert np.isnan(out[1]) and out[[0, 2, 3]].tolist() == [0.0, 1.0, 0.5]
    assert minmax(np.array([5.0, 5.0])).tolist() == [1.0, 1.0]
