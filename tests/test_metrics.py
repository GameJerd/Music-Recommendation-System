import numpy as np
import pytest

from recsys.eval.metrics import evaluate, hit_at_k, ndcg_at_k, recall_at_k


def test_ranking_metrics():
    recs, rel = ["a", "b", "c", "d"], {"b", "z"}
    assert recall_at_k(recs, rel, 3) == 0.5
    assert hit_at_k(recs, rel, 1) == 0.0 and hit_at_k(recs, rel, 2) == 1.0
    assert ndcg_at_k(recs, rel, 3) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert ndcg_at_k(["b", "z"], rel, 3) == pytest.approx(1.0)


def test_evaluate_averages_over_users():
    res = evaluate(lambda u: {"u1": ["a"], "u2": ["x"]}[u], {"u1": {"a"}, "u2": {"b"}}, k=10)
    assert res["users"] == 2 and res["hit@10"] == 0.5 and res["n_distinct_recommended"] == 2
