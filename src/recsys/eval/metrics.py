from collections.abc import Callable

import numpy as np


def recall_at_k(recs: list[str], relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant)


def hit_at_k(recs: list[str], relevant: set, k: int) -> float:
    return float(bool(set(recs[:k]) & relevant))


def ndcg_at_k(recs: list[str], relevant: set, k: int) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, t in enumerate(recs[:k]) if t in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def evaluate(recommend: Callable[[str], list[str]], targets: dict[str, set], k: int = 10) -> dict:
    """Mean Recall/Hit/NDCG@k over users with targets, plus catalog coverage of all recommended items."""
    rows, recommended = [], set()
    for uid, rel in targets.items():
        recs = recommend(uid)[:k]
        recommended.update(recs)
        rows.append((recall_at_k(recs, rel, k), hit_at_k(recs, rel, k), ndcg_at_k(recs, rel, k)))
    r = np.array(rows) if rows else np.zeros((0, 3))
    return {"users": len(rows), f"recall@{k}": r[:, 0].mean() if rows else np.nan,
            f"hit@{k}": r[:, 1].mean() if rows else np.nan, f"ndcg@{k}": r[:, 2].mean() if rows else np.nan,
            "n_distinct_recommended": len(recommended)}
