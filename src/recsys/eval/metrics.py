"""List-quality proxies for seed-track queries. With no user data these describe *what kind* of list the
engine returns (on-genre? same era? varied? popular?); none of them measures whether listeners like it."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Catalog:
    genres: list[set]        # per row
    decade: np.ndarray       # per row
    artist: np.ndarray       # primary artist per row
    popularity: np.ndarray   # 0-100 per row
    audio_unit: np.ndarray   # standardised audio features, L2-normalised per row (diversity space)


def list_metrics(seed: int, recs: list[int], cat: Catalog) -> dict:
    """Metrics for one top-k list of catalog rows recommended for a seed row."""
    k = len(recs)
    a = cat.audio_unit[recs]
    pair = (a @ a.T)[np.triu_indices(k, 1)] if k > 1 else np.array([1.0])
    return {
        "genre_match": np.mean([bool(cat.genres[r] & cat.genres[seed]) for r in recs]),
        "decade_match": np.mean(cat.decade[recs] == cat.decade[seed]),
        "same_artist": np.mean(cat.artist[recs] == cat.artist[seed]),
        "distinct_artists": len(set(cat.artist[recs])),
        "audio_diversity": 1 - pair.mean(),
        "mean_popularity": cat.popularity[recs].mean(),
    }
