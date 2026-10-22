import pandas as pd


class PopularityEngine:
    """Rank tracks by play count in the training interactions."""

    def __init__(self, interactions: pd.DataFrame):
        plays = interactions[interactions["event_type"] == "play"]
        counts = plays["track_id"].value_counts()
        self.ranked = counts.index.tolist()
        self.score_of = (counts / counts.iloc[0]).to_dict() if len(counts) else {}

    def recommend(self, k: int, exclude=frozenset()) -> list[tuple[str, float]]:
        out = []
        for tid in self.ranked:
            if tid not in exclude:
                out.append((tid, self.score_of[tid]))
                if len(out) == k:
                    break
        return out
