import pandas as pd


class PopularityEngine:
    """Rank tracks by play count."""

    def __init__(self, play_counts: pd.Series):
        counts = play_counts[play_counts > 0].sort_values(ascending=False, kind="stable")
        self.ranked = counts.index.tolist()
        self.score_of = (counts / counts.iloc[0]).to_dict() if len(counts) else {}

    @classmethod
    def from_interactions(cls, interactions: pd.DataFrame) -> "PopularityEngine":
        return cls(interactions.loc[interactions["event_type"] == "play", "track_id"].value_counts())

    def recommend(self, k: int, exclude=frozenset()) -> list[tuple[str, float]]:
        out = []
        for tid in self.ranked:
            if tid not in exclude:
                out.append((tid, self.score_of[tid]))
                if len(out) == k:
                    break
        return out
