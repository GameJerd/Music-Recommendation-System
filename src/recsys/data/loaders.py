import ast
from pathlib import Path

import pandas as pd

from recsys.data.schema import ARTISTS_FILE, TRACKS_FILE


def load_tracks(raw_dir: Path) -> pd.DataFrame:
    """data.csv with `artists` parsed from its stringified list."""
    df = pd.read_csv(raw_dir / TRACKS_FILE)
    df["artists"] = df["artists"].map(ast.literal_eval)
    return df


def load_artist_genres(raw_dir: Path) -> dict[str, list[str]]:
    a = pd.read_csv(raw_dir / ARTISTS_FILE, usecols=["artists", "genres"])
    return dict(zip(a["artists"], a["genres"].map(ast.literal_eval)))


def attach_genres(tracks: pd.DataFrame, genre_map: dict[str, list[str]]) -> pd.DataFrame:
    """Track genres = union of every listed artist's genres, primary artist's first."""
    tracks = tracks.copy()
    tracks["genres"] = tracks["artists"].map(
        lambda artists: list(dict.fromkeys(g for a in artists for g in genre_map.get(a, [])))
    )
    return tracks
