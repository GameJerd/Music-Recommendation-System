"""Combine feature blocks into one sparse, L2-normalised content vector per track.

Each block is L2-normalised per row and scaled by sqrt(weight), so with weights summing
to 1 the cosine of two tracks is the weighted sum of their per-block cosines. A track with
an empty block (no genres) is renormalised over the blocks it has.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from recsys.features.audio import audio_block
from recsys.features.text import decade_block, genre_block

TRACK_COLUMNS = ["track_id", "name", "artists", "genres", "year", "popularity"]


def build_content(tracks: pd.DataFrame, block_weights: dict[str, float]) -> tuple[sparse.csr_matrix, dict]:
    genres, genre_vocab = genre_block(tracks)
    decades, decade_vocab = decade_block(tracks)
    blocks = {"audio": sparse.csr_matrix(audio_block(tracks)), "genre": genres, "decade": decades}
    used = {name: w for name, w in block_weights.items() if w > 0}
    X = sparse.hstack([normalize(blocks[name]) * np.sqrt(w) for name, w in used.items()]).tocsr()
    X = normalize(X).astype(np.float32)
    info = {
        "blocks": list(used),
        "block_weights": used,
        "dims": {name: blocks[name].shape[1] for name in used},
        "genre_vocab_size": len(genre_vocab),
        "decades": decade_vocab,
    }
    return X, info


def save_content(out_dir: Path, X: sparse.csr_matrix, tracks: pd.DataFrame, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "vectors.npz", X)
    tracks[TRACK_COLUMNS].to_parquet(out_dir / "tracks.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_content(out_dir: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, dict]:
    X = sparse.load_npz(out_dir / "vectors.npz").tocsr()
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return X, tracks, meta
