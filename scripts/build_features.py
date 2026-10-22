"""Build content vectors from data/raw and load the track catalog into SQLite.

Usage: python scripts/build_features.py [--config config/default.yaml]
Writes artifacts/content/{vectors.npz, tracks.parquet, meta.json} and the tracks table.
"""
import argparse
import time
from datetime import datetime, timezone

from recsys.config import DEFAULT_CONFIG, load_config, resolve
from recsys.data.loaders import attach_genres, load_artist_genres, load_tracks
from recsys.features.build import build_content, save_content
from recsys.store.db import make_engine, write_tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    cfg = load_config(ap.parse_args().config)
    t0 = time.perf_counter()

    raw = resolve(cfg, "raw")
    tracks = attach_genres(load_tracks(raw), load_artist_genres(raw)).rename(columns={"id": "track_id"})
    X, info = build_content(tracks, cfg["features"]["block_weights"])
    built_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"version": f"content-{built_at}", "n_tracks": X.shape[0], "n_dims": X.shape[1],
            "nnz": int(X.nnz), "scale_tier": cfg["scale"]["tier"], **info}
    save_content(resolve(cfg, "content"), X, tracks, meta)
    print(f"content vectors: {X.shape[0]:,} x {X.shape[1]:,}, nnz {X.nnz:,} "
          f"({X.data.nbytes / 1e6:.1f} MB data) -> {resolve(cfg, 'content')}")

    write_tracks(make_engine(resolve(cfg, "db")), tracks)
    print(f"tracks table: {len(tracks):,} rows -> {resolve(cfg, 'db')}")
    print(f"version {meta['version']}, done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
