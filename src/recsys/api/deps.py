"""Artifacts the API serves from. Loaded at startup and by /admin/reload; the API never trains."""
from dataclasses import dataclass

import pandas as pd
from fastapi import Request
from sqlalchemy import Engine

from recsys.config import resolve
from recsys.engines.collaborative import ALSModel
from recsys.engines.content import ContentEngine, dup_keys
from recsys.engines.hybrid import HybridEngine
from recsys.features.build import load_content
from recsys.store.db import make_engine


@dataclass
class AppState:
    cfg: dict
    db: Engine
    hybrid: HybridEngine
    version: str


def load_state(cfg: dict) -> AppState:
    content_dir, cf_dir = resolve(cfg, "content"), resolve(cfg, "cf")
    if not (content_dir / "vectors.npz").exists():
        raise RuntimeError(f"No content artifacts in {content_dir}; run `python scripts/build_features.py` first.")
    if not (cf_dir / "als.npz").exists():
        raise RuntimeError(f"No CF artifacts in {cf_dir}; run `python scripts/train_cf.py` first.")
    X, tracks, meta = load_content(content_dir)
    content = ContentEngine(X, tracks["track_id"].tolist(), keys=dup_keys(tracks))
    cf_meta = pd.read_json(cf_dir / "meta.json", typ="series")
    hybrid = HybridEngine(content, ALSModel.load(cf_dir), pd.read_parquet(cf_dir / "history.parquet"), cfg)
    return AppState(cfg, make_engine(resolve(cfg, "db")), hybrid, f"{meta['version']}+{cf_meta['version']}")


def get_state(request: Request) -> AppState:
    return request.app.state.recsys
