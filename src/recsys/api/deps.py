"""Artifacts the API serves from. Loaded at startup and by /admin/reload; the API never builds features."""
from dataclasses import dataclass

from fastapi import Request
from sqlalchemy import Engine

from recsys.config import resolve
from recsys.engines.content import ContentEngine
from recsys.features.build import load_content
from recsys.store.db import make_engine


@dataclass
class AppState:
    cfg: dict
    db: Engine
    content: ContentEngine
    version: str


def load_state(cfg: dict) -> AppState:
    content_dir = resolve(cfg, "content")
    if not (content_dir / "vectors.npz").exists():
        raise RuntimeError(f"No content artifacts in {content_dir}; run `python scripts/build_features.py` first.")
    X, tracks, meta = load_content(content_dir)
    content = ContentEngine(X, tracks, **cfg["recommend"])
    return AppState(cfg, make_engine(resolve(cfg, "db")), content, meta["version"])


def get_state(request: Request) -> AppState:
    return request.app.state.recsys
