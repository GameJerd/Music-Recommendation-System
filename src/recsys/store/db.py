import json
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, String, cast, create_engine, insert, or_, select
from sqlalchemy.orm import Session

from recsys.store.models import Base, Track


def make_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False so LIKE search matches non-ASCII artist names stored as JSON text
    return create_engine(f"sqlite:///{db_path}", json_serializer=lambda o: json.dumps(o, ensure_ascii=False))


def write_tracks(engine: Engine, tracks: pd.DataFrame) -> None:
    """Replace the tracks table (other tables, e.g. events, are kept)."""
    Track.__table__.drop(engine, checkfirst=True)
    Base.metadata.create_all(engine)
    rows = tracks[["track_id", "name", "artists", "genres", "year", "popularity"]].to_dict("records")
    for r in rows:  # parquet round-trips lists as numpy arrays
        r["artists"], r["genres"] = list(r["artists"]), list(r["genres"])
    with Session(engine) as s, s.begin():
        s.execute(insert(Track), rows)


def get_track(engine: Engine, track_id: str) -> Track | None:
    with Session(engine) as s:
        return s.get(Track, track_id)


def get_tracks(engine: Engine, track_ids: list[str]) -> dict[str, Track]:
    with Session(engine) as s:
        return {t.track_id: t for t in s.scalars(select(Track).where(Track.track_id.in_(track_ids)))}


def search_tracks(engine: Engine, q: str, limit: int) -> list[Track]:
    """Case-insensitive substring match on title or artist, most popular first."""
    pattern = f"%{q}%"
    stmt = (
        select(Track)
        .where(or_(Track.name.ilike(pattern), cast(Track.artists, String).ilike(pattern)))
        .order_by(Track.popularity.desc())
        .limit(limit)
    )
    with Session(engine) as s:
        return list(s.scalars(stmt))
