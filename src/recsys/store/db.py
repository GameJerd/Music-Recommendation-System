import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, String, cast, create_engine, insert, or_, select
from sqlalchemy.orm import Session

from recsys.store.models import Base, Event, Track


def make_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False so LIKE search matches non-ASCII artist names stored as JSON text
    engine = create_engine(f"sqlite:///{db_path}", json_serializer=lambda o: json.dumps(o, ensure_ascii=False))
    Base.metadata.create_all(engine)
    return engine


def write_tracks(engine: Engine, tracks: pd.DataFrame) -> None:
    """Replace the tracks table (other tables, e.g. events, are kept)."""
    Track.__table__.drop(engine, checkfirst=True)
    Base.metadata.create_all(engine)
    rows = tracks[["track_id", "name", "artists", "genres", "year", "popularity"]].to_dict("records")
    for r in rows:  # parquet round-trips lists as numpy arrays
        r["artists"], r["genres"] = list(r["artists"]), list(r["genres"])
    with Session(engine) as s, s.begin():
        s.execute(insert(Track), rows)


def add_event(engine: Engine, user_id: str, track_id: str, event_type: str, ts: datetime) -> Event:
    with Session(engine, expire_on_commit=False) as s, s.begin():
        e = Event(user_id=user_id, track_id=track_id, event_type=event_type, weight=1.0, ts=ts, source="live")
        s.add(e)
    return e


def user_events(engine: Engine, user_id: str) -> list[tuple[str, str]]:
    """(track_id, event_type) of a user's live events, oldest first."""
    with Session(engine) as s:
        rows = s.execute(select(Event.track_id, Event.event_type).where(Event.user_id == user_id).order_by(Event.id))
        return [tuple(r) for r in rows]


def live_events_frame(engine: Engine) -> pd.DataFrame:
    """All live events in the interaction-record layout, for retraining."""
    cols = ["user_id", "track_id", "event_type", "weight", "ts", "source"]
    df = pd.read_sql(select(*[getattr(Event, c) for c in cols]), engine)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


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
