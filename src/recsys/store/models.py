from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Track(Base):
    __tablename__ = "tracks"

    track_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    artists: Mapped[list] = mapped_column(JSON)
    genres: Mapped[list] = mapped_column(JSON)
    year: Mapped[int] = mapped_column(Integer)
    popularity: Mapped[int] = mapped_column(Integer)


class Event(Base):
    """Logged interaction; same columns as the interaction record in the PRD."""
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    track_id: Mapped[str] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    ts: Mapped[datetime] = mapped_column(DateTime)  # UTC
    source: Mapped[str] = mapped_column(String, default="live")
