from sqlalchemy import JSON, Integer, String
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
