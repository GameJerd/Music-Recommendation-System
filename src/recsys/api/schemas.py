from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    model_version: str
    scale_tier: str
    n_tracks: int


class TrackOut(BaseModel):
    track_id: str
    track_name: str
    artists: list[str]
    genres: list[str]
    year: int
    popularity: int


class RecItem(BaseModel):
    track_id: str
    track_name: str
    artist_name: str
    genre: str
    year: int
    popularity: int
    score: float = Field(description="Ranking score: similarity + popularity_boost * popularity/100")
    similarity: float | None = Field(description="Cosine similarity to the seed or profile; null for popular picks")


class SimilarResponse(BaseModel):
    seed_track_id: str
    items: list[RecItem]


class UserRecResponse(BaseModel):
    user_id: str
    mode: str = Field(description="Mode actually served; a user with no history and no seeds gets `popular`")
    cold_start: bool
    n_history: int = Field(description="Distinct tracks the user played or liked (logged via /events)")
    items: list[RecItem]


class EventIn(BaseModel):
    user_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    event_type: Literal["play", "like", "skip"]


class EventOut(EventIn):
    id: int
    weight: float
    ts: datetime
    source: str
