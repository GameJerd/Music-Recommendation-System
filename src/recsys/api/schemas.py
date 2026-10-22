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
    score: float
    content_score: float | None = None
    cf_score: float | None = None


class Weights(BaseModel):
    content: float
    cf: float


class SimilarResponse(BaseModel):
    seed_track_id: str
    weights: Weights
    items: list[RecItem]


class UserRecResponse(BaseModel):
    user_id: str
    mode: str = Field(description="Mode actually served: a cold-start user falls back to content or popular")
    cold_start: bool
    n_plays: int = Field(description="Play events in the user's history (training + live); drives the CF weight")
    weights: Weights
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
