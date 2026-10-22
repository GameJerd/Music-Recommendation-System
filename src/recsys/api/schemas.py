from pydantic import BaseModel


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


class SimilarResponse(BaseModel):
    seed_track_id: str
    items: list[RecItem]
