from fastapi import APIRouter, Depends, HTTPException, Query

from recsys.api.deps import AppState, get_state
from recsys.api.schemas import Health, RecItem, SimilarResponse, TrackOut
from recsys.store import db
from recsys.store.models import Track

router = APIRouter()


def track_out(t: Track) -> TrackOut:
    return TrackOut(track_id=t.track_id, track_name=t.name, artists=t.artists, genres=t.genres,
                    year=t.year, popularity=t.popularity)


def rec_items(state: AppState, scored: list[tuple[str, float]]) -> list[RecItem]:
    meta = db.get_tracks(state.db, [tid for tid, _ in scored])
    return [
        RecItem(track_id=tid, track_name=meta[tid].name, artist_name=", ".join(meta[tid].artists),
                genre=", ".join(meta[tid].genres[:3]), score=s, content_score=s)
        for tid, s in scored
    ]


@router.get("/health", response_model=Health)
def health(state: AppState = Depends(get_state)):
    return Health(status="ok", model_version=state.meta["version"], scale_tier=state.cfg["scale"]["tier"],
                  n_tracks=len(state.content.track_ids))


@router.get("/tracks/search", response_model=list[TrackOut])
def search(q: str = Query(min_length=1), limit: int = Query(10, ge=1, le=50), state: AppState = Depends(get_state)):
    return [track_out(t) for t in db.search_tracks(state.db, q, limit)]


@router.get("/tracks/{track_id}", response_model=TrackOut)
def get_track(track_id: str, state: AppState = Depends(get_state)):
    t = db.get_track(state.db, track_id)
    if t is None:
        raise HTTPException(404, f"Unknown track_id {track_id}")
    return track_out(t)


@router.get("/recommend/similar/{track_id}", response_model=SimilarResponse)
def similar(track_id: str, k: int = Query(10, ge=1, le=100), state: AppState = Depends(get_state)):
    if track_id not in state.content:
        raise HTTPException(404, f"Unknown track_id {track_id}")
    return SimilarResponse(seed_track_id=track_id, items=rec_items(state, state.content.similar(track_id, k)))
