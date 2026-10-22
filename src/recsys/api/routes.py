from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from recsys.api.deps import AppState, get_state, load_state
from recsys.api.schemas import (EventIn, EventOut, Health, RecItem, SimilarResponse, TrackOut, UserRecResponse,
                                Weights)
from recsys.engines.hybrid import Rec
from recsys.store import db
from recsys.store.models import Track

router = APIRouter()


def track_out(t: Track) -> TrackOut:
    return TrackOut(track_id=t.track_id, track_name=t.name, artists=t.artists, genres=t.genres,
                    year=t.year, popularity=t.popularity)


def rec_items(state: AppState, recs: list[Rec]) -> list[RecItem]:
    meta = db.get_tracks(state.db, [r.track_id for r in recs])
    return [
        RecItem(track_id=r.track_id, track_name=meta[r.track_id].name, artist_name=", ".join(meta[r.track_id].artists),
                genre=", ".join(meta[r.track_id].genres[:3]), score=r.score, content_score=r.content_score,
                cf_score=r.cf_score)
        for r in recs
    ]


def health_of(state: AppState) -> Health:
    return Health(status="ok", model_version=state.version, scale_tier=state.cfg["scale"]["tier"],
                  n_tracks=len(state.hybrid.content.track_ids))


@router.get("/health", response_model=Health)
def health(state: AppState = Depends(get_state)):
    return health_of(state)


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
    if track_id not in state.hybrid.content:
        raise HTTPException(404, f"Unknown track_id {track_id}")
    items, w = state.hybrid.similar(track_id, k)
    return SimilarResponse(seed_track_id=track_id, weights=Weights(content=1 - w, cf=w),
                           items=rec_items(state, items))


@router.get("/recommend/user/{user_id}", response_model=UserRecResponse)
def recommend_user(
    user_id: str,
    k: int = Query(10, ge=1, le=100),
    mode: Literal["hybrid", "content", "cf", "popular"] = "hybrid",
    seed: list[str] = Query(default=[], description="Track ids a new user picked (cold-start onboarding)"),
    exclude_played: bool = True,
    state: AppState = Depends(get_state),
):
    unknown = [s for s in seed if s not in state.hybrid.content]
    if unknown:
        raise HTTPException(404, f"Unknown seed track_id(s) {unknown}")
    res = state.hybrid.recommend(user_id, k, mode, db.user_events(state.db, user_id), seed, exclude_played)
    return UserRecResponse(user_id=user_id, mode=res.mode, cold_start=res.cold_start, n_plays=res.n_plays,
                           weights=Weights(**res.weights), items=rec_items(state, res.items))


@router.post("/events", response_model=EventOut, status_code=201)
def log_event(event: EventIn, state: AppState = Depends(get_state)):
    if event.track_id not in state.hybrid.content:
        raise HTTPException(404, f"Unknown track_id {event.track_id}")
    e = db.add_event(state.db, event.user_id, event.track_id, event.event_type,
                     datetime.now(timezone.utc).replace(tzinfo=None))
    return EventOut(id=e.id, user_id=e.user_id, track_id=e.track_id, event_type=e.event_type, weight=e.weight,
                    ts=e.ts.replace(tzinfo=timezone.utc), source=e.source)


@router.post("/admin/reload", response_model=Health)
def reload(request: Request):
    """Load the latest artifacts from disk (after build_features / train_cf) without restarting."""
    request.app.state.recsys = load_state(request.app.state.recsys.cfg)
    return health_of(request.app.state.recsys)
