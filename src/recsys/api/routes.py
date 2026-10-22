from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from recsys.api.deps import AppState, get_state, load_state
from recsys.api.schemas import (EventIn, EventOut, Health, HistoryItem, RecItem, SimilarResponse, TrackOut,
                                UserRecResponse)
from recsys.engines.content import Rec
from recsys.store import db
from recsys.store.models import Track

router = APIRouter()


def track_out(t: Track) -> TrackOut:
    return TrackOut(track_id=t.track_id, track_name=t.name, artists=t.artists, genres=t.genres,
                    year=t.year, popularity=t.popularity, audio=t.audio)


def rec_items(state: AppState, recs: list[Rec]) -> list[RecItem]:
    meta = db.get_tracks(state.db, [r.track_id for r in recs])
    return [
        RecItem(track_id=r.track_id, track_name=meta[r.track_id].name, artist_name=", ".join(meta[r.track_id].artists),
                genre=", ".join(meta[r.track_id].genres[:3]), year=meta[r.track_id].year,
                popularity=meta[r.track_id].popularity, audio=meta[r.track_id].audio, score=r.score,
                similarity=r.similarity)
        for r in recs
    ]


def health_of(state: AppState) -> Health:
    return Health(status="ok", model_version=state.version, scale_tier=state.cfg["scale"]["tier"],
                  n_tracks=len(state.content.track_ids))


def require_known(state: AppState, track_ids: list[str]) -> None:
    unknown = [t for t in track_ids if t not in state.content]
    if unknown:
        raise HTTPException(404, f"Unknown track_id(s) {unknown}")


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
    require_known(state, [track_id])
    return SimilarResponse(seed_track_id=track_id, items=rec_items(state, state.content.similar(track_id, k)))


@router.get("/recommend/user/{user_id}", response_model=UserRecResponse)
def recommend_user(
    user_id: str,
    k: int = Query(10, ge=1, le=100),
    mode: Literal["content", "popular"] = "content",
    seed: list[str] = Query(default=[], description="Track ids the user picked (onboarding); added to the profile"),
    exclude_played: bool = True,
    state: AppState = Depends(get_state),
):
    """Content profile = mean vector of the user's played/liked tracks (from /events) plus any seeds.
    Skipped tracks are never recommended. No history and no seeds -> most popular tracks."""
    require_known(state, seed)
    events = db.user_events(state.db, user_id)
    history = [t for t, e in events if e in ("play", "like")]
    skipped = [t for t, e in events if e == "skip"]
    res = state.content.for_user(history, k, seeds=seed, avoid=skipped, mode=mode, exclude_history=exclude_played)
    return UserRecResponse(user_id=user_id, mode=res.mode, cold_start=res.cold_start,
                           n_history=len(set(history)), items=rec_items(state, res.items))


@router.get("/users/{user_id}/history", response_model=list[HistoryItem])
def history(user_id: str, limit: int = Query(50, ge=1, le=500), state: AppState = Depends(get_state)):
    """The user's logged events (plays, likes, skips), newest first."""
    return [HistoryItem(event_type=e.event_type, ts=e.ts.replace(tzinfo=timezone.utc), track=track_out(t))
            for e, t in db.user_history(state.db, user_id, limit)]


@router.post("/events", response_model=EventOut, status_code=201)
def log_event(event: EventIn, state: AppState = Depends(get_state)):
    require_known(state, [event.track_id])
    e = db.add_event(state.db, event.user_id, event.track_id, event.event_type,
                     datetime.now(timezone.utc).replace(tzinfo=None))
    return EventOut(id=e.id, user_id=e.user_id, track_id=e.track_id, event_type=e.event_type, weight=e.weight,
                    ts=e.ts.replace(tzinfo=timezone.utc), source=e.source)


@router.post("/admin/reload", response_model=Health)
def reload(request: Request):
    """Load the latest artifacts from disk (after build_features) without restarting."""
    request.app.state.recsys = load_state(request.app.state.recsys.cfg)
    return health_of(request.app.state.recsys)
