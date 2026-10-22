"""Run with: uvicorn recsys.api.main:app  — the web app is at /, the API contract at /docs."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from recsys.api.deps import load_state
from recsys.api.routes import router
from recsys.config import load_config

STATIC = Path(__file__).parent / "static"


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recsys = load_state(cfg)
        yield

    app = FastAPI(title="Content-based music recommender", version="0.2.0", lifespan=lifespan)
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")  # after the API routes, so they win
    return app


app = create_app()
