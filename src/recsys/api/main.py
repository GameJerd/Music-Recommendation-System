"""Run with: uvicorn recsys.api.main:app"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from recsys.api.deps import load_state
from recsys.api.routes import router
from recsys.config import load_config


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recsys = load_state(cfg)
        yield

    app = FastAPI(title="Hybrid music recommender", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
