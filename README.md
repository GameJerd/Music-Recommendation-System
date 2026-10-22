# Hybrid music recommender

Content-based + collaborative-filtering music recommender with a FastAPI backend. Spec: [docs/PRD.md](docs/PRD.md). Decisions: [docs/decisions.md](docs/decisions.md).

> **The user data is simulated.** No real listening data exists for this catalog, so collaborative filtering trains on users we generate (`scripts/simulate_users.py`). Results show whether the pipeline recovers the taste structure we built in. They don't measure accuracy on real listeners.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e .[dev]
```

Download the Kaggle dataset [vatsalmavani/spotify-dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset) (a Kaggle login is needed) and put `data.csv` and `data_w_genres.csv` in `data/raw/`.

## Commands

| Step | Command |
| --- | --- |
| Audit dataset | `python scripts/audit_dataset.py` |
| Build content features + track DB | `python scripts/build_features.py` |
| Simulate users (seeded; writes `data/synthetic/` + `manifest.json`) | `python scripts/simulate_users.py` |
| Train CF, check it beats popularity, save serving model | `python scripts/train_cf.py` |
| Run the API (then open http://127.0.0.1:8000/docs) | `uvicorn recsys.api.main:app --port 8000` |
| Retrain including events logged via the API, then hot-reload | `python scripts/train_cf.py --include-live` then `POST /admin/reload` |
| Tests | `pytest -q` |

## API

Open `/docs` for the live, typed contract. Main calls:

- `GET /tracks/search?q=queen`: find a `track_id`.
- `GET /recommend/similar/{track_id}?k=10`: similar songs (content, with item-item CF blended in when the seed has plays).
- `GET /recommend/user/{user_id}?k=10&mode=hybrid`: personalised list with `weights` and a per-item `content_score` / `cf_score`. `mode` is `hybrid`, `content`, `cf` or `popular`. A new user can pass `&seed=<track_id>` to start from picked songs. Simulated users are `u_00000` … `u_09999`.
- `POST /events` `{"user_id": "...", "track_id": "...", "event_type": "play|like|skip"}`: logs a live event. It counts towards that user's next recommendations right away, with no retraining.
- `POST /admin/reload`: loads newly trained artifacts.
