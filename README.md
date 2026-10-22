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
| Tests | `pytest -q` |
