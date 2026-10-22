# Decision log

Non-obvious choices and their reasons. The PRD's nine key decisions are in `docs/PRD.md`; this file records what we decided while building.

## Phase 0

- **Python 3.13 venv.** It is the default interpreter on the dev machine, and `implicit 0.7.3` ships a prebuilt `cp313-win_amd64` wheel, so no C++ compiler is needed and there is no fallback for `implicit`.
- **Installed versions (2026-10-22):** implicit 0.7.3, fastapi 0.159.1, uvicorn 0.54.0, pydantic 2.14.1, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, sqlalchemy 2.1.6, pyarrow 26.0.0, pyyaml 6.0.3, pytest 9.1.1.
- **No Makefile.** `make` isn't installed on the Windows dev machine, so the README lists plain `python scripts/...` commands, which run the same everywhere.
- **`httpx` as a dev dependency.** FastAPI's `TestClient` needs it. Only the tests use it.
- **Column checks compare sets, not order.** CSV column order isn't part of the contract. The audit fails on *missing* expected columns and only reports extra ones.
- **Required files are `data.csv` and `data_w_genres.csv`.** These are the tracks with audio features and the artist→genres map. The other files in the Kaggle dataset (`data_by_artist`, `data_by_genres`, `data_by_year`) are aggregates we don't use. The audit still prints them if they're present.

- **Pinned dataset (Gate 1, 2026-10-22).** Kaggle `vatsalmavani/spotify-dataset`, `archive.zip` sha256 `f6a94ca95697e0f792af2fa2876e0fcc0d64ef90c7d6d2677185bcdae1903130`. `data.csv`: 170,653 rows, 19 columns, all `id` values unique and non-null, years 1921–2020, no nulls, sha256 `c97c4234…77ded7f8`. `data_w_genres.csv`: 28,680 artists, unique names, sha256 `4b0a9548…0de3a829`. The columns match the PRD. The licence is not yet confirmed (the Kaggle page needs a login).
- **Genre coverage.** 9,857 of 28,680 artists have an empty genre list. After joining on every listed artist, 154,535 of 170,653 tracks (90.6%) get at least one of 2,972 genres. The other 9.4% rely on audio and decade only.
- **Duplicate titles confirmed.** 56,118 rows share a name with another row, across 19,103 distinct names. This backs PRD decision 8 (key everything by `track_id`).

## PRD amendments (approved with the plan, 2026-10-22)

1. **Split for light users.** Per user, ordered by time: test = last `max(1, round(0.1n))` if n ≥ 2, validation = the `max(1, round(0.1n))` before it if n ≥ 3, the rest is train. A plain 10% would give the 1–4 play bucket no test items. Evaluation buckets use the train count.
2. **Cold-item set.** A seeded 2% of interacted tracks, with all their interactions moved to test. "No train interactions" would cover about 150k tracks.
3. **Blend scores the union.** Both engines score every candidate in the union of their top lists, then each is min-maxed over the union. Missing scores aren't treated as 0.
4. **n in w_cf is the count of training play events.** Likes add ALS confidence but don't move the blend weight.
5. **Multi-artist tracks get the union of all their artists' genres.**
6. **Unit tests use tiny hand-written fixtures.** Gate checks and scripts run only on the real CSVs.
