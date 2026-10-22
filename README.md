# Content-based music recommender

Recommends songs from the Kaggle Spotify 1921–2020 dataset (170,653 tracks) by how similar they are: audio features, artist genres and release decade. It's served by a FastAPI backend with interactive docs.

Original spec: [docs/PRD.md](docs/PRD.md). It described a hybrid with collaborative filtering; see the scope note at its top. Decisions and measured results: [docs/decisions.md](docs/decisions.md).

> **No user data.** There's no real listening data for this catalog, so the system is content-based only. It works from the first request without any users. Personal recommendations come from songs a user plays, likes or picks through the API. Quality is reported as list-quality proxies (genre/era match, artist variety, popularity, coverage), not as accuracy on real listeners. The earlier hybrid version, which trained on simulated users, is preserved at git tag `hybrid-v1`.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e .[dev]
```

Download the Kaggle dataset [vatsalmavani/spotify-dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset) (a Kaggle login is needed) and put `data.csv` and `data_w_genres.csv` in `data/raw/`.

## Commands

| Step | Command | Time |
| --- | --- | --- |
| Audit the dataset (Gate 1: unique ids, expected columns) | `python scripts/audit_dataset.py` | ~5 s |
| Build content features and the track database | `python scripts/build_features.py` | ~10 s |
| Evaluate: proxies and ablations, written to `reports/content_eval.md` | `python scripts/evaluate.py` | ~1 min |
| Run the app: web app at http://127.0.0.1:8000/, API docs at `/docs` | `uvicorn recsys.api.main:app --port 8000` | |
| Tests | `pytest -q` | ~2 s |

Settings are in [config/default.yaml](config/default.yaml): block weights, artist cap, popularity boost and paths. After rebuilding features, `POST /admin/reload` picks them up without restarting the API.

## Web app

Open http://127.0.0.1:8000/ after starting the server. **Vibify** is a small music-app interface over the API (plain HTML/CSS/JS in `src/recsys/api/static/`, no build step):

- **Home:** popular songs and a search box for a new listener. Once you play or like something, it shows *Your mix* and *More like* your last song.
- **Search:** songs or artists as you type.
- **Song page:** *Listen on Spotify* (opens the track on Spotify and counts it as played), like, *Not for me* (hides it from your mix), the similar songs with their match percentage, and how the song sounds.
- **Covers are generated from each song's audio features**, because the dataset has no artwork. Mood sets the colour (sad blue to happy amber), energy the size of the disc, tempo the grooves, danceability the stripes and acousticness the grain. Songs that sound alike look alike.
- **"Listening as"** in the sidebar is the user ID your events are logged under. Change it to start a fresh listener.

There's no audio playback; streaming was out of scope in the original PRD.

## API

Open `/docs` for the live, typed contract. Main calls:

- `GET /tracks/search?q=queen`: find a `track_id` by title or artist.
- `GET /tracks/{track_id}`: track metadata.
- `GET /recommend/similar/{track_id}?k=10`: songs similar to a seed song.
- `POST /events` `{"user_id": "...", "track_id": "...", "event_type": "play|like|skip"}`: logs what a user did. Played and liked songs shape their profile, and skipped songs are never recommended to them.
- `GET /recommend/user/{user_id}?k=10`: personal list from the user's logged songs. A new user can add `&seed=<track_id>` (repeatable) to start from songs they pick. With no history and no seeds, they get the most popular songs (`cold_start: true`).
- `GET /users/{user_id}/history`: that user's logged plays, likes and skips, newest first.
- `GET /health`, `POST /admin/reload`.

## How it works

1. Each track becomes one sparse vector made of three blocks: standardised audio features (danceability, energy, valence, acousticness, instrumentalness, liveness, speechiness, loudness, tempo), the multi-hot genres of its artists, and its release decade. They're weighted 0.5 / 0.35 / 0.15, so similarity is the weighted sum of the per-block cosines.
2. Exact cosine nearest-neighbour search runs over all tracks (`BruteForceIndex` behind a `NeighborIndex` interface, so a faster index can be swapped in later). There's no song×song matrix, and a request takes about 7 ms.
3. Results are re-ranked so each recording appears once (re-releases share a title and artist) and no artist takes more than 2 of the slots.

## Limitations

- Genres are per artist, not per song, and 9% of tracks have none.
- Nine audio numbers are a coarse description of sound. There are no lyrics or audio embeddings.
- A user's taste is a single average, so mixed tastes blur together.
- Weights are hand-set. Without listener feedback there's no honest signal to tune them on.
- Title variants ("… - Remastered 2011") show as separate songs.
