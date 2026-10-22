# Decision log

Non-obvious choices and their reasons. The PRD's nine key decisions are in `docs/PRD.md`; this file records what we decided while building.

## Scope change: content-based only (your decision, 2026-10-22)

- **Why.** There is no real listening data for this catalog. Collaborative filtering could only train on users we simulated, so its numbers showed whether the pipeline recovered what we built in, not quality for real listeners. Content-based recommendation works on real data alone, from the first request. The hybrid also had an open blend issue (see Phase 3).
- **What was removed.** The user simulator, ALS, item-KNN, the hybrid blend and its 70/30 anchor, `train_cf.py`, Gates 2–3, and the `implicit` dependency. The full hybrid is preserved at git tag **`hybrid-v1`** (`git checkout hybrid-v1`). The Phase 2–3 entries below are kept as history.
- **What `/recommend/user` does now.** The profile is the mean content vector of the tracks the user played or liked through `POST /events`, plus any `seed` tracks they pick. Skipped tracks are never recommended. With no history and no seeds, the user gets the dataset's most popular tracks (its real `popularity` column), with `cold_start: true`.
- **Artist cap (default 2 per primary artist per list).** Genres are per artist, so before the cap 49% of a "similar" list was the seed's own artist (4.5 distinct artists in 10). With the cap it's 16%, with 7.4 distinct artists (`reports/content_eval.md`). The cap applies to the primary (first-listed) artist, so features like "Tony Yayo, Eminem" still pass for an Eminem seed.
- **Popularity boost: built, off by default.** `score = cosine + boost · popularity/100`. At 0.1 it raises the mean popularity of results from 32.6 to 37.1 and barely changes the other proxies. Without users there's no evidence either way, so the default stays at pure similarity and the knob is in config.
- **Block weights stay at 0.5 / 0.35 / 0.15, untuned.** Tuning needs a signal of what listeners like, and the only signals left are proxies that the weights themselves decide (for example, a bigger genre weight always means more genre match). The ablation shows what each block contributes instead.
- **Evaluation is list-quality proxies, not accuracy** (`scripts/evaluate.py`). It uses 2,000 seeded random seed tracks and a random-tracks reference row. Genre match is a fair test only for variants that don't use genre as a feature, and the report marks the others as circular. The fair result is **audio-only lists share a genre with the seed 27% of the time, against 7% for random tracks**, so audio features alone carry real genre signal.
- **The era block is very strong.** With decade in the features, 97–100% of results come from the seed's decade (11% for random tracks). The one-hot decade cosine is all-or-nothing. If cross-era results matter, lower the weight or use a smoother year feature.
- **Known gaps.** Title variants ("Lose Yourself" vs "Lose Yourself - Soundtrack Version") still appear as separate songs. A user's profile is a single mean, so mixed tastes blur. Search is substring-only.

## Web app (Slopify)

- **Plain HTML/CSS/JS served by FastAPI at `/`.** There's no framework, build step or new dependency, and it runs from the same `uvicorn` command. Static files are mounted after the API routes, so `/docs` and every endpoint still win.
- **Its own brand, a music-app layout.** The brief asked for a Spotify or Apple Music feel. We borrowed the *kind* of interface (sidebar, cover shelves, a song header backed by its own blurred cover, a pill play action) but not their branding.
- **Redesign, round 2 (your feedback: covers and icons looked awful, the purple gradient had to go).** We dropped the generated discs, rings and stripes. A cover is now a colour from an 11-colour curated palette, picked by mood × energy (earthy tones for very acoustic songs), so similar songs still look alike without the noise. The purple song header became an Apple Music-style backdrop: the song's own cover colours, blurred and faded out with a mask, behind translucent (`backdrop-filter`) panels and over a soft ambient glow. Icons are one consistent outline set. Type moved to the system font (SF Pro / Inter / Segoe).
- **Final look: Glass dark + Aura covers (your choice, 2026-10-22).** We built four themes (Glass dark, Glass light, Studio, Mono) and three cover styles (Aura, Monogram, Gradient) as live-switchable options. After comparing full-size screenshots (headless Edge, 1280×800, because the browser pane was too small), you picked Glass dark + Aura. The other options, the Appearance picker and the `?theme=&cover=` URL override were removed to keep one maintained look. They're in commit `d0a9cbd` if ever wanted.
- **"Listen on Spotify" instead of a play button.** There's no audio, and the PRD lists streaming as a non-goal. Spotify track IDs are the dataset's IDs, so the button opens the real song and logs a `play` event, which is what makes the recommendations personal. Likes can't be undone, because the API has no delete. That's acceptable for a demo.
- **Long titles split at " - "** ("Bohemian Rhapsody" plus a "Remastered 2011" line), the way music apps show versions, rather than being cut off.
- **Track responses now include `audio`** (raw features) for the covers and the "How it sounds" panel. They're stored as JSON on the `tracks` table, so the track DB must be rebuilt once with `build_features.py`. We added `GET /users/{id}/history` for "Recently played".
- **Checked in a browser** at 1280×800 and 375×812: no horizontal scroll at phone width, no console errors, and like, play and skip events update the mix and the sidebar.

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

## Phase 1

- **Content vectors are sparse CSR, not dense.** 2,972 genre columns make a dense float32 matrix of about 2 GB. Sparse, the full 170,653 × 2,992 matrix holds 2.5M non-zeros (10 MB). `BruteForceIndex` is one sparse matrix-vector product per query: exact cosine, about 7.5 ms per `/recommend/similar` call end to end.
- **Block weighting.** Each block (audio, genre, decade) is L2-normalised per row, then scaled by √w, then the whole row is renormalised. With weights summing to 1, the cosine is exactly Σ w_b · cos_b. A unit test checks this. Tracks with no genres are renormalised over audio + decade. Starting weights are audio 0.5, genre 0.35, decade 0.15, to be tuned on validation in Phase 4. A weight of 0 drops a block, which is how the ablation works.
- **Genres are multi-hot, not TF-IDF.** It's the simplest option the PRD allows. IDF weighting is a cheap Phase 4 ablation if common genres like "rock" swamp similarity.
- **No per-column audio weights.** The PRD says "weighted", but every column starts at 1 and nothing tunes them, so for now the block weight is the only knob. We'll add per-column weights if the ablation shows one feature dominating.
- **The SQLite DB lives at `data/recsys.db`, not `artifacts/`.** From Phase 3 it holds logged events, which are data, while `artifacts/` holds rebuildable outputs. Rebuilding features replaces only the `tracks` table.
- **JSON stored with `ensure_ascii=False`.** Otherwise "Beyoncé" is stored as `Beyoncé` and the LIKE search misses it. A test covers this.
- **The API fails fast at startup if artifacts are missing.** The error message names the build command.
- **Search is a substring LIKE on title or artist, sorted by popularity.** It's a full scan, fine at tier S.

- **Dedupe at serving time, catalog untouched (your decision, 2026-10-22).** 24,656 rows (14.4%) share a normalised (title, artists) key with another row, and 6,879 of those have identical audio. Results keep the best-scoring copy per key, and excluding a track (the seed, or later the played tracks) also excludes its copies. Title variants such as "- Remastered 2011" are different keys and are deliberately not merged, because that would need guessing at title cleanup. The catalog and the evaluation stay keyed by `track_id`.

## Phase 2 (historical: removed in the scope change, code at tag `hybrid-v1`)

- **Simulator model** (`src/recsys/data/simulate.py`). There are 12 hand-written personas. Each is defined by targets on standardised audio features, preferred decades, and 5 favourite artists drawn with a seed from the 50 best-matching artists that have at least 10 tracks. A user mixes 1 persona (60%) or 2 (40%, weights 0.5–0.9 / rest), plus a personal tilt on the audio features. Hidden taste uses 8-d random factors per track and per primary artist. A user's hidden vector is their personas' hidden centres plus noise, so same-persona users share hidden taste that only co-plays reveal. Visible and hidden scores are each z-scored per user over the catalog, then mixed by `visible_share`.
- **Sharpness, `beta_affinity` 5.0 rather than the first guess of 2.5.** At 2.5 each user spread plays over about 36k tracks. 488k plays landed on 129k tracks with almost no repeats, and the long-tail self-check failed (Gini 0.40). A sweep in a scratch script (128 users) gave effective support per user ≈ 36k / 5k / 1.1k at β = 2.5 / 4 / 5. We chose 5: the top 1% of tracks take about 23% of plays, Gini is 0.66, same-persona co-play is 4.3× chance, and 367k distinct user-track pairs over 488k plays means repeats exist. `beta_popularity` (2.5) matters much less, because the z-scored log prior is compressed at the top. Mean popularity of played tracks only moves from 43 to 48.
- **Activity buckets.** 1–4 is uniform. 5–19 and 20–99 are log-uniform. 100+ is 100 × (1 + Lomax(2.0)), capped at 2,000, chosen so the mean total is about 50 plays per user, which gives about 490k play rows for 10k users.
- **Events.** One row per play (weight 1). A like follows a play with probability `0.4 · sigmoid(2(a − 1))`. Skips are drawn from popularity-only exposures, at 0.2 per play, and become skips with probability `sigmoid(−2a)`. That gives 164k likes (34% of plays, on the high side; `like_max` is the knob) and 48k skips. Sessions average 8 plays at 2.5–4.3 minutes apart, inside a 90-day window starting 2026-07-01.
- **Self-check thresholds** (written to `manifest.json`): every bucket within 2 percentage points of target; Gini over played tracks > 0.5 and top-1% share > 5%; density < 1%; same-persona Jaccard > 1.5× random pairs. The script exits 1 if any check fails.
- **`implicit` confidence.** `implicit` 0.7.3 uses the stored matrix value itself as the confidence c, scaled by its own `alpha` (we read `implicit/cpu/als.py`, where the solver adds `(confidence − 1)·yyᵀ`). To get the PRD's c = 1 + α·r, we store `1 + α·r` and pass `alpha=1` to implicit.
- **r_ui = 1 × plays + 2 × likes.** A like is a stronger signal than a single play. Skips are unused in v1, as the PRD says.
- **Gate 2 is checked on validation, not test.** Test is reported once, in Phase 4, after tuning on validation. The model the API serves is refit on all positive events.
- **Evaluation doesn't dedupe copies.** Simulated plays are keyed by `track_id`, so the ground truth already treats copies as different tracks. Dedupe applies only when serving.
- **ALS defaults** (not yet tuned): 64 factors, regularisation 0.05, 15 iterations, α = 10. BLAS is limited to 1 thread while fitting, as `implicit` recommends.
- **Item-KNN scores below popularity** (NDCG@10 0.0023 vs 0.0044 on validation). Most items have only a few plays, so cosine neighbours of rare items are noise, and its recommendations scatter over 19.6k distinct items. It stays as the PRD's baseline. Phase 4 can try K, BM25 weighting or a minimum support.

## Phase 3 (historical: removed in the scope change, code at tag `hybrid-v1`)

- **The API serves user histories from `artifacts/cf/history.parquet`** (per user and track: plays, likes), written by `train_cf.py` next to the model. Live events from SQLite are merged in on each request, so n, the played-track exclusion and the content profile include them right away.
- **Fold-in for live users.** A user with live events (or a new user) gets an ALS vector from one closed-form ALS user step against the fixed item factors, with no retraining. On the full data it reproduces implicit's trained user vectors at cosine ≥ 0.999 (20 sampled users), which also confirms implicit uses λ unscaled. A unit test covers it.
- **Content profile = unweighted mean of the vectors of the user's distinct played or liked tracks.** It's simple. Weighting by r_ui is a possible Phase 4 ablation.
- **The response's `mode` is the mode actually served.** A cold-start user asking for `hybrid` gets `content` (with picked `seed` tracks) or `popular`, and `cold_start: true`. That reads more honestly than echoing the requested mode.
- **Cold-start seeding is by track ids only** (`?seed=<id>&seed=<id>`, found with `/tracks/search`). Seeding by genre (PRD) isn't built. It would need a genre → content-vector mapping, and track seeds cover the onboarding flow.
- **`/recommend/similar` blends item-item CF** (cosine of ALS item factors) at a fixed `similar_cf_weight` 0.3 when the seed has a CF vector, and is content-only otherwise. The PRD gives no weight for this endpoint, so 0.3 mirrors the user-side anchor.
- **Event `weight` is always 1 per row.** The PRD's example row `play,3` reads as an aggregated count. We log raw events and aggregate when training.
- **`train_cf.py --include-live`** adds SQLite live events to the *serving* model and history only. Reported metrics stay simulated-only, as the PRD says.
- **Open issue, proposed change pending (see the Phase 3 report):** the PRD's rule that a track with no CF vector gets its content score only makes those tracks win the blend. About 88k of 170k tracks were never played in the simulation, and 97% of hybrid top-10 slots go to them (validation NDCG@10 0.0022, vs 0.0380 for CF-only). Imputing a neutral CF score fixes it: 0 gives 0.0386 with 2% unplayed tracks in the top 10, and the pool mean gives 0.0373 with 16%.

## PRD amendments (approved with the plan, 2026-10-22; 1–4 applied to the hybrid only)

1. **Split for light users.** Per user, ordered by time: test = last `max(1, round(0.1n))` if n ≥ 2, validation = the `max(1, round(0.1n))` before it if n ≥ 3, the rest is train. A plain 10% would give the 1–4 play bucket no test items. Evaluation buckets use the train count.
2. **Cold-item set.** A seeded 2% of interacted tracks, with all their interactions moved to test. "No train interactions" would cover about 150k tracks.
3. **Blend scores the union.** Both engines score every candidate in the union of their top lists, then each is min-maxed over the union. Missing scores aren't treated as 0.
4. **n in w_cf is the count of training play events.** Likes add ALS confidence but don't move the blend weight.
5. **Multi-artist tracks get the union of all their artists' genres.**
6. **Unit tests use tiny hand-written fixtures.** Gate checks and scripts run only on the real CSVs.
