"""Synthetic users and plays over the real track catalog (PRD: Data strategy).

Affinity of user u for track i:
    a_ui = visible_share * vis_ui + (1 - visible_share) * hid_ui
vis: match on audio features, decade and favourite artists (what the content model can see).
hid: match on random per-track and per-artist factors that no content feature shows.
Both are z-scored per user over the catalog. Plays are sampled with
    P(i | u) ∝ exp(beta_affinity * a_ui) * exp(beta_popularity * z(log popularity prior)).
Persona labels go to users.csv for self-checks only; they are never model input.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.data.schema import AUDIO_COLUMNS

# name: (audio targets on standardised features, preferred decades)
PERSONAS: dict[str, tuple[dict[str, float], list[int]]] = {
    "dance_pop":    ({"danceability": 1.0, "energy": 0.8, "valence": 0.5}, [2000, 2010]),
    "quiet_acoustic": ({"acousticness": 1.2, "energy": -1.0, "loudness": -0.8}, [1960, 1970, 2010]),
    "classical":    ({"instrumentalness": 1.5, "acousticness": 1.0, "speechiness": -0.5}, [1920, 1930, 1940, 1950]),
    "hip_hop":      ({"speechiness": 1.5, "danceability": 0.8}, [1990, 2000, 2010]),
    "rock":         ({"energy": 1.2, "loudness": 1.0, "acousticness": -1.0}, [1970, 1980, 1990]),
    "mellow_jazz":  ({"acousticness": 0.8, "instrumentalness": 0.5, "tempo": -0.5, "energy": -0.5}, [1940, 1950, 1960]),
    "happy_oldies": ({"valence": 1.2, "danceability": 0.5}, [1950, 1960]),
    "sad_ballads":  ({"valence": -1.0, "energy": -0.8, "acousticness": 0.5}, [1980, 1990, 2000, 2010]),
    "edm":          ({"energy": 1.0, "danceability": 0.8, "instrumentalness": 0.6, "tempo": 0.5}, [2010, 2020]),
    "live_lovers":  ({"liveness": 1.5}, [1960, 1970, 1980]),
    "metal":        ({"energy": 1.5, "loudness": 1.2, "valence": -0.8}, [1980, 1990, 2000]),
    "chill_lofi":   ({"energy": -0.6, "instrumentalness": 0.8, "danceability": 0.3}, [2010, 2020]),
}
BUCKETS = [(1, 4), (5, 19), (20, 99), (100, None)]


def zscore(x: np.ndarray, axis=0) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-9)


@dataclass
class World:
    """Per-track inputs shared by all users."""
    track_ids: np.ndarray
    vis: np.ndarray        # (n_tracks, n_personas) z-scored persona match
    audio: np.ndarray      # (n_tracks, 9) standardised audio, for per-user noise
    hidden: np.ndarray     # (n_tracks, d) hidden track + artist factors
    log_prior: np.ndarray  # (n_tracks,) z-scored log popularity prior
    favourites: dict[str, list[str]]


def build_world(tracks: pd.DataFrame, p: dict, rng: np.random.Generator) -> World:
    audio = zscore(tracks[AUDIO_COLUMNS].to_numpy(np.float32))
    decade = (tracks["year"].to_numpy() // 10) * 10
    primary = tracks["artists"].map(lambda a: a[0]).to_numpy()

    vis, favourites = [], {}
    for name, (targets, decades) in PERSONAS.items():
        cols = [AUDIO_COLUMNS.index(f) for f in targets]
        audio_match = zscore(-((audio[:, cols] - np.array(list(targets.values()))) ** 2).sum(axis=1))
        era = np.isin(decade, decades)
        # favourite artists: a seeded draw from the 50 best-matching artists with >= 10 tracks
        by_artist = pd.Series(audio_match + era).groupby(primary).agg(["mean", "size"])
        top = by_artist[by_artist["size"] >= 10].sort_values("mean", ascending=False).index[:50]
        favourites[name] = sorted(map(str, rng.choice(top, size=min(p["favourite_artists"], len(top)), replace=False)))
        fav = np.isin(primary, favourites[name])
        vis.append(zscore(audio_match + p["decade_bonus"] * era + p["artist_bonus"] * fav))

    d = p["hidden_dims"]
    _, artist_idx = np.unique(primary, return_inverse=True)
    artist_f = rng.normal(size=(artist_idx.max() + 1, d))
    hidden = ((rng.normal(size=(len(tracks), d)) + artist_f[artist_idx]) / np.sqrt(2)).astype(np.float32)

    log_prior = zscore(np.log((tracks["popularity"].to_numpy() + 1) / 101))
    return World(tracks["track_id"].to_numpy(), np.column_stack(vis).astype(np.float32), audio, hidden,
                 log_prior, favourites)


def draw_users(n_users: int, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    names = list(PERSONAS)
    shares = np.array(p["bucket_shares"])
    bucket = rng.choice(len(BUCKETS), size=n_users, p=shares / shares.sum())
    n_plays = np.empty(n_users, dtype=np.int64)
    for b, (lo, hi) in enumerate(BUCKETS):
        m = bucket == b
        if hi is None:  # heavy tail: lo * (1 + Lomax(a)), capped
            n = lo * (1 + rng.pareto(p["tail_shape"], m.sum()))
            n_plays[m] = np.minimum(n, p["max_plays"]).astype(int)
        elif lo == 1:
            n_plays[m] = rng.integers(lo, hi + 1, m.sum())
        else:  # log-uniform
            n_plays[m] = np.exp(rng.uniform(np.log(lo), np.log(hi + 1), m.sum())).astype(int)
    rows = []
    for u in range(n_users):
        k = 2 if rng.random() < p["two_persona_share"] else 1
        ps = rng.choice(len(names), size=k, replace=False)
        w = [1.0] if k == 1 else [(w0 := rng.uniform(0.5, 0.9)), 1 - w0]
        rows.append({"user_id": f"u_{u:05d}", "primary_persona": names[ps[0]],
                     "personas": "|".join(f"{names[i]}:{wi:.2f}" for i, wi in zip(ps, w)),
                     "bucket": f"{BUCKETS[bucket[u]][0]}+" if BUCKETS[bucket[u]][1] is None
                     else f"{BUCKETS[bucket[u]][0]}-{BUCKETS[bucket[u]][1]}",
                     "n_plays": int(n_plays[u]), "_mix": dict(zip(ps.tolist(), w))})
    return pd.DataFrame(rows)


def simulate(tracks: pd.DataFrame, p: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, World]:
    """Returns (interactions, users, world). Deterministic for a given seed and params."""
    rng = np.random.default_rng(seed)
    world = build_world(tracks, p, rng)
    users = draw_users(p["n_users"], p, rng)
    n_p, d = world.vis.shape[1], world.hidden.shape[1]
    persona_hidden = rng.normal(size=(n_p, d)) / np.sqrt(d)
    vs = p["visible_share"]
    start = pd.Timestamp(p["start"]).value // 10**9
    window = p["days"] * 86400
    prior_logit = p["beta_popularity"] * world.log_prior
    prior_p = np.exp(prior_logit - prior_logit.max())
    prior_p /= prior_p.sum()

    events = []
    chunk = 64  # users per batch: a few (n_tracks x chunk) float32 matrices, ~45 MB each
    for c0 in range(0, len(users), chunk):
        part = users.iloc[c0:c0 + chunk]
        mix = np.zeros((len(part), n_p), dtype=np.float32)
        for r, m in enumerate(part["_mix"]):
            for i, w in m.items():
                mix[r, i] = w
        noise = rng.normal(scale=p["user_noise"], size=(len(part), world.audio.shape[1])).astype(np.float32)
        vis = zscore(world.vis @ mix.T + world.audio @ noise.T)                       # (tracks, users)
        z = mix @ persona_hidden + rng.normal(scale=p["hidden_user_noise"] / np.sqrt(d), size=(len(part), d))
        hid = zscore(world.hidden @ z.T.astype(np.float32))
        aff = vs * vis + (1 - vs) * hid
        logits = p["beta_affinity"] * aff + prior_logit[:, None]
        for r, (uid, n) in enumerate(zip(part["user_id"], part["n_plays"])):
            col = logits[:, r].astype(np.float64)
            prob = np.exp(col - col.max())
            prob /= prob.sum()
            items = rng.choice(len(prob), size=n, p=prob)
            events.append(_user_events(uid, items, aff[:, r], world, prior_p, p, rng, start, window))

    interactions = pd.concat(events, ignore_index=True)
    return interactions, users.drop(columns="_mix"), world


def _user_events(uid, items, aff, world, prior_p, p, rng, start, window) -> pd.DataFrame:
    """Plays in sessions, likes by affinity, and skips on popularity-driven exposures."""
    n = len(items)
    n_sessions = max(1, int(np.ceil(n / p["session_length"])))
    session_start = np.sort(rng.uniform(0, window - 6 * 3600, n_sessions))
    session = np.sort(rng.integers(0, n_sessions, n))
    pos = np.arange(n) - np.searchsorted(session, session)  # position within session
    t = start + session_start[session] + pos * rng.uniform(150, 260, n)
    like = rng.random(n) < p["like_max"] / (1 + np.exp(-2 * (aff[items] - p["like_threshold"])))

    # skips: tracks surfaced by popularity alone; skipped more often the lower the affinity
    m = rng.binomial(n, p["skip_exposure_rate"])
    exp_items = rng.choice(len(prior_p), size=m, p=prior_p)
    skip = rng.random(m) < 1 / (1 + np.exp(2 * aff[exp_items]))
    skip_t = start + rng.uniform(0, window, m)

    ts = np.concatenate([t, t[like] + 30, skip_t[skip]])
    track = np.concatenate([items, items[like], exp_items[skip]])
    kind = ["play"] * n + ["like"] * int(like.sum()) + ["skip"] * int(skip.sum())
    return pd.DataFrame({"user_id": uid, "track_id": world.track_ids[track], "event_type": kind,
                         "weight": 1, "ts": pd.to_datetime(ts.astype(np.int64), unit="s", utc=True),
                         "source": "simulated"})


def self_checks(inter: pd.DataFrame, users: pd.DataFrame, n_tracks: int, p: dict, seed: int) -> dict:
    """Bucket histogram, popularity long tail, density, same-persona co-play vs chance."""
    checks = {}
    plays = inter[inter["event_type"] == "play"]
    per_user = plays.groupby("user_id").size().reindex(users["user_id"], fill_value=0)
    edges = [(lo, hi or np.inf) for lo, hi in BUCKETS]
    observed = [float(((per_user >= lo) & (per_user <= hi)).mean()) for lo, hi in edges]
    target = [s / sum(p["bucket_shares"]) for s in p["bucket_shares"]]
    checks["activity_buckets"] = {"buckets": ["1-4", "5-19", "20-99", "100+"], "observed": observed,
                                  "target": target,
                                  "pass": bool(max(abs(o - t) for o, t in zip(observed, target)) < 0.02)}

    per_track = plays.groupby("track_id").size().sort_values(ascending=False).to_numpy()
    top1 = per_track[: max(1, len(per_track) // 100)].sum() / per_track.sum()
    cum = np.cumsum(np.sort(per_track))
    gini = float(1 - 2 * (cum / cum[-1]).sum() / len(per_track) + 1 / len(per_track))
    checks["popularity_long_tail"] = {"tracks_played": int(len(per_track)), "top1pct_play_share": float(top1),
                                      "share_tracks_played_once": float((per_track == 1).mean()),
                                      "gini": gini, "pass": bool(gini > 0.5 and top1 > 0.05)}

    pairs = plays[["user_id", "track_id"]].drop_duplicates()
    density = len(pairs) / (len(users) * n_tracks)
    checks["density"] = {"user_track_pairs": int(len(pairs)), "density": density, "pass": bool(density < 0.01)}

    rng = np.random.default_rng(seed + 1)
    sets = pairs.groupby("user_id")["track_id"].apply(set)
    active = users[users["user_id"].isin(sets.index[sets.map(len) >= 20])]
    by_persona = active.groupby("primary_persona")["user_id"].apply(list)

    def jaccard(a, b):
        return len(sets[a] & sets[b]) / len(sets[a] | sets[b])

    same, rand = [], []
    ids = active["user_id"].to_numpy()
    for _ in range(3000):
        group = by_persona.iloc[rng.integers(len(by_persona))]
        if len(group) >= 2:
            a, b = rng.choice(group, 2, replace=False)
            same.append(jaccard(a, b))
        a, b = rng.choice(ids, 2, replace=False)
        rand.append(jaccard(a, b))
    ratio = float(np.mean(same) / max(np.mean(rand), 1e-12))
    checks["persona_coplay"] = {"same_persona_jaccard": float(np.mean(same)), "random_pair_jaccard": float(np.mean(rand)),
                                "ratio": ratio, "pass": bool(ratio > 1.5)}
    return checks
