"""Generate synthetic users and interactions over the real catalog.

Usage: python scripts/simulate_users.py [--visible-share 0.5] [--seed 42] [--n-users 10000] [--out data/synthetic]
Writes interactions.parquet, users.csv and manifest.json (seed, params, self-checks) to --out.
Exits 1 if any self-check fails.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from recsys.config import DEFAULT_CONFIG, load_config, resolve
from recsys.data.loaders import load_tracks
from recsys.data.simulate import self_checks, simulate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--visible-share", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-users", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = dict(cfg["simulation"])
    if args.visible_share is not None:
        p["visible_share"] = args.visible_share
    if args.n_users is not None:
        p["n_users"] = args.n_users
    seed = cfg["seed"] if args.seed is None else args.seed
    out = args.out or resolve(cfg, "synthetic")

    t0 = time.perf_counter()
    tracks = load_tracks(resolve(cfg, "raw")).rename(columns={"id": "track_id"})
    inter, users, world = simulate(tracks, p, seed)
    out.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(out / "interactions.parquet", index=False)
    users.to_csv(out / "users.csv", index=False)

    checks = self_checks(inter, users, len(tracks), p, seed)
    manifest = {
        "seed": seed, "params": p, "n_tracks_catalog": len(tracks), "n_users": len(users),
        "rows": int(len(inter)), "rows_by_event": inter["event_type"].value_counts().to_dict(),
        "favourite_artists": world.favourites, "checks": checks,
        "all_checks_pass": all(c["pass"] for c in checks.values()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(users):,} users, {len(inter):,} rows {manifest['rows_by_event']} -> {out}  "
          f"({time.perf_counter() - t0:.0f}s)")
    for name, c in checks.items():
        detail = {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v] if isinstance(v, list)
                      and v and isinstance(v[0], float) else v) for k, v in c.items() if k != "pass"}
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name}: {detail}")
    return 0 if manifest["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
