"""Train collaborative filtering, check it against popularity (Gate 2), save the serving model.

Usage: python scripts/train_cf.py [--data data/synthetic]
1. Temporal split of the simulated interactions (see recsys/eval/split.py).
2. Fit popularity, item-KNN and ALS on train; report Recall/Hit/NDCG@10 on VALIDATION by user bucket.
   The test split is not touched here; it is reported once, by scripts/evaluate.py.
3. Refit ALS on all interactions (plus live events with --include-live) and save it, with the per-user
   history the API serves from, to artifacts/cf/. Then POST /admin/reload to serve it.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import DEFAULT_CONFIG, load_config, resolve
from recsys.engines.collaborative import ALSModel, ItemKNN
from recsys.engines.hybrid import aggregate_history
from recsys.engines.popularity import PopularityEngine
from recsys.store.db import live_events_frame, make_engine
from recsys.eval.metrics import evaluate
from recsys.eval.split import POSITIVE, targets, temporal_split

BUCKETS = [("1-4", 1, 4), ("5-19", 5, 19), ("20+", 20, np.inf)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--include-live", action="store_true", help="add live events from SQLite to the serving model")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = args.data or resolve(cfg, "synthetic")
    p, k, seed = cfg["cf"], cfg["eval"]["k"], cfg["seed"]
    t0 = time.perf_counter()

    inter = pd.read_parquet(data_dir / "interactions.parquet")
    inter = inter[inter["source"] == "simulated"]
    split = temporal_split(inter, cfg["eval"]["cold_item_share"], seed)
    val_targets = targets(split.val, split.train)
    print(f"data: {data_dir} | tier {cfg['scale']['tier']} | {inter['user_id'].nunique():,} users, "
          f"{len(inter):,} rows (layer: simulated)")
    print(f"split: train {len(split.train):,} / val {len(split.val):,} / test {len(split.test):,} positive events; "
          f"{len(split.cold_items):,} cold items; {len(val_targets):,} users with validation targets")

    seen = split.train.groupby("user_id")["track_id"].apply(set).to_dict()
    pop = PopularityEngine.from_interactions(split.train)
    knn = ItemKNN(split.train, p["event_weights"], p["knn_neighbours"])
    als = ALSModel.fit(split.train, p, seed)
    knn_recs = knn.recommend_batch(list(val_targets), k)
    models = {
        "popularity": lambda u: [t for t, _ in pop.recommend(k, seen.get(u, set()))],
        "item-knn": lambda u: knn_recs.get(u, []),
        "als": lambda u: [t for t, _ in als.recommend(u, k, seen.get(u, ()))],
    }

    train_plays = split.train[split.train["event_type"] == "play"].groupby("user_id").size()
    rows = []
    for bucket, lo, hi in [("all", 0, np.inf)] + BUCKETS:
        tg = {u: s for u, s in val_targets.items() if lo <= train_plays.get(u, 0) <= hi}
        for name, rec in models.items():
            rows.append({"bucket": bucket, "model": name, **evaluate(rec, tg, k)})
    table = pd.DataFrame(rows)
    print(f"\nValidation, k={k}:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    overall = table[table["bucket"] == "all"].set_index("model")[f"ndcg@{k}"]
    ok = overall["als"] > overall["popularity"]
    print(f"\nGate 2: ALS NDCG@{k} {overall['als']:.4f} vs popularity {overall['popularity']:.4f} -> "
          f"{'PASS' if ok else 'FAIL'}")

    serve = inter
    if args.include_live:  # live events train the served model only; reported numbers stay simulated-only
        live = live_events_frame(make_engine(resolve(cfg, "db")))
        serve = pd.concat([inter, live], ignore_index=True)
        print(f"including {len(live):,} live events from {resolve(cfg, 'db')} in the serving model")
    full = ALSModel.fit(serve[serve["event_type"].isin(POSITIVE)], p, seed)
    out = resolve(cfg, "cf")
    full.save(out)
    aggregate_history(serve).to_parquet(out / "history.parquet", index=False)
    meta = {"version": f"cf-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "data": str(data_dir),
            "sources": sorted(serve["source"].unique()), "n_users": len(full.user_ids),
            "n_items": len(full.item_ids), "params": p, "validation": table.to_dict("records")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"serving model ({len(full.user_ids):,} users x {len(full.item_ids):,} items) -> {out}  "
          f"({time.perf_counter() - t0:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
