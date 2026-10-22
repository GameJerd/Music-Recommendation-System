"""Phase 0 audit of the Kaggle Spotify 1921-2020 CSVs in data/raw/.

Usage: python scripts/audit_dataset.py [--raw data/raw]
Exits 1 if a required file is missing, track ids are not unique, or columns differ.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from recsys.config import ROOT
from recsys.data.schema import EXPECTED_COLUMNS, REQUIRED_FILES, TRACKS_FILE


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_file(path: Path) -> list[str]:
    """Print the audit for one CSV and return a list of problems."""
    problems = []
    df = pd.read_csv(path)
    print(f"\n=== {path.name} ===")
    print(f"sha256: {sha256(path)}")
    print(f"rows: {len(df):,}   columns: {len(df.columns)}")
    print("\ncolumn dtypes and nulls:")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum()}).to_string())

    if "name" in df.columns:
        dup_names = df["name"].duplicated(keep=False)
        print(f"\nrows sharing a track name with another row: {dup_names.sum():,} "
              f"({df.loc[dup_names, 'name'].nunique():,} distinct names)")
    if "id" in df.columns:
        n_unique = df["id"].nunique()
        print(f"id: {n_unique:,} unique of {len(df):,} rows, {df['id'].isna().sum()} null")
        if n_unique != len(df) or df["id"].isna().any():
            problems.append(f"{path.name}: id is not a unique, non-null key")
    if "year" in df.columns:
        print(f"year range: {df['year'].min()} - {df['year'].max()}")

    expected = EXPECTED_COLUMNS.get(path.name)
    if expected is not None:
        missing, extra = expected - set(df.columns), set(df.columns) - expected
        print(f"columns vs PRD/schema: missing={sorted(missing) or 'none'} extra={sorted(extra) or 'none'}")
        if missing:
            problems.append(f"{path.name}: missing expected columns {sorted(missing)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    csvs = sorted(args.raw.glob("*.csv"))
    print(f"files in {args.raw}: {[p.name for p in csvs] or 'none'}")
    missing = [f for f in REQUIRED_FILES if not (args.raw / f).exists()]
    if missing:
        print(f"\nMISSING required files: {missing}")
        print("Download them from https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset "
              f"and place them in {args.raw}")
        return 1

    problems = []
    for path in csvs:
        problems += audit_file(path)

    print("\n=== Gate 1 ===")
    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1
    print(f"PASS  {TRACKS_FILE}: id unique; required columns present in {REQUIRED_FILES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
