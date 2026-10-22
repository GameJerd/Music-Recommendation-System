from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """cfg['paths'][key] as a Path; relative paths are taken from the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def anchor_k(w_max: float, n_anchor: int = 5, cf_at_anchor: float = 0.3) -> float:
    """k that keeps w_cf(n_anchor) == cf_at_anchor for a given w_max (PRD: k = 5 * (w_max/0.3 - 1))."""
    return n_anchor * (w_max / cf_at_anchor - 1)
