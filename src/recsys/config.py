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
