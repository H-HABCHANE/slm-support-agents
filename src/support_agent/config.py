"""Chargement de la configuration YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Chemin relatif à la racine du projet."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
