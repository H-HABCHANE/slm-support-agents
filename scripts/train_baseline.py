"""Entraîne la baseline TF-IDF + régression logistique sur le split d'entraînement Bitext."""

from __future__ import annotations

import argparse
import time

from support_agent.classifier import TfidfClassifier
from support_agent.config import load_config, resolve
from support_agent.data import augment_split, load_bitext_splits


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--output", default=cfg["classifier"]["baseline_path"])
    args = ap.parse_args()
    splits = load_bitext_splits(cfg)
    x, y = splits.train.texts, splits.train.labels
    if args.augment:
        x, y = augment_split(x, y, cfg["seed"], ratio=0.5)
    t0 = time.time()
    clf = TfidfClassifier().fit(x, y)
    path = resolve(args.output)
    clf.save(path)
    print(f"Baseline entraînée sur {len(x)} exemples en {time.time() - t0:.1f} s -> {path}")


if __name__ == "__main__":
    main()
