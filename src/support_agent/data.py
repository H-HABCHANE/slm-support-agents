"""Chargement du dataset Bitext et découpage train / validation / test reproductible."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from .config import load_config, resolve


@dataclass
class Split:
    texts: list[str]
    labels: list[str]


@dataclass
class Splits:
    train: Split
    val: Split
    test: Split
    labels: list[str]          # liste ordonnée des intentions


def load_bitext_splits(cfg: dict | None = None) -> Splits:
    from datasets import load_dataset

    cfg = cfg or load_config()
    d = cfg["data"]
    ds = load_dataset(d["dataset"], split="train")
    texts, labels = list(ds["instruction"]), list(ds["intent"])
    seed = cfg["seed"]

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        texts, labels, test_size=d["test_size"], stratify=labels, random_state=seed
    )
    val_frac = d["val_size"] / (1 - d["test_size"])
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed
    )
    return Splits(
        Split(x_train, y_train), Split(x_val, y_val), Split(x_test, y_test), sorted(set(labels))
    )


def load_realistic_tickets(cfg: dict | None = None) -> list[dict]:
    """Tickets écrits à la main, plus longs et plus bruités que Bitext (évaluation hors distribution)."""
    cfg = cfg or load_config()
    path = resolve(cfg["data"]["realistic_tickets"])
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- augmentation
GREETINGS = ["", "Hi,", "Hello,", "Hello NovaShop team,", "Good morning,", "Dear customer service,", "hey"]
CONTEXT = [
    "", "", "I've been a customer for years.", "I ordered a {product} last week (order NS-{num}).",
    "This is quite urgent for me.", "I already tried your website but couldn't find the answer.",
    "My order number is NS-{num}.", "I bought a {product} from you recently.",
    "Sorry if this is the wrong place to ask.",
]
CLOSINGS = ["", "", "Thanks.", "Thank you in advance!", "Best regards, {name}", "Thanks, {name}", "cheers"]
PRODUCTS = ["blender", "jacket", "laptop", "phone case", "coffee machine", "pair of shoes", "sofa", "toaster"]
NAMES = ["Julie", "Marc", "Sam", "Nadia", "Paul", "Lea"]


def augment_ticket(text: str, rng) -> str:
    """Enveloppe une phrase Bitext dans un e-mail réaliste : le modèle doit apprendre à ignorer ce bruit."""
    def fill(s: str) -> str:
        return s.format(product=rng.choice(PRODUCTS), num=rng.randint(100000, 999999), name=rng.choice(NAMES))

    text = re.sub(r"\{\{Order Number\}\}", lambda _: f"NS-{rng.randint(100000, 999999)}", text)
    text = re.sub(r"\{\{([^}]+)\}\}", lambda m: m.group(1).lower(), text)  # autres champs : texte simple
    parts = [rng.choice(GREETINGS), fill(rng.choice(CONTEXT)), text[0].upper() + text[1:] + ".",
             fill(rng.choice(CONTEXT)), fill(rng.choice(CLOSINGS))]
    return " ".join(p for p in parts if p)


def augment_split(texts: list[str], labels: list[str], seed: int, ratio: float = 0.5) -> tuple[list[str], list[str]]:
    """Ajoute une version augmentée pour une partie des exemples (les originaux sont conservés)."""
    import random

    rng = random.Random(seed)
    idx = [i for i in range(len(texts)) if rng.random() < ratio]
    return texts + [augment_ticket(texts[i], rng) for i in idx], labels + [labels[i] for i in idx]
