"""Classification d'intention : baseline TF-IDF, SLM zero-shot et SLM fine-tuné en LoRA.

Les trois classifieurs partagent la même interface : predict(texts) -> list[(intention, confiance)].
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from .config import load_config, resolve

Prediction = tuple[str, float]


class IntentClassifier(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


# --------------------------------------------------------------------------- baseline
class TfidfClassifier:
    """Baseline NLP classique : TF-IDF (mots + n-grammes de caractères) et régression logistique."""

    def __init__(self, pipeline=None, name: str = "tfidf_logreg"):
        self.pipeline = pipeline
        self.name = name

    @staticmethod
    def build():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])
        return Pipeline([("tfidf", features), ("clf", LogisticRegression(max_iter=2000, C=5.0))])

    def fit(self, texts: list[str], labels: list[str]) -> TfidfClassifier:
        self.pipeline = self.build().fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        proba = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        return [(str(classes[p.argmax()]), float(p.max())) for p in proba]

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path | None = None, name: str = "tfidf_logreg") -> TfidfClassifier:
        import joblib

        path = resolve(path or load_config()["classifier"]["baseline_path"])
        return cls(joblib.load(path), name=name)


# --------------------------------------------------------------------------- SLM zero-shot
ZERO_SHOT_PROMPT = (
    "You classify customer support tickets. Possible intents:\n{intents}\n\n"
    "Ticket: {ticket}\n\nAnswer with exactly one intent from the list and nothing else."
)


def parse_intent(output: str, labels: list[str]) -> str | None:
    """Récupère l'intention citée dans la sortie du modèle (tolère majuscules, espaces, ponctuation)."""
    norm = re.sub(r"[^a-z_ ]", " ", output.lower()).replace(" ", "_")
    found = [(norm.find(lbl), lbl) for lbl in labels if lbl in norm]
    if found:
        return min(found)[1]
    return None


class ZeroShotSLMClassifier:
    """Le SLM génératif classe le ticket avec un simple prompt, sans entraînement."""

    def __init__(self, chat_model, labels: list[str], name: str = "slm_zero_shot"):
        self.chat_model = chat_model
        self.labels = labels
        self.name = name

    def predict(self, texts: list[str]) -> list[Prediction]:
        intents = "\n".join(f"- {lbl}" for lbl in self.labels)
        preds = []
        for t in texts:
            out = self.chat_model.invoke(ZERO_SHOT_PROMPT.format(intents=intents, ticket=t)).content
            label = parse_intent(out, self.labels)
            # pas de probabilité calibrée en génération : confiance 1 si l'intention est reconnue
            preds.append((label, 1.0) if label else ("unknown", 0.0))
        return preds


# --------------------------------------------------------------------------- SLM + LoRA
class LoraSLMClassifier:
    """SLM (Qwen2.5-0.5B) avec une tête de classification, fine-tuné en LoRA (voir scripts/train_lora.py)."""

    def __init__(self, adapter_dir: str | Path | None = None, base_model: str | None = None,
                 max_length: int | None = None, batch_size: int = 16, name: str = "slm_lora"):
        import json

        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = load_config()["classifier"]
        self.name = name
        adapter_dir = resolve(adapter_dir or cfg["adapter_dir"])
        base_model = base_model or cfg["base_model"]
        self.max_length = max_length or cfg["max_length"]
        self.batch_size = batch_size

        labels = json.loads((adapter_dir / "labels.json").read_text())
        self.id2label = dict(enumerate(labels))
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(labels), dtype=torch.float32
        )
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().eval()

    def predict(self, texts: list[str]) -> list[Prediction]:
        import torch

        preds: list[Prediction] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                probs = self.model(**batch).logits.softmax(-1)
            conf, idx = probs.max(-1)
            preds += [(self.id2label[int(k)], float(c)) for k, c in zip(idx, conf, strict=True)]
        return preds


CLASSIFIERS = {
    "tfidf": ("tfidf", "models/baseline_tfidf.joblib"),
    "tfidf_aug": ("tfidf", "models/baseline_tfidf_aug.joblib"),
    "lora": ("lora", "models/lora-intent"),
    "lora_aug": ("lora", "models/lora-intent-aug"),
}


def load_classifier(key: str) -> IntentClassifier:
    kind, path = CLASSIFIERS[key]
    return TfidfClassifier.load(path, name=key) if kind == "tfidf" else LoraSLMClassifier(path, name=key)


def load_best_classifier() -> IntentClassifier:
    """Classifieur utilisé par l'agent (configs/config.yaml : classifier.agent_classifier)."""
    return load_classifier(load_config()["classifier"].get("agent_classifier", "tfidf"))
