"""Compare les classifieurs d'intention sur deux jeux de test :
- Bitext test (même distribution que l'entraînement, phrases courtes et générées) ;
- tickets réalistes écrits à la main (hors distribution : plus longs, bruités, contexte superflu).

Classifieurs : TF-IDF + régression logistique, SLM zero-shot (prompt), SLM fine-tuné en LoRA.
Résultats : reports/classification.json
"""

from __future__ import annotations

import argparse
import json
import random

from support_agent.classifier import LoraSLMClassifier, TfidfClassifier, ZeroShotSLMClassifier
from support_agent.config import load_config, resolve
from support_agent.data import load_bitext_splits, load_realistic_tickets
from support_agent.metrics import classification_scores, rss_mb, timed_predict


def evaluate(clf, sets: dict[str, tuple[list[str], list[str]]], extra: dict) -> dict:
    res = {"classifier": clf.name, **extra}
    for name, (x, y) in sets.items():
        preds, med_ms = timed_predict(clf, x)
        scores = classification_scores(y, [p for p, _ in preds])
        scores["median_latency_ms"] = round(med_ms, 1)
        res[name] = scores
        if name == "realistic":
            res["realistic_errors"] = [
                {"text": t[:90], "gold": g, "pred": p} for t, g, (p, _) in zip(x, y, preds, strict=True) if g != p
            ]
        print(f"  {clf.name:28s} {name:10s} acc={scores['accuracy']:.3f} f1={scores['macro_f1']:.3f} "
              f"latence={med_ms:.0f} ms")
    return res


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bitext", type=int, default=1000, help="échantillon du test Bitext")
    ap.add_argument("--n-bitext-zeroshot", type=int, default=100, help="échantillon pour le zero-shot (lent)")
    ap.add_argument("--zero-shot-models", nargs="*",
                    default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument("--rerun-zero-shot", action="store_true")
    args = ap.parse_args()

    splits = load_bitext_splits(cfg)
    rng = random.Random(cfg["seed"])
    idx = rng.sample(range(len(splits.test.texts)), args.n_bitext)
    bitext = ([splits.test.texts[i] for i in idx], [splits.test.labels[i] for i in idx])
    real = load_realistic_tickets(cfg)
    realistic = ([t["text"] for t in real], [t["intent"] for t in real])
    sets = {"bitext": bitext, "realistic": realistic}
    small_sets = {"bitext": (bitext[0][: args.n_bitext_zeroshot], bitext[1][: args.n_bitext_zeroshot]),
                  "realistic": realistic}

    out = resolve("reports/classification.json")
    previous = {r["classifier"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    results = []
    print("Évaluation des classifieurs")
    for path, name, training in [
        (cfg["classifier"]["baseline_path"], "tfidf_logreg", "train Bitext complet"),
        ("models/baseline_tfidf_aug.joblib", "tfidf_logreg+augmentation", "train Bitext complet + augmentation"),
    ]:
        if resolve(path).exists():
            results.append(evaluate(TfidfClassifier.load(path, name), sets,
                                    {"params": "modèle linéaire", "training": training}))

    for adapter, name in [(cfg["classifier"]["adapter_dir"], "slm_lora"),
                          ("models/lora-intent-aug", "slm_lora+augmentation")]:
        adir = resolve(adapter)
        if args.skip_lora or not (adir / "adapter_config.json").exists():
            continue
        before = rss_mb()
        lora = LoraSLMClassifier(adir, name=name)
        info = json.loads((adir / "training_info.json").read_text())
        results.append(evaluate(lora, sets, {
            "params": f"{info['total_params'] / 1e6:.0f} M, {info['trainable_params'] / 1e6:.1f} M entraînés",
            "training": f"{info['n_train']} exemples, {info['epochs']:g} époque, {info['device']}",
            "ram_mb": round(rss_mb() - before),
        }))
        del lora

    from support_agent.llm import get_chat_model

    for model_id in args.zero_shot_models:
        name = f"zero_shot:{model_id.split('/')[-1]}"
        if name in previous and not args.rerun_zero_shot:
            results.append(previous[name])          # déjà calculé (lent) : on réutilise
            continue
        chat = get_chat_model(model_id, max_new_tokens=12)
        clf = ZeroShotSLMClassifier(chat, splits.labels, name=name)
        results.append(evaluate(clf, small_sets, {"params": model_id, "training": "aucun (prompt)"}))

    # on conserve les résultats précédents qui n'ont pas été recalculés
    merged = {**previous, **{r["classifier"]: r for r in results}}
    out.write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
