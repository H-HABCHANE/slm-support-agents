"""Assemble les fichiers reports/*.json en tableaux Markdown (reports/RESULTS.md), repris dans le README."""

from __future__ import annotations

import json

from support_agent.config import resolve

R = resolve("reports")


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def classification() -> list[str]:
    path = R / "classification.json"
    if not path.exists():
        return []
    lines = ["## Classification d'intention (27 intentions)\n",
             "| Classifieur | Entraînement | Bitext test : acc. / F1 macro | Tickets réalistes : acc. / F1 macro "
             "| Latence médiane |",
             "|---|---|---|---|---|"]
    for r in json.loads(path.read_text()):
        b, s = r["bitext"], r["realistic"]
        lines.append(f"| {r['classifier']} | {r['training']} | {pct(b['accuracy'])} / {pct(b['macro_f1'])} "
                     f"(n={b['n']}) | **{pct(s['accuracy'])}** / {pct(s['macro_f1'])} (n={s['n']}) "
                     f"| {s['median_latency_ms']:.0f} ms |")
    return lines + [""]


def retrieval() -> list[str]:
    path = R / "retrieval.json"
    if not path.exists():
        return []
    ret = json.loads(path.read_text())
    lines = [f"## Recherche RAG ({ret['n_tickets']} tickets réalistes, {ret['embedding_model'].split('/')[-1]})\n",
             "| Requête | hit@1 | hit@3 |", "|---|---|---|"]
    lines += [f"| {k} | {pct(v['hit@1'])} | {pct(v['hit@3'])} |" for k, v in ret.items() if isinstance(v, dict)]
    return lines + [""]


def calibration() -> list[str]:
    path = R / "judge_calibration.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    lines = [f"## Calibration des vérificateurs ({rows[0]['n']} cas : moitié fidèles, moitié aux chiffres modifiés)\n",
             "| Vérificateur | Méthode | Accepte les réponses fidèles | Rejette les réponses fausses "
             "| Exactitude équilibrée |",
             "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["balanced_accuracy"]):
        lines.append(f"| {r['model'].split('/')[-1]} | {r['prompt']} | {pct(r['accepts_faithful'])} "
                     f"| {pct(r['rejects_corrupted'])} | **{pct(r['balanced_accuracy'])}** |")
    return lines + [""]


def agents() -> list[str]:
    files = sorted(R.glob("agent_*.json"), key=lambda p: json.loads(p.read_text())["summary"]["params_m"])
    if not files:
        return []
    s0 = json.loads(files[0].read_text())["summary"]
    lines = [f"## Agent complet : comparaison des SLM générateurs (CPU 8 cœurs, sans GPU, "
             f"classifieur {s0['classifier']}, vérificateur {s0['verifier']})\n",
             "| SLM | Paramètres | RAM | Réponses acceptées | Dont au 1er essai | Confirmées par le juge "
             "| Similarité / référence | Latence médiane (p90) | Tokens/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in files:
        s = json.loads(p.read_text())["summary"]
        lines.append(
            f"| {s['model'].split('/')[-1]} | {s['params_m']} M | {s['ram_model_mb']} Mo "
            f"| {pct(s['answered_rate'])} | {pct(s['first_try_accepted_rate'])} "
            f"| {pct(s.get('judge_supported_rate'))} | {s['mean_similarity_to_reference']} "
            f"| {s['median_seconds_per_ticket']} s ({s['p90_seconds_per_ticket']} s) "
            f"| {s['median_tokens_per_second']} |")
    lines.append(f"\n{s0['n_with_llm']} tickets sur {s0['n_tickets']} passent par le SLM, les autres sont "
                 f"escaladés directement (intention réservée aux humains ou confiance trop faible). "
                 f"Exactitude du classifieur sur ces tickets : {pct(s0['intent_accuracy'])}. "
                 f"Appels à une API externe : 0.\n")
    return lines


def main() -> None:
    lines = ["# Résultats\n", *classification(), *retrieval(), *calibration(), *agents()]
    out = R / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
