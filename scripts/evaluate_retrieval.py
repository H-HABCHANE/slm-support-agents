"""Évalue la recherche RAG sur les tickets réalistes : la section de la base de connaissances
correspondant à l'intention réelle est-elle retrouvée parmi les k premiers résultats ?

Trois stratégies de requête : texte seul, texte + intention prédite (ce que fait l'agent), texte + intention réelle.
Résultats : reports/retrieval.json
"""

from __future__ import annotations

import json

from support_agent.classifier import load_best_classifier
from support_agent.config import load_config, resolve
from support_agent.data import load_realistic_tickets
from support_agent.kb import build_retriever, get_embeddings, load_kb_documents


def hit_at(retriever, queries: list[str], gold: list[str], k: int) -> float:
    hits = 0
    for q, g in zip(queries, gold, strict=True):
        docs = retriever.invoke(q)[:k]
        hits += any(d.metadata.get("intent") == g for d in docs)
    return round(hits / len(queries), 4)


def main() -> None:
    cfg = load_config()
    tickets = load_realistic_tickets(cfg)
    texts, gold = [t["text"] for t in tickets], [t["intent"] for t in tickets]
    pred = [p for p, _ in load_best_classifier().predict(texts)]
    retriever = build_retriever(load_kb_documents(), get_embeddings(), k=3)

    strategies = {
        "texte seul": texts,
        "texte + intention prédite (agent)": [f"{p.replace('_', ' ')}: {t}" for p, t in zip(pred, texts, strict=True)],
        "texte + intention réelle (plafond)": [f"{g.replace('_', ' ')}: {t}" for g, t in zip(gold, texts, strict=True)],
    }
    results = {name: {"hit@1": hit_at(retriever, q, gold, 1), "hit@3": hit_at(retriever, q, gold, 3)}
               for name, q in strategies.items()}
    results["n_tickets"] = len(texts)
    results["embedding_model"] = cfg["kb"]["embedding_model"]
    out = resolve("reports/retrieval.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
