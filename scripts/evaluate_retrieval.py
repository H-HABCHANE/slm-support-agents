"""Évalue la recherche RAG sur les tickets réalistes : la section de la base de connaissances
correspondant à l'intention réelle est-elle retrouvée parmi les k premiers résultats ?

Trois formulations de requête (texte seul, + intention prédite, + intention réelle), avec et sans reranking.
Résultats : reports/retrieval.json
"""

from __future__ import annotations

import json

from support_agent.classifier import load_best_classifier
from support_agent.config import load_config, resolve
from support_agent.data import load_realistic_tickets
from support_agent.kb import build_retriever, get_embeddings, load_kb_documents


def hit_at(retriever, queries: list[str], gold: list[str]) -> dict:
    """hit@1, hit@3, MRR et latence médiane d'une requête."""
    import statistics
    import time

    h1 = h3 = rr = 0.0
    times = []
    for q, g in zip(queries, gold, strict=True):
        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        times.append((time.perf_counter() - t0) * 1000)
        ranks = [i for i, d in enumerate(docs) if d.metadata.get("intent") == g]
        if ranks:
            h1 += ranks[0] == 0
            h3 += ranks[0] < 3
            rr += 1 / (ranks[0] + 1)
    n = len(queries)
    return {"hit@1": round(h1 / n, 4), "hit@3": round(h3 / n, 4), "mrr@3": round(rr / n, 4),
            "median_latency_ms": round(statistics.median(times), 1)}


def main() -> None:
    cfg = load_config()
    tickets = load_realistic_tickets(cfg)
    texts, gold = [t["text"] for t in tickets], [t["intent"] for t in tickets]
    pred = [p for p, _ in load_best_classifier().predict(texts)]
    docs, emb = load_kb_documents(), get_embeddings()
    retrievers = {"vectoriel": build_retriever(docs, emb, k=3, rerank=False),
                  "vectoriel + reranking": build_retriever(docs, emb, k=3, rerank=True)}

    queries = {
        "texte seul": texts,
        "texte + intention prédite": [f"{p.replace('_', ' ')}: {t}" for p, t in zip(pred, texts, strict=True)],
        "texte + intention réelle (plafond)": [f"{g.replace('_', ' ')}: {t}" for g, t in zip(gold, texts, strict=True)],
    }
    results = {f"{qname} | {rname}": hit_at(r, q, gold)
               for qname, q in queries.items() for rname, r in retrievers.items()}
    results["n_tickets"] = len(texts)
    results["embedding_model"] = cfg["kb"]["embedding_model"]
    results["rerank_model"] = cfg["kb"]["rerank_model"]
    results["classifier"] = load_best_classifier().name
    out = resolve("reports/retrieval.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
