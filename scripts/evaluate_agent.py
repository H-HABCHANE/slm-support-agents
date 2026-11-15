"""Compare plusieurs SLM dans l'agent complet (LangGraph) sur les tickets réalistes.

Pour chaque SLM (un processus séparé, pour mesurer la mémoire proprement) :
  qualité   : taux de réponses acceptées par le vérificateur NLI, similarité avec la section de référence,
              puis contre-vérification par un juge SLM calibré (étape --judge) ;
  coût      : latence par ticket et par agent, tokens/s, mémoire, temps de chargement ;
  souveraineté : tout tourne en local, aucune donnée ne sort de la machine.

  python scripts/evaluate_agent.py --all          # tous les SLM de configs/config.yaml, puis le juge
  python scripts/evaluate_agent.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time

from support_agent.config import load_config, resolve
from support_agent.data import load_realistic_tickets

REPORTS = resolve("reports")


def slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def pick_tickets(n: int) -> list[dict]:
    """Un ticket par intention (le premier de chaque paire), dans l'ordre du fichier."""
    seen, out = set(), []
    for t in load_realistic_tickets():
        if t["intent"] not in seen:
            seen.add(t["intent"])
            out.append(t)
    return out[:n]


def run_model(model_id: str, n: int) -> None:
    from support_agent.classifier import load_best_classifier
    from support_agent.graph import build_graph
    from support_agent.kb import build_retriever, format_context, get_embeddings, load_kb_documents
    from support_agent.llm import count_tokens, get_chat_model
    from support_agent.metrics import percentile, rss_mb

    cfg = load_config()
    classifier = load_best_classifier()
    embeddings = get_embeddings()
    kb_docs = load_kb_documents()
    retriever = build_retriever(kb_docs, embeddings)
    section_by_intent = {d.metadata["intent"]: d.page_content for d in kb_docs if d.metadata["intent"]}

    ram_before, t0 = rss_mb(), time.perf_counter()
    chat = get_chat_model(model_id)
    load_s, ram_model = time.perf_counter() - t0, rss_mb() - ram_before
    n_params = sum(p.numel() for p in chat.llm.pipeline.model.parameters())
    graph = build_graph(classifier, retriever, chat, cfg=cfg)

    rows = []
    for t in pick_tickets(n):
        t1 = time.perf_counter()
        state = graph.invoke({"ticket_text": t["text"]})
        total = time.perf_counter() - t1
        gen_s = sum(x["seconds"] for x in state["trace"] if x["node"] == "generate")
        answer = state.get("answer", "")
        generated = answer if state["status"] == "answered" else state.get("draft_answer", "")
        ntok = count_tokens(chat, generated) if state.get("context_docs") else 0
        sim = None
        if state["status"] == "answered" and t["intent"] in section_by_intent:
            a, b = embeddings.embed_documents([answer, section_by_intent[t["intent"]]])
            sim = round(sum(x * y for x, y in zip(a, b, strict=True)), 4)
        rows.append({
            "id": t["id"], "gold_intent": t["intent"], "pred_intent": state["intent"],
            "confidence": round(state["confidence"], 3), "status": state["status"],
            "generations": sum(1 for x in state["trace"] if x["node"] == "generate"),
            "verdict": state.get("verdict"), "escalation_reason": state.get("escalation_reason"),
            "seconds": round(total, 2), "generate_seconds": round(gen_s, 2),
            "answer_tokens": ntok, "similarity_to_reference": sim,
            "trace": state["trace"], "ticket": t["text"], "answer": answer,
            "draft_answer": state.get("draft_answer", ""),
            "sources": [d.metadata.get("source") for d in state.get("context_docs", [])],
            "context": format_context(state.get("context_docs", [])),
        })
        print(f"  {t['id']} {state['status']:9s} {total:6.1f}s  {t['intent']}", flush=True)

    with_llm = [r for r in rows if r["generations"] > 0]
    answered = [r for r in rows if r["status"] == "answered"]
    secs = [r["seconds"] for r in with_llm]
    tok_s = [r["answer_tokens"] / (r["generate_seconds"] / r["generations"])
             for r in with_llm if r["generate_seconds"] > 0]
    sims = [r["similarity_to_reference"] for r in answered if r["similarity_to_reference"] is not None]
    summary = {
        "model": model_id, "params_m": round(n_params / 1e6), "load_seconds": round(load_s, 1),
        "ram_model_mb": round(ram_model), "n_tickets": len(rows), "n_with_llm": len(with_llm),
        "intent_accuracy": round(sum(r["gold_intent"] == r["pred_intent"] for r in rows) / len(rows), 3),
        "answered_rate": round(len(answered) / len(with_llm), 3) if with_llm else None,
        "first_try_accepted_rate": round(sum(r["status"] == "answered" and r["generations"] == 1
                                             for r in with_llm) / len(with_llm), 3) if with_llm else None,
        "median_seconds_per_ticket": round(statistics.median(secs), 1) if secs else None,
        "p90_seconds_per_ticket": round(percentile(secs, 0.9), 1) if secs else None,
        "median_tokens_per_second": round(statistics.median(tok_s), 1) if tok_s else None,
        "mean_similarity_to_reference": round(statistics.mean(sims), 3) if sims else None,
        "classifier": classifier.name, "verifier": cfg["agent"]["verifier"],
        "device": "cpu", "external_api_calls": 0,
    }
    out = REPORTS / f"agent_{slug(model_id)}.json"
    out.write_text(json.dumps({"summary": summary, "tickets": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


def judge(judge_model: str) -> None:
    """Contre-vérification indépendante du NLI : Qwen2.5-1.5B avec le prompt de vérification calibré
    (94,7 % d'exactitude équilibrée, voir reports/judge_calibration.json), appliqué aux réponses finales."""
    from langchain_core.output_parsers import StrOutputParser

    from support_agent.grounding import VERIFY_PROMPT, parse_verdict
    from support_agent.llm import get_chat_model

    chain = VERIFY_PROMPT | get_chat_model(judge_model, max_new_tokens=8) | StrOutputParser()
    for path in sorted(REPORTS.glob("agent_*.json")):
        data = json.loads(path.read_text())
        rows = [r for r in data["tickets"] if r["status"] == "answered"]
        for r in rows:
            out = chain.invoke({"context": r["context"], "answer": r["answer"]})
            r["judge_supported"] = parse_verdict(out) == "SUPPORTED"
        ok = sum(r["judge_supported"] for r in rows)
        data["summary"].update({"judge_model": judge_model,
                                "judge_supported_rate": round(ok / len(rows), 3) if rows else None})
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"{path.name}: réponses finales confirmées par le juge = {ok}/{len(rows)}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=cfg["eval"]["n_tickets"])
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    if args.model:
        run_model(args.model, args.n)
    if args.all:
        for m in cfg["eval"]["candidate_llms"]:
            print(f"=== {m}", flush=True)
            subprocess.run([sys.executable, __file__, "--model", m, "--n", str(args.n)], check=True)
        args.judge = True
    if args.judge:
        judge(args.judge_model)


if __name__ == "__main__":
    main()
