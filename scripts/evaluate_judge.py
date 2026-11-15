"""Évaluer l'évaluateur : un SLM vérificateur sait-il distinguer une réponse fidèle d'une réponse fausse ?

Jeu de calibration construit automatiquement depuis la base de connaissances :
  - réponse fidèle   = texte exact d'une section ;
  - réponse corrompue = même texte avec les chiffres modifiés (délais, prix, durées faux).
On mesure le taux d'acceptation des réponses fidèles et le taux de rejet des réponses corrompues.
Résultats : reports/judge_calibration.json
"""

from __future__ import annotations

import argparse
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from support_agent.config import resolve
from support_agent.grounding import VERIFY_PROMPT, NLIVerifier, parse_verdict
from support_agent.kb import format_context, load_kb_documents
from support_agent.llm import get_chat_model

# Question inversée : « y a-t-il une erreur ? ». Les petits modèles ont tendance à répondre toujours
# la même chose ; on compare donc deux formulations.
ERROR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You compare an answer with the documentation. Look carefully at every number, delay, price "
               "and procedure."),
    ("human", "Documentation:\n{context}\n\nAnswer:\n{answer}\n\nDoes the answer contain a number, delay, "
              "price or procedure that is different from the documentation? Reply YES or NO."),
])


def corrupt(text: str) -> str:
    return re.sub(r"\d+", lambda m: str(int(m.group()) * 2 + 1), text)


def build_cases() -> list[dict]:
    cases = []
    for d in load_kb_documents():
        if not d.metadata["intent"] or not re.search(r"\d", d.page_content):
            continue
        body = d.page_content.split("\n", 1)[1]
        ctx = format_context([d])
        base = {"intent": d.metadata["intent"], "context": ctx, "doc": d}
        cases.append({**base, "answer": body, "faithful": True})
        cases.append({**base, "answer": corrupt(body), "faithful": False})
    return cases


def score(cases, accepts: list[bool]) -> dict:
    faithful = [a for c, a in zip(cases, accepts, strict=True) if c["faithful"]]
    corrupted = [a for c, a in zip(cases, accepts, strict=True) if not c["faithful"]]
    acc_f = sum(faithful) / len(faithful)
    rej_c = sum(not a for a in corrupted) / len(corrupted)
    return {"accepts_faithful": round(acc_f, 3), "rejects_corrupted": round(rej_c, 3),
            "balanced_accuracy": round((acc_f + rej_c) / 2, 3), "n": len(cases)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["HuggingFaceTB/SmolLM2-360M-Instruct",
                                                    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
    args = ap.parse_args()
    cases = build_cases()
    print(f"{len(cases)} cas de calibration ({len(cases) // 2} fidèles, {len(cases) // 2} corrompus)")
    results = []
    for name, verifier in [("NLI DeBERTa-v3-xsmall seul", NLIVerifier(check_numbers=False)),
                           ("NLI + contrôle des chiffres", NLIVerifier())]:
        accepts = [verifier(c["answer"], [c["doc"]]) is None for c in cases]
        res = {"model": "cross-encoder/nli-deberta-v3-xsmall", "prompt": name, **score(cases, accepts)}
        results.append(res)
        print(json.dumps(res, ensure_ascii=False), flush=True)
    for model_id in args.models:
        chat = get_chat_model(model_id, max_new_tokens=8)
        prompts = {
            "verify (SUPPORTED / NOT_SUPPORTED)": (VERIFY_PROMPT | chat | StrOutputParser(),
                                                   lambda out: parse_verdict(out) == "SUPPORTED"),
            "error? (YES / NO)": (ERROR_PROMPT | chat | StrOutputParser(),
                                  lambda out: not out.strip().upper().startswith("YES")),
        }
        for pname, (chain, accept) in prompts.items():  # noqa: B007
            accepts = [accept(chain.invoke({"context": c["context"], "answer": c["answer"]})) for c in cases]
            res = {"model": model_id, "prompt": pname, **score(cases, accepts)}
            results.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
        get_chat_model.cache_clear()
    out = resolve("reports/judge_calibration.json")
    merged = {(r["model"], r["prompt"]): r for r in (json.loads(out.read_text()) if out.exists() else [])}
    merged.update({(r["model"], r["prompt"]): r for r in results})
    out.write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
