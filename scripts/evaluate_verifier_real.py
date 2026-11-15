"""Vérificateur NLI sur de vraies réponses de l'agent (Qwen2.5-1.5B), annotées à la main.

Le jeu de calibration de evaluate_judge.py est facile (sections recopiées vs chiffres modifiés). Ici, on
mesure le vérificateur sur des brouillons réellement générés : fidèles (paraphrases correctes, formules de
politesse) ou faux (procédure inventée, règle contredite). Annotation : lecture de la réponse face au contexte.
Résultats : reports/verifier_real.json
"""

from __future__ import annotations

import json
import re

from langchain_core.documents import Document

from support_agent.config import resolve
from support_agent.grounding import NLIVerifier
from support_agent.kb import get_embeddings

SOURCE = "reports/agent_Qwen2.5-1.5B-Instruct.json"
# Annotations manuelles (True = fidèle au contexte, False = contient une information fausse ou inventée)
LABELS = {
    "R01": True, "R11": True, "R13": True, "R29": True, "R31": True, "R41": True, "R43": True, "R45": True,
    "R25": True, "R35": True, "R51": True,
    "R03": False,  # invente un retour au statut « Received »
    "R09": False,  # invente un montant attendu et une remise
    "R33": False,  # affirme qu'un grille-pain est un produit sur mesure
    "R39": False,  # « pas de commande en invité » : contredit la documentation
}


def context_docs(context: str) -> list[Document]:
    parts = re.split(r"\n\n(?=\[\d+\] \()", context)
    return [Document(page_content=re.sub(r"^\[\d+\] \([^)]*\) ", "", p)) for p in parts]


def main() -> None:
    tickets = {t["id"]: t for t in json.loads(resolve(SOURCE).read_text())["tickets"]}
    emb, base = get_embeddings(), "cross-encoder/nli-deberta-v3-base"
    verifiers = {"xsmall : NLI + chiffres (toutes prémisses)": NLIVerifier(),
                 "xsmall : NLI + chiffres + prémisses proches": NLIVerifier(embeddings=emb),
                 "base : NLI + chiffres (toutes prémisses)": NLIVerifier(model_name=base),
                 "base : NLI + chiffres + prémisses proches": NLIVerifier(model_name=base, embeddings=emb)}
    results = {"n": len(LABELS), "n_faithful": sum(LABELS.values()), "source": SOURCE, "verifiers": {}}
    for name, v in verifiers.items():
        accepted = {}
        for i in LABELS:
            t = tickets[i]
            accepted[i] = v(t["draft_answer"] or t["answer"], context_docs(t["context"]), t["ticket"]) is None
        good = [accepted[i] for i, ok in LABELS.items() if ok]
        bad = [not accepted[i] for i, ok in LABELS.items() if not ok]
        acc_f, rej_c = sum(good) / len(good), sum(bad) / len(bad)
        results["verifiers"][name] = {"accepts_faithful": round(acc_f, 3), "rejects_wrong": round(rej_c, 3),
                                      "balanced_accuracy": round((acc_f + rej_c) / 2, 3),
                                      "wrongly_rejected": [i for i, ok in LABELS.items() if ok and not accepted[i]],
                                      "wrongly_accepted": [i for i, ok in LABELS.items() if not ok and accepted[i]]}
    out = resolve("reports/verifier_real.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
