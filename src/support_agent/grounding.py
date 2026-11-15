"""Vérification de la fidélité d'une réponse à la documentation retrouvée.

Deux vérificateurs, interchangeables dans le graphe :
- NLIVerifier (par défaut) : contrôle des chiffres + modèle NLI (DeBERTa-v3) qui détecte les phrases de la
  réponse contredites par la documentation. Les formules de politesse sont « neutres » et ne sont pas pénalisées.
- LLMJudgeVerifier : un SLM répond SUPPORTED / NOT_SUPPORTED. Moins fiable avec de petits modèles
  (voir scripts/evaluate_judge.py et reports/judge_calibration.json).

Un vérificateur renvoie None si la réponse est acceptée, sinon la raison du rejet.
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)  # « 1. », « 2) » en début de ligne
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def numbers(text: str) -> set[str]:
    """Chiffres d'un texte, hors numéros d'une liste numérotée."""
    return {n.replace(",", ".") for n in NUMBER_RE.findall(LIST_MARKER_RE.sub("", text))}


def split_sentences(text: str) -> list[str]:
    return [s.strip(" -*•") for s in SENTENCE_RE.split(text) if len(s.strip(" -*•")) > 3]


def unsupported_numbers(answer: str, docs: list[Document], ticket: str = "") -> set[str]:
    """Chiffres de la réponse absents de la documentation et du ticket (n° de commande, montants cités…)."""
    allowed = numbers(ticket)
    for d in docs:
        allowed |= numbers(d.page_content)
    return numbers(answer) - allowed


@lru_cache(maxsize=2)
def _nli_model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class NLIVerifier:
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall", threshold: float = 0.5,
                 check_numbers: bool = True, embeddings=None, top_premises: int = 3):
        self.model_name, self.threshold, self.check_numbers = model_name, threshold, check_numbers
        # Une contradiction ne compte que si elle vient d'une des `top_premises` prémisses les plus proches
        # de la phrase (similarité d'embeddings) : un petit modèle NLI « contredit » volontiers une phrase
        # par un passage hors sujet (ex. une formule de politesse contre les CGV). None = toutes les prémisses.
        self.embeddings, self.top_premises = embeddings, top_premises

    def contradicted_sentences(self, answer: str, docs: list[Document]) -> list[tuple[str, float]]:
        model = _nli_model(self.model_name)
        labels = {v.lower(): k for k, v in model.model.config.id2label.items()}
        sentences = split_sentences(answer)
        if not sentences or not docs:
            return []
        # Prémisses à plusieurs granularités (approche SummaC) : documents entiers, phrases et paires de
        # phrases consécutives. Un petit modèle NLI se trompe sur les prémisses longues.
        premises: list[str] = []
        for d in docs:
            doc_sents = split_sentences(d.page_content)
            premises += [d.page_content, *doc_sents]
            premises += [f"{a} {b}" for a, b in zip(doc_sents, doc_sents[1:], strict=False)]
        pairs = [(p, s) for s in sentences for p in premises]
        probs = model.predict(pairs, apply_softmax=True, batch_size=64).reshape(len(sentences), len(premises), -1)
        relevant = self._relevant_premises(sentences, premises)
        flagged = []
        for s, p, idx in zip(sentences, probs, relevant, strict=True):
            contra, entail = p[idx, labels["contradiction"]].max(), p[:, labels["entailment"]].max()
            # contredite par une prémisse et impliquée par aucune
            if contra > self.threshold and entail < self.threshold:
                flagged.append((s, float(contra)))
        return flagged

    def _relevant_premises(self, sentences: list[str], premises: list[str]):
        """Indices des prémisses les plus proches de chaque phrase (toutes si pas d'embeddings)."""
        import numpy as np

        if self.embeddings is None or not self.top_premises:
            return [slice(None)] * len(sentences)
        s_emb = np.asarray(self.embeddings.embed_documents(sentences))
        p_emb = np.asarray(self.embeddings.embed_documents(premises))
        sims = s_emb @ p_emb.T  # embeddings normalisés : produit scalaire = cosinus
        return [np.argsort(-row)[: self.top_premises] for row in sims]

    def __call__(self, answer: str, docs: list[Document], ticket: str = "") -> str | None:
        if self.check_numbers:
            bad = unsupported_numbers(answer, docs, ticket)
            if bad:
                return f"numbers not found in the documentation: {', '.join(sorted(bad))}"
        flagged = self.contradicted_sentences(answer, docs)
        if flagged:
            s, p = flagged[0]
            return f"sentence contradicted by the documentation (p={p:.2f}): {s[:80]}"
        return None


VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check answers written by a support assistant. Reply with one word: SUPPORTED if every fact "
     "in the answer (numbers, delays, prices, procedures) appears in the context, otherwise NOT_SUPPORTED."),
    ("human", "Context:\n{context}\n\nAnswer to check:\n{answer}\n\nVerdict:"),
])

VERDICT_RE = re.compile(r"NOT[\s_-]*SUPPORTED|SUPPORTED", re.IGNORECASE)


def parse_verdict(output: str) -> str:
    """Premier verdict cité par le juge ; en cas de sortie illisible, on rejette (prudence)."""
    m = VERDICT_RE.search(output)
    if m and not m.group(0).upper().startswith("NOT"):
        return "SUPPORTED"
    return "NOT_SUPPORTED"


class LLMJudgeVerifier:
    name = "llm_judge"

    def __init__(self, chat_model):
        self.chain = VERIFY_PROMPT | chat_model | StrOutputParser()

    def __call__(self, answer: str, docs: list[Document], ticket: str = "") -> str | None:
        from .kb import format_context

        out = self.chain.invoke({"context": format_context(docs), "answer": answer})
        return None if parse_verdict(out) == "SUPPORTED" else "judge: answer not supported by the context"
