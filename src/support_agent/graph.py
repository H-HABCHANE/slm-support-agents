"""Orchestration multi-agents avec LangGraph.

    ingestion -> classification -> (routage) -> recherche (RAG) -> rédaction -> vérification (NLI)
                                       |                                ^            |
                                       v                                └─ nouvel essai (1 max)
                                   escalade vers un humain  <───────────────── échec ┘

Chaque nœud enregistre sa durée dans `trace`, pour mesurer la latence agent par agent.
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .config import load_config
from .grounding import LLMJudgeVerifier, NLIVerifier
from .ingestion import load_file
from .kb import format_context


class TicketState(TypedDict, total=False):
    ticket_text: str
    source_file: str
    modality: str
    intent: str
    confidence: float
    context_docs: list[Document]
    answer: str
    draft_answer: str                            # réponse rejetée par le vérificateur (pour analyse)
    verdict: str
    feedback: str
    retries: int
    status: str                                  # "answered" | "escalated"
    escalation_reason: str
    trace: Annotated[list[dict[str, Any]], operator.add]


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are NovaShop's customer support assistant. Answer the customer using ONLY the facts in the "
     "context below. Be concise (at most 5 sentences), polite, and give concrete steps. If the context "
     "does not contain the answer, say that a human agent will follow up. Never invent prices, delays "
     "or policies.\n\nContext:\n{context}"),
    ("human", "Detected intent: {intent}\nTicket:\n{ticket}{feedback}"),
])

PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\[(?:insert|your|name)[^\]]*\]", re.IGNORECASE)


def rule_checks(answer: str) -> str | None:
    """Contrôles déterministes avant le vérificateur. Renvoie la raison de l'échec, ou None."""
    if len(answer.strip()) < 20:
        return "answer too short"
    if PLACEHOLDER_RE.search(answer):
        return "answer contains a template placeholder"
    return None


def _timed(name: str, fn):
    def wrapper(state: TicketState) -> dict:
        t0 = time.perf_counter()
        out = fn(state) or {}
        out["trace"] = [{"node": name, "seconds": round(time.perf_counter() - t0, 3)}]
        return out
    return wrapper


def make_verifier(kind: str, chat_model):
    if kind == "nli":
        return NLIVerifier()
    if kind == "llm_judge":
        return LLMJudgeVerifier(chat_model)
    raise ValueError(f"Vérificateur inconnu : {kind}")


def build_graph(classifier, retriever, chat_model, verifier=None, cfg: dict | None = None):
    """Assemble le graphe. Les dépendances sont injectées : les tests utilisent des modèles factices.

    verifier : fonction (réponse, documents, ticket) -> raison du rejet ou None.
    """
    cfg = cfg or load_config()
    acfg = cfg["agent"]
    verifier = verifier or make_verifier(acfg.get("verifier", "nli"), chat_model)
    answer_chain = ANSWER_PROMPT | chat_model | StrOutputParser()

    def ingest(state: TicketState) -> dict:
        if state.get("source_file"):
            text, modality = load_file(state["source_file"])
            return {"ticket_text": text, "modality": modality, "retries": 0}
        return {"modality": state.get("modality", "text"), "retries": 0}

    def classify(state: TicketState) -> dict:
        intent, conf = classifier.predict([state["ticket_text"]])[0]
        return {"intent": intent, "confidence": conf}

    def route(state: TicketState) -> str:
        if state["intent"] in acfg["escalate_intents"]:
            return "escalate"
        if state["confidence"] < acfg["min_confidence"]:
            return "escalate"
        return "retrieve"

    def retrieve(state: TicketState) -> dict:
        query = f"{state['intent'].replace('_', ' ')}: {state['ticket_text']}"
        return {"context_docs": retriever.invoke(query)}

    def generate(state: TicketState) -> dict:
        answer = answer_chain.invoke({
            "context": format_context(state["context_docs"]),
            "intent": state["intent"],
            "ticket": state["ticket_text"],
            "feedback": state.get("feedback", ""),
        })
        return {"answer": answer.strip()}

    def verify(state: TicketState) -> dict:
        reason = rule_checks(state["answer"]) or verifier(state["answer"], state["context_docs"],
                                                          state["ticket_text"])
        verdict = "NOT_SUPPORTED" if reason else "SUPPORTED"
        update: dict = {"verdict": verdict, "escalation_reason": None}
        if reason:
            update["retries"] = state.get("retries", 0) + 1
            update["feedback"] = (f"\n\nYour previous answer was rejected ({reason}). Rewrite it using only "
                                  "facts from the context.")
            update["escalation_reason"] = reason
        return update

    def after_verify(state: TicketState) -> str:
        if state["verdict"] == "SUPPORTED":
            return "finalize"
        if state.get("retries", 0) <= acfg["max_retries"]:
            return "generate"
        return "escalate"

    def finalize(state: TicketState) -> dict:
        return {"status": "answered"}

    def escalate(state: TicketState) -> dict:
        if state.get("escalation_reason"):
            reason = state["escalation_reason"]
        elif state["intent"] in acfg["escalate_intents"]:
            reason = f"intent '{state['intent']}' is always handled by a human"
        else:
            reason = f"low classification confidence ({state['confidence']:.2f})"
        return {
            "status": "escalated",
            "escalation_reason": reason,
            "draft_answer": state.get("answer", ""),
            "answer": "Thank you for your message. A NovaShop agent will take over your request and reply "
                      "shortly (Monday to Friday, 9 am to 6 pm).",
        }

    g = StateGraph(TicketState)
    for name, fn in [("ingest", ingest), ("classify", classify), ("retrieve", retrieve),
                     ("generate", generate), ("verify", verify), ("finalize", finalize), ("escalate", escalate)]:
        g.add_node(name, _timed(name, fn))
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")
    g.add_conditional_edges("classify", route, {"retrieve": "retrieve", "escalate": "escalate"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify,
                            {"finalize": "finalize", "generate": "generate", "escalate": "escalate"})
    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g.compile()


def build_default_graph():
    """Graphe avec les vrais modèles locaux (classifieur, embeddings, SLM)."""
    from .classifier import load_best_classifier
    from .kb import build_retriever
    from .llm import get_chat_model

    return build_graph(load_best_classifier(), build_retriever(), get_chat_model())
