"""Doublures de test : aucun modèle n'est téléchargé, les tests tournent en quelques secondes."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class FakeClassifier:
    name = "fake"

    def __init__(self, intent: str = "track_order", confidence: float = 0.95):
        self.intent, self.confidence = intent, confidence

    def predict(self, texts):
        return [(self.intent, self.confidence) for _ in texts]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return [Document(page_content="Standard delivery takes 3 to 5 business days.",
                         metadata={"source": "delivery.md", "intent": "delivery_period"})]


def fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


@pytest.fixture
def fake_retriever():
    return FakeRetriever()
