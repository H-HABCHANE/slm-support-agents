from pathlib import Path

import pytest
from langchain_core.documents import Document

from support_agent.classifier import TfidfClassifier, parse_intent
from support_agent.ingestion import clean_text, detect_modality, load_file
from support_agent.kb import load_kb_documents, split_markdown

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
ALL_INTENTS = {
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee", "check_invoice",
    "check_payment_methods", "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account", "delivery_options", "delivery_period",
    "edit_account", "get_invoice", "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
}


def test_clean_text():
    assert clean_text("a   b\r\n\n\n\nc ") == "a b\n\nc"


def test_detect_modality():
    assert detect_modality("x.txt") == "text"
    assert detect_modality("x.PDF") == "pdf"
    assert detect_modality("x.png") == "image"
    with pytest.raises(ValueError):
        detect_modality("x.docx")


def test_load_text_and_pdf():
    text, modality = load_file(SAMPLES / "ticket_email.txt")
    assert modality == "text" and "NS-482913" in text
    text, modality = load_file(SAMPLES / "ticket_letter.pdf")
    assert modality == "pdf" and "NS-118204" in text


def test_load_image_ocr():
    text, modality = load_file(SAMPLES / "ticket_screenshot.png")
    assert modality == "image"
    assert "declined" in text.lower() and "charged twice" in text.lower()


def test_split_markdown_metadata():
    docs = split_markdown("# T\n\n## Track an order (track_order)\nUse the link.\n\n## Other\nText", "o.md")
    assert [d.metadata["intent"] for d in docs] == ["track_order", ""]
    assert docs[0].page_content.startswith("Track an order")


def test_kb_covers_every_intent():
    docs = load_kb_documents()
    covered = {d.metadata["intent"] for d in docs if d.metadata["intent"]}
    assert covered == ALL_INTENTS
    assert any(d.metadata["source"] == "terms_of_sale.pdf" for d in docs)


def test_parse_intent():
    labels = ["track_order", "track_refund"]
    assert parse_intent("Intent: track_refund.", labels) == "track_refund"
    assert parse_intent("TRACK ORDER", labels) == "track_order"
    assert parse_intent("no idea", labels) is None


def test_tfidf_classifier_roundtrip(tmp_path):
    texts = ["where is my parcel", "track my order", "refund status please", "where is my refund"] * 3
    labels = ["track_order", "track_order", "track_refund", "track_refund"] * 3
    clf = TfidfClassifier().fit(texts, labels)
    clf.save(tmp_path / "m.joblib")
    pred, conf = TfidfClassifier.load(tmp_path / "m.joblib").predict(["where is my order"])[0]
    assert pred == "track_order" and 0 < conf <= 1


def test_rerank_retriever_reorders_candidates():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.vectorstores import InMemoryVectorStore

    from support_agent.kb import RerankRetriever

    docs = [Document(page_content=f"passage {i}", metadata={"intent": f"i{i}"}) for i in range(5)]
    store = InMemoryVectorStore.from_documents(docs, DeterministicFakeEmbedding(size=16))

    class FakeReranker:  # score = numéro du passage : le passage 4 doit sortir en premier
        def predict(self, pairs):
            return [int(p[1].split()[-1]) for p in pairs]

    out = RerankRetriever(store=store, reranker=FakeReranker(), fetch_k=5, k=2).invoke("q")
    assert [d.metadata["intent"] for d in out] == ["i4", "i3"]
