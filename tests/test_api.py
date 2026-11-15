from pathlib import Path

from conftest import FakeClassifier, FakeRetriever, fake_llm
from fastapi.testclient import TestClient

from support_agent import api
from support_agent.graph import build_graph

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
ANSWER = "Standard delivery takes 3 to 5 business days after shipping."


def client_with(intent="delivery_period"):
    graph = build_graph(FakeClassifier(intent), FakeRetriever(), fake_llm(ANSWER), verifier=lambda a, d, t="": None)
    api.set_graph(graph)
    return TestClient(api.app)


def test_health():
    assert client_with().get("/health").json() == {"status": "ok"}


def test_text_ticket():
    r = client_with().post("/tickets", data={"text": "When will my parcel arrive?"})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "answered" and body["answer"] == ANSWER
    assert body["sources"] == ["delivery.md"]
    assert [t["node"] for t in body["trace"]][-1] == "finalize"


def test_pdf_ticket_upload():
    with open(SAMPLES / "ticket_letter.pdf", "rb") as f:
        r = client_with("track_refund").post("/tickets", files={"file": ("letter.pdf", f, "application/pdf")})
    assert r.status_code == 200 and r.json()["modality"] == "pdf"


def test_missing_input_and_bad_type():
    c = client_with()
    assert c.post("/tickets").status_code == 422
    r = c.post("/tickets", files={"file": ("x.docx", b"abc", "application/octet-stream")})
    assert r.status_code == 415
