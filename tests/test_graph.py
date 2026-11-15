from conftest import FakeClassifier, fake_llm

from support_agent.graph import build_graph, rule_checks
from support_agent.grounding import parse_verdict

GOOD = "Standard delivery takes 3 to 5 business days. You can follow your parcel in Account > Orders."


class ScriptedVerifier:
    """Renvoie les verdicts dans l'ordre : None = acceptée, sinon raison du rejet."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def __call__(self, answer, docs, ticket=""):
        v = self.verdicts.pop(0)
        return None if v == "SUPPORTED" else "judge: answer not supported by the context"


def run(classifier, retriever, answers, verdicts, text="When will my parcel arrive?"):
    graph = build_graph(classifier, retriever, fake_llm(*answers), verifier=ScriptedVerifier(verdicts))
    return graph.invoke({"ticket_text": text})


def nodes(state):
    return [t["node"] for t in state["trace"]]


def test_answered_path(fake_retriever):
    state = run(FakeClassifier("delivery_period"), fake_retriever, [GOOD], ["SUPPORTED"])
    assert state["status"] == "answered"
    assert state["answer"] == GOOD
    assert nodes(state) == ["ingest", "classify", "retrieve", "generate", "verify", "finalize"]
    assert fake_retriever.queries[0].startswith("delivery period:")
    assert all(t["seconds"] >= 0 for t in state["trace"])


def test_human_intent_is_escalated_without_llm(fake_retriever):
    state = run(FakeClassifier("contact_human_agent"), fake_retriever, [], [])
    assert state["status"] == "escalated"
    assert "always handled by a human" in state["escalation_reason"]
    assert "retrieve" not in nodes(state)


def test_low_confidence_is_escalated(fake_retriever):
    state = run(FakeClassifier("track_order", confidence=0.2), fake_retriever, [], [])
    assert state["status"] == "escalated"
    assert "low classification confidence" in state["escalation_reason"]


def test_retry_then_success(fake_retriever):
    state = run(FakeClassifier("delivery_period"), fake_retriever, ["Delivery takes 1 day.", GOOD],
                ["NOT_SUPPORTED", "SUPPORTED"])
    assert state["status"] == "answered"
    assert nodes(state).count("generate") == 2
    assert state["escalation_reason"] is None


def test_retry_then_escalate(fake_retriever):
    state = run(FakeClassifier("delivery_period"), fake_retriever, ["Delivery takes 1 day."] * 2,
                ["NOT_SUPPORTED"] * 2)
    assert state["status"] == "escalated"
    assert "not supported" in state["escalation_reason"]


def test_placeholder_rejected_before_judge(fake_retriever):
    bad = "Please visit {{Order Page URL}} to follow your order and wait for delivery."
    state = run(FakeClassifier("track_order"), fake_retriever, [bad, bad], [])
    assert state["status"] == "escalated"
    assert "placeholder" in state["escalation_reason"]


def test_parse_verdict():
    assert parse_verdict("SUPPORTED") == "SUPPORTED"
    assert parse_verdict("Supported. Nothing to add.") == "SUPPORTED"
    assert parse_verdict("NOT_SUPPORTED") == "NOT_SUPPORTED"
    assert parse_verdict("not supported: the delay is wrong") == "NOT_SUPPORTED"
    assert parse_verdict("I don't know") == "NOT_SUPPORTED"


def test_rule_checks():
    assert rule_checks("ok") == "answer too short"
    assert rule_checks(GOOD) is None
