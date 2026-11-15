import pytest
from langchain_core.documents import Document

from support_agent.grounding import NLIVerifier, split_sentences, unsupported_numbers

DOCS = [Document(page_content="Customers can return most items within 30 days of delivery for a full refund. "
                              "Return shipping costs 4.90 EUR.")]


def test_split_sentences():
    assert split_sentences("Hello. You can return it!\n- Step two here") == ["Hello.", "You can return it!",
                                                                            "Step two here"]


def test_unsupported_numbers_allows_ticket_numbers():
    assert unsupported_numbers("Within 30 days, cost 4.90 EUR.", DOCS) == set()
    assert unsupported_numbers("Within 60 days.", DOCS) == {"60"}
    # le numéro de commande vient du ticket : il est autorisé
    assert unsupported_numbers("Your order 482913 can be returned.", DOCS, ticket="order NS-482913") == set()


@pytest.mark.slow
def test_nli_verifier_detects_contradiction():
    v = NLIVerifier()
    assert v("Thank you for your message. You can return the item within 30 days of delivery.", DOCS) is None
    assert "numbers" in v("You can return the item within 60 days.", DOCS)
    reason = NLIVerifier(check_numbers=False)("Items cannot be returned for a refund.", DOCS)
    assert reason and "contradicted" in reason
