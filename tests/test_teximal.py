"""
Checks against a real model folder:  FORT_TEST_MODEL=<folder> python -m pytest tests -q
(TEXIMAL_BACKEND=torch runs them on PyTorch.)
"""
import os

import pytest

from teximal import Fort

MODEL = os.environ.get("FORT_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set FORT_TEST_MODEL to a Fort model folder")
OPTS = {"billing": "invoices, payments, refunds", "technical": "bugs, outages, error messages"}


@pytest.fixture(scope="module")
def fort():
    return Fort(MODEL)


def test_choice_is_a_distribution(fort):
    d = fort.task(OPTS, "Which team should handle this?").decide("I was charged twice this month.")
    assert d.choice in OPTS and abs(sum(d.probs.values()) - 1) < 1e-3 and d.confidence == max(d.probs.values())


def test_typed_questions(fort):
    out = fort.decide("The app crashes every time I open it. Fix it today or I cancel.", {
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": OPTS},
        "churn": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "critical"]}})
    assert out["team"]["choice"] in OPTS
    assert 0 <= out["churn"]["noul"] <= 1
    assert 0 <= out["urgency"]["score"] <= 2 and set(out["urgency"]["legend"]) == {"0", "1", "2"}


def test_bulk_matches_one_at_a_time(fort):
    texts = ["Refund please, I was billed twice.", "The login page shows error 500.", "Why is my invoice higher?",
             "Nothing loads since the update.", "Can I get my money back?", "The export button does nothing."]
    task = fort.task(OPTS, "Which team should handle this?")
    assert [d.choice for d in task.decide_many(texts, batch=4)] == [task.decide(t).choice for t in texts]


def test_long_label_list(fort):
    labels = [f"topic_{i}" for i in range(30)] + ["card_arrival"]
    d = fort.task(labels).decide("My new card still hasn't arrived after two weeks.")
    assert d.choice in labels and 0 < d.confidence <= 1
