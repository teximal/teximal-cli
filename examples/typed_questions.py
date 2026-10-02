"""Several typed questions about one message: a choice, a yes/no and a scale."""
import json

from teximal import Fort

fort = Fort("teximal/fort-1-2b")
print(json.dumps(fort.decide({"customer_tier": "enterprise", "message": "Checkout is down for all our users!"}, {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors"}},
    "escalate": {"type": "noul", "instructions": "Should this go to the on-call engineer now?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "critical"]},
}), indent=2))
