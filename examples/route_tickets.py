"""Route support tickets to a team, and send unsure ones to a person."""
from teximal import Fort

fort = Fort("teximal/fort-1-0.8b")
route = fort.task({"billing": "invoices, payments, refunds", "technical": "bugs, outages, error messages",
                   "account": "login, password, profile, closing the account"}, question="Which team should handle this?")
for text in ["I was charged twice for March.", "The dashboard shows error 502 since this morning.",
             "How do I change the email on my profile?", "Hello?"]:
    d = route.decide(text)
    print(f"{d.choice if d.confidence >= 0.8 else 'a person (unsure)':<20} {d.confidence:.2f}  {text}")
