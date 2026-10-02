"""Label every row of a CSV: python examples/bulk_csv.py reviews.csv text > labeled.csv"""
import csv, sys

from teximal import Fort

path, column = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path)))
task = Fort("teximal/fort-1-0.8b").task({"negative": "the writer is unhappy or critical",
                                         "positive": "the writer is pleased or approving"})
out = csv.writer(sys.stdout)
out.writerow([column, "label", "confidence"])
for r, d in zip(rows, task.decide_many([r[column] for r in rows])):
    out.writerow([r[column], d.choice, d.confidence])
