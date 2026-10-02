"""
`teximal eval`: how a model does on your own labeled examples, and the one number that makes its confidence
honest on them. Accuracy, macro-F1, calibration error (ECE: the gap between how sure it says it is and how
often it is right), and a temperature fitted on half the examples and checked on the other half; pass it back
with --temperature to run and serve.
"""
import csv, json, math


def read(path, text_col, label_col):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    missing = {text_col, label_col} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path}: no column(s) {', '.join(sorted(missing))}; use --text-column / --label-column")
    return [(r[text_col], r[label_col].strip()) for r in rows if r[text_col].strip() and r[label_col].strip()]


def ece(conf, ok, bins=10):
    total, n = 0.0, len(conf)
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if min(int(c * bins), bins - 1) == b]
        if idx:
            total += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(ok[i] for i in idx) / len(idx))
    return total


def soften(probs, T):
    z = {k: math.log(max(v, 1e-12)) / T for k, v in probs.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def fit_temperature(pairs):
    """The temperature with the lowest log loss of the right label (grid from 0.25 to 8)."""
    grid = [0.25 * (32 ** (i / 80)) for i in range(81)]
    loss = lambda T: -sum(math.log(max(soften(p, T)[g], 1e-12)) for p, g in pairs) / len(pairs)
    return min(grid, key=loss)


def macro_f1(golds, picks, labels):
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, picks))
        fp = sum(g != lab and p == lab for g, p in zip(golds, picks))
        fn = sum(g == lab and p != lab for g, p in zip(golds, picks))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def evaluate(model, rows, options, question):
    task = model.task(options, question)
    names = list(task.options)
    unknown = sorted({g for _, g in rows} - set(names))
    if unknown:
        raise ValueError(f"labels in the data that are not options: {', '.join(unknown[:10])}")
    decisions = task.decide_many([t for t, _ in rows])
    golds = [g for _, g in rows]
    picks = [d.choice for d in decisions]
    ok = [p == g for p, g in zip(picks, golds)]
    conf = [d.confidence for d in decisions]
    sure = [i for i, c in enumerate(conf) if c >= 0.9]
    out = dict(n=len(rows), accuracy=sum(ok) / len(ok), macro_f1=macro_f1(golds, picks, names), ece=ece(conf, ok),
               sure_share=len(sure) / len(rows), sure_wrong=(sum(not ok[i] for i in sure) / len(sure)) if sure else 0.0,
               ms=sorted(d.ms for d in decisions)[len(decisions) // 2])
    if len(names) <= 26 and len(rows) >= 40:   # long label lists return the chosen label's probability only
        fit = [(d.probs, g) for i, (d, g) in enumerate(zip(decisions, golds)) if i % 2 == 0]
        check = [(d.probs, g) for i, (d, g) in enumerate(zip(decisions, golds)) if i % 2 == 1]
        T = fit_temperature(fit)
        before = ece([max(p.values()) for p, _ in check], [max(p, key=p.get) == g for p, g in check])
        soft = [soften(p, T) for p, _ in check]
        after = ece([max(p.values()) for p in soft], [max(p, key=p.get) == g for p, (_, g) in zip(soft, check)])
        out.update(temperature=T, ece_check_before=before, ece_check_after=after)
    return out


def report(out):
    lines = [f"examples            {out['n']}",
             f"accuracy            {100 * out['accuracy']:.1f}%",
             f"macro-F1            {100 * out['macro_f1']:.1f}%",
             f"calibration (ECE)   {out['ece']:.3f}   (0 is perfect)",
             f"90%+ confident      {100 * out['sure_share']:.0f}% of answers, {100 * out['sure_wrong']:.1f}% of those wrong",
             f"median ms           {out['ms']:.1f}"]
    if "temperature" in out:
        gain = out["ece_check_after"] < out["ece_check_before"]
        lines.append(f"fitted temperature  {out['temperature']:.2f}: ECE on held-out half {out['ece_check_before']:.3f} -> "
                     f"{out['ece_check_after']:.3f}   " + (f"(use --temperature {out['temperature']:.2f})" if gain else
                                                       "(no gain on the held-out half: keep the default)"))
    return "\n".join(lines)


def save(out, path):
    json.dump(out, open(path, "w"), indent=2)
