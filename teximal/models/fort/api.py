"""
The Fort API.

    from teximal import Fort
    fort = Fort("teximal/fort-1-0.8b")                 # a Hugging Face repo or a local folder
    route = fort.task({"billing": "invoices, payments, refunds",
                       "technical": "bugs, outages, error messages"}, question="Which team should handle this?")
    d = route.decide("I was charged twice.")           # Decision(choice, confidence, probs, ms)
    ds = route.decide_many(texts)                      # bulk

    fort.decide(text, {"team": {"type": "choice", "instructions": "...", "criteria": {...}},
                       "urgent": {"type": "noul", "instructions": "Does this need an answer today?"},
                       "priority": {"type": "score", "instructions": "How urgent is this?",
                                    "criteria": ["not urgent", "soon", "critical"]}})
                                                       # the request shape of OpenRouter's Decisions API
"""
import os, time
from dataclasses import dataclass

from ...hub import config, default_backend, resolve

DEFAULT_QUESTION = "Which category best describes the text? Pick exactly one."


@dataclass
class Decision:
    choice: str
    confidence: float
    probs: dict
    ms: float


def _shown(options):
    return [f"{k.replace('_', ' ')}: {v}" if v else k.replace("_", " ") for k, v in options.items()]


class Task:
    """One question with fixed options, asked of any number of texts."""

    def __init__(self, fort, options, question):
        self.fort, self.question = fort, question
        self.options = options if isinstance(options, dict) else {k: "" for k in options}
        self.names = list(self.options)
        self.labels = _shown(self.options)

    def decide(self, text):
        t0 = time.perf_counter()
        f = self.fort
        if len(self.names) > 26:
            ins = "" if self.question == DEFAULT_QUESTION else self.question
            k, p = f.engine.by_names(text, [n.replace("_", " ") for n in self.names], ins, f.t_names)
            return Decision(self.names[k], p, {self.names[k]: p}, round(1000 * (time.perf_counter() - t0), 1))
        p, _ = f.engine.ask(text, self.question, self.labels, f.t_choice)
        probs = dict(zip(self.names, p))
        top = max(probs, key=probs.get)
        return Decision(top, probs[top], probs, round(1000 * (time.perf_counter() - t0), 1))

    def decide_many(self, texts, batch=32):
        """Texts side by side through the model, in batches. Each Decision's ms is its share of its batch's
        time."""
        f = self.fort
        if len(self.names) > 26:
            return [self.decide(t) for t in texts]
        out = []
        for p, ms in f.engine.bulk(texts, self.question, self.labels, f.t_choice, batch):
            probs = dict(zip(self.names, p))
            top = max(probs, key=probs.get)
            out.append(Decision(top, probs[top], probs, ms))
        return out


class Fort:
    def __init__(self, model, temperature=None, label_temperature=None, backend=None, device=None, dtype=None):
        """model: a Hugging Face repo id (teximal/fort-1-2b, or fort-1-2b) or a local folder. The prompt order the
        model was trained in and its confidence defaults come from the folder's teximal.json. temperature
        overrides the default for lettered answers (choice, yes/no, score), label_temperature the one for long
        label lists; `teximal eval` fits the first on your own labeled examples.

        backend: "mlx" (the default on Apple silicon) or "torch" (everywhere else). device and dtype are for
        torch: by default the GPU in bfloat16 when there is one, else the CPU in float32."""
        path = resolve(model)
        self.config = config(path)
        order = self.config.get("prompt_order", "state_first")
        temps = self.config.get("temperature", {})
        self.t_choice = temperature or temps.get("choice", 1.0)
        self.t_names = label_temperature or temps.get("label_names", 1.0)
        self.name = self.config.get("model", os.path.basename(path.rstrip("/")))
        self.backend = backend or default_backend()
        if self.backend not in self.config["backends"]:
            full = self.name.replace("-mlx-4bit", "") if self.name.endswith("-mlx-4bit") else None
            raise ValueError(f"{self.name} runs with {' or '.join(self.config['backends'])}, not {self.backend}"
                             + (f"; use {full} here" if full else ""))
        if self.backend == "mlx":
            from .engine import Engine
            self.engine = Engine(path, order == "question_first")
        elif self.backend == "torch":
            try:
                from .engine_torch import Engine
            except ImportError as e:            # Apple silicon installs MLX only
                raise ValueError(f"the PyTorch backend needs torch and transformers ({e}): "
                                 "pip install 'teximal[torch]'") from e
            self.engine = Engine(path, order == "question_first", device=device, dtype=dtype)
        else:
            raise ValueError(f"backend {self.backend!r}: mlx or torch")

    def task(self, options, question=DEFAULT_QUESTION):
        """options: a list of names, or a dict of name -> short definition (definitions help)."""
        return Task(self, options, question)

    def decide(self, state, questions):
        """Typed questions about one text, in the Decisions API's shape: {name: {"type": "choice" | "noul" |
        "score", "instructions": "...", "criteria": {...} or [...]}}. State may be text or a dict of fields."""
        text = "\n".join(f"{k}: {v}" for k, v in state.items()) if isinstance(state, dict) else str(state)
        out, plain = {}, []
        for name, spec in questions.items():
            t, ins = spec.get("type", "choice"), spec.get("instructions") or spec.get("prompt") or ""
            crit = spec.get("criteria") or {}
            if t == "noul":
                plain.append((name, t, ins, ["no", "yes"], None))
            elif t == "score":
                plain.append((name, t, ins, list(crit) if isinstance(crit, list) else list(crit.values()), None))
            elif t == "choice":
                crit = {c: "" for c in crit} if isinstance(crit, list) else crit
                if len(crit) > 26:
                    d = Task(self, crit, ins).decide(text)
                    out[name] = {"choice": d.choice, "probabilities": d.probs, "confidence": d.confidence}
                else:
                    plain.append((name, t, ins, _shown(crit), list(crit)))
            else:
                raise ValueError(f"question {name!r}: unknown type {t!r} (choice, noul or score)")
        answers = self.engine.ask_many(text, [(ins, labels) for _, _, ins, labels, _ in plain], self.t_choice)
        for (name, t, _, labels, keys), (p, _) in zip(plain, answers):
            top = max(range(len(p)), key=p.__getitem__)
            if t == "noul":
                out[name] = {"noul": p[1]}
            elif t == "score":
                out[name] = {"score": sum(i * x for i, x in enumerate(p)),
                             "probabilities": {str(i): x for i, x in enumerate(p)},
                             "confidence": p[top], "legend": {str(i): s for i, s in enumerate(labels)}}
            else:
                out[name] = {"choice": keys[top], "probabilities": dict(zip(keys, p)), "confidence": p[top]}
        return out
