# Teximal

Run, serve and evaluate Teximal's models. The first family is **Fort**: give it a text and a question with
options, and it tells you which option applies and how sure it is. Fort never writes free text: every answer
is one of your options, read from the model's probabilities, so the answer always fits and the confidence is a
probability you can act on: take confident answers, and send unsure ones to a person or a larger model.

## Install

Python 3.10 or later:

```bash
pip install teximal
```

On Apple silicon it runs on MLX; everywhere else on PyTorch, on an NVIDIA GPU when there is one and on the
CPU otherwise. For a CPU-only machine, installing PyTorch's CPU build first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) saves a large download.

## The command line

```bash
teximal pull fort-1-0.8b                                     # download once; cached after
teximal list                                                 # Teximal models on this machine
teximal run fort-1-0.8b "I was charged twice." --options billing,technical,sales
cat tickets.txt | teximal run fort-1-0.8b --task route.json  # one JSON line per input line, in batches
teximal serve fort-1-0.8b                                    # the Decisions API at http://127.0.0.1:8766
teximal eval fort-1-0.8b --data labeled.csv                  # accuracy and calibration on your examples
```

A task file names the question and the options, with a short definition for each (definitions help):

```json
{"question": "Which team should handle this?",
 "options": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, error messages",
             "sales": "plans, upgrades, pricing"}}
```

`teximal eval` reads a CSV with `text` and `label` columns and reports accuracy, macro-F1 and calibration
error (ECE: how far the confidence is from how often Fort is right). It also fits one number, a temperature,
on half your examples and checks it on the other half; when it helps there, pass it to `run` or `serve` with
`--temperature` to make the confidence honest on your task.

## Python

```python
from teximal import Fort

fort = Fort("teximal/fort-1-0.8b")           # or fort-1-2b, or a local folder
route = fort.task({"billing": "invoices, payments, refunds",
                   "technical": "bugs, outages, error messages",
                   "sales": "plans, upgrades, pricing"}, question="Which team should handle this?")

d = route.decide("I was charged twice for my subscription this month.")
d.choice, d.confidence, d.probs              # the option, its probability, every option's probability
decisions = route.decide_many(["Refund please.", "The app crashes on start."])   # bulk, in batches
```

Several questions about one text, in the request shape of OpenRouter's Decisions API (the one TypeSafe's Jev
is served through):

```python
fort.decide("The app crashes every time I open it. Fix it today or I cancel.", {
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors"}},
    "churn":   {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "critical"]}})
```

- **choice:** one of up to 26 options, with a probability for each. Longer label lists (77 bank intents, say)
  are read by writing the label's name, one allowed token at a time, so the answer is always a listed label;
  only the chosen label's probability is returned.
- **noul:** the probability of yes.
- **score:** an ordered scale, low to high: the expected position, each level's probability and the legend.

`teximal.load("teximal/fort-1-2b")` returns the right class for any Teximal model, read from the model's
`teximal.json`.

## The server

`teximal serve` answers Decisions API requests (`model`, `state`, `questions`), so switching from Jev is a
change of URL. Send `"states": [...]` instead of `"state"` for many texts in one request.

## Models

| Model | Reads | Best for |
|---|---|---|
| [teximal/fort-1-2b](https://huggingface.co/teximal/fort-1-2b) | the text first | accuracy |
| [teximal/fort-1-0.8b](https://huggingface.co/teximal/fort-1-0.8b) | the options first | speed, small devices |

The weights are in the standard Qwen3.5 layout, so transformers and MLX load them directly. Each size also
has a 4-bit MLX version (`teximal/fort-1-2b-mlx-4bit`, `teximal/fort-1-0.8b-mlx-4bit`) at about a third of the
memory. The model cards hold every score, the settings behind them, and the limitations. Each model's
`teximal.json` tells this package which family it is, which order it reads in, its confidence defaults and
which backends can run it: nothing to set by hand.

## Backends

`--backend mlx|torch` (or `backend=` in Python, or `TEXIMAL_BACKEND`) picks one; the default is MLX on Apple
silicon and PyTorch elsewhere. For PyTorch, `--device` (`cuda`, `cpu`, `mps`, ...) and `--dtype` (`bfloat16`,
`float32`) override the defaults: the GPU in bfloat16 when there is one, else the CPU in float32. Both backends
use the same prompts and readouts; MLX reuses the shared beginning of prompts, PyTorch reads each prompt
whole, so their probabilities differ only by floating-point detail: on 900 test decisions across both sizes,
by 0.003 on average, with the same choice and yes/no answers throughout (four 1-to-5 scores split differently
where MLX had an exact tie). On NVIDIA GPUs, `pip install flash-linear-attention
causal-conv1d` gives transformers its fast kernels for Fort's linear-attention layers.

## Notes

- Bulk decisions run texts side by side; batched arithmetic can move a probability by a few hundredths, so a
  pick can differ from a one-at-a-time decision only near a tie.
- Fine-tuning on your own labeled examples (`teximal finetune`) comes next.

## License

Apache-2.0. The models are fine-tuned from Qwen3.5 (Apache-2.0); see each model's NOTICE.
