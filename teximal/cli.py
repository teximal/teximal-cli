"""
The teximal command.

  teximal pull fort-1-0.8b                                    download (once; cached after)
  teximal list                                                Teximal models on this machine
  teximal run fort-1-0.8b "I was charged twice" --options billing,technical,sales
  cat tickets.txt | teximal run fort-1-0.8b --task route.json one JSON line per input line, in batches
  teximal serve fort-1-0.8b                                   the Decisions API at http://127.0.0.1:8766
  teximal eval fort-1-0.8b --data labeled.csv                 accuracy and calibration on your examples

A task file is {"question": "...", "options": {"billing": "invoices, payments, refunds", ...}} (or a list of
option names). Model names without a slash are teximal/<name>; a local folder works too. run, serve and eval take
--backend mlx|torch (default: MLX on Apple silicon, PyTorch elsewhere), and --device and --dtype for PyTorch.
"""
import argparse, json, os, sys

from .hub import cached, load, pull, resolve


def task_spec(args, labels=None):
    if args.task:
        spec = json.load(open(args.task))
        return spec["options"], spec.get("question") or args.question
    if args.options:
        return [o.strip() for o in args.options.split(",") if o.strip()], args.question
    if labels:
        return sorted(set(labels)), args.question
    sys.exit("teximal: give the options with --options a,b,c or --task task.json")


def cmd_pull(args):
    print(pull(args.model))


def cmd_list(args):
    rows = cached()
    if not rows:
        print("no Teximal models downloaded yet (teximal pull fort-1-0.8b)")
    for repo, gb, path in rows:
        print(f"{repo:<32} {gb:6.2f} GB  {path}")


def runtime(args):
    return dict(backend=args.backend, device=args.device, dtype=args.dtype)


def open_model(args, temperature):
    """Downloaded first if it has to be (with progress bars), then loaded quietly: answers on stdout, nothing else."""
    path = resolve(args.model)
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()                     # transformers draws its weight-loading bar with these
    return load(path, temperature=temperature, **runtime(args))


def cmd_run(args):
    model = open_model(args, args.temperature)
    options, question = task_spec(args)
    task = model.task(options, question)
    texts = [args.text] if args.text and args.text != "-" else [l.rstrip("\n") for l in sys.stdin if l.strip()]
    decisions = task.decide_many(texts, batch=args.batch) if len(texts) > 1 else [task.decide(t) for t in texts]
    for text, d in zip(texts, decisions):
        if args.json or len(texts) > 1:
            print(json.dumps({"text": text, "choice": d.choice, "confidence": round(d.confidence, 4),
                              "probabilities": {k: round(v, 4) for k, v in d.probs.items()}}))
        else:
            print(f"{d.choice}  {d.confidence:.2f}")


def cmd_serve(args):
    from .serve import serve
    serve(open_model(args, args.temperature), args.host, args.port)


def cmd_eval(args):
    from .evaluate import evaluate, read, report, save
    rows = read(args.data, args.text_column, args.label_column)
    options, question = task_spec(args, [g for _, g in rows])
    out = evaluate(open_model(args, 1.0), rows, options, question)
    print(report(out))
    if args.save:
        save(out, args.save)


def add_runtime(p):
    p.add_argument("--backend", choices=["mlx", "torch"], help="default: mlx on Apple silicon, torch elsewhere")
    p.add_argument("--device", help="torch: cuda, cpu, mps... (default: the GPU if there is one)")
    p.add_argument("--dtype", help="torch: bfloat16 or float32 (default: bfloat16 on a GPU, float32 on a CPU)")


def main():
    ap = argparse.ArgumentParser(prog="teximal", description="Run, serve and evaluate Teximal's models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull", help="download a model")
    p.add_argument("model")
    p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("list", help="Teximal models on this machine")
    p.set_defaults(fn=cmd_list)
    for name, fn, helptext in (("run", cmd_run, "decide one text, or one per stdin line"),
                               ("eval", cmd_eval, "accuracy and calibration on your labeled examples")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("model")
        if name == "run":
            p.add_argument("text", nargs="?", help="the text to decide (omit, or -, to read lines from stdin)")
            p.add_argument("--json", action="store_true", help="JSON output for a single text too")
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--temperature", type=float, help="confidence temperature (from teximal eval)")
        else:
            p.add_argument("--data", required=True, help="CSV with a text column and a label column")
            p.add_argument("--text-column", default="text")
            p.add_argument("--label-column", default="label")
            p.add_argument("--save", help="write the results as JSON")
        p.add_argument("--options", help="comma-separated option names (eval defaults to the labels in --data)")
        p.add_argument("--task", help="task file: {\"question\": ..., \"options\": {name: definition}}")
        p.add_argument("--question", default="Which category best describes the text? Pick exactly one.")
        add_runtime(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("serve", help="answer Decisions API requests locally")
    p.add_argument("model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--temperature", type=float, help="confidence temperature (from teximal eval)")
    add_runtime(p)
    p.set_defaults(fn=cmd_serve)
    args = ap.parse_args()
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")     # no library notes on stderr (kernel hints: README)
    from huggingface_hub.errors import HfHubHTTPError
    try:
        args.fn(args)
    except (ValueError, OSError, HfHubHTTPError) as e:     # a clear line, not a traceback
        sys.exit(f"teximal: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
