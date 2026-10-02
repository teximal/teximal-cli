"""
Fort's decision engine on PyTorch, through transformers: NVIDIA and AMD GPUs, and any CPU. The prompts and
the readouts are the MLX engine's; every prompt is read whole here (no reuse of a shared beginning), so
answers match the MLX engine's up to the two libraries' floating-point differences.
"""
import math, time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import Prompts

TOKENS_PER_PASS = 16384          # texts side by side, up to this many tokens (padding included) per forward pass


def _shift_conv(x, w, bias, activation):
    """Valid depthwise cross-correlation of x (batch, channels, time) with w (channels, k), as k shifted
    multiply-adds."""
    k = w.shape[-1]
    n = x.shape[-1] - k + 1
    out = x[:, :, :n] * w[:, 0, None]
    for i in range(1, k):
        out = out + x[:, :, i:i + n] * w[:, i, None]
    if bias is not None:
        out = out + bias[:, None]
    return F.silu(out) if activation in ("silu", "swish") else out


def _fast_convs():
    """Without the causal-conv1d package, transformers' Qwen3.5 layers fall back to PyTorch's depthwise
    convolution, which on a CPU takes about half a second per linear-attention layer for every prompt and every
    generated token. Off CUDA, the same numbers come from shifted multiply-adds instead; patched only when both
    match the originals on a random probe, so a changed transformers keeps its own code."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as Q
        fn0, up0 = Q.causal_conv1d_fn, Q.causal_conv1d_update
    except (ImportError, AttributeError):
        return
    if getattr(fn0, "teximal", False) or fn0 is None or up0 is None:
        return

    def fn(hidden_states, weight, bias=None, activation=None, **kw):
        if hidden_states.is_cuda or weight.dim() != 2 or any(v is not None for k, v in kw.items() if k != "use_cache"):
            return fn0(hidden_states, weight, bias=bias, activation=activation, **kw)
        return _shift_conv(F.pad(hidden_states, (weight.shape[-1] - 1, 0)), weight, bias, activation)

    def up(hidden_states, conv_state, weight, bias=None, activation=None):
        if hidden_states.is_cuda or weight.dim() != 2:
            return up0(hidden_states, conv_state, weight, bias, activation)
        x = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(x[:, :, -conv_state.shape[-1]:])
        return _shift_conv(x, weight, bias, activation)[:, :, -hidden_states.shape[-1]:].to(hidden_states.dtype)

    g = torch.Generator().manual_seed(0)
    x, w, s = (torch.randn(2, 8, 7, generator=g), torch.randn(8, 4, generator=g), torch.randn(2, 8, 4, generator=g))
    t = torch.randn(2, 8, 1, generator=g)
    s0, s1 = s.clone(), s.clone()
    same = (torch.allclose(fn(x, w, None, "silu", use_cache=False), fn0(x, w, None, "silu", use_cache=False), atol=1e-5)
            and torch.allclose(up(t, s0, w, None, "silu"), up0(t, s1, w, None, "silu"), atol=1e-5)
            and torch.equal(s0, s1))
    if same:
        fn.teximal = up.teximal = True
        Q.causal_conv1d_fn, Q.causal_conv1d_update = fn, up


def pick_device():
    if torch.cuda.is_available():                       # NVIDIA, and AMD through ROCm
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class Engine(Prompts):
    def __init__(self, path, question_first, device=None, dtype=None):
        self.device = device or pick_device()
        if dtype is None:                               # half precision on accelerators; float32 on CPUs, most
            dtype = "float32" if self.device == "cpu" else "bfloat16"     # of which have no fast bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype, dtype))
        self.dtype = dtype
        self.model, info = AutoModelForCausalLM.from_pretrained(path, dtype=self.dtype, output_loading_info=True)
        info = info if isinstance(info, dict) else vars(info)
        missing = [k for k in info.get("missing_keys") or [] if not k.endswith("lm_head.weight")]
        if missing:
            raise ValueError(f"{path}: no weights for {', '.join(sorted(missing)[:3])}...: this folder is not "
                             "in the Hugging Face layout (an MLX-only build?)")
        self.model.to(self.device).eval()
        _fast_convs()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.body = self.model.get_decoder()
        self.head = self.model.get_output_embeddings().weight
        self.question_first = question_first
        self._letters, self._names = {}, {}

    def letter_ids(self, n):
        if n not in self._letters:
            self._letters[n] = torch.tensor(self.letter_token_ids(n), device=self.device)
        return self._letters[n]

    # ---- forward passes
    @torch.inference_mode()
    def last_hidden(self, seqs):
        """Each sequence's final hidden state at its last token. Sequences go side by side, padded at the end:
        Fort reads left to right, so padding after a text's last token cannot change what is read there."""
        out, i = [], 0
        while i < len(seqs):
            j, longest = i, 0
            while j < len(seqs) and (j == i or (j + 1 - i) * max(longest, len(seqs[j])) <= TOKENS_PER_PASS):
                longest = max(longest, len(seqs[j]))
                j += 1
            group = seqs[i:j]
            x = torch.tensor([s + [0] * (longest - len(s)) for s in group], device=self.device)
            h = self.body(input_ids=x, use_cache=False).last_hidden_state
            out.append(h[torch.arange(len(group), device=self.device),
                         torch.tensor([len(s) - 1 for s in group], device=self.device)])
            i = j
        return torch.cat(out)

    @torch.inference_mode()
    def letter_probs(self, h, n, temperature=1.0):
        rows = self.head[self.letter_ids(n)]                     # Fort-1 ties its word tables
        return torch.softmax((h @ rows.T).float() / temperature, dim=-1).tolist()

    def ask(self, state, question, labels, temperature=1.0):
        ids = self.ids(self.prompt(state, question, labels))
        return self.letter_probs(self.last_hidden([ids]), len(labels), temperature)[0], len(ids)

    def ask_many(self, state, questions, temperature=1.0):
        """Several questions about one text, one prompt each, read side by side."""
        if not questions:
            return []
        if len(questions) == 1:
            return [self.ask(state, questions[0][0], questions[0][1], temperature)]
        seqs = [self.ids(self.prompt(state, q, labels)) for q, labels in questions]
        h = self.last_hidden(seqs)
        return [(self.letter_probs(h[i:i + 1], len(labels), temperature)[0], len(seqs[i]))
                for i, (_, labels) in enumerate(questions)]

    def bulk(self, texts, question, labels, temperature=1.0, batch=32):
        """One question about many texts, side by side. (probabilities, ms) per text; ms is its share of its
        batch's time."""
        out = []
        for i in range(0, len(texts), batch):
            t0 = time.perf_counter()
            seqs = [self.ids(self.prompt(t, question, labels)) for t in texts[i:i + batch]]
            p = self.letter_probs(self.last_hidden(seqs), len(labels), temperature)
            ms = round(1000 * (time.perf_counter() - t0) / len(seqs), 2)
            out += [(row, ms) for row in p]
        return out

    @torch.inference_mode()
    def by_names(self, text, names, instructions="", temperature=1.0):
        key = (tuple(names), instructions)
        if key not in self._names:
            self._names[key] = (self.ids(self.names_prefix(names, instructions)), self.trie(names))
        pre, node = self._names[key]
        step = lambda ids, cache: self.model(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache,
                                             use_cache=True, logits_to_keep=1)
        out = step(pre + self.ids(self.names_suffix(text)), None)
        last, cache, logp, stop = out.logits[0, -1], out.past_key_values, 0.0, self.stop_id()
        while True:
            allowed = [t for t in node if t is not None]
            if None in node:
                if not allowed:
                    break
                allowed.append(stop)
            p = torch.softmax(last[allowed].float() / temperature, dim=-1)
            j = int(torch.argmax(p))
            logp += float(torch.log(p[j]))
            if allowed[j] == stop:
                break
            node = node[allowed[j]]
            out = step([allowed[j]], cache)
            last, cache = out.logits[0, -1], out.past_key_values
        return node[None], math.exp(logp)
