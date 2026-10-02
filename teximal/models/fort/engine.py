"""
Fort's decision engine on MLX (Apple silicon). A question and its options become Fort's typed prompt, the
model reads it once, and the answer is read at a single position, restricted to the option letters: every
answer is one of the options and its probability comes straight from the model.
"""
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .prompts import Prompts


def _copy(st):
    if isinstance(st, list):
        return [None if a is None else mx.array(a) for a in st]
    if isinstance(st, tuple):
        return tuple(mx.array(a) for a in st)
    return mx.array(st)


def snapshot(cache):
    return [(_copy(c.state), getattr(c, "meta_state", None)) for c in cache]


def restore(cache, snap):
    for c, (arrs, meta) in zip(cache, snap):
        c.state = _copy(arrs)          # linear-attention layers replace their state lists, so copy, never share
        if meta:
            c.meta_state = meta


class Engine(Prompts):
    def __init__(self, path, question_first):
        self.model, self.tok = load(path)
        self.question_first = question_first
        self._letters, self._memo, self._names = {}, None, {}

    def letter_ids(self, n):
        if n not in self._letters:
            self._letters[n] = mx.array(self.letter_token_ids(n))
        return self._letters[n]

    # ---- forward passes
    def last_logits(self, ids, cache):
        out = self.model(mx.array([ids]), cache=cache)
        return (out.logits if hasattr(out, "logits") else out)[0, -1, :]

    def letter_probs(self, logits, n, temperature=1.0):
        p = mx.softmax(logits[self.letter_ids(n)].astype(mx.float32) / temperature)
        mx.eval(p)
        return [float(x) for x in p]

    def ask(self, state, question, labels, temperature=1.0):
        """Probabilities over the options for one question about one text. Options first, the part before
        the text is read once per question and options, and reused."""
        ids = self.ids(self.prompt(state, question, labels))
        cache = make_prompt_cache(self.model)
        if self.question_first and state:
            pre = self.ids(self.question_prefix(question, labels))
            if ids[:len(pre)] == pre:
                if not self._memo or self._memo[0] != pre:
                    self.last_logits(pre, cache)
                    mx.eval([c.state for c in cache])
                    self._memo = (pre, snapshot(cache))
                restore(cache, self._memo[1])
                return self.letter_probs(self.last_logits(ids[len(pre):], cache), len(labels), temperature), len(ids)
        return self.letter_probs(self.last_logits(ids, cache), len(labels), temperature), len(ids)

    def ask_many(self, state, questions, temperature=1.0):
        """Several questions about one text. Text first, the text is read once and every question branches
        off it; options first, each question reuses its own options."""
        if self.question_first or not state or len(questions) == 1:      # one question: the scoreboard's exact path
            return [self.ask(state, q, labels, temperature) for q, labels in questions]
        cache = make_prompt_cache(self.model)
        pre = self.ids(self.state_prefix(state))
        self.last_logits(pre, cache)
        mx.eval([c.state for c in cache])
        snap, out = snapshot(cache), []
        for q, labels in questions:
            full = self.ids(self.prompt(state, q, labels))       # the prompt tokenized whole, as in training
            if full[:len(pre)] != pre:                          # a token merged across the seam: read it whole
                out.append(self.ask(state, q, labels, temperature))
                continue
            out.append((self.letter_probs(self.last_logits(full[len(pre):], cache), len(labels), temperature), len(full)))
            restore(cache, snap)
        return out

    def bulk(self, texts, question, labels, temperature=1.0, batch=32):
        """One question about many texts, side by side, padded at the end (each answer is read at its own
        text's last token, which padding after it cannot change). Options first, the question and options are
        read once for the whole batch. (probabilities, ms) per text; ms is its share of its batch's time."""
        body = getattr(self.model, "language_model", self.model).model
        emb, ids = body.embed_tokens, self.letter_ids(len(labels))         # Fort-1 ties its word tables
        if hasattr(emb, "scales"):                                          # 4-bit: the table is packed
            letters = lambda h: emb.as_linear(h)[:, ids]
        else:
            rows = emb.weight[ids]
            letters = lambda h: h @ rows.T
        pre = self.ids(self.question_prefix(question, labels)) if self.question_first else []
        if pre:
            cache = make_prompt_cache(self.model)
            self.last_logits(pre, cache)
            snap = snapshot(cache)
        out = []
        for i in range(0, len(texts), batch):
            t0 = time.perf_counter()
            seqs = [self.ids(self.prompt(t, question, labels)) for t in texts[i:i + batch]]
            if pre and not all(s[:len(pre)] == pre for s in seqs):         # a token merged across the boundary
                for t in texts[i:i + batch]:
                    t1 = time.perf_counter()
                    p, _ = self.ask(t, question, labels, temperature)
                    out.append((p, round(1000 * (time.perf_counter() - t1), 1)))
                continue
            seqs = [s[len(pre):] for s in seqs]
            B, L = len(seqs), max(map(len, seqs))
            cache = make_prompt_cache(self.model)
            if pre:
                for c, (st, _) in zip(cache, snap):
                    c.state = (tuple(mx.repeat(a, B, axis=0) for a in st) if isinstance(st, tuple)
                               else [None if a is None else mx.repeat(a, B, axis=0) for a in st])
            h = body(mx.array([s + [0] * (L - len(s)) for s in seqs]), cache=cache)
            h = h[mx.arange(B), mx.array([len(s) - 1 for s in seqs])]
            p = mx.softmax(letters(h).astype(mx.float32) / temperature, axis=-1)
            mx.eval(p)
            ms = round(1000 * (time.perf_counter() - t0) / B, 2)
            out += [(row, ms) for row in p.tolist()]
        return out

    def by_names(self, text, names, instructions="", temperature=1.0):
        key = (tuple(names), instructions)
        if key not in self._names:
            cache = make_prompt_cache(self.model)
            self.last_logits(self.ids(self.names_prefix(names, instructions)), cache)
            mx.eval([c.state for c in cache])
            self._names[key] = (cache, snapshot(cache), self.trie(names))
        cache, snap, node = self._names[key]
        restore(cache, snap)
        stop = self.stop_id()
        last, logp = self.last_logits(self.ids(self.names_suffix(text)), cache), 0.0
        while True:
            allowed = [t for t in node if t is not None]
            if None in node:
                if not allowed:
                    break
                allowed.append(stop)
            z = last[mx.array(allowed)].astype(mx.float32) / temperature
            p = mx.softmax(z)
            j = int(mx.argmax(p))
            logp += float(mx.log(p[j]))
            if allowed[j] == stop:
                break
            node = node[allowed[j]]
            last = self.last_logits([allowed[j]], cache)
        return node[None], float(mx.exp(mx.array(logp)))
