"""
Fort's prompts, shared by every backend. They match the ones Fort was trained on character for character;
change them and accuracy drops.
"""
OPEN = "<|im_start|>user\n"
CLOSE = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
LETTERS = [chr(ord("A") + i) for i in range(26)]
HEAD = "You answer questions about the STATE below. Reply with a single letter.\n\n"


class Prompts:
    """Needs self.tok (a Hugging Face tokenizer) and self.question_first."""

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def letter_token_ids(self, n):
        return [self.ids(L)[-1] for L in LETTERS[:n]]

    # ---- lettered options: one question about one text
    @staticmethod
    def _lines(labels):
        return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(labels))

    def question_prefix(self, question, labels):
        return OPEN + HEAD + "### QUESTION\n" + question.strip() + "\n" + self._lines(labels) + "\n\n### STATE\n"

    def state_prefix(self, state):
        if not state:
            return OPEN + "Answer the question. Reply with a single letter.\n\n"
        return OPEN + HEAD + "### STATE\n" + state.strip() + "\n\n"

    def question_block(self, question, labels):
        return "### QUESTION\n" + question.strip() + "\n" + self._lines(labels) + "\nAnswer with one letter." + CLOSE

    def prompt(self, state, question, labels):
        if self.question_first and state:
            return self.question_prefix(question, labels) + state.strip() + "\n\nAnswer with one letter." + CLOSE
        return self.state_prefix(state) + self.question_block(question, labels)

    # ---- long label lists: the model writes a label's name, one allowed token at a time
    @staticmethod
    def names_prefix(names, instructions=""):
        # the trained shape: one "Classify ..." sentence, then "Reply with the category name only."
        head = (instructions.strip() or "Classify the message into exactly one of these categories.") + \
            " Reply with the category name only.\n\n"
        return OPEN + head + "\n".join(names) + '\n\nMessage: "'

    @staticmethod
    def names_suffix(text):
        return text.strip() + '"' + CLOSE

    def names_ids(self, names, instructions, text):
        """The two parts are tokenized apart, as in training."""
        return self.ids(self.names_prefix(names, instructions)), self.ids(self.names_suffix(text))

    def trie(self, names):
        root = {}
        for i, n in enumerate(names):
            node = root
            for t in self.ids(n):
                node = node.setdefault(t, {})
            node[None] = i
        return root

    def stop_id(self):
        return self.ids("<|im_end|>")[-1]
