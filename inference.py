"""
inference.py — Local inference for the Tanglish DistilGPT2 model.
"""

import logging
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model path — relative to this file's location
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "weights"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_suggester = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tok = None
_EOS = None


def tokenize_words(text):
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", str(text).lower())
    return re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", text)


BOS, EOS_W = "<s>", "</s>"

class NGram:
    def __init__(self, sents, vocab, discount=0.75):
        self.vocab = vocab
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.w2i[EOS_W] = len(vocab)
        self.V = len(vocab)
        self.d = discount
        self.c1 = defaultdict(Counter)
        self.c2 = defaultdict(Counter)
        left = defaultdict(set)
        for s in sents:
            seq = [BOS, BOS] + s + [EOS_W]
            for i in range(2, len(seq)):
                self.c1[(seq[i - 1],)][seq[i]] += 1
                self.c2[(seq[i - 2], seq[i - 1])][seq[i]] += 1
                left[seq[i]].add(seq[i - 1])
        uni = np.full(self.V + 1, 0.1)
        for w, i in self.w2i.items():
            uni[i] += len(left.get(w, ()))
        self.uni = uni / uni.sum()

    def _interp(self, lower, counts):
        if not counts:
            return lower
        total = sum(counts.values())
        out = lower * (self.d * len(counts) / total)
        for w, n in counts.items():
            j = self.w2i.get(w)
            if j is not None:
                out[j] += max(n - self.d, 0.0) / total
        return out

    def probs(self, ctx):
        seq = [BOS, BOS] + list(ctx)
        p = self.uni.copy()
        p = self._interp(p, self.c1.get((seq[-1],)))
        p = self._interp(p, self.c2.get((seq[-2], seq[-1])))
        p = p[:self.V]
        return p / p.sum()


class LMScorer:
    def __init__(self, model, vocab, chunk=128):
        self.model, self.vocab, self.chunk = model.eval(), vocab, chunk
        self.first = [_tok.encode(w) for w in vocab]
        self.later = [_tok.encode(" " + w) for w in vocab]
        n_tok = model.get_output_embeddings().weight.shape[0]
        boundary = []
        for i in range(min(n_tok, len(_tok))):
            s = _tok.decode([i])
            if i == _EOS or (s and (s[0].isspace() or not (s[0].isalnum() or s[0] == "'"))):
                boundary.append(i)
        self.boundary = torch.tensor(boundary, device=_device)
        self.cache = {}

    @torch.no_grad()
    def log_probs(self, ctx):
        key = tuple(ctx)
        if key in self.cache:
            return self.cache[key]
        prefix = [_EOS] + (_tok.encode(" ".join(ctx)) if ctx else [])
        P = len(prefix)
        cands = self.later if ctx else self.first
        order = sorted(range(len(cands)), key=lambda i: len(cands[i]))
        scores = np.zeros(len(cands))
        for a in range(0, len(order), self.chunk):
            idx = order[a:a + self.chunk]
            n, L = len(idx), max(len(cands[i]) for i in idx)
            ids = torch.full((n, P + L), _EOS, dtype=torch.long)
            att = torch.zeros((n, P + L), dtype=torch.long)
            tgt = torch.zeros((n, L), dtype=torch.long)
            msk = torch.zeros((n, L))
            last = torch.zeros(n, dtype=torch.long)
            for r, i in enumerate(idx):
                seq = prefix + cands[i]
                ids[r, :len(seq)] = torch.tensor(seq)
                att[r, :len(seq)] = 1
                tgt[r, :len(cands[i])] = torch.tensor(cands[i])
                msk[r, :len(cands[i])] = 1
                last[r] = len(seq) - P
            ids, att, tgt, msk, last = (t.to(_device) for t in (ids, att, tgt, msk, last))
            logits = self.model(input_ids=ids, attention_mask=att).logits[:, P - 1:, :]
            lp = F.log_softmax(logits.float(), dim=-1)
            tok_lp = lp[:, :L, :].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            word_lp = (tok_lp * msk).sum(1)
            end_lp = torch.logsumexp(lp[torch.arange(n, device=_device), last][:, self.boundary], dim=-1)
            scores[idx] = (word_lp + end_lp).cpu().numpy()
        scores = scores - np.logaddexp.reduce(scores)
        self.cache[key] = scores
        return scores

    def probs(self, ctx):
        return np.exp(self.log_probs(ctx))


class Suggester:
    def __init__(self, vocab, ngram, lm=None, alpha=0.5):
        self.vocab, self.ng, self.lm, self.alpha = vocab, ngram, lm, alpha
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self._starts = {}

    def blend(self, p_lm, p_ng, alpha=None):
        a = self.alpha if alpha is None else alpha
        return a * p_lm + (1 - a) * p_ng

    def prefix_mask(self, prefix):
        if prefix not in self._starts:
            self._starts[prefix] = np.array([w.startswith(prefix) for w in self.vocab])
        return self._starts[prefix]

    def probs(self, ctx):
        p_ng = self.ng.probs(ctx)
        if self.lm is None:
            return p_ng
        return self.blend(self.lm.probs(ctx), p_ng)

    def suggest(self, text, k=3):
        words = tokenize_words(text)
        typing_word = bool(text) and not text[-1].isspace() and bool(words)
        prefix = words[-1] if typing_word else ""
        ctx = words[:-1] if typing_word else words
        p = self.probs(ctx)
        if prefix:
            mask = self.prefix_mask(prefix)
            if not mask.any():
                return [prefix]
            p = np.where(mask, p, -1.0)
        top = np.argsort(-p)[:k]
        return [self.vocab[i] for i in top if p[i] >= 0]


def _load_model(model_dir: Path = _DEFAULT_MODEL_DIR) -> None:
    global _suggester, _tok, _EOS, _device

    if _suggester is not None:
        return

    if not model_dir.exists():
        raise RuntimeError(
            f"Model directory not found: {model_dir}\n"
            "Extract tanglish-distilgpt2.zip (from Kaggle output) into models/."
        )

    logger.info("Loading tokenizer from %s ...", model_dir)
    _tok = AutoTokenizer.from_pretrained(str(model_dir))

    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _EOS = _tok.eos_token_id

    logger.info("Loading model on %s ...", _device.upper())
    lm = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device).eval()

    meta_path = model_dir / "tanglish_meta.json"
    if not meta_path.exists():
        raise RuntimeError("tanglish_meta.json not found in model_dir")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab = sorted({w for s in meta["sentences"] for w in s})
    ngram = NGram(meta["sentences"], vocab, meta["discount"])
    lm_scorer = LMScorer(lm, vocab)
    _suggester = Suggester(vocab, ngram, lm_scorer, alpha=meta["alpha"])
    logger.info("Model ready.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_completion(prompt: str, max_new_tokens: int = 15, **kwargs) -> str:
    """Compatibility method for old route"""
    _load_model()
    suggestions = _suggester.suggest(prompt, k=1)
    return suggestions[0] if suggestions else ""


def generate_suggestions(prompt: str, k: int = 3) -> list[str]:
    """Return top k next words using the hybrid N-gram + LM model."""
    _load_model()
    return _suggester.suggest(prompt, k=k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test_prompts = [
        "naan gym ",
        "naan gy",
        "enna ",
        "machi ",
    ]

    print("\n── Tanglish completion test ──────────────────────────")
    for p in test_prompts:
        c = generate_suggestions(p)
        print(f"  [{p!r}]  →  {c!r}")
