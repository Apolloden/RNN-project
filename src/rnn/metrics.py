"""Evaluation metrics for synthesised text.

Perplexity and self-BLEU are computed locally; spelling accuracy needs
`pyspellchecker` and BERTScore needs `bert-score`, both of which are imported
lazily so that the rest of the package stays usable without them.
"""

from __future__ import annotations

import functools
import re

import torch
import torch.nn.functional as F

# Contractions a dictionary spell checker rejects but that are correct English.
CONTRACTIONS = re.compile(
    r"^(?:I'll|I've|I'm|I'd|don't|can't|won't|wasn't|hasn't|you're|you've|you'll|you'd|"
    r"she's|it's|that'll|should've|aren't|couldn't|didn't|doesn't|"
    r"hadn't|haven't|isn't|mightn't|mustn't|needn't|here's|shouldn't|"
    r"weren't|wouldn't|let's|he'd|he'll|he's|shan't|she'd|she'll|that's|there's|they'd|"
    r"they'll|they're|they've|we'd|we're|we've|what's|where's|who'd|who'll|"
    r"who're|who's|who've)$"
)


@torch.no_grad()
def text_perplexity(model, vocab, text: str, device: torch.device | str | None = None) -> float:
    """Character-level perplexity of `text` under `model`.

    Perplexity is exp of the mean negative log-likelihood the model assigns to
    each next character; lower means the model finds the text less surprising.
    """
    if len(text) < 2:
        raise ValueError("need at least two characters to score a next-character prediction")
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    ids = torch.tensor(vocab.encode(text), device=device).unsqueeze(0)
    logits, _ = model(ids[:, :-1])
    nll = F.cross_entropy(logits.squeeze(0), ids[0, 1:])

    model.train(was_training)
    return float(torch.exp(nll))


@functools.lru_cache(maxsize=1)
def _spell_checker():
    from spellchecker import SpellChecker

    return SpellChecker()


def spelling_accuracy(text: str) -> float:
    """Fraction of whitespace-separated tokens that are correctly spelled."""
    # Drop punctuation that would otherwise make a correct word look misspelled.
    text = re.sub(r"(?<=[A-Za-z])[”\.\,]", "", text)
    text = re.sub(r"(?=[A-Za-z])”", "", text)
    text = re.sub(r"\s&\s|(?<!\s)\?\s", " ", text)

    words = text.lower().split()
    if not words:
        return 0.0

    spell = _spell_checker()
    correct = sum(1 for w in words if w in spell or CONTRACTIONS.match(w))
    return correct / len(words)


def self_bleu(texts: list[str], max_n: int = 4) -> float:
    """Mean BLEU-n of every text against all the others.

    Used as a diversity measure: a high self-BLEU means the samples repeat each
    other, a low one means they differ.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if len(texts) < 2:
        raise ValueError("self-BLEU needs at least two samples to compare")

    tokenised = [t.split() for t in texts]
    weights = tuple(1.0 / max_n for _ in range(max_n))
    smoother = SmoothingFunction().method1
    scores = [
        sentence_bleu(
            tokenised[:i] + tokenised[i + 1 :], hyp, weights=weights, smoothing_function=smoother
        )
        for i, hyp in enumerate(tokenised)
    ]
    return sum(scores) / len(scores)


def bert_score_f1(
    candidates: list[str],
    references: list[str],
    model_type: str = "bert-base-uncased",
    device: str | None = None,
) -> dict[str, float]:
    """Mean BERTScore precision/recall/F1 of candidates against references.

    Requires `bert-score`, which downloads the BERT weights on first use.
    """
    from bert_score import score

    precision, recall, f1 = score(
        cands=candidates,
        refs=references,
        lang="en",
        model_type=model_type,
        device=device,
        rescale_with_baseline=True,
    )
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
