"""Corpus loading, character vocabulary and train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class Vocab:
    """Character <-> index mapping.

    The characters are stored in sorted order so that the mapping is identical
    for every run on the same corpus.
    """

    itos: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        return cls(tuple(sorted(set(text))))

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.itos)}

    def encode(self, text: str) -> np.ndarray:
        """Map a string to an array of character indices."""
        stoi = self.stoi
        return np.array([stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        """Map an iterable of character indices back to a string."""
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self) -> int:
        return len(self.itos)


def load_text(path: str | Path, limit_chars: int | None = None) -> str:
    """Read a corpus as UTF-8, optionally truncated to the first `limit_chars`.

    Line endings are normalised so that a Windows copy of a corpus does not add
    a carriage return to the vocabulary.
    """
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return text[:limit_chars] if limit_chars else text


def make_sequences(ids: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut a stream of indices into non-overlapping (input, target) sequences.

    Targets are the inputs shifted by one character; the tail that does not fill
    a whole sequence is discarded.
    """
    num_seq = (len(ids) - 1) // seq_length
    if num_seq < 1:
        raise ValueError(f"corpus of {len(ids)} chars is too short for seq_length={seq_length}")
    usable = num_seq * seq_length
    x = ids[:usable].reshape(num_seq, seq_length)
    y = ids[1 : usable + 1].reshape(num_seq, seq_length)
    return x, y


def split_sequences(
    x: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split sequences into train/val/test without shuffling.

    The split is contiguous so that validation and test text is never seen
    during training, which shuffling of overlapping text would not guarantee.
    """
    num_seq = len(x)
    num_test = int(num_seq * test_frac)
    num_val = int(num_seq * val_frac)
    num_train = num_seq - num_val - num_test
    if min(num_train, num_val, num_test) < 1:
        raise ValueError(f"{num_seq} sequences are too few for a {val_frac}/{test_frac} split")

    bounds = [(0, num_train), (num_train, num_train + num_val), (num_train + num_val, num_seq)]
    return tuple(
        TensorDataset(torch.from_numpy(x[a:b]), torch.from_numpy(y[a:b])) for a, b in bounds
    )


def build_datasets(
    path: str | Path,
    seq_length: int,
    limit_chars: int | None = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[Vocab, TensorDataset, TensorDataset, TensorDataset]:
    """Load a corpus and return its vocabulary and the three dataset splits.

    The vocabulary is built from the whole corpus, so a character that only
    occurs in the held-out text is still a known symbol at evaluation time.
    """
    text = load_text(path, limit_chars)
    vocab = Vocab.from_text(text)
    x, y = make_sequences(vocab.encode(text), seq_length)
    train_ds, val_ds, test_ds = split_sequences(x, y, val_frac, test_frac)
    return vocab, train_ds, val_ds, test_ds
