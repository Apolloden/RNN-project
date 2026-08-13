"""Character-level recurrent language model and text synthesis."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class CharRNN(nn.Module):
    """A recurrent layer stack followed by a linear projection to the vocabulary.

    Inputs are character indices that are one-hot encoded on the fly, so the
    model has no embedding table: it sees exactly the representation described
    in the report while keeping the datasets small enough to hold in memory.

    Args:
        vocab_size: number of distinct characters (input and output size).
        hidden_size: neurons per recurrent layer.
        num_layers: number of stacked recurrent layers.
        cell_type: one of "rnn", "lstm", "gru".
        dropout: dropout between recurrent layers; ignored when num_layers == 1.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = "lstm",
        dropout: float = 0.0,
    ):
        super().__init__()
        if cell_type not in CELLS:
            raise ValueError(f"cell_type must be one of {sorted(CELLS)}, got {cell_type!r}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.rnn = CELLS[cell_type](
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None) -> tuple[torch.Tensor, object]:
        """Run the model over a batch of index sequences.

        Args:
            x: character indices, shape (batch, seq_length).
            hidden: previous state, or None to start from zeros.

        Returns:
            logits of shape (batch, seq_length, vocab_size) and the new state.
        """
        one_hot = F.one_hot(x, num_classes=self.vocab_size).float()
        out, hidden = self.rnn(one_hot, hidden)
        return self.fc(out), hidden


def parameter_count(model: CharRNN) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training_flops(model: CharRNN, tokens: int) -> float:
    """Estimated FLOPs to train on `tokens` characters.

    Counts the multiply-accumulates in the gate matrices and the output
    projection at two FLOPs each, and takes the backward pass as twice the cost
    of the forward pass. Element-wise gate activations are ignored, so this is a
    lower bound dominated by the matrix multiplications.
    """
    gates = {"rnn": 1, "lstm": 4, "gru": 3}[model.cell_type]
    per_token = 2 * model.hidden_size * model.vocab_size  # output projection
    for layer in range(model.num_layers):
        in_size = model.vocab_size if layer == 0 else model.hidden_size
        per_token += 2 * gates * model.hidden_size * (in_size + model.hidden_size)
    return 3 * per_token * tokens


def detach_hidden(hidden):
    """Cut the graph between batches while keeping the state values."""
    if isinstance(hidden, tuple):
        return tuple(h.detach() for h in hidden)
    return hidden.detach()


def nucleus_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of characters whose probability mass reaches top_p.

    Everything outside that nucleus is zeroed out and the remainder is
    renormalised, as in Holtzman et al. (2020).
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep every character up to and including the one that crosses top_p.
    keep = (cumulative - sorted_probs) < top_p
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()


@torch.no_grad()
def generate(
    model: CharRNN,
    vocab,
    prompt: str,
    length: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: torch.device | str | None = None,
) -> str:
    """Synthesise `length` characters continuing `prompt`, one character at a time.

    Args:
        temperature: divides the logits; low values make the text repetitive,
            high values make it incoherent.
        top_p: nucleus threshold. None samples from the full distribution.

    Returns:
        The prompt followed by the synthesised text.
    """
    if not prompt:
        raise ValueError("prompt must contain at least one character")
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    ids = torch.tensor(vocab.encode(prompt), device=device).unsqueeze(0)
    logits, hidden = model(ids)
    out = [prompt]
    for _ in range(length):
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        if top_p is not None:
            probs = nucleus_probs(probs, top_p)
        next_id = torch.multinomial(probs, num_samples=1)
        out.append(vocab.itos[int(next_id)])
        logits, hidden = model(next_id.view(1, 1), hidden)

    model.train(was_training)
    return "".join(out)
