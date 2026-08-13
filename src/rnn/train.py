"""Train a character-level RNN/LSTM/GRU from a YAML config.

Example:
    python -m rnn.train --config configs/lstm.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn.data import build_datasets, load_text
from rnn.metrics import self_bleu, spelling_accuracy, text_perplexity
from rnn.models import CharRNN, detach_hidden, generate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: CharRNN, loader: DataLoader, device: torch.device) -> float:
    """Mean cross-entropy of the model over a data loader."""
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    total, batches = 0.0, 0
    hidden = None
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, hidden = model(inputs, hidden)
        total += criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1)).item()
        batches += 1

    model.train(was_training)
    return total / batches


def fit(
    model: CharRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[CharRNN, dict]:
    """Train with Adam, keeping the parameters with the lowest validation loss.

    Training stops early when the validation loss has not improved for
    `patience` consecutive epochs.

    Returns:
        The best model and a history dict of the recorded losses.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss_steps": [], "train_loss_epochs": [], "val_loss_epochs": []}
    best_state, best_val, best_epoch, epochs_without_gain = None, float("inf"), 0, 0

    model.train()
    for epoch in range(num_epochs):
        hidden = None
        epoch_losses = []
        for inputs, targets in tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, hidden = model(inputs, hidden)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            # The state carries context across batches, the graph must not.
            hidden = detach_hidden(hidden)
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss = evaluate(model, val_loader, device)
        history["train_loss_steps"].extend(epoch_losses)
        history["train_loss_epochs"].append(train_loss)
        history["val_loss_epochs"].append(val_loss)
        print(f"epoch {epoch + 1}: train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_epoch, epochs_without_gain = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                print(f"no improvement for {patience} epochs, stopping early")
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch + 1
    history["best_val_loss"] = best_val
    return model, history


def sample_texts(model, vocab, cfg, device) -> list[str]:
    """Synthesise `num_samples` independent continuations of the prompt."""
    return [
        generate(
            model,
            vocab,
            prompt=cfg["prompt"],
            length=cfg["sample_length"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            device=device,
        )
        for _ in range(cfg["num_samples"])
    ]


def score_samples(model, vocab, samples: list[str], device) -> dict:
    """Perplexity, spelling accuracy and self-BLEU for a set of samples."""
    return {
        "perplexity": float(np.mean([text_perplexity(model, vocab, s, device) for s in samples])),
        "spelling_accuracy": float(np.mean([spelling_accuracy(s) for s in samples])),
        "self_bleu": self_bleu(samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--epochs", type=int, help="override num_epochs")
    parser.add_argument("--limit-chars", type=int, help="use only the first N characters")
    parser.add_argument("--device", help="cuda, mps or cpu (auto-detected by default)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--bertscore", action="store_true", help="also compute BERTScore")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.limit_chars is not None:
        cfg["limit_chars"] = args.limit_chars

    device = pick_device(args.device)
    set_seed(cfg["seed"])
    print(f"config {args.config}  device {device}")

    vocab, train_ds, val_ds, test_ds = build_datasets(
        cfg["data"], cfg["seq_length"], cfg.get("limit_chars")
    )
    loaders = [
        DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
        for ds in (train_ds, val_ds, test_ds)
    ]
    train_loader, val_loader, test_loader = loaders
    print(f"{len(vocab)} characters, {len(train_ds)}/{len(val_ds)}/{len(test_ds)} sequences")

    model = CharRNN(
        vocab_size=len(vocab),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        cell_type=cfg["cell_type"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)

    start = time.time()
    model, history = fit(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        patience=cfg["patience"],
    )
    train_minutes = (time.time() - start) / 60

    test_loss = evaluate(model, test_loader, device)
    samples = sample_texts(model, vocab, cfg, device)
    metrics = {
        "name": cfg["name"],
        "cell_type": cfg["cell_type"],
        "num_layers": cfg["num_layers"],
        "hidden_size": cfg["hidden_size"],
        "best_epoch": history["best_epoch"],
        "val_loss": history["best_val_loss"],
        "test_loss": test_loss,
        **score_samples(model, vocab, samples, device),
        "train_minutes": round(train_minutes, 2),
    }

    if args.bertscore:
        from rnn.metrics import bert_score_f1

        # Compare each sample against a real passage of the same length.
        text = load_text(cfg["data"], cfg.get("limit_chars"))
        step = len(text) // (len(samples) + 1)
        length = cfg["sample_length"]
        references = [text[i * step : i * step + length] for i in range(len(samples))]
        metrics["bert_score"] = bert_score_f1(samples, references, device=str(device))

    out_dir = args.out / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "config": cfg, "vocab": vocab.itos}
    torch.save(checkpoint, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps({"config": cfg, **history}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "samples.txt").write_text(f"\n\n{'=' * 70}\n\n".join(samples), encoding="utf-8")

    print(
        f"\ntest loss {test_loss:.4f}  perplexity {metrics['perplexity']:.2f}  "
        f"spelling {metrics['spelling_accuracy']:.2%}  self-BLEU {metrics['self_bleu']:.2f}"
    )
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
