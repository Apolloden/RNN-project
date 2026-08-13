"""Random hyperparameter search over one or more architectures.

    python scripts/hparam_search.py                       # all three, 10 trials each
    python scripts/hparam_search.py --models lstm --trials 20
    python scripts/hparam_search.py --lr-range 0.002 0.006 # fine search around a coarse winner

Each trial starts from the architecture's config and overrides the searched
keys. Only the validation loss is measured, so no text is synthesised and no
checkpoint is written; the winning values belong in `configs/<model>.yaml`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from rnn.data import build_datasets
from rnn.models import CharRNN
from rnn.train import fit, pick_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# The ranges the report searched over; the learning rate is sampled log-uniformly.
HIDDEN_SIZES = [128, 150, 256, 306, 450, 576]
BATCH_SIZES = [16, 28, 50, 64, 128]
NUM_LAYERS = [1, 2]
LR_RANGE = (1e-4, 1e-1)


def sample_trial(rng: np.random.Generator, lr_range: tuple[float, float]) -> dict:
    """Draw one point from the search space."""
    low, high = (np.log10(bound) for bound in lr_range)
    return {
        "hidden_size": int(rng.choice(HIDDEN_SIZES)),
        "batch_size": int(rng.choice(BATCH_SIZES)),
        "num_layers": int(rng.choice(NUM_LAYERS)),
        "learning_rate": float(10 ** rng.uniform(low, high)),
    }


def run_trial(cfg: dict, datasets, device) -> float:
    """Train one configuration and return its best validation loss."""
    train_ds, val_ds = datasets
    loaders = [
        DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
        for ds in (train_ds, val_ds)
    ]
    model = CharRNN(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        cell_type=cfg["cell_type"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    _, history = fit(
        model,
        *loaders,
        device,
        num_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        patience=cfg["patience"],
    )
    return history["best_val_loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["rnn", "lstm", "gru"], help="which configs")
    parser.add_argument("--trials", type=int, default=10, help="trials per architecture")
    parser.add_argument("--epochs", type=int, default=5, help="epochs per trial")
    parser.add_argument("--patience", type=int, default=2, help="epochs before abandoning a trial")
    parser.add_argument("--limit-chars", type=int, help="use only the first N characters")
    parser.add_argument("--lr-range", nargs=2, type=float, default=LR_RANGE, metavar=("MIN", "MAX"))
    parser.add_argument("--device", help="cuda, mps or cpu (auto-detected by default)")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--out", type=Path, default=Path("results/search"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Datasets are shared between trials and between architectures that read the
    # same corpus, so the corpus is encoded once rather than once per trial.
    datasets: dict[tuple, tuple] = {}

    for model_name in args.models:
        cfg = yaml.safe_load((CONFIG_DIR / f"{model_name}.yaml").read_text())
        key = (cfg["data"], cfg["seq_length"], args.limit_chars)
        if key not in datasets:
            vocab, train_ds, val_ds, _ = build_datasets(*key)
            datasets[key] = (len(vocab), (train_ds, val_ds))
        vocab_size, splits = datasets[key]

        print(f"\n=== {model_name}: {args.trials} trials on {cfg['data']} ===")
        trials = []
        for trial in range(args.trials):
            set_seed(args.seed)
            values = sample_trial(rng, tuple(args.lr_range))
            print(f"\ntrial {trial + 1}/{args.trials}: {values}")
            val_loss = run_trial(
                {
                    **cfg,
                    **values,
                    "vocab_size": vocab_size,
                    "num_epochs": args.epochs,
                    "patience": args.patience,
                },
                splits,
                device,
            )
            trials.append({**values, "val_loss": val_loss})
            best = min(trials, key=lambda t: t["val_loss"])
            print(f"val loss {val_loss:.4f}  (best so far {best['val_loss']:.4f})")

        trials.sort(key=lambda t: t["val_loss"])
        out_file = args.out / f"{model_name}.json"
        out_file.write_text(json.dumps({"epochs": args.epochs, "trials": trials}, indent=2))
        print(f"\nbest {model_name}: {trials[0]}\nwrote {out_file}")


if __name__ == "__main__":
    main()
