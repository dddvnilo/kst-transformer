"""
Random search za hiperparametre

Primer pokretanja:
  python random_search.py
  python random_search.py --trials 30 --epochs 40
"""

import argparse
import csv
import os
import sys
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kst.model import KSTTransformer
from kst.dataset import make_dataloaders, make_loss_mask

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPTS_DIR, "..")


def parse_args():
    parser = argparse.ArgumentParser(description="Random search za hiperparametre")
    parser.add_argument("--data",        type=str,   default=os.path.join(_ROOT_DIR, "data", "kst_dataset_weighted.npz"))
    parser.add_argument("--trials",      type=int,   default=20,  help="Broj nasumicnih kombinacija (default: 20)")
    parser.add_argument("--epochs",      type=int,   default=30,  help="Epohe po kombinaciji (default: 30)")
    parser.add_argument("--patience",    type=int,   default=10,  help="Early stopping patience (default: 10)")
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--test-ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output",      type=str,   default=os.path.join(_ROOT_DIR, "checkpoints", "random_search.csv"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prostor hiperparametara
# ---------------------------------------------------------------------------

D_MODELS       = [64, 128, 256]
NUM_LAYERS     = [2, 3, 4]
DIM_FEEDFORWARD = [128, 256, 512]
DROPOUTS       = [0.1, 0.2, 0.3]
BATCH_SIZES    = [32, 64, 128]
LR_RANGE       = (1e-4, 1e-3)   # log-uniform


def sample_hparams(rng: random.Random) -> dict:
    d_model = rng.choice(D_MODELS)
    # nhead mora da deli d_model
    valid_nheads = [h for h in [2, 4, 8] if d_model % h == 0]
    nhead = rng.choice(valid_nheads)

    lr = 10 ** rng.uniform(*[torch.log10(torch.tensor(x)).item() for x in LR_RANGE])

    return {
        "d_model":         d_model,
        "nhead":           nhead,
        "num_layers":      rng.choice(NUM_LAYERS),
        "dim_feedforward": rng.choice(DIM_FEEDFORWARD),
        "dropout":         rng.choice(DROPOUTS),
        "batch_size":      rng.choice(BATCH_SIZES),
        "lr":              round(lr, 6),
    }


# ---------------------------------------------------------------------------
# Pomocne funkcije
# ---------------------------------------------------------------------------

def compute_pos_weight(loader, max_items: int, device) -> torch.Tensor:
    total_pos = total_neg = 0
    for _, Y, item_counts in loader:
        mask = make_loss_mask(item_counts, max_items)
        total_pos += Y[mask].sum().item()
        total_neg += (~Y[mask].bool()).sum().item()
    pw = total_neg / total_pos if total_pos > 0 else 1.0
    return torch.tensor([pw], device=device)


def run_epoch(model, loader, optimizer, device, max_items, pos_weight, train: bool):
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for X, Y, item_counts in loader:
            X, Y, item_counts = X.to(device), Y.to(device), item_counts.to(device)
            pred = model(X, item_counts)
            mask = make_loss_mask(item_counts, max_items)
            loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask], pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train_trial(hparams, train_loader, val_loader, students, max_items, device, epochs, patience):
    model = KSTTransformer(
        max_items=max_items,
        students=students,
        d_model=hparams["d_model"],
        nhead=hparams["nhead"],
        num_encoder_layers=hparams["num_layers"],
        dim_feedforward=hparams["dim_feedforward"],
        dropout=hparams["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    pos_weight = compute_pos_weight(train_loader, max_items, device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, optimizer, device, max_items, pos_weight, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, max_items, pos_weight, train=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val_loss, epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Trials: {args.trials}  |  Max epoha po trialu: {args.epochs}  |  Patience: {args.patience}\n")

    base_loader_args = dict(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    results = []

    for trial in range(1, args.trials + 1):
        hparams = sample_hparams(rng)

        train_loader, val_loader, _ = make_dataloaders(
            args.data,
            batch_size=hparams["batch_size"],
            **base_loader_args,
        )

        X_sample, _, _ = next(iter(train_loader))
        _, students, max_items = X_sample.shape

        print(f"Trial {trial:>2}/{args.trials} | {hparams}", end="  ", flush=True)

        best_val_loss, stopped_epoch = train_trial(
            hparams, train_loader, val_loader,
            students, max_items, device,
            epochs=args.epochs, patience=args.patience,
        )

        print(f"-> val_loss={best_val_loss:.4f}  (epoha {stopped_epoch})")

        results.append({"val_loss": best_val_loss, "stopped_epoch": stopped_epoch, **hparams})

    # Sortiraj po val_loss
    results.sort(key=lambda r: r["val_loss"])

    # Sacuvaj CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["val_loss", "stopped_epoch", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "batch_size", "lr"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRezultati sacuvani: {args.output}")
    print(f"\nTop 5 kombinacija:")
    print(f"{'val_loss':>10} {'d_model':>8} {'nhead':>6} {'layers':>7} {'ff':>6} {'dropout':>8} {'bs':>4} {'lr':>10}")
    print("-" * 75)
    for r in results[:5]:
        print(f"{r['val_loss']:>10.4f} {r['d_model']:>8} {r['nhead']:>6} {r['num_layers']:>7} "
              f"{r['dim_feedforward']:>6} {r['dropout']:>8} {r['batch_size']:>4} {r['lr']:>10.6f}")


if __name__ == "__main__":
    main()
