"""
Random search za hiperparametre

Primer pokretanja:
  python random_search.py
  python random_search.py --trials 30 --epochs 40
"""

import argparse
import csv
import os
import random

import torch
import torch.nn.functional as F

from kst.model import KSTTransformer
from kst.dataset import make_dataloaders, make_loss_mask, make_lenient_loss_mask
from util.metrics import compute_pos_weight
from util.paths import DATA_DIR, CHECKPOINT_DIR, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Random search za hiperparametre")
    parser.add_argument("--data",        type=str,   default="kst_dataset_2-10items_80k_weighted.npz",
                        help="Ime fajla u data/ ili puna putanja")
    parser.add_argument("--trials",      type=int,   default=30,  help="Broj nasumicnih kombinacija (default: 20)")
    parser.add_argument("--epochs",      type=int,   default=30,  help="Epohe po kombinaciji (default: 30)")
    parser.add_argument("--patience",    type=int,   default=10,  help="Early stopping patience (default: 10)")
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--test-ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output",      type=str,   default="random_search_10items.csv",
                        help="Ime fajla u checkpoints/ ili puna putanja")
    parser.add_argument("--history",     type=str,   default="random_search_history.csv",
                        help="Ime fajla u checkpoints/ ili puna putanja - koristi se da se izbegnu ponovljene "
                             "kombinacije")
    parser.add_argument("--lenient-loss", action="store_true", default=False,
                        help="Ne kaznjava u loss-u tranzitivno validne, ali nedirektne predikcije")
    args = parser.parse_args()
    args.data = str(resolve_path(args.data, DATA_DIR))
    args.output = str(resolve_path(args.output, CHECKPOINT_DIR))
    args.history = str(resolve_path(args.history, CHECKPOINT_DIR))
    return args


# ---------------------------------------------------------------------------
# Prostor hiperparametara
# ---------------------------------------------------------------------------

D_MODELS        = [64, 128, 256, 384]
NUM_LAYERS      = [2, 3, 4, 5, 6]
DIM_FEEDFORWARD = [128, 256, 512, 1024]
DROPOUTS        = [0.1, 0.2, 0.3]
BATCH_SIZES     = [32, 64, 128]
LR_RANGE        = (1e-4, 1e-3)   # log-uniform

# Diskretni deo prostora (bez lr, koji je kontinualan) - koristi se za detekciju
# vec isprobanih kombinacija preko --history fajla.
DISCRETE_KEYS = ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "batch_size"]


def discrete_key(hparams: dict) -> tuple:
    return tuple(hparams[k] for k in DISCRETE_KEYS)


def sample_hparams(rng: random.Random, exclude: set = frozenset(), max_attempts: int = 1000) -> dict:
    for _ in range(max_attempts):
        d_model = rng.choice(D_MODELS)
        # nhead mora da deli d_model
        valid_nheads = [h for h in [2, 4, 8, 16] if d_model % h == 0]
        nhead = rng.choice(valid_nheads)

        lr = 10 ** rng.uniform(*[torch.log10(torch.tensor(x)).item() for x in LR_RANGE])

        hparams = {
            "d_model":         d_model,
            "nhead":           nhead,
            "num_layers":      rng.choice(NUM_LAYERS),
            "dim_feedforward": rng.choice(DIM_FEEDFORWARD),
            "dropout":         rng.choice(DROPOUTS),
            "batch_size":      rng.choice(BATCH_SIZES),
            "lr":              round(lr, 6),
        }

        if discrete_key(hparams) not in exclude:
            return hparams

    raise RuntimeError(
        f"Nije pronadjena nova kombinacija hiperparametara posle {max_attempts} pokusaja - "
        "prostor pretrage je verovatno skoro/potpuno iscrpljen (videti --history fajl)."
    )


# ---------------------------------------------------------------------------
# Pomocne funkcije
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, max_items, pos_weight, train: bool, lenient: bool = False):
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for X, Y, item_counts in loader:
            X, Y, item_counts = X.to(device), Y.to(device), item_counts.to(device)
            pred = model(X, item_counts)
            mask = make_loss_mask(item_counts, max_items)
            if lenient:
                mask = make_lenient_loss_mask(Y, mask)
            loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask], pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train_trial(hparams, train_loader, val_loader, students, max_items, device, epochs, patience, lenient: bool = False):
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
    pw = compute_pos_weight(train_loader, max_items, lenient=lenient)
    pos_weight = torch.tensor([pw], device=device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, optimizer, device, max_items, pos_weight, train=True, lenient=lenient)
        val_loss = run_epoch(model, val_loader, optimizer, device, max_items, pos_weight, train=False, lenient=lenient)

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

def save_results_csv(output_path: str, results: list) -> None:
    """Upisuje trenutne rezultate u CSV - poziva se posle svakog trial-a"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["val_loss", "stopped_epoch", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "batch_size", "lr"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_tried_combos(history_path: str) -> set:
    """Cita --history fajl (akumulira se kroz sva pokretanja) i vraca skup vec isprobanih
    diskretnih kombinacija hiperparametara."""
    if not os.path.exists(history_path):
        return set()
    tried = set()
    with open(history_path, newline="") as f:
        for row in csv.DictReader(f):
            tried.add(tuple(
                float(row[k]) if k == "dropout" else int(row[k]) for k in DISCRETE_KEYS
            ))
    return tried


def append_history(history_path: str, hparams: dict) -> None:
    """Dodaje jednu isprobanu kombinaciju u --history fajl (append)."""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    file_exists = os.path.exists(history_path)
    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DISCRETE_KEYS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: hparams[k] for k in DISCRETE_KEYS})


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

    tried = load_tried_combos(args.history)
    print(f"Vec isprobanih kombinacija (iz {args.history}): {len(tried)}\n")

    for trial in range(1, args.trials + 1):
        hparams = sample_hparams(rng, exclude=tried)
        tried.add(discrete_key(hparams))
        append_history(args.history, hparams)

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
            lenient=args.lenient_loss,
        )

        print(f"-> val_loss={best_val_loss:.4f}  (epoha {stopped_epoch})")

        results.append({"val_loss": best_val_loss, "stopped_epoch": stopped_epoch, **hparams})
        save_results_csv(args.output, results)

    # Sortiraj po val_loss i sacuvaj
    results.sort(key=lambda r: r["val_loss"])
    save_results_csv(args.output, results)

    print(f"\nRezultati sacuvani: {args.output}")
    print(f"\nTop 5 kombinacija:")
    print(f"{'val_loss':>10} {'d_model':>8} {'nhead':>6} {'layers':>7} {'ff':>6} {'dropout':>8} {'bs':>4} {'lr':>10}")
    print("-" * 75)
    for r in results[:5]:
        print(f"{r['val_loss']:>10.4f} {r['d_model']:>8} {r['nhead']:>6} {r['num_layers']:>7} "
              f"{r['dim_feedforward']:>6} {r['dropout']:>8} {r['batch_size']:>4} {r['lr']:>10.6f}")


if __name__ == "__main__":
    main()
