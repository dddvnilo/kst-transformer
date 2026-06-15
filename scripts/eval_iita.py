"""
Evaluacija IITA vs KST Transformer na test skupu.

Primer pokretanja:
  python eval_iita.py
  python eval_iita.py --num-samples 500
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from learning_spaces.kst.iita import iita_exclude_transitive
from kst.dataset import make_dataloaders
from kst.model import KSTTransformer

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPTS_DIR, "..")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluacija IITA vs KST Transformer")
    parser.add_argument("--data",        type=str,   default=os.path.join(_ROOT_DIR, "data", "kst_dataset_weighted.npz"))
    parser.add_argument("--checkpoint",  type=str,   default=os.path.join(_ROOT_DIR, "checkpoints", "best.pt"))
    parser.add_argument("--num-samples", type=int,   default=200,  help="Broj uzoraka za IITA")
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--test-ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--batch-size",  type=int,   default=64)
    return parser.parse_args()


def metrics_np(pred_adj: np.ndarray, true_adj: np.ndarray, n_items: int):
    """F1 i Hamming na n_items x n_items matrici, bez dijagonale."""

    # Maska - Invertovana identity matrica (ne gledamo refleksivne relacije)
    mask = ~np.eye(n_items, dtype=bool)
    pred_flat = pred_adj[mask].astype(bool)
    true_flat = true_adj[mask].astype(bool)

    tp = ( pred_flat &  true_flat).sum()
    fp = ( pred_flat & ~true_flat).sum()
    fn = (~pred_flat &  true_flat).sum()
    f1      = (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    hamming = (pred_flat != true_flat).sum() / len(pred_flat)
    return float(f1), float(hamming)


def run_iita(X_np: np.ndarray, Y_np: np.ndarray, item_counts_np: np.ndarray, num_samples: int, v: int = 1):
    """Pokrece IITA (exclude_transitive) nad test uzorcima i vraca prosecne metrike."""
    f1s, hammings = [], []
    skipped = 0

    for i in range(num_samples):
        n = int(item_counts_np[i])
        x      = X_np[i, :, :n]   # (students, n_items) — bez paddinga
        y_true = Y_np[i, :n, :n]  # (n_items, n_items)

        try:
            result   = iita_exclude_transitive(x, v=v)
            pred_adj = np.zeros((n, n), dtype=np.float32)
            for (a, b) in result['implications']:
                if a < n and b < n:
                    pred_adj[a][b] = 1.0
        except Exception as e:
            skipped += 1
            continue

        f1, hamming = metrics_np(pred_adj, y_true, n)
        f1s.append(f1)
        hammings.append(hamming)

        if (i + 1) % 50 == 0:
            print(f"  IITA v={v}: {i + 1}/{num_samples}")

    if skipped:
        print(f"  Preskoceno uzoraka (IITA greska): {skipped}")

    return np.mean(f1s), np.mean(hammings)


def run_transformer(model, X: torch.Tensor, Y: torch.Tensor, item_counts: torch.Tensor,
                    device, batch_size: int):
    """Pokrece transformer na svim uzorcima i vraca prosecne metrike po uzorku."""
    model.eval()
    all_pred, all_target, all_ic = [], [], []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end  = min(start + batch_size, len(X))
            X_b  = X[start:end].to(device)
            Y_b  = Y[start:end].to(device)
            ic_b = item_counts[start:end].to(device)

            pred = model(X_b, ic_b)

            all_pred.append(pred.cpu())
            all_target.append(Y_b.cpu())
            all_ic.append(ic_b.cpu())

    all_pred   = torch.cat(all_pred).numpy()
    all_target = torch.cat(all_target).numpy()
    all_ic     = torch.cat(all_ic).numpy()

    f1s, hammings = [], []
    for i in range(len(all_pred)):
        n        = int(all_ic[i])
        pred_adj = (1 / (1 + np.exp(-all_pred[i, :n, :n])) > 0.5).astype(np.float32)
        true_adj = all_target[i, :n, :n]
        f1, hamming = metrics_np(pred_adj, true_adj, n)
        f1s.append(f1)
        hammings.append(hamming)

    return float(np.mean(f1s)), float(np.mean(hammings))


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Isti split kao u treningu
    _, _, test_loader = make_dataloaders(
        args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Liste batch-eva iz tenzora test_loader-a
    X_list, Y_list, ic_list = [], [], []
    for X, Y, ic in test_loader:
        X_list.append(X)
        Y_list.append(Y)
        ic_list.append(ic)

    # Svi batchevi zajedno
    X_all  = torch.cat(X_list)
    Y_all  = torch.cat(Y_list)
    ic_all = torch.cat(ic_list)

    # Uzimamo deo dataset-a
    n = min(args.num_samples, len(X_all))
    X_sub  = X_all[:n]
    Y_sub  = Y_all[:n]
    ic_sub = ic_all[:n]

    _, students, max_items = X_sub.shape

    print(f"\n{'='*55}")
    print(f"  Evaluacija: IITA vs KST Transformer")
    print(f"{'='*55}")
    print(f"  Uzoraka:    {n}  (test skup)")
    print(f"  Device:     {device}")
    print(f"{'='*55}\n")

    # IITA v=1, v=2, v=3
    iita_results = {}
    for v in [1, 2, 3]:
        print(f"Pokrecem IITA (v={v}, exclude_transitive)...")
        f1, hamming = run_iita(X_sub.numpy(), Y_sub.numpy(), ic_sub.numpy(), n, v=v)
        iita_results[v] = (f1, hamming)
        print(f"  -> F1={f1:.3f}  Hamming={hamming:.3f}\n")

    # Transformer
    print("Pokrecem KST Transformer...")
    checkpoint  = torch.load(args.checkpoint, map_location=device)
    saved_args  = checkpoint.get("args", {})

    model = KSTTransformer(
        max_items=max_items,
        students=students,
        d_model=saved_args.get("d_model"),
        nhead=saved_args.get("nhead"),
        num_encoder_layers=saved_args.get("num_layers"),
        dim_feedforward=saved_args.get("dim_feedforward"),
        dropout=saved_args.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    transformer_f1, transformer_hamming = run_transformer(
        model, X_sub, Y_sub, ic_sub, device, args.batch_size
    )
    print(f"  -> F1={transformer_f1:.3f}  Hamming={transformer_hamming:.3f}\n")

    # Rezultati
    print(f"{'='*55}")
    print(f"{'Metod':<22} {'F1':>10} {'Hamming':>10}")
    print(f"{'-'*55}")
    for v, (f1, hamming) in iita_results.items():
        print(f"{f'IITA (v={v})':<22} {f1:>10.3f} {hamming:>10.3f}")
    print(f"{'KST Transformer':<22} {transformer_f1:>10.3f} {transformer_hamming:>10.3f}")
    print(f"{'='*55}")
    print(f"\nCheckpoint: epoha {checkpoint.get('epoch', '?')} | "
          f"val_loss={checkpoint.get('val_loss', 0):.4f} | "
          f"val_F1={checkpoint.get('val_f1', 0):.3f}")


if __name__ == "__main__":
    main()
