"""
KST Transformer - Trening skripta

Primer pokretanja:
  python train.py \
      --data ../data/kst_dataset.npz \
      --epochs 50 \
      --batch-size 32 \
      --lr 1e-3 \
      --checkpoint-dir ../checkpoints
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kst.model import KSTTransformer
from kst.dataset import make_dataloaders, make_loss_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Trening skripta")
    parser.add_argument("--data",            type=str,   default="../data/kst_dataset.npz")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--d-model",         type=int,   default=64)
    parser.add_argument("--nhead",           type=int,   default=4)
    parser.add_argument("--num-layers",      type=int,   default=3)
    parser.add_argument("--dim-feedforward", type=int,   default=256)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--val-ratio",       type=float, default=0.1)
    parser.add_argument("--test-ratio",      type=float, default=0.1)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--checkpoint-dir",  type=str,   default="../checkpoints")
    return parser.parse_args()


def compute_pos_weight(loader, max_items: int) -> float:
    """Racuna pos_weight = (broj nula) / (broj jedinica) u trening setu."""
    total_pos = total_neg = 0
    for _, Y, item_counts in loader:
        mask = make_loss_mask(item_counts, max_items)
        total_pos += Y[mask].sum().item()
        total_neg += (~Y[mask].bool()).sum().item()
    return total_neg / total_pos if total_pos > 0 else 1.0


def compute_f1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """F1 score samo na maskiranim celijama (ignorise padding i dijagonalu)."""
    pred_bin   = (torch.sigmoid(pred[mask]) > 0.5)  # logit -> verovatnoca -> binarno
    target_bin = target[mask].bool()
    tp = (pred_bin &  target_bin).sum().item()
    fp = (pred_bin & ~target_bin).sum().item()
    fn = (~pred_bin & target_bin).sum().item()
    return (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def run_epoch(model, loader, optimizer, device, max_items, pos_weight: torch.Tensor, train: bool):
    model.train(train)
    total_loss = 0.0
    all_pred, all_target, all_mask = [], [], []

    with torch.set_grad_enabled(train):
        for X, Y, item_counts in loader:
            X, Y, item_counts = X.to(device), Y.to(device), item_counts.to(device)

            pred = model(X)
            mask = make_loss_mask(item_counts, max_items)

            loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask], pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_pred.append(pred.detach())
            all_target.append(Y)
            all_mask.append(mask)

    avg_loss = total_loss / len(loader)
    f1 = compute_f1(
        torch.cat(all_pred),
        torch.cat(all_target),
        torch.cat(all_mask),
    )
    return avg_loss, f1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_dataloaders(
        args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Citamo max_items i students iz prvog batcha da ne moramo da ih prosledjujemo rucno
    X_sample, _, _ = next(iter(train_loader))
    _, students, max_items = X_sample.shape

    model = KSTTransformer(
        max_items=max_items,
        students=students,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pw = compute_pos_weight(train_loader, max_items)
    pos_weight = torch.tensor([pw], device=device)
    print(f"pos_weight: {pw:.2f}  (nula/jedinica odnos u trening setu)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "best.pt")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, device, max_items, pos_weight, train=True)
        val_loss,   val_f1   = run_epoch(model, val_loader,   optimizer, device, max_items, pos_weight, train=False)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Train F1: {train_f1:.3f} | "
            f"Val loss: {val_loss:.4f} | Val F1: {val_f1:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "val_f1":          val_f1,
                "args":            vars(args),
            }, checkpoint_path)
            print(f"  -> Checkpoint sacuvan (val_loss={val_loss:.4f})")

    # Test evaluacija sa najboljim modelom
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_f1 = run_epoch(model, test_loader, optimizer, device, max_items, pos_weight, train=False)
    print(f"\nTest | Loss: {test_loss:.4f} | F1: {test_f1:.3f}")


if __name__ == "__main__":
    main()
