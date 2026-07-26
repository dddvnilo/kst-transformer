import numpy as np
import torch

from kst.dataset import make_loss_mask, make_lenient_loss_mask


def compute_pos_weight(loader, max_items: int, lenient: bool = False) -> float:
    """Racuna pos_weight = (broj nula) / (broj jedinica) u trening setu."""
    total_pos = total_neg = 0
    for _, Y, item_counts in loader:
        mask = make_loss_mask(item_counts, max_items)
        if lenient:
            mask = make_lenient_loss_mask(Y, mask)
        total_pos += Y[mask].sum().item()
        total_neg += (~Y[mask].bool()).sum().item()
    return total_neg / total_pos if total_pos > 0 else 1.0


def compute_f1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """F1 score samo na maskiranim celijama (ignorise padding i dijagonalu)."""
    pred_bin   = (torch.sigmoid(pred[mask]) > 0.5)
    target_bin = target[mask].bool()
    tp = (pred_bin &  target_bin).sum().item()
    fp = (pred_bin & ~target_bin).sum().item()
    fn = (~pred_bin & target_bin).sum().item()
    return (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 1.0


def compute_hamming(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Hamming loss = udeo pogresno klasifikovanih celija (FP + FN) / ukupno."""
    pred_bin   = (torch.sigmoid(pred[mask]) > 0.5)
    target_bin = target[mask].bool()
    wrong = (pred_bin != target_bin).sum().item()
    total = mask.sum().item()
    return wrong / total if total > 0 else 0.0


def metrics_np(pred_adj: np.ndarray, true_adj: np.ndarray, n_items: int):
    """F1 i Hamming na n_items x n_items matrici, bez dijagonale."""
    mask = ~np.eye(n_items, dtype=bool)
    pred_flat = pred_adj[mask].astype(bool)
    true_flat = true_adj[mask].astype(bool)

    tp = ( pred_flat &  true_flat).sum()
    fp = ( pred_flat & ~true_flat).sum()
    fn = (~pred_flat &  true_flat).sum()
    f1      = (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    hamming = (pred_flat != true_flat).sum() / len(pred_flat)
    return float(f1), float(hamming)


def transitive_closure_matrix(adj: np.ndarray) -> np.ndarray:
    """Floyd-Warshall tranzitivno zatvorenje nad bool adj matricom."""
    tc = adj.astype(bool).copy()
    n  = tc.shape[0]
    for k in range(n):
        tc = tc | (tc[:, k:k+1] & tc[k:k+1, :])
    return tc


def metrics_closure(pred_adj: np.ndarray, true_adj: np.ndarray, n_items: int):
    """
    Full transitive closure F1 i Hamming - poredi SIROV output (bez ikakve
    transformacije nad predikcijom) sa punim tranzitivnim zatvorenjem ground
    truth-a. Za razliku od lenient, promasene tranzitivne veze se ovde kaznjavaju
    (FN), jer je ground truth kompletna relacija dostiznosti:
      TP: predvidjeno i u zatvorenju Y
      FP: predvidjeno ali van zatvorenja Y
      FN: nije predvidjeno a jeste u zatvorenju Y (i direktne i tranzitivne)
    """
    true_closed = transitive_closure_matrix(true_adj[:n_items, :n_items])
    return metrics_np(pred_adj, true_closed.astype(np.float32), n_items)


def metrics_lenient(pred_adj: np.ndarray, true_adj: np.ndarray, n_items: int):
    """
    Lenient F1 i Hamming - tranzitivne veze se ne kažnjavaju:
      TP: predvidjeno i validno (u tranzitivnom zatvorenju Y)
      FP: predvidjeno ali ne moze se izvesti iz Y
      FN: nije predvidjeno a jeste direktna veza u Y (Hasse)
    """
    true_closed = transitive_closure_matrix(true_adj[:n_items, :n_items])

    mask             = ~np.eye(n_items, dtype=bool)
    pred_flat        = pred_adj[mask].astype(bool)
    true_flat        = true_adj[mask].astype(bool)
    true_closed_flat = true_closed[mask]

    tp = ( pred_flat &  true_closed_flat).sum()
    fp = ( pred_flat & ~true_closed_flat).sum()
    fn = (~pred_flat &  true_flat).sum()

    f1      = (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    hamming = (fp + fn) / len(pred_flat) if len(pred_flat) > 0 else 0.0
    return float(f1), float(hamming)
