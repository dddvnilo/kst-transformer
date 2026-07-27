"""
Evaluacija IITA vs KST Transformer na test skupu.

Rezultati se cuvaju kao plotovi u checkpoints/eval plots/, za tri moda (strict / lenient / closure), po 3 plota:
  - "_overall"       : tabela F1 / Hamming / Vreme za IITA (v=1,2,3) i transformer
  - "_f1_by_items"    : F1 po broju pitanja u testu, za sve 4 metode
  - "_hamming_by_items": Hamming po broju pitanja u testu, za sve 4 metode

Modovi evaluacije:
  - strict  : ground truth = Hasse (min. skup); IITA output se tranzitivno
redukuje (transitive_reduction) pre poredjenja 
(obzirom da je transformer vec treniran na tranzitivno redukovanom skupu podataka)

  - lenient : tacne tranzitivne veze se ne kaznjavaju (metrics_lenient)

  - closure : ground truth = puno tranzitivno zatvorenje; sirov output se poredi
takav kakav jeste, promasene tranzitivne veze SE kaznjavaju (FN)

Primer pokretanja:
  python eval_iita.py
  python eval_iita.py --num-samples 500
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from learning_spaces.kst.iita import iita
from kst.dataset import make_dataloaders
from kst.model import KSTTransformer
from util.metrics import metrics_np, metrics_lenient, metrics_closure
from util.generate_dataset_util import transitive_reduction
from util.paths import DATA_DIR, CHECKPOINT_DIR, resolve_path

# Paleta po metodi
COLORS = {
    "IITA (v=1)":      "#2a78d6",  # plava
    "IITA (v=2)":      "#eb6834",  # narandzasta
    "IITA (v=3)":      "#1baf7a",  # tirkizna
    "KST Transformer": "#eda100",  # zuta
}
SURFACE  = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS     = "#c3c2b7"
INK      = "#0b0b0b"
INK_2    = "#52514e"

# Rezimi evaluacije: (tag za ime fajla, naslov, f1_idx, hamming_idx u record tuple-u)
# record = (n, f1, hamming, f1_l, hamming_l, f1_c, hamming_c)
MODES = [
    ("strict",  "Strict (ground truth = Hasse dijagram)",               1, 2),
    ("lenient", "Lenient (tranzitivne veze se ne kaznjavaju)",           3, 4),
    ("closure", "Full transitive closure (puno zatvorenje se kaznjava)", 5, 6),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluacija IITA vs KST Transformer")
    parser.add_argument("--data",        type=str,   default="kst_dataset_2-10items_80k_weighted.npz",
                        help="Ime fajla u data/ ili puna putanja")
    parser.add_argument("--checkpoint",  type=str,   default="best_10items_80k_weighted_200epochs_lenient.pt",
                        help="Ime fajla u checkpoints/ ili puna putanja")
    parser.add_argument("--num-samples", type=int,   default=200,  help="Broj uzoraka iz test skupa")
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--test-ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--output-dir",  type=str,   default=str(CHECKPOINT_DIR / "eval plots"),
                        help="Direktorijum u koji se cuvaju plotovi (default: checkpoints/eval plots/)")
    args = parser.parse_args()
    args.data = str(resolve_path(args.data, DATA_DIR))
    args.checkpoint = str(resolve_path(args.checkpoint, CHECKPOINT_DIR))
    return args


def run_iita(X_np: np.ndarray, item_counts_np: np.ndarray, Y_np: np.ndarray, num_samples: int, v: int = 1):
    """Pokrece IITA nad test uzorcima, vraca listu po-uzorak rezultata:
    (n, f1, hamming, f1_l, hamming_l, f1_c, hamming_c) - strict, lenient, closure."""
    records = []
    skipped = 0

    for i in range(num_samples):
        n = int(item_counts_np[i])
        x      = X_np[i, :, :n]
        y_true = Y_np[i, :n, :n]

        try:
            result = iita(x, v=v)
            impl   = [(a, b) for (a, b) in result['implications'] if a < n and b < n]
        except Exception:
            skipped += 1
            continue

        # Strict: IITA output se tranzitivno redukuje (Hasse dijagram, uporedivo sa Y)
        pred_adj_reduced = np.zeros((n, n), dtype=np.float32)
        for (a, b) in transitive_reduction(impl):
            pred_adj_reduced[a][b] = 1.0

        # Lenient: sirov IITA output (tranzitivne veze se ionako ne kaznjavaju)
        pred_adj_raw = np.zeros((n, n), dtype=np.float32)
        for (a, b) in impl:
            pred_adj_raw[a][b] = 1.0

        f1, hamming     = metrics_np(pred_adj_reduced, y_true, n)
        f1_l, hamming_l = metrics_lenient(pred_adj_raw, y_true, n)
        f1_c, hamming_c = metrics_closure(pred_adj_raw, y_true, n)
        records.append((n, f1, hamming, f1_l, hamming_l, f1_c, hamming_c))

        if (i + 1) % 50 == 0:
            print(f"  IITA v={v}: {i + 1}/{num_samples}")

    if skipped:
        print(f"  Preskoceno uzoraka (IITA greska): {skipped}")

    return records


def run_transformer(model, X: torch.Tensor, Y: torch.Tensor, item_counts: torch.Tensor,
                    device, batch_size: int):
    """Pokrece transformer na svim uzorcima, vraca listu po-uzorak rezultata:
    (n, f1, hamming, f1_l, hamming_l, f1_c, hamming_c) - strict, lenient, closure."""
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

    records = []
    for i in range(len(all_pred)):
        n        = int(all_ic[i])
        # sigmoid(x) > 0.5  <=>  x > 0  (izbegava overflow u exp za velike negativne logite)
        pred_adj = (all_pred[i, :n, :n] > 0).astype(np.float32)
        true_adj = all_target[i, :n, :n]
        f1, hamming     = metrics_np(pred_adj, true_adj, n)
        f1_l, hamming_l = metrics_lenient(pred_adj, true_adj, n)
        f1_c, hamming_c = metrics_closure(pred_adj, true_adj, n)
        records.append((n, f1, hamming, f1_l, hamming_l, f1_c, hamming_c))

    return records


def aggregate_overall(records, f1_idx: int, ham_idx: int):
    """records -> (mean_f1, mean_hamming) za dati rezim (indeksi kolona)."""
    arr = np.array(records)
    return float(arr[:, f1_idx].mean()), float(arr[:, ham_idx].mean())


def aggregate_by_items(records, f1_idx: int, ham_idx: int):
    """records -> (f1_by_n, hamming_by_n) dict-ovi za dati rezim (indeksi kolona)."""
    by_n = defaultdict(list)
    for rec in records:
        by_n[rec[0]].append((rec[f1_idx], rec[ham_idx]))
    f1_by_n      = {n: float(np.mean([v[0] for v in vals])) for n, vals in by_n.items()}
    hamming_by_n = {n: float(np.mean([v[1] for v in vals])) for n, vals in by_n.items()}
    return f1_by_n, hamming_by_n


def plot_overall_table(rows, title, save_path):
    """rows: lista (metod, f1, hamming, vreme_s)."""
    fig, ax = plt.subplots(figsize=(9.5, 0.7 * len(rows) + 1.4))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=16)

    col_labels = ["Metod", "F1", "Hamming", "Vreme (s)"]
    cell_text  = [[name, f"{f1:.3f}", f"{hamming:.3f}", f"{t:.1f}"] for name, f1, hamming, t in rows]

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRIDLINE)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(fontweight="bold", color=INK)
        else:
            cell.set_facecolor(SURFACE)
            method_name = cell_text[r - 1][0]
            if c == 0:
                cell.set_text_props(color=COLORS.get(method_name, INK), fontweight="bold")
            else:
                cell.set_text_props(color=INK)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_metric_by_items(data_by_method, metric_label, title, save_path):
    """data_by_method: dict metod -> dict n -> vrednost."""
    methods = list(data_by_method.keys())
    all_ns  = sorted(set(n for d in data_by_method.values() for n in d.keys()))
    x       = np.arange(len(all_ns))
    width   = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals   = [data_by_method[method].get(n, np.nan) for n in all_ns]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=method,
               color=COLORS.get(method), edgecolor=SURFACE, linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.set_xlabel("Broj pitanja")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1)
    ax.set_title(title, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, labelcolor=INK_2, ncol=len(methods),
              loc="upper center", bbox_to_anchor=(0.5, -0.14))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = make_dataloaders(
        args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    X_list, Y_list, ic_list = [], [], []
    for X, Y, ic in test_loader:
        X_list.append(X)
        Y_list.append(Y)
        ic_list.append(ic)

    X_all  = torch.cat(X_list)
    Y_all  = torch.cat(Y_list)
    ic_all = torch.cat(ic_list)

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

    method_records = {}

    for v in [1, 2, 3]:
        print(f"Pokrecem IITA (v={v})...")
        t0 = time.perf_counter()
        records = run_iita(X_sub.numpy(), ic_sub.numpy(), Y_sub.numpy(), n, v=v)
        elapsed = time.perf_counter() - t0
        method_records[f"IITA (v={v})"] = (records, elapsed)
        print(f"  -> gotovo za {elapsed:.1f}s\n")

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

    t0 = time.perf_counter()
    tr_records = run_transformer(model, X_sub, Y_sub, ic_sub, device, args.batch_size)
    tr_elapsed = time.perf_counter() - t0
    method_records["KST Transformer"] = (tr_records, tr_elapsed)
    print(f"  -> gotovo za {tr_elapsed:.1f}s\n")

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    for tag, label, f1_idx, ham_idx in MODES:
        overall_rows  = []
        f1_by_method  = {}
        ham_by_method = {}

        for method, (records, elapsed) in method_records.items():
            f1, hamming = aggregate_overall(records, f1_idx, ham_idx)
            overall_rows.append((method, f1, hamming, elapsed))

            f1_by_n, ham_by_n = aggregate_by_items(records, f1_idx, ham_idx)
            f1_by_method[method]  = f1_by_n
            ham_by_method[method] = ham_by_n

        plot_overall_table(
            overall_rows,
            f"IITA vs KST Transformer - {label}",
            os.path.join(args.output_dir, f"iita_comparison_{stem}_{tag}_overall.png"),
        )
        plot_metric_by_items(
            f1_by_method, "F1 score",
            f"F1 po broju pitanja - {label}",
            os.path.join(args.output_dir, f"iita_comparison_{stem}_{tag}_f1_by_items.png"),
        )
        plot_metric_by_items(
            ham_by_method, "Hamming loss",
            f"Hamming loss po broju pitanja - {label}",
            os.path.join(args.output_dir, f"iita_comparison_{stem}_{tag}_hamming_by_items.png"),
        )
        print(f"Sacuvano: iita_comparison_{stem}_{tag}_{{overall,f1_by_items,hamming_by_items}}.png")

    print(f"\nCheckpoint: epoha {checkpoint.get('epoch', '?')} | "
          f"val_loss={checkpoint.get('val_loss', 0):.4f} | "
          f"val_F1={checkpoint.get('val_f1', 0):.3f}")


if __name__ == "__main__":
    main()
