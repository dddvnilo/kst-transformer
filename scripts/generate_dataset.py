"""
KST Dataset Generator

Generise podatke u obliku:
  INPUT:  response matrica studenata (size x items) - binarna
  OUTPUT: adjacency matrica prerequisita (items x items) - binarna

Koristi:
  python generate_dataset.py \
      --num-samples 10000 \
      --max-items 5 \
      --min-items 2 \
      --size 200 \
      --ce 0.1 \
      --lg 0.05 \
      --output dataset.npz
"""

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from util.generate_dataset_util import generate_dataset

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPTS_DIR, "..")


def write_dataset_card(output_path: Path, args: argparse.Namespace, item_counts: np.ndarray) -> None:
    """Pise <output_path bez .npz>.md - YAML frontmatter + markdown, HF Dataset Card stil."""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_commit = None

    n = len(item_counts)
    frontmatter = {
        "dataset_name":   output_path.stem,
        "generated_by":   "scripts/generate_dataset.py",
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator_args": vars(args),
        "git_commit":     git_commit,
    }

    distribution_rows = []
    for k in range(args.min_items, args.max_items + 1):
        cnt = int((item_counts == k).sum())
        pct = 100 * cnt / n if n > 0 else 0.0
        distribution_rows.append(f"| {k} | {cnt} | {pct:.1f}% |")

    body = (
        f"\n# {output_path.stem}\n\n"
        "## Summary\n"
        "Sinteticki dataset za KST Transformer - simulirani odgovori studenata, "
        "generisani sa `scripts/generate_dataset.py`.\n\n"
        "## Structure\n"
        f"- `X`: (N, {args.size}, {args.max_items}) float32 - binarna response matrica, paddovana nulama\n"
        f"- `Y`: (N, {args.max_items}, {args.max_items}) float32 - Hasse dijagram (tranzitivna redukcija)\n"
        "- `item_counts`: (N,) int64 - stvarni broj pitanja po uzorku\n\n"
        "## Distribucija broja pitanja\n"
        "| items | count | % |\n"
        "|---|---|---|\n"
        + "\n".join(distribution_rows) + "\n"
    )

    card_path = output_path.with_suffix(".md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.safe_dump(frontmatter, f, sort_keys=False, allow_unicode=True)
        f.write("---\n")
        f.write(body)

    print(f"Kartica sacuvana: {card_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KST Dataset Generator"
    )
    parser.add_argument("--num-samples", type=int,   default=5000,
                        help="Broj primera u datasetu (default: 5000)")
    parser.add_argument("--max-items",   type=int,   default=5,
                        help="Maksimalan broj pitanja (default: 5)")
    parser.add_argument("--min-items",   type=int,   default=2,
                        help="Minimalan broj pitanja (default: 2)")
    parser.add_argument("--size",        type=int,   default=500,
                        help="Broj studenata po simulaciji (default: 500)")
    parser.add_argument("--ce",          type=float, default=0.1,
                        help="Careless error verovatnoca (default: 0.1)")
    parser.add_argument("--lg",          type=float, default=0.05,
                        help="Lucky guess verovatnoca (default: 0.05)")
    parser.add_argument("--ce-std",      type=float, default=0.03,
                        help="Std za CE noise (default: 0.03)")
    parser.add_argument("--lg-std",      type=float, default=0.02,
                        help="Std za LG noise (default: 0.02)")
    parser.add_argument("--output",      type=str,   default=os.path.join(_ROOT_DIR, "data", "kst_dataset.npz"),
                        help="Output fajl (.npz format)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--weighted",    action="store_true", default=False,
                        help="Linearne tezine — vise pitanja = veci udeo uzoraka")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  KST Dataset Generator")
    print(f"{'='*55}")
    print(f"  Broj primera:            {args.num_samples}")
    print(f"  Items:                   {args.min_items} - {args.max_items}")
    if args.weighted:
        n = args.max_items - args.min_items + 1
        w = list(range(1, n + 1))
        probs = [round(x / sum(w) * 100, 1) for x in w]
        print(f"  Tezine (linearne):       {dict(zip(range(args.min_items, args.max_items+1), probs))}%")
    print(f"  Broj studenata po primeru: {args.size}")
    print(f"  Careless error:          {args.ce} (std={args.ce_std})")
    print(f"  Lucky guess:             {args.lg} (std={args.lg_std})")
    print(f"  Output:                  {args.output}")
    print(f"{'='*55}\n")

    print("- Pocetak generisanja -")
    responses, adj_matrices, item_counts = generate_dataset(
        num_samples=args.num_samples,
        max_items=args.max_items,
        min_items=args.min_items,
        student_size=args.size,
        ce=args.ce,
        lg=args.lg,
        ce_std=args.ce_std,
        lg_std=args.lg_std,
        weighted=args.weighted,
    )
    print("- Kraj generisanja -")

    n = len(responses)
    pad = args.max_items

    X = np.stack(responses,    axis=0)        # (N, size_students, max_items)
    Y = np.stack(adj_matrices, axis=0)        # (N, max_items, max_items)
    C = np.array(item_counts,  dtype=np.int64) # (N,)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        Y=Y,
        item_counts=C,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"\nDataset sacuvan: {output_path}  ({size_mb:.2f} MB)")
    print(f"\nOblik tenzora:")
    print(f"  X (responses):   {X.shape}  - input za transformer")
    print(f"  Y (adj matrix):  {Y.shape} - output 2D")
    print(f"  item_counts:     {C.shape}  - za masking padding-a")
    print(f"\nDistribucija broja pitanja/item-a:")
    for k in range(args.min_items, args.max_items + 1):
        cnt = (C == k).sum()
        print(f"  {k} items: {cnt} primera ({100*cnt/n:.1f}%)")

    write_dataset_card(output_path, args, C)

    print(f"\nQuick check (prvi primer):")
    print(f"  item_count = {C[0]}")
    print(f"  Adj matrica:\n{Y[0]}")
    unique_responses = np.unique(X[0], axis=0)
    print(f"  Unique response pattern-i: {len(unique_responses)} / {args.size}")
    print(f"\n- Kraj -")


if __name__ == "__main__":
    main()
