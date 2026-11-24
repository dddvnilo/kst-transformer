"""
KST Dataset Generator

Generise podatke u obliku:
  INPUT:  response matrica studenata (size x items) - binarna
  OUTPUT: adjacency matrica prerequisita (items x items) - binarna, puno tranzitivno zatvorenje
          (kvazi-poredak)

Generator i izvor parametara: src/util/generate_dataset_util.py.

Koristi:
  python generate_dataset.py \
      --num-samples 80000 \
      --max-items 10 \
      --min-items 2 \
      --size 500 \
      --weighted
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from util.generate_dataset_util import generate_dataset, DELTA_MEAN, DELTA_STD, DELTA_MAX, ERROR_RATES
from util.paths import DATA_DIR, resolve_path


def write_dataset_card(output_path: Path, args: argparse.Namespace, item_counts: np.ndarray,
                       Y: np.ndarray, error_rates: np.ndarray) -> None:
    """Pise <output_path bez .npz>.md - YAML frontmatter + markdown, HF Dataset Card stil."""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_commit = None

    n = len(item_counts)
    generator_args = dict(vars(args))
    generator_args["output"] = output_path.name

    frontmatter = {
        "dataset_name":   output_path.stem,
        "generated_by":   "scripts/generate_dataset.py",
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator_args": generator_args,
        "git_commit":     git_commit,
    }

    distribution_rows = []
    for k in range(args.min_items, args.max_items + 1):
        cnt = int((item_counts == k).sum())
        pct = 100 * cnt / n if n > 0 else 0.0
        distribution_rows.append(f"| {k} | {cnt} | {pct:.1f}% |")

    error_rows = []
    for tau in ERROR_RATES:
        cnt = int(np.isclose(error_rates, tau).sum())
        pct = 100 * cnt / n if n > 0 else 0.0
        error_rows.append(f"| {tau:.2f} | {cnt} | {pct:.1f}% |")

    graph_rows = []
    for k in range(args.min_items, args.max_items + 1):
        Yk = Y[item_counts == k][:, :k, :k].astype(bool)
        if len(Yk) == 0:
            continue
        sizes = Yk.sum(axis=(1, 2))
        has_equivalent = (Yk & Yk.transpose(0, 2, 1)).any(axis=(1, 2))
        graph_rows.append(
            f"| {k} | {sizes.mean():.1f} / {k * (k - 1)} | {100 * (sizes == 0).mean():.1f}% "
            f"| {100 * has_equivalent.mean():.1f}% |"
        )

    body = (
        f"\n# {output_path.stem}\n\n"
        "## Summary\n"
        "Sinteticki dataset za KST Transformer - simulirani odgovori studenata, "
        "generisani sa `scripts/generate_dataset.py`.\n\n"
        "## Generator\n"
        f"- Grafovi: normalna sema (Sargin & Unlu, 2009a) - delta ~ N({DELTA_MEAN}, {DELTA_STD}), "
        f"odseceno na [0, {DELTA_MAX}]; svaki par (i, j), i != j, ulazi sa verovatnocom delta; "
        "pa tranzitivno zatvorenje. Rezultat je kvazi-poredak (moguca ekvivalentna pitanja).\n"
        f"- Sum: careless error = lucky guess = tau, tau uniformno iz {list(ERROR_RATES)} po uzorku.\n"
        "- Izvor: Sargin (2009), disertacija, sek. 3.4.1 "
        "(`notes/learning spaces learning material/35097045.pdf`).\n\n"
        "## Structure\n"
        f"- `X`: (N, {args.size}, {args.max_items}) float32 - binarna response matrica, paddovana nulama\n"
        f"- `Y`: (N, {args.max_items}, {args.max_items}) float32 - PUNO TRANZITIVNO ZATVORENJE "
        "(kvazi-poredak; ekvivalentna pitanja imaju veze u oba smera), dijagonala = 0\n"
        "- `item_counts`: (N,) int64 - stvarni broj pitanja po uzorku\n"
        "- `error_rates`: (N,) float32 - stopa greske tau po uzorku\n\n"
        "## Distribucija broja pitanja\n"
        "| items | count | % |\n"
        "|---|---|---|\n"
        + "\n".join(distribution_rows) + "\n\n"
        "## Distribucija stope greske tau\n"
        "| tau | count | % |\n"
        "|---|---|---|\n"
        + "\n".join(error_rows) + "\n\n"
        "## Grafovi po broju pitanja\n"
        "| items | prosecno implikacija / max | prazna relacija | sa ekvivalentnim pitanjima |\n"
        "|---|---|---|---|\n"
        + "\n".join(graph_rows) + "\n"
    )

    card_path = output_path.with_suffix(".md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.safe_dump(frontmatter, f, sort_keys=False, allow_unicode=True)
        f.write("---\n")
        f.write(body)

    print(f"Kartica sacuvana: {card_path}")


def build_default_output(args: argparse.Namespace) -> str:
    """Konstruise default ime .npz fajla iz parametara generisanja
    (npr. kst_dataset_2-10items_80k_weighted_daks.npz). Sufiks "_daks" sprecava prepisivanje
    legacy dataset-a sa diplomskog."""
    samples_tag = f"{args.num_samples // 1000}k" if args.num_samples % 1000 == 0 else str(args.num_samples)
    items_tag = f"{args.max_items}items" if args.min_items == args.max_items else f"{args.min_items}-{args.max_items}items"
    weighted_tag = "weighted" if args.weighted else "uniform"
    filename = f"kst_dataset_{items_tag}_{samples_tag}_{weighted_tag}_daks.npz"
    return str(DATA_DIR / filename)


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
    parser.add_argument("--output",      type=str,   default=None,
                        help="Output fajl (.npz format) - samo ime (upisuje se u data/) ili puna putanja; "
                             "ako nije zadat, konstruise se automatski iz parametara (broj pitanja, broj uzoraka, weighted/uniform)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--weighted",    action="store_true", default=False,
                        help="Linearne tezine — vise pitanja = veci udeo uzoraka")
    args = parser.parse_args()

    if args.output is None:
        args.output = build_default_output(args)
    else:
        args.output = str(resolve_path(args.output, DATA_DIR))

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
    print(f"  Grafovi:                 delta ~ N({DELTA_MEAN}, {DELTA_STD}), odseceno na [0, {DELTA_MAX}]")
    print(f"  Stopa greske tau:        {list(ERROR_RATES)} (careless error = lucky guess)")
    print(f"  Output:                  {args.output}")
    print(f"{'='*55}\n")

    print("- Pocetak generisanja -")
    responses, adj_matrices, item_counts, error_rates = generate_dataset(
        num_samples=args.num_samples,
        max_items=args.max_items,
        min_items=args.min_items,
        student_size=args.size,
        weighted=args.weighted,
    )
    print("- Kraj generisanja -")

    n = len(responses)

    X = np.stack(responses,    axis=0)            # (N, size_students, max_items)
    Y = np.stack(adj_matrices, axis=0)            # (N, max_items, max_items)
    C = np.array(item_counts,  dtype=np.int64)    # (N,)
    T = np.array(error_rates,  dtype=np.float32)  # (N,)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        Y=Y,
        item_counts=C,
        error_rates=T,
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

    write_dataset_card(output_path, args, C, Y, T)

    print(f"\nQuick check (prvi primer):")
    print(f"  item_count = {C[0]}")
    print(f"  Adj matrica:\n{Y[0]}")
    unique_responses = np.unique(X[0], axis=0)
    print(f"  Unique response pattern-i: {len(unique_responses)} / {args.size}")
    print(f"\n- Kraj -")


if __name__ == "__main__":
    main()
