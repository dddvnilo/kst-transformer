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
import numpy as np
from pathlib import Path
from learning_spaces.kst import simu

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPTS_DIR, "..")


# ---------------------------------------------------------------------------
# Tranzitivno zatvorenje
# ---------------------------------------------------------------------------

def transitive_closure(impl_list, items):
    """
    Floyd-Warshall: prosiruje listu implikacija na puno tranzitivno zatvorenje.
    Npr. ako postoji (0,1) i (1,2), dodaje i (0,2).
    Ne ukljucuje refleksivne parove (i,i).
    """
    reach = [[False] * items for _ in range(items)]
    for (i, j) in impl_list:
        reach[i][j] = True

    for k in range(items):
        for i in range(items):
            for j in range(items):
                if reach[i][k] and reach[k][j]:
                    reach[i][j] = True

    return [(i, j) for i in range(items) for j in range(items) if reach[i][j] and i != j]


# ---------------------------------------------------------------------------
# Tranzitivna redukcija
# ---------------------------------------------------------------------------

def transitive_reduction(imp):
    """
    Vraca minimalni skup implikacija (bez refleksivnih i tranzitivnih).
    Npr. ako postoji (0,1) i (1,2), uklanja (0,2).
    """
    implications = list(imp)

    for i in list(implications):
        if i[0] == i[1]:
            implications.remove(i)

    for i in list(implications):
        for j in list(implications):
            for k in list(implications):
                if i[1] == j[0] and j[1] == k[1] and i[0] == k[0]:
                    if k in implications:
                        implications.remove(k)

    return implications


# ---------------------------------------------------------------------------
# Generisanje random implikacija
# ---------------------------------------------------------------------------

def random_implications(items, min_impl=0, max_impl=None):
    """
    Generise nasumican skup implikacija izmedju `items` pitanja.
    Implikacija (i, j) znači: j zahteva i kao prerequisit.

    Koristi samo parove gde i < j (gornji trougao) - garantuje DAG,
    nikad (i,j) i (j,i) istovremeno.

    Vraca:
        impl_closed:  puno tranzitivno zatvorenje - za simu
        impl_reduced: minimalne implikacije - za adj matricu
    """

    topo = np.random.permutation(items)
    possible = [(int(topo[i]), int(topo[j])) for i in range(items) for j in range(i + 1, items)]

    if max_impl is None:
        max_impl = len(possible)

    num_impl = np.random.randint(min_impl, min(max_impl, len(possible)) + 1)
    chosen = np.random.choice(len(possible), size=num_impl, replace=False)
    impl_base = [possible[k] for k in chosen]

    impl_closed = transitive_closure(impl_base, items)
    impl_reduced = transitive_reduction(impl_closed)

    return impl_closed, impl_reduced


# ---------------------------------------------------------------------------
# Generator dataseta
# ---------------------------------------------------------------------------

def generate_dataset(
    num_samples: int,
    max_items: int,
    min_items: int,
    student_size: int,
    ce: float,
    lg: float,
    ce_std: float = 0.03,
    lg_std: float = 0.02,
    pad_to: int = None,
):
    """
    Generise `num_samples` primera.

    Svaki primer:
      - nasumično bira broj pitanja iz [min_items, max_items]
      - generise nasumicne implikacije (bez ciklusa, sa tranzitivnim zatvorenjem)
      - simulira response matricu studenata
      - padduje response matricu na `pad_to` kolona (ako je zadato)
      - adj matrica sadrzi samo minimalne implikacije (tranzitivna redukcija)

    Return:
        responses:    list of np.array oblika (student_size, pad_to)
        adj_matrices: list of np.array oblika (pad_to, pad_to)
        item_counts:  list of int - stvarni broj pitanja u svakom primeru
    """
    if pad_to is None:
        pad_to = max_items

    responses = []
    adj_matrices = []
    item_counts = []

    for idx in range(num_samples):
        # Nasumican broj pitanja
        n_items = np.random.randint(min_items, max_items + 1)

        ce_rand = np.clip(np.random.normal(ce, ce_std), 0.0, 0.3)
        lg_rand = np.clip(np.random.normal(lg, lg_std), 0.0, 0.2)

        # Generisi implikacije
        impl_closed, impl_reduced = random_implications(n_items)

        # Simuliraj odgovore - simu dobija puno tranzitivno zatvorenje
        try:
            result = simu(
                items=n_items,
                size=student_size,
                ce=ce_rand,
                lg=lg_rand,
                delta=0.0,       # delta nije relevantna kad prosledjujemo imp
                imp=impl_closed if impl_closed else []
            )
        except Exception:
            continue

        response = result['dataset']  # (student_size, n_items)

        # Adj matrica od minimalnih implikacija (tranzitivna redukcija)
        adj = np.zeros((n_items, n_items), dtype=np.int8)
        for (pre, item) in impl_reduced:
            adj[pre][item] = 1

        # Padovanje response matrice na pad_to kolona
        if n_items < pad_to:
            pad_cols = np.zeros((student_size, pad_to - n_items), dtype=np.int8)
            response = np.hstack([response, pad_cols])

        # Padovanje adj matrice na (pad_to x pad_to)
        full_adj = np.zeros((pad_to, pad_to), dtype=np.int8)
        full_adj[:n_items, :n_items] = adj

        responses.append(response)
        adj_matrices.append(full_adj)
        item_counts.append(n_items)

        # Ispis na svakih 10%
        log_step = max(1, num_samples // 10)
        if (idx + 1) % log_step == 0:
            print(f"  Generisano {idx + 1}/{num_samples} primera...")

    return responses, adj_matrices, item_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  KST Dataset Generator")
    print(f"{'='*55}")
    print(f"  Broj primera:            {args.num_samples}")
    print(f"  Items:                   {args.min_items} - {args.max_items}")
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
    )
    print("- Kraj generisanja -")

    n = len(responses)
    pad = args.max_items

    X = np.stack(responses,    axis=0)        # (N, size, pad_to)
    Y = np.stack(adj_matrices, axis=0)        # (N, pad_to, pad_to)
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

    print(f"\nQuick check (prvi primer):")
    print(f"  item_count = {C[0]}")
    print(f"  Adj matrica:\n{Y[0]}")
    unique_responses = np.unique(X[0], axis=0)
    print(f"  Unique response pattern-i: {len(unique_responses)} / {args.size}")
    print(f"\n- Kraj -")


if __name__ == "__main__":
    main()