"""
KST Dataset Generator

Generise sintetičke podatke u obliku:
  INPUT:  response matrica studenata (size x items) - binarna
  OUTPUT: adjacency matrica prerequisita (items x items) - binarna

Koristi:
  python generate_kst_dataset.py \
      --num-samples 10000 \
      --max-items 5 \
      --min-items 2 \
      --size 200 \
      --ce 0.1 \
      --lg 0.05 \
      -- delta 0.5 \
      --output dataset.npz
"""

import argparse
import numpy as np
from itertools import product
from pathlib import Path
from learning_spaces.kst import simu

# # ---------------------------------------------------------------------------
# # Generisanje random implikacija
# # ---------------------------------------------------------------------------

# def random_implications(items, min_impl=0, max_impl=None):
#     """
#     Generiše nasumičan skup implikacija između `items` pitanja.
#     Implikacija (i, j) znači: j zahteva i kao prerequisit.
#     Izbegava cikluse (DAG struktura).

#     Vraća:
#         impl_list: lista tuplova (prerequisite, item)
#         adj_matrix: numpy matrica oblika (items x items),
#                     adj[i][j] = 1 znači j zahteva i (i je prerequisit za j)
#     """
#     if max_impl is None:
#         max_impl = items * (items - 1) // 2

#     # Radimo na gornjetrougaonoj matrici da garantujemo DAG
#     # (pitanje i je prerequisit za pitanje j samo ako i < j)
#     # Ovo je pojednostavljivanje - možeš koristiti topološki sort za opštiji slučaj
#     possible = [(i, j) for i in range(items) for j in range(items) if i != j]

#     num_impl = np.random.randint(min_impl, max_impl + 1)
#     num_impl = min(num_impl, len(possible))

#     chosen = np.random.choice(len(possible), size=num_impl, replace=False)
#     impl_list = [possible[k] for k in chosen]


#     return impl_list, adj


# ---------------------------------------------------------------------------
# Glavni generator dataseta
# ---------------------------------------------------------------------------

def generate_dataset(
    num_samples: int,
    max_items: int,
    min_items: int,
    student_size: int,
    ce: float,
    lg: float,
    delta: float,
    ce_std: float = 0.03,
    lg_std: float = 0.02,
    delta_std: float = 0.1,
    pad_to: int = None,
):
    """
    Generise `num_samples` primera

    Svaki primer:
      - nasumično bira broj pitanja iz [min_items, max_items]
      - generise nasumicne implikacije (kroz simulator)
      - simulira response matricu studenata
      - padduje response matricu na `pad_to` kolona (ako je zadato)

    Vraca:
        responses:   list of np.array oblika (student_size, pad_to)
        adj_matrices: list of np.array oblika (pad_to, pad_to)
        item_counts: list of int - stvarni broj pitanja u svakom primeru
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
        delta_rand = np.clip(np.random.normal(delta, delta_std), 0.0, 1.0)

        # Simuliraj odgovore
        try:
            result = simu(
                items=n_items,
                size=student_size,
                ce=ce_rand,
                lg=lg_rand,
                delta=delta_rand,
                imp=None
            )
        except ValueError:
            # Edge case: nema validnih stanja (jako restriktivne impl.) - preskoci
            continue

        response = result['dataset']  # (student_size, n_items)
        implications = result['implications']

        # TODO: video sam ovde da radi ciklicne veze tipa (0,1) i (1,0)
        # PREDLOG: napraviti skriptu moju koja generise implikacije
        # print(implications)

        # Napravi adjacency matricu
        adj = np.zeros((n_items, n_items), dtype=np.int8)
        for (pre, item) in implications:
            adj[pre][item] = 1

        # Padovanje na pad_to kolona (desno, nulama)
        if n_items < pad_to:
            pad_cols = np.zeros((student_size, pad_to - n_items), dtype=np.int8)
            response = np.hstack([response, pad_cols])

        # Padovanje adjacency matrice na (pad_to x pad_to)
        full_adj = np.zeros((pad_to, pad_to), dtype=np.int8)
        full_adj[:n_items, :n_items] = adj

        responses.append(response)
        adj_matrices.append(full_adj)
        item_counts.append(n_items)

        if (idx + 1) % 500 == 0:
            print(f"  Generisano {idx + 1}/{num_samples} primera...")

    return responses, adj_matrices, item_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KST Dataset Generator"
    )
    parser.add_argument("--num-samples",    type=int,   default=5000,
                        help="Broj primera u datasetu (default: 5000)")
    parser.add_argument("--max-items",      type=int,   default=5,
                        help="Maksimalan broj pitanja (default: 5)")
    parser.add_argument("--min-items",      type=int,   default=2,
                        help="Minimalan broj pitanja (default: 2)")
    parser.add_argument("--size",           type=int,   default=500,
                        help="Broj studenata po simulaciji (default: 500)")
    parser.add_argument("--ce",             type=float, default=0.1,
                        help="Careless error verovatnoca (default: 0.1)")
    parser.add_argument("--lg",             type=float, default=0.05,
                        help="Lucky guess verovatnoca (default: 0.05)")
    parser.add_argument("--delta",             type=float, default=0.5,
                        help="Delta (default: 0.5)")
    parser.add_argument("--ce-std", type=float, default=0.03,
                    help="Std za CE noise (default: 0.03)")
    parser.add_argument("--lg-std", type=float, default=0.02,
                        help="Std za LG noise (default: 0.02)")
    parser.add_argument("--delta-std", type=float, default=0.1,
                        help="Std za delta noise (default: 0.1)")
    parser.add_argument("--output",         type=str,   default="data/kst_dataset.npz",
                        help="Output fajl (.npz format, default: data/kst_dataset.npz)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  KST Dataset Generator")
    print(f"{'='*55}")
    print(f"  Broj primera:     {args.num_samples}")
    print(f"  Items:            {args.min_items} – {args.max_items}")
    print(f"  Studenata/primer: {args.size}")
    print(f"  Careless error:   {args.ce}")
    print(f"  Lucky guess:      {args.lg}")
    print(f"  Delta:            {args.delta}")
    print(f"  CE std:           {args.ce_std}")
    print(f"  LG std:           {args.lg_std}")
    print(f"  Delta std:        {args.delta_std}")
    print(f"  Output:           {args.output}")
    print(f"{'='*55}\n")

    print("- Pocetak generacije -")
    responses, adj_matrices, item_counts = generate_dataset(
        num_samples=args.num_samples,
        max_items=args.max_items,
        min_items=args.min_items,
        student_size=args.size,
        ce=args.ce,
        lg=args.lg,
        delta=args.delta,
        ce_std=args.ce_std,
        lg_std=args.lg_std,
        delta_std=args.delta_std
    )

    n = len(responses)
    pad = args.max_items

    # Stack u numpy array-eve
    # responses:    (N, student_size, pad_to)
    # adj_matrices: (N, pad_to, pad_to)  - flattened -> (N, pad_to*pad_to)
    X = np.stack(responses,    axis=0)           # (N, size, pad_to)
    Y = np.stack(adj_matrices, axis=0)           # (N, pad_to, pad_to)
    C = np.array(item_counts,  dtype=np.int8)    # (N,) - stvarni broj items

    output_path = Path(args.output)
    np.savez_compressed(
        output_path,
        X=X,           # response matrice
        Y=Y,           # adjacency matrice (2D)
        item_counts=C, # stvarni broj pitanja po primeru
    )

    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"\nDataset sacuvan: {output_path}  ({size_mb:.2f} MB)")
    print(f"\nOblik tenzora:")
    print(f"  X (responses):   {X.shape}   - input za transformer")
    print(f"  Y (adj matrix):  {Y.shape}   - output 2D")
    print(f"  item_counts:     {C.shape}   - za masking padding-a")
    print(f"\nDistribucija broja pitanja/item-a:")
    for k in range(args.min_items, args.max_items + 1):
        cnt = (C == k).sum()
        print(f"  {k} items: {cnt} primera ({100*cnt/n:.1f}%)")

    # Provera prvog primera
    print(f"\nQuick check (prvi primer):")
    print(f"  item_count = {C[0]}")
    print(f"  Adj matrica:\n{Y[0]}")
    unique_responses = np.unique(X[0], axis=0)
    print(f"  Unique response pattern-i: {len(unique_responses)} / {args.size}")
    print(f"\n- Kraj. -")


if __name__ == "__main__":
    main()