import numpy as np
from learning_spaces.kst import simu


# Tranzitivno zatvorenje

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


# Tranzitivna redukcija

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


# Generisanje random implikacija

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


# Generator dataseta

def generate_dataset(
    num_samples: int,
    max_items: int,
    min_items: int,
    student_size: int,
    ce: float,
    lg: float,
    ce_std: float = 0.03,
    lg_std: float = 0.02,
    weighted: bool = False,
):
    """
    Generise `num_samples` primera.

    Svaki primer:
      - nasumično bira broj pitanja iz [min_items, max_items] (uniformno ili tezinski)
      - generise nasumicne implikacije (bez ciklusa, sa tranzitivnim zatvorenjem)
      - simulira response matricu studenata
      - padduje response matricu na `max_items` (pad_to) kolona
      - adj matrica sadrzi samo minimalne implikacije (tranzitivna redukcija)

    Return:
        responses:    list of np.array oblika (student_size, max_items)
        adj_matrices: list of np.array oblika (max_items, max_items)
        item_counts:  list of int - stvarni broj pitanja u svakom primeru
    """

    pad_to = max_items

    item_range = list(range(min_items, max_items + 1))
    if weighted:
        w = np.arange(1, len(item_range) + 1, dtype=float)  # [1, 2, 3, ...]
        probs = w / w.sum()
    else:
        probs = None  # uniformno

    responses = []
    adj_matrices = []
    item_counts = []

    for idx in range(num_samples):
        # Nasumican broj pitanja
        n_items = int(np.random.choice(item_range, p=probs))

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

        # Padovanje response matrice na pad_to (max_items) kolona
        if n_items < pad_to:
            pad_cols = np.zeros((student_size, pad_to - n_items), dtype=np.int8)
            response = np.hstack([response, pad_cols])

        # Padovanje adj matrice na (pad_to x pad_to, odnosno max_items x max_items)
        full_adj = np.zeros((pad_to, pad_to), dtype=np.int8)
        full_adj[:n_items, :n_items] = adj

        responses.append(response)
        adj_matrices.append(full_adj)
        item_counts.append(n_items)

        # Ispis na svakih 10%
        log_step = max(1, num_samples // 10)
        if (idx + 1) % log_step == 0:
            print(f"  Generisano {idx + 1}/{num_samples} primera")

    return responses, adj_matrices, item_counts
