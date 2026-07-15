"""
Testovi za src/util/generate_dataset_util.py - tranzitivno zatvorenje/redukcija
i generisanje sintetickog dataseta.
"""

import numpy as np

from util.generate_dataset_util import (
    transitive_closure,
    transitive_reduction,
    random_implications,
    generate_dataset,
)


def test_closure_chain_adds_transitive_edge():
    # (0,1) i (1,2) u ulazu -> zatvorenje mora sadrzati i (0,2)
    result = transitive_closure([(0, 1), (1, 2)], items=3)
    assert (0, 2) in result


def test_closure_full_chain_all_pairs_reachable():
    # lanac (0,1),(1,2),...,(n-2,n-1) -> zatvorenje sadrzi SVE parove (i,j), i<j
    items = 5
    chain = [(i, i + 1) for i in range(items - 1)]
    result = set(transitive_closure(chain, items))
    expected = {(i, j) for i in range(items) for j in range(items) if i < j}
    assert result == expected


def test_closure_is_idempotent():
    # transitive_closure(transitive_closure(x)) == transitive_closure(x)
    items = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    once = set(transitive_closure(edges, items))
    twice = set(transitive_closure(list(once), items))
    assert once == twice


def test_closure_empty_input_returns_empty():
    # prazna lista implikacija -> prazna lista na izlazu
    assert transitive_closure([], items=4) == []


def test_closure_never_contains_self_loops():
    # nijedan (i, i) par ne sme da se pojavi u zatvorenju
    items = 4
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    result = transitive_closure(edges, items)
    assert all(i != j for (i, j) in result)


def test_reduction_removes_redundant_edge_from_chain():
    # zatvorenje lanca 0->1->2 (sadrzi i (0,2)) -> redukcija vraca samo {(0,1), (1,2)}
    closed = transitive_closure([(0, 1), (1, 2)], items=3)
    assert (0, 2) in closed  # sanity check da je (0,2) zaista u zatvorenju

    reduced = transitive_reduction(closed)
    assert set(reduced) == {(0, 1), (1, 2)}


def test_reduction_is_idempotent():
    # transitive_reduction(transitive_reduction(x)) == transitive_reduction(x)
    closed = transitive_closure([(0, 1), (1, 2), (2, 3)], items=4)
    once = set(transitive_reduction(closed))
    twice = set(transitive_reduction(list(once)))
    assert once == twice


def test_reduction_never_contains_self_loops():
    # redukcija nikad ne vraca (i, i) parove
    edges = [(0, 0), (0, 1), (1, 2)]
    result = transitive_reduction(edges)
    assert all(i != j for (i, j) in result)
    assert set(result) == {(0, 1), (1, 2)}


def test_closure_then_reduction_roundtrip():
    # za DAG bez redundantnih ivica na pocetku:
    # transitive_reduction(transitive_closure(edges)) == edges
    items = 4
    edges = [(0, 1), (1, 2), (0, 3)]
    closed = transitive_closure(edges, items)
    reduced = transitive_reduction(closed)
    assert set(reduced) == set(edges)


def test_random_implications_produces_dag():
    # impl_closed i impl_reduced nikad ne sadrze (i,j) i (j,i) istovremeno (nema ciklusa)
    for seed in range(20):
        np.random.seed(seed)
        impl_closed, impl_reduced = random_implications(6)
        assert all((j, i) not in impl_closed for (i, j) in impl_closed)
        assert all((j, i) not in impl_reduced for (i, j) in impl_reduced)


def test_random_implications_closed_is_actually_closed():
    # impl_closed mora biti fiksna tacka transitive_closure funkcije
    for seed in range(20):
        np.random.seed(seed)
        impl_closed, _ = random_implications(6)
        assert set(transitive_closure(impl_closed, 6)) == set(impl_closed)


def test_random_implications_reduced_is_actually_minimal():
    # nijedna ivica u impl_reduced ne sme biti izvodljiva iz preostalih ivica
    for seed in range(20):
        np.random.seed(seed)
        _, impl_reduced = random_implications(6)
        roundtrip = transitive_reduction(transitive_closure(impl_reduced, 6))
        assert set(roundtrip) == set(impl_reduced)


def test_generate_dataset_output_shapes():
    # responses: (student_size, max_items) po uzorku
    # adj_matrices: (max_items, max_items)
    # item_counts: svaka vrednost u [min_items, max_items]
    np.random.seed(0)
    responses, adj_matrices, item_counts = generate_dataset(
        num_samples=10, max_items=4, min_items=2, student_size=15,
        ce=0.1, lg=0.05,
    )
    assert len(responses) == len(adj_matrices) == len(item_counts)
    for r, a, n in zip(responses, adj_matrices, item_counts):
        assert r.shape == (15, 4)
        assert a.shape == (4, 4)
        assert 2 <= n <= 4


def test_generate_dataset_padding_is_zero():
    # kolone u response matrici i redovi/kolone u adj matrici van item_count
    # moraju biti sve nule
    np.random.seed(0)
    responses, adj_matrices, item_counts = generate_dataset(
        num_samples=10, max_items=5, min_items=2, student_size=15,
        ce=0.1, lg=0.05,
    )
    for r, a, n in zip(responses, adj_matrices, item_counts):
        if n < 5:
            assert np.all(r[:, n:] == 0)
            assert np.all(a[n:, :] == 0)
            assert np.all(a[:, n:] == 0)


def test_generate_dataset_adj_matches_impl_reduced(monkeypatch):
    # adj[:n, :n] restrikcija mora se tacno poklapati sa impl_reduced (bez viska/manjka ivica)
    fixed_reduced = [(0, 1), (1, 2)]
    fixed_closed = transitive_closure(fixed_reduced, 3)

    def fake_random_implications(items, min_impl=0, max_impl=None):
        return fixed_closed, fixed_reduced

    monkeypatch.setattr(
        "util.generate_dataset_util.random_implications", fake_random_implications
    )

    _, adj_matrices, item_counts = generate_dataset(
        num_samples=3, max_items=3, min_items=3, student_size=10,
        ce=0.1, lg=0.05,
    )

    expected = np.zeros((3, 3), dtype=np.int8)
    for (i, j) in fixed_reduced:
        expected[i, j] = 1

    for a, n in zip(adj_matrices, item_counts):
        assert np.array_equal(a[:n, :n], expected)


def test_generate_dataset_ce_lg_within_clip_bounds(monkeypatch):
    # ce_rand uvek u [0, 0.3], lg_rand uvek u [0, 0.2], nezavisno od normalnog suma
    captured = []

    def fake_simu(items, size, ce, lg, delta, imp):
        captured.append((ce, lg))
        return {"dataset": np.zeros((size, items), dtype=np.int8)}

    monkeypatch.setattr("util.generate_dataset_util.simu", fake_simu)

    generate_dataset(
        num_samples=20, max_items=3, min_items=3, student_size=5,
        ce=10.0, lg=10.0, ce_std=5.0, lg_std=5.0,
    )

    assert captured  # sanity check da je simu uopste pozvan
    for ce_val, lg_val in captured:
        assert 0.0 <= ce_val <= 0.3
        assert 0.0 <= lg_val <= 0.2
