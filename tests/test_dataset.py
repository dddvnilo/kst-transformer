"""
Testovi za src/kst/dataset.py - KSTDataset, make_loss_mask,
batched_transitive_closure, make_lenient_loss_mask, make_dataloaders.
"""

import numpy as np
import torch
import pytest

from kst.dataset import (
    KSTDataset,
    make_loss_mask,
    batched_transitive_closure,
    make_lenient_loss_mask,
    make_dataloaders,
)


@pytest.fixture
def tiny_npz(tmp_path):
    """Mali sinteticki .npz fajl (20 uzoraka, 15 studenata, max 4 pitanja)."""
    n, students, max_items = 20, 15, 4
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(n, students, max_items)).astype(np.float32)
    Y = np.zeros((n, max_items, max_items), dtype=np.float32)
    item_counts = rng.integers(2, max_items + 1, size=n).astype(np.int64)
    path = tmp_path / "tiny.npz"
    np.savez_compressed(path, X=X, Y=Y, item_counts=item_counts)
    return str(path)


# --- make_loss_mask ---

def test_make_loss_mask_diagonal_always_false():
    # dijagonala (i == j) mora biti False za svaki uzorak, nezavisno od item_count
    item_counts = torch.tensor([5, 3])
    mask = make_loss_mask(item_counts, max_items=5)
    for b in range(2):
        assert not mask[b].diagonal().any()


def test_make_loss_mask_padding_cells_false():
    # celije gde je i >= item_count ili j >= item_count moraju biti False
    item_counts = torch.tensor([3])
    mask = make_loss_mask(item_counts, max_items=5)[0]
    assert not mask[3:, :].any()
    assert not mask[:, 3:].any()


def test_make_loss_mask_valid_cells_true():
    # ne-dijagonalne celije unutar item_count granice moraju biti True
    item_counts = torch.tensor([3])
    mask = make_loss_mask(item_counts, max_items=5)[0]
    for i in range(3):
        for j in range(3):
            if i != j:
                assert mask[i, j]


def test_make_loss_mask_zero_item_count():
    # item_count == 0 -> citava maska za taj uzorak je False
    item_counts = torch.tensor([0])
    mask = make_loss_mask(item_counts, max_items=4)[0]
    assert not mask.any()


def test_make_loss_mask_full_item_count():
    # item_count == max_items -> samo dijagonala je False, sve ostalo True
    item_counts = torch.tensor([4])
    mask = make_loss_mask(item_counts, max_items=4)[0]
    expected = ~torch.eye(4, dtype=torch.bool)
    assert torch.equal(mask, expected)


# --- batched_transitive_closure ---

def test_batched_closure_chain_adds_transitive_edge():
    # (0,1) i (1,2) u ulazu -> zatvorenje mora sadrzati i (0,2)
    Y = torch.zeros(1, 3, 3, dtype=torch.bool)
    Y[0, 0, 1] = True
    Y[0, 1, 2] = True
    closed = batched_transitive_closure(Y)
    assert closed[0, 0, 2]


def test_batched_closure_empty_stays_empty():
    # prazna adjacency matrica -> zatvorenje je i dalje prazno
    Y = torch.zeros(1, 4, 4, dtype=torch.bool)
    closed = batched_transitive_closure(Y)
    assert not closed.any()


def test_batched_closure_is_idempotent():
    # closure(closure(x)) == closure(x)
    Y = torch.zeros(1, 4, 4, dtype=torch.bool)
    Y[0, 0, 1] = True
    Y[0, 1, 2] = True
    Y[0, 2, 3] = True
    once = batched_transitive_closure(Y)
    twice = batched_transitive_closure(once)
    assert torch.equal(once, twice)


def test_batched_closure_batch_samples_independent():
    # ivice jednog uzorka u batchu ne smeju da uticu na drugi uzorak
    Y = torch.zeros(2, 3, 3, dtype=torch.bool)
    Y[0, 0, 1] = True
    Y[0, 1, 2] = True
    # uzorak 1 ostaje bez ijedne ivice
    closed = batched_transitive_closure(Y)
    assert closed[0, 0, 2]
    assert not closed[1].any()


# --- make_lenient_loss_mask ---

def test_lenient_mask_excludes_transitive_only_cell():
    # lanac 0->1->2: (0,2) je tranzitivna, ne direktna Hasse ivica -> iskljucena
    Y = torch.zeros(1, 3, 3)
    Y[0, 0, 1] = 1.0
    Y[0, 1, 2] = 1.0
    base_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    base_mask[0].fill_diagonal_(False)
    lenient = make_lenient_loss_mask(Y, base_mask)
    assert not lenient[0, 0, 2]


def test_lenient_mask_keeps_direct_edges():
    # direktne Hasse ivice (0,1) i (1,2) moraju ostati u maski
    Y = torch.zeros(1, 3, 3)
    Y[0, 0, 1] = 1.0
    Y[0, 1, 2] = 1.0
    base_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    base_mask[0].fill_diagonal_(False)
    lenient = make_lenient_loss_mask(Y, base_mask)
    assert lenient[0, 0, 1]
    assert lenient[0, 1, 2]


def test_lenient_mask_keeps_true_negatives():
    # (1,0) - obrnut smer, nikad implicirana - mora ostati u maski (prava negativna veza)
    Y = torch.zeros(1, 3, 3)
    Y[0, 0, 1] = 1.0
    Y[0, 1, 2] = 1.0
    base_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    base_mask[0].fill_diagonal_(False)
    lenient = make_lenient_loss_mask(Y, base_mask)
    assert lenient[0, 1, 0]


def test_lenient_mask_is_subset_of_base_mask():
    # lenient maska nikad ne sme da vrati True tamo gde je base_mask vec bila False
    item_counts = torch.tensor([3])
    base_mask = make_loss_mask(item_counts, max_items=5)
    Y = torch.zeros(1, 5, 5)
    Y[0, 0, 1] = 1.0
    Y[0, 1, 2] = 1.0
    lenient = make_lenient_loss_mask(Y, base_mask)
    assert torch.equal(lenient & base_mask, lenient)


# --- KSTDataset ---

def test_kst_dataset_len_matches_npz(tiny_npz):
    # KSTDataset.__len__() mora vratiti tacan broj uzoraka N iz .npz fajla
    ds = KSTDataset(tiny_npz)
    assert len(ds) == 20


def test_kst_dataset_getitem_shapes_and_dtypes(tiny_npz):
    # __getitem__ vraca X (students, max_items) float32, Y (max_items, max_items)
    # float32, item_count int64 skalar
    ds = KSTDataset(tiny_npz)
    X, Y, item_count = ds[0]
    assert X.shape == (15, 4)
    assert Y.shape == (4, 4)
    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32
    assert item_count.dtype == torch.int64


# --- make_dataloaders ---

def test_make_dataloaders_splits_are_disjoint_and_complete(tiny_npz):
    # indeksi train/val/test se ne preklapaju i zajedno pokrivaju sve uzorke
    train_loader, val_loader, test_loader = make_dataloaders(
        tiny_npz, batch_size=4, val_ratio=0.2, test_ratio=0.2, seed=0,
    )
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    test_idx = set(test_loader.dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(range(20))


def test_make_dataloaders_reproducible_with_seed(tiny_npz):
    # isti seed -> identican train/val/test split
    _, val1, _ = make_dataloaders(tiny_npz, batch_size=4, val_ratio=0.2, test_ratio=0.2, seed=7)
    _, val2, _ = make_dataloaders(tiny_npz, batch_size=4, val_ratio=0.2, test_ratio=0.2, seed=7)
    assert list(val1.dataset.indices) == list(val2.dataset.indices)


def test_make_dataloaders_respects_ratios(tiny_npz):
    # velicine val/test setova odgovaraju val_ratio/test_ratio parametrima
    train_loader, val_loader, test_loader = make_dataloaders(
        tiny_npz, batch_size=4, val_ratio=0.2, test_ratio=0.1, seed=0,
    )
    assert len(val_loader.dataset) == int(20 * 0.2)
    assert len(test_loader.dataset) == int(20 * 0.1)
    assert len(train_loader.dataset) == 20 - int(20 * 0.2) - int(20 * 0.1)
