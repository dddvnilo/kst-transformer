"""
Testovi za metrike u src/util/metrics.py,
i njihovu medjusobnu konzistentnost.
"""

import numpy as np
import torch

from util.metrics import (
    compute_f1,
    compute_hamming,
    compute_pos_weight,
    metrics_np,
    metrics_lenient,
    metrics_closure,
    metrics_closure_pred,
    transitive_closure_matrix,
)
from util.generate_dataset_util import transitive_closure


def test_compute_f1_perfect_prediction():
    # predikcija identicna target-u -> F1 == 1.0
    target = torch.tensor([[0., 1.], [1., 0.]])
    pred = target * 20 - 10  # sigmoid(10) > 0.5, sigmoid(-10) < 0.5
    mask = torch.ones_like(target, dtype=torch.bool)
    assert compute_f1(pred, target, mask) == 1.0


def test_compute_f1_all_wrong():
    # sve celije pogresne -> F1 == 0.0
    target = torch.tensor([[0., 1.], [1., 0.]])
    pred = (1 - target) * 20 - 10  # inverzna, sigurna predikcija
    mask = torch.ones_like(target, dtype=torch.bool)
    assert compute_f1(pred, target, mask) == 0.0


def test_compute_f1_empty_mask_edge_case():
    # mask je svuda False (nema validnih celija) -> definisano ponasanje (trenutno vraca 1.0)
    target = torch.zeros(2, 2)
    pred = torch.zeros(2, 2)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    assert compute_f1(pred, target, mask) == 1.0


def test_compute_pos_weight_known_ratio():
    # na rucno kreiranom Y sa poznatim brojem 0/1 vrednosti, pos_weight
    # mora odgovarati tacnom odnosu negativnih/pozitivnih (total_neg / total_pos)
    max_items = 3
    Y = torch.zeros(1, max_items, max_items)
    Y[0, 0, 1] = 1.0  # jedna pozitivna van dijagonale, ostatak (5) su nule
    item_counts = torch.tensor([max_items])
    fake_loader = [(None, Y, item_counts)]

    pw = compute_pos_weight(fake_loader, max_items)
    assert pw == 5.0  # 5 nula / 1 jedinica


def test_metrics_np_empty_graph_returns_one():
    # metrics_np: kada je tp+fp+fn == 0 (prazan graf tacno predvidjen) -> F1 vraca 1.0
    n = 3
    pred_adj = np.zeros((n, n))
    true_adj = np.zeros((n, n))
    f1, hamming = metrics_np(pred_adj, true_adj, n)
    assert f1 == 1.0
    assert hamming == 0.0


def test_strict_f1_always_less_or_equal_lenient_f1():
    # za istu predikciju/target: lenient F1 (metrics_lenient) mora biti >= strict F1 (metrics_np)
    n = 3
    true_adj = np.zeros((n, n))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1

    pred_adj = np.zeros((n, n))
    pred_adj[0, 1] = 1
    pred_adj[1, 2] = 1
    pred_adj[0, 2] = 1  # tranzitivna veza, nije direktna Hasse ivica

    f1_strict, _ = metrics_np(pred_adj, true_adj, n)
    f1_lenient, _ = metrics_lenient(pred_adj, true_adj, n)

    assert f1_lenient >= f1_strict


def test_transitive_closure_matrix_agrees_with_transitive_closure_list():
    # eval_iita.transitive_closure_matrix i generate_dataset.transitive_closure su
    # dve nezavisne implementacije istog koncepta (matrica vs lista ivica) -
    # moraju se slagati na istim ulazima
    n = 4
    edges = [(0, 1), (1, 2), (2, 3)]

    closed_list = set(transitive_closure(edges, n))

    adj = np.zeros((n, n), dtype=bool)
    for (i, j) in edges:
        adj[i, j] = True
    closed_matrix = transitive_closure_matrix(adj)

    closed_from_matrix = {
        (i, j) for i in range(n) for j in range(n) if closed_matrix[i, j] and i != j
    }

    assert closed_from_matrix == closed_list


def test_metrics_lenient_fn_only_counts_direct_hasse_edges():
    # FN u lenient metrici sme da broji samo direktne (Hasse) ivice koje nisu
    # predvidjene, ne i tranzitivne
    n = 3
    true_adj = np.zeros((n, n))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1
    # true_adj[0, 2] namerno NIJE direktna Hasse ivica (izvodi se tranzitivno)

    pred_adj = np.zeros((n, n))  # ne predvidja nista

    f1, hamming = metrics_lenient(pred_adj, true_adj, n)

    # tp=0, fp=0, fn=2 (samo direktne (0,1) i (1,2)) -> f1 = 0
    assert f1 == 0.0
    assert hamming == 2 / 6


def test_metrics_closure_penalizes_missed_transitive_edges():
    # closure kaznjava promasenu tranzitivnu vezu (FN), za razliku od lenient
    n = 3
    true_adj = np.zeros((n, n))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1
    # zatvorenje sadrzi i (0,2)

    pred_adj = np.zeros((n, n))
    pred_adj[0, 1] = 1
    pred_adj[1, 2] = 1
    # (0,2) NIJE predvidjeno

    f1_c, hamming_c = metrics_closure(pred_adj, true_adj, n)
    f1_l, _         = metrics_lenient(pred_adj, true_adj, n)

    # closure: tp=2, fp=0, fn=1 (promasena tranzitivna (0,2)) -> f1 = 4/5
    assert f1_c == 4 / 5
    assert hamming_c == 1 / 6
    # lenient ne kaznjava tu tranzitivnu vezu -> strogo popustljivije
    assert f1_l > f1_c


def test_metrics_closure_perfect_when_prediction_is_full_closure():
    # ako predikcija sadrzi celo zatvorenje, closure F1 == 1.0
    n = 3
    true_adj = np.zeros((n, n))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1

    pred_adj = transitive_closure_matrix(true_adj).astype(np.float32)

    f1_c, hamming_c = metrics_closure(pred_adj, true_adj, n)
    assert f1_c == 1.0
    assert hamming_c == 0.0


def test_metrics_closure_pred_closes_prediction_before_compare():
    # closure_pred zatvara predikciju pre poredjenja: minimalna (Hasse) predikcija
    # koja je tacna dobija F1 = 1.0, dok je obican closure kaznjava za (0,2)
    n = 3
    true_adj = np.zeros((n, n))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1

    pred_adj = np.zeros((n, n))
    pred_adj[0, 1] = 1
    pred_adj[1, 2] = 1
    # (0,2) NIJE predvidjeno - ali se izvodi zatvaranjem predikcije

    f1_cp, hamming_cp = metrics_closure_pred(pred_adj, true_adj, n)
    f1_c,  _          = metrics_closure(pred_adj, true_adj, n)

    # zatvaranjem predikcije (0,2) postaje TP -> savrseno poklapanje
    assert f1_cp == 1.0
    assert hamming_cp == 0.0
    # bez zatvaranja predikcije, isti ulaz je strogo losije ocenjen
    assert f1_c < f1_cp
