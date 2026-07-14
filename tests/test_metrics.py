"""
Testovi za metrike u scripts/train.py i scripts/eval_iita.py,
i njihovu medjusobnu konzistentnost.
"""

from train import compute_f1, compute_hamming, compute_pos_weight
from eval_iita import metrics_np, metrics_lenient, transitive_closure_matrix
from generate_dataset import transitive_closure


def test_compute_f1_perfect_prediction():
    # predikcija identicna target-u -> F1 == 1.0
    pass


def test_compute_f1_all_wrong():
    # sve celije pogresne -> F1 == 0.0
    pass


def test_compute_f1_empty_mask_edge_case():
    # mask je svuda False -> definisano ponasanje (trenutno vraca 0.0)
    pass


def test_compute_pos_weight_known_ratio():
    # na rucno kreiranom Y sa poznatim brojem 0/1 vrednosti, pos_weight
    # mora odgovarati tacnom odnosu negativnih/pozitivnih (total_neg / total_pos)
    pass


def test_metrics_np_empty_graph_returns_one():
    # metrics_np: kada je tp+fp+fn == 0 (prazan graf tacno predvidjen) -> F1 vraca 1.0
    pass


def test_compute_f1_vs_metrics_np_disagree_on_empty_graph():
    # POZNATA NEKONZISTENTNOST: train.py.compute_f1 vraca 0.0, a
    # eval_iita.py.metrics_np vraca 1.0 za isti rubni slucaj (tp+fp+fn == 0).
    # TODO: odluciti koji pristup je ispravan
    pass


def test_strict_f1_always_less_or_equal_lenient_f1():
    # za istu predikciju/target: lenient F1 (metrics_lenient) mora biti >= strict F1 (metrics_np)
    pass


def test_transitive_closure_matrix_agrees_with_transitive_closure_list():
    # eval_iita.transitive_closure_matrix i generate_dataset.transitive_closure su
    # dve nezavisne implementacije istog koncepta (matrica vs lista ivica) -
    # moraju se slagati na istim ulazima
    pass


def test_metrics_lenient_fn_only_counts_direct_hasse_edges():
    # FN u lenient metrici sme da broji samo direktne (Hasse) ivice koje nisu
    # predvidjene, ne i tranzitivne
    pass
