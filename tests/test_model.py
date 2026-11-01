"""
Testovi za KSTTransformer u src/kst/model.py
"""

import pytest
import torch

from kst.model import KSTTransformer


STUDENTS = 20


def make_model(pos_encoding, max_items=6):
    torch.manual_seed(0)
    model = KSTTransformer(
        max_items=max_items,
        students=STUDENTS,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        pos_encoding=pos_encoding,
    )
    return model.eval()


def permute_output(out, perm):
    # izlaz za permutovan ulaz treba da bude out[perm[a], perm[b]] na mestu (a, b)
    return out[:, perm][:, :, perm]


def test_no_pos_encoding_is_permutation_equivariant():
    # bez pozicionog kodiranja permutacija pitanja na ulazu samo permutuje izlaz
    model = make_model("none")
    torch.manual_seed(1)
    x = torch.randint(0, 2, (3, STUDENTS, 6)).float()
    perm = torch.randperm(6)

    with torch.no_grad():
        out = model(x)
        out_perm = model(x[:, :, perm])

    torch.testing.assert_close(out_perm, permute_output(out, perm), atol=1e-5, rtol=1e-5)


def test_no_pos_encoding_is_permutation_equivariant_with_padding():
    # permutuju se samo validna pitanja; padding ostaje na kraju, kao u datasetu
    model = make_model("none")
    torch.manual_seed(1)
    x = torch.randint(0, 2, (3, STUDENTS, 6)).float()
    x[:, :, 4:] = 0
    item_counts = torch.tensor([4, 4, 4])
    perm = torch.cat([torch.randperm(4), torch.arange(4, 6)])

    with torch.no_grad():
        out = model(x, item_counts)
        out_perm = model(x[:, :, perm], item_counts)

    torch.testing.assert_close(
        out_perm[:, :4, :4], permute_output(out, perm)[:, :4, :4], atol=1e-5, rtol=1e-5
    )


def test_learned_pos_encoding_is_not_permutation_equivariant():
    # kontrast: nauceno poziciono kodiranje cini izlaz zavisnim od redosleda pitanja
    model = make_model("learned")
    torch.manual_seed(1)
    x = torch.randint(0, 2, (3, STUDENTS, 6)).float()
    perm = torch.tensor([1, 0, 2, 3, 4, 5])

    with torch.no_grad():
        out = model(x)
        out_perm = model(x[:, :, perm])

    assert not torch.allclose(out_perm, permute_output(out, perm), atol=1e-5, rtol=1e-5)


def test_no_pos_encoding_accepts_more_items_than_max_items():
    model = make_model("none", max_items=6)
    x = torch.randint(0, 2, (2, STUDENTS, 9)).float()

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 9, 9)


def test_state_dict_keys_follow_pos_encoding():
    # checkpointi sa diplomskog imaju positional_encoding.weight, pa ga "learned" mora zadrzati
    assert "positional_encoding.weight" in make_model("learned").state_dict()
    assert not any(k.startswith("positional_encoding") for k in make_model("none").state_dict())


def test_unknown_pos_encoding_raises():
    with pytest.raises(ValueError):
        make_model("sinusoidal")
