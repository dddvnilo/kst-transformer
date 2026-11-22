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


# ----------------------------------------------------------------------
# "set" ulazni enkoder - studenti kao skup
# ----------------------------------------------------------------------

def make_set_model(max_items=6):
    torch.manual_seed(0)
    model = KSTTransformer(
        max_items=max_items,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        pos_encoding="none",
        input_encoder="set",
    )
    return model.eval()


def test_set_encoder_is_invariant_to_student_permutation():
    model = make_set_model()
    torch.manual_seed(1)
    x = torch.randint(0, 2, (3, STUDENTS, 6)).float()
    perm = torch.randperm(STUDENTS)

    with torch.no_grad():
        out = model(x)
        out_perm = model(x[:, perm, :])

    torch.testing.assert_close(out_perm, out, atol=1e-5, rtol=1e-5)


def test_set_encoder_is_equivariant_to_question_permutation_with_padding():
    model = make_set_model()
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


def test_set_encoder_accepts_any_number_of_students():
    # isti test sa svakim studentom dupliranim: udeli su isti, pa i izlaz mora biti isti
    model = make_set_model()
    torch.manual_seed(1)
    x = torch.randint(0, 2, (2, STUDENTS, 6)).float()

    with torch.no_grad():
        out = model(x)
        out_doubled = model(torch.cat([x, x], dim=1))
        out_other = model(torch.randint(0, 2, (2, 7, 6)).float())

    torch.testing.assert_close(out_doubled, out, atol=1e-5, rtol=1e-5)
    assert out_other.shape == (2, 6, 6)


def test_set_encoder_padding_does_not_change_output():
    model = make_set_model()
    torch.manual_seed(1)
    x = torch.randint(0, 2, (2, STUDENTS, 4)).float()
    x_padded = torch.cat([x, torch.zeros(2, STUDENTS, 2)], dim=2)
    item_counts = torch.tensor([4, 4])

    with torch.no_grad():
        out = model(x)
        out_padded = model(x_padded, item_counts)

    torch.testing.assert_close(out_padded[:, :4, :4], out, atol=1e-5, rtol=1e-5)


def test_set_encoder_distinguishes_pairs_with_identical_item_statistics():
    # Polovina studenata resila A i B, polovina C i D: A ide uz B, a ne uz C. Sva pitanja imaju
    # istu tezinu i svi koji su ih resili imaju isti zbir, pa su tokeni pitanja identicni -
    # razliku medju parovima nosi samo matrica parova S. (Dizajn sa row attention-om nad
    # anonimnim pitanjima bi ovde za (A, B) i (A, C) morao dati isti odgovor.)
    model = make_set_model(max_items=4)
    x = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]] * 5).float().unsqueeze(0)  # (1, 10, 4)
    a, b, c = 0, 1, 2

    with torch.no_grad():
        tokens, _ = model.encode_set(x)
        out = model(x)

    torch.testing.assert_close(tokens[0, a], tokens[0, c])
    assert not torch.isclose(out[0, a, b], out[0, a, c], atol=1e-4)


def test_state_dict_keys_follow_input_encoder():
    # legacy checkpointi imaju input_projection.weight; "set" nema projekciju po studentima
    assert "input_projection.weight" in make_model("none").state_dict()
    set_keys = make_set_model().state_dict()
    assert not any(k.startswith("input_projection") for k in set_keys)
    assert "answer_embedding.weight" in set_keys and "pair_coef" in set_keys


def test_linear_input_encoder_requires_students():
    with pytest.raises(ValueError):
        KSTTransformer(max_items=6, input_encoder="linear")


def test_unknown_input_encoder_raises():
    with pytest.raises(ValueError):
        KSTTransformer(max_items=6, students=STUDENTS, input_encoder="axial")
