import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from learning_spaces.kst import simu, hasse
from kst.model import KSTTransformer

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPTS_DIR, "..")


def parse_args():
    parser = argparse.ArgumentParser(description="KST Transformer demo")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_ROOT_DIR, "model", "best.pt"))
    parser.add_argument("--ce",         type=float, default=0.1,  help="Careless error stopa")
    parser.add_argument("--lg",         type=float, default=0.1, help="Lucky guess stopa")
    parser.add_argument("--threshold",  type=float, default=0.5,  help="Threshold za sigmoid")
    return parser.parse_args()

TRUE_IMP = [
    (1, 2),   # c ne moze bez b
    (0, 4),   # e ne moze bez a
    (2, 4),   # e ne moze bez c
    (3, 4),   # e ne moze bez d
]
N_ITEMS = 5


def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    state      = checkpoint["model_state"]

    # Ucitavamo dimenzije sa broj studenata i broj pitanja da bi simulirali unos bez hardkodovanih vrednosti
    students  = state["input_projection.weight"].shape[1]
    max_items = state["positional_encoding.weight"].shape[0]

    model = KSTTransformer(
        max_items=max_items,
        students=students,
        d_model=saved_args.get("d_model"),
        nhead=saved_args.get("nhead"),
        num_encoder_layers=saved_args.get("num_layers"),
        dim_feedforward=saved_args.get("dim_feedforward"),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    return model, students, max_items, checkpoint


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, students, max_items, checkpoint = load_model(args.checkpoint, device)

    assert N_ITEMS <= max_items, (
        f"N_ITEMS={N_ITEMS} je vece od max_items={max_items} na kojima je model treniran."
    )

    print(f"{'='*55}")
    print(f"  KST Transformer Demo")
    print(f"{'='*55}")
    print(f"  Checkpoint: epoha {checkpoint.get('epoch','?')} | "
          f"val_loss={checkpoint.get('val_loss',0):.4f} | "
          f"val_F1={checkpoint.get('val_f1',0):.3f}")
    print(f"  students={students}, max_items={max_items}, N_ITEMS={N_ITEMS}")
    print(f"{'='*55}\n")

    print(f"Prave implikacije: {sorted(TRUE_IMP)}")
    print(f"Simuliram odgovore ({students} studenata, {N_ITEMS} pitanja)...\n")

    # Simulacija odgovora studenata
    sim = simu(
        items=N_ITEMS,
        size=students,
        ce=args.ce,
        lg=args.lg,
        delta=0.5,
        imp=TRUE_IMP,
    )

    # Odgovori studenata iz simulacije
    student_responses = np.array(sim["dataset"], dtype=np.float32)  # (students, N_ITEMS)

    # Priprema ulaza - tenzor odgovora studenata sa paddovanim nulama za pitanja koja ne postoje
    X_padded = np.zeros((students, max_items), dtype=np.float32)
    X_padded[:, :N_ITEMS] = student_responses

    X_tensor   = torch.from_numpy(X_padded).unsqueeze(0).to(device)  # (1, students, max_items)
    item_count = torch.tensor([N_ITEMS], dtype=torch.long, device=device)

    # Predikcija modela
    with torch.no_grad():
        logits = model(X_tensor, item_count)   # (1, max_items, max_items)

    # Logits u verovatnoce
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (max_items, max_items)

    # Adjacency matrica relacija izmedju pitanja sa vrednostima True i False
    adj = (probs[:N_ITEMS, :N_ITEMS] > args.threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    # Implikacije iz adjacency matrice
    pred_imp = {(i, j) for i in range(N_ITEMS) for j in range(N_ITEMS) if adj[i, j] > 0}
    true_set = set(map(tuple, TRUE_IMP))

    # Prikaz verovatnoca
    print("Matrica verovatnoca (i -> j znaci j zahteva i kao prerequisit):")
    header = "      " + "  ".join(f"  {j}" for j in range(N_ITEMS))
    print(header)
    for i in range(N_ITEMS):
        row = f"  {i}  |"
        for j in range(N_ITEMS):
            if i == j:
                row += "   - "
            else:
                row += f" {probs[i,j]:.2f}"
        print(row)

    # Rezultati
    total_cells = N_ITEMS * (N_ITEMS - 1)  # bez dijagonale
    wrong = len(pred_imp.symmetric_difference(true_set))
    hamming = wrong / total_cells if total_cells > 0 else 0.0

    tp = pred_imp & true_set
    fp = pred_imp - true_set
    fn = true_set - pred_imp

    print(f"\n{'='*55}")
    print(f"  Prave implikacije:         {sorted(true_set)}")
    print(f"  Predvidjene:               {sorted(pred_imp)}")
    print(f"  Tacno predvidjene (TP):    {sorted(tp)}")
    print(f"  Pogresno predvidjene (FP): {sorted(fp)}")
    print(f"  Promasene (FN):            {sorted(fn)}")
    print(f"{'='*55}")
    print(f"  Hamming loss: {hamming:.3f}  ({wrong}/{total_cells} pogresnih predikcija)")
    print(f"{'='*55}\n")

    # Hasse dijagram
    if pred_imp:
        print("Hasse dijagram predvidjenih implikacija:")
        try:
            hasse_out = hasse(pred_imp, N_ITEMS)
            print(hasse_out)
        except PermissionError:
            pass  # Windows bug: hasse() ne moze da obrize temp fajl pre stampanja
    else:
        print("Model nije predvideo nijednu implikaciju.")


if __name__ == "__main__":
    main()
