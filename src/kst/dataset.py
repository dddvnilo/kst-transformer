import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class KSTDataset(Dataset):
    """
    Input:  putanja do .npz fajla generisanog sa scripts/generate_dataset.py
    Output: (X, Y, item_count) po sample-u
        X:          (students, max_items) float32 - binarna response matrica
        Y:          (max_items, max_items) float32 - adj matrica prerequisita
        item_count: int64 skalar - stvarni broj pitanja (ostalo je padding od nula)
    """

    def __init__(self, path: str):
        data = np.load(path)
        self.X = torch.from_numpy(data["X"]).float()                  # (N, students, max_items)
        self.Y = torch.from_numpy(data["Y"]).float()                  # (N, max_items, max_items)
        self.item_counts = torch.from_numpy(data["item_counts"]).long() # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.item_counts[idx]


def make_loss_mask(item_counts: torch.Tensor, max_items: int) -> torch.Tensor:
    """
    Pravi bool masku za BCELoss — True tamo gde treba racunati loss.

    Maskiramo:
      - padding celije (i >= item_count ili j >= item_count); ovo su nule u adjacency matrici
      - dijagonalu (i == j, nema self-prerequisita)

    :param item_counts: (batch,) int64 - stvarni broj pitanja po primeru
    :param max_items:   int - dimenzija paddovane matrice
    :return:            (batch, max_items, max_items) bool
    """
    idx = torch.arange(max_items, device=item_counts.device)

    # valid[b, i] = True ako je pitanje i stvarno (nije padding)
    valid = idx.unsqueeze(0) < item_counts.unsqueeze(1)              # (batch, max_items)

    # valid_pair[b, i, j] = True ako su oba pitanja stvarna
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)             # (batch, max_items, max_items)

    # ukloni dijagonalu
    diag_mask = ~torch.eye(max_items, dtype=torch.bool, device=item_counts.device).unsqueeze(0)

    return valid_pair & diag_mask                                     # (batch, max_items, max_items)


def make_dataloaders(
    path: str,
    batch_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Ucitava dataset i deli ga na train/val/test DataLoadere.

    :return: (train_loader, val_loader, test_loader)
    """
    dataset = KSTDataset(path)
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    '''
    Main je cisto quick check da sve radi
    '''
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kst_dataset.npz")

    train_loader, val_loader, test_loader = make_dataloaders(path, batch_size=32)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    X, Y, item_counts = next(iter(train_loader))
    print(f"\nX shape:           {X.shape}")
    print(f"Y shape:           {Y.shape}")
    print(f"item_counts shape: {item_counts.shape}")
    print(f"item_counts[:5]:   {item_counts[:5].tolist()}")

    mask = make_loss_mask(item_counts, max_items=X.shape[-1])
    print(f"\nMask shape:        {mask.shape}")
    print(f"Mask primer [0]:\n{mask[0].int()}")
