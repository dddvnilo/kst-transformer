import torch
import torch.nn as nn


class KSTTransformer(nn.Module):
    """
    Input:  (batch, students, items)  - binarna response matrica
    Output: (batch, items, items)     - adj matrica prerequisita (logiti, bez sigmoida)
    """

    def __init__(
        self,
        max_items: int,
        students: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        :param max_items:          maksimalan broj pitanja (= veličina adj matrice)
        :param students:           broj studenata (= dimenzija sirovog input tokena)
        :param d_model:            dimenzija internog prostora transformera
        :param nhead:              broj attention glava
        :param num_encoder_layers: broj encoder slojeva
        :param dim_feedforward:    dimenzija feedforward sloja unutar encodera
        :param dropout:            dropout stopa
        """
        super().__init__()

        self.max_items = max_items
        self.d_model = d_model

        # ------------------------------------------------------------------
        # 1. Projection: (students,) -> (d_model,)
        #    Mapira sirovi response vektor iz prostora broja studenata
        #    svakog pitanja u d_model prostor.
        # ------------------------------------------------------------------
        self.input_projection = nn.Linear(students, d_model)

        # ------------------------------------------------------------------
        # 2. Positional Encoding
        #    Naucen jer redosled pitanja nije prirodan poredak kao u jeziku za NLP.
        # ------------------------------------------------------------------
        self.positional_encoding = nn.Embedding(max_items, d_model)

        # ------------------------------------------------------------------
        # 3. Transformer Encoder
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (batch, seq, d_model) - prirodniji format
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ------------------------------------------------------------------
        # 4. Glava: MLP za binarnu klasifikaciju po parovima
        #    Prima konkatenirane vektore pitanja i i j,
        #    vraca verovatnocu da j zahteva i kao prerequisit.
        # ------------------------------------------------------------------
        self.pair_classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, students, items) - response matrica
        :return:  (batch, items, items)    - logiti (primeni sigmoid za verovatnoce)
        """
        batch_size, students, items = x.shape

        # (batch, students, items) -> (batch, items, students)
        # Svako pitanje je sada vektor duzine `students`
        x = x.permute(0, 2, 1)

        # Projection: (batch, items, students) -> (batch, items, d_model)
        x = self.input_projection(x)

        # Positional encoding: dodaj pozicijsku informaciju za svaki token
        positions = torch.arange(items, device=x.device)          # (items,)
        x = x + self.positional_encoding(positions)                # broadcast po batchu

        # Transformer Encoder: (batch, items, d_model) -> (batch, items, d_model)
        h = self.encoder(x)

        # Glava - klasifikacija po parovima
        # Za svaki par (i, j) konkateniramo h_i i h_j
        h_i = h.unsqueeze(2).expand(-1, -1, items, -1)  # (batch, items, items, d_model)
        h_j = h.unsqueeze(1).expand(-1, items, -1, -1)  # (batch, items, items, d_model)

        pairs = torch.cat([h_i, h_j], dim=-1)           # (batch, items, items, 2*d_model)

        # (batch, items, items, 2*d_model) -> (batch, items, items)
        return self.pair_classifier(pairs).squeeze(-1)   # logiti (bez originalnog sigmoida)


if __name__ == "__main__":
    # Sanity check
    BATCH    = 8
    STUDENTS = 200
    ITEMS    = 5

    model = KSTTransformer(
        max_items=ITEMS,
        students=STUDENTS,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    )

    x = torch.randint(0, 2, (BATCH, STUDENTS, ITEMS)).float()
    out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (BATCH, ITEMS, ITEMS), "Pogresan oblik outputa."

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Ukupno parametara: {total_params:,}")
