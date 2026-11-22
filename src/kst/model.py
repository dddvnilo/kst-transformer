from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def mha_fastpath_disabled():
    """
    PyTorch "fast path" za TransformerEncoderLayer (eval rezim bez gradijenata) daje NaN kada se
    kao `mask` prosledi aditivna float maska po glavi (provereno na torch 2.6), a sporija putanja
    radi ispravno. Iskljucuje se samo za vreme bloka, pa se prethodno stanje vraca.
    """
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


# Podrzane vrste pozicionog kodiranja. Nova vrsta (npr. sinusno) se dodaje ovde i u
# odgovarajuce grane u __init__ i forward.
POS_ENCODINGS = ("learned", "none")

# Podrzani ulazni enkoderi - nacin na koji se od odgovora studenata formira token pitanja:
#   "linear": legacy - kolona odgovora svih studenata ide kroz Linear(students, d_model).
#             Broj studenata je fiksiran arhitekturom, a redosled studenata nosi znacenje.
#   "set":    studenti su skup - tokeni i statistike parova pitanja su proseci preko studenata,
#             pa model prima proizvoljan broj studenata i invarijantan je na njihov redosled.
INPUT_ENCODERS = ("linear", "set")


class KSTTransformer(nn.Module):
    """
    Input:  (batch, students, items)  - binarna response matrica
    Output: (batch, items, items)     - adj matrica prerequisita (logiti, bez sigmoida)
    """

    def __init__(
        self,
        max_items: int,
        students: int | None = None,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
        input_encoder: str = "linear",
    ):
        """
        :param max_items:          maksimalan broj pitanja (= veličina adj matrice); ogranicava
                                   ulaz samo uz pos_encoding="learned"
        :param students:           broj studenata (= dimenzija sirovog input tokena); potreban
                                   samo za legacy input_encoder="linear"
        :param d_model:            dimenzija internog prostora transformera
        :param nhead:              broj attention glava
        :param num_encoder_layers: broj encoder slojeva
        :param dim_feedforward:    dimenzija feedforward sloja unutar encodera
        :param dropout:            dropout stopa
        :param pos_encoding:       vrsta pozicionog kodiranja, jedna od POS_ENCODINGS
        :param input_encoder:      ulazni enkoder, jedan od INPUT_ENCODERS
        """
        super().__init__()

        if pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding mora biti jedno od {POS_ENCODINGS}, dobijeno: {pos_encoding!r}")
        if input_encoder not in INPUT_ENCODERS:
            raise ValueError(f"input_encoder mora biti jedno od {INPUT_ENCODERS}, dobijeno: {input_encoder!r}")
        if input_encoder == "linear" and students is None:
            raise ValueError("legacy input_encoder='linear' zahteva broj studenata (students)")

        self.max_items = max_items
        self.d_model = d_model
        self.nhead = nhead
        self.pos_encoding = pos_encoding
        self.input_encoder = input_encoder

        # ------------------------------------------------------------------
        # 1. Ulazni enkoder - formiranje tokena pitanja
        # ------------------------------------------------------------------
        if input_encoder == "linear":
            # Legacy: (students,) -> (d_model,)
            # Mapira response vektor odgovora na pitanje sa testa iz prostora broja studenata
            # svakog pitanja u d_model prostor. Svaki student ima svoju tezinu, pa je broj
            # studenata fiksiran, a redosled studenata nosi znacenje.
            self.input_projection = nn.Linear(students, d_model)
        else:
            # "set": embedding odgovora 0/1; token pitanja je prosek embedding-a odgovora
            # svih studenata na to pitanje.
            self.answer_embedding = nn.Embedding(2, d_model)
            # Po glavi naucena 2x2 matrica c[h, a, b] - koliko vredi kombinacija odgovora
            # (a na pitanje i, b na pitanje j) jednog studenta. Matrica parova je
            # S[h, i, j] = sum_ab P_ij(a, b) * c[h, a, b] (tied row attention logit u prvom sloju).
            # Nasumicna inicijalizacija: sa nulama bi S na pocetku bio 0 i parovi se ne bi
            # razlikovali.
            self.pair_coef = nn.Parameter(torch.randn(nhead, 2, 2))

        # ------------------------------------------------------------------
        # 2. Positional Encoding
        #    "learned": naucen embedding po poziciji - ogranicava broj pitanja na max_items.
        #    "none":    bez pozicione informacije. Redosled pitanja u testu ne nosi informaciju
        #               (ni IITA ga ne koristi), pa je model tada permutaciono ekvivarijantan po
        #               pitanjima i prima proizvoljan broj pitanja.
        # ------------------------------------------------------------------
        if pos_encoding == "learned":
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
            enable_nested_tensor=False,
        )

        # ------------------------------------------------------------------
        # 4. Glava: MLP za binarnu klasifikaciju po parovima
        #    Prima konkatenirane vektore pitanja i i j (uz "set" i matricu parova S_ij za
        #    svaku glavu), vraca verovatnocu da j zahteva i kao prerequisit.
        # ------------------------------------------------------------------
        pair_features = 2 * d_model + (nhead if input_encoder == "set" else 0)
        self.pair_classifier = nn.Sequential(
            nn.Linear(pair_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode_set(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        "set" ulazni enkoder. Sve velicine su proseci preko studenata, pa ne zavise ni od broja
        ni od redosleda studenata.

        :param x: (batch, students, items) - binarna response matrica
        :return:  tokeni pitanja (batch, items, d_model) i matrica parova S (batch, nhead, items, items)
        """
        students = x.shape[1]

        # One-hot odgovora: (batch, students, items, 2), poslednja osa je odgovor a iz {0, 1}
        answers = F.one_hot(x.long(), num_classes=2).to(x.dtype)

        # Udeo studenata sa odgovorom a na pitanje i: P_i(a), oblik (batch, items, 2)
        item_tables = answers.mean(dim=1)

        # Token pitanja: t_i = sum_a P_i(a) * E[a] -> (batch, items, d_model)
        tokens = item_tables @ self.answer_embedding.weight

        # 2x2 tabela po paru pitanja - udeo studenata sa odgovorom a na i i b na j:
        # P_ij(a, b), oblik (batch, items, items, 2, 2). Isti student s za oba pitanja cuva vezu
        # medju odgovorima istog studenta.
        pair_tables = torch.einsum("bsia,bsjc->bijac", answers, answers) / students

        # Matrica parova po glavi: S[h, i, j] = sum_ab P_ij(a, b) * c[h, a, b]
        pair_scores = torch.einsum("bijac,hac->bhij", pair_tables, self.pair_coef)

        return tokens, pair_scores

    def forward(self, x: torch.Tensor, item_counts: torch.Tensor = None) -> torch.Tensor:
        """
        :param x:           (batch, students, items) - response matrica
        :param item_counts: (batch,) int64 - stvarni broj pitanja; padding pozicije se maskiraju u attention-u
        :return:            (batch, items, items)    - logiti (primeni sigmoid za verovatnoce)
        """
        batch_size, students, items = x.shape

        pair_scores = None
        if self.input_encoder == "linear":
            # Legacy: (batch, students, items) -> (batch, items, students)
            # Svako pitanje je sada vektor duzine `students` -> pitanja su ulazni tokeni
            # Projection: (batch, items, students) -> (batch, items, d_model)
            tokens = self.input_projection(x.permute(0, 2, 1))
        else:
            tokens, pair_scores = self.encode_set(x)

        # Positional encoding: dodaj pozicijsku informaciju za svaki token
        if self.pos_encoding == "learned":
            positions = torch.arange(items, device=x.device)               # (items,)
            tokens = tokens + self.positional_encoding(positions)           # broadcast po batchu

        # Padding maska za attention: True = ignorisi ovu poziciju
        padding_mask = None
        if item_counts is not None:
            idx = torch.arange(items, device=x.device)
            padding_mask = idx.unsqueeze(0) >= item_counts.unsqueeze(1)  # (batch, items)

        # Transformer Encoder: (batch, items, d_model) -> (batch, items, d_model)
        if pair_scores is None:
            h = self.encoder(tokens, src_key_padding_mask=padding_mask)
        else:
            # Matrica parova se dodaje na attention logite svake glave. Padding pitanja ide u
            # istu float masku kao -inf (kljuc j se ignorise), da maske ne bi bile razlicitog tipa.
            attn_bias = pair_scores
            if padding_mask is not None:
                attn_bias = attn_bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
            # (batch, nhead, items, items) -> (batch * nhead, items, items), redosled koji ocekuje nn.MultiheadAttention
            attn_bias = attn_bias.reshape(batch_size * self.nhead, items, items)
            with mha_fastpath_disabled():
                h = self.encoder(tokens, mask=attn_bias)

        # Glava - klasifikacija po parovima
        # Za svaki par (i, j) konkateniramo h_i i h_j
        h_i = h.unsqueeze(2).expand(-1, -1, items, -1)  # (batch, items, items, d_model)
        h_j = h.unsqueeze(1).expand(-1, items, -1, -1)  # (batch, items, items, d_model)

        pairs = torch.cat([h_i, h_j], dim=-1)           # (batch, items, items, 2*d_model)
        if pair_scores is not None:
            # glava direktno vidi i S_ij za svaku glavu -> (batch, items, items, 2*d_model + nhead)
            pairs = torch.cat([pairs, pair_scores.permute(0, 2, 3, 1)], dim=-1)

        # (batch, items, items, pair_features) -> (batch, items, items)
        return self.pair_classifier(pairs).squeeze(-1)   # logiti (bez originalnog sigmoida, numericka stabilnost pri racunanju loss-a)


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
