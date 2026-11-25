# Dnevnik rada: promenljiv broj studenata i složeniji dataset

Projekat: KST Transformer. Model iz matrice odgovora studenata (ko je rešio koje pitanje)
predviđa relaciju preduslova među pitanjima (surmise relaciju, tj. kvazi-poredak), isto što radi
IITA. Ovaj dnevnik opisuje dva zadatka iz proširenja na masteru (novembar 2026):

1. **Uklanjanje ograničenja broja studenata.** Završeno, spojeno u `main`.
2. **Složeniji dataset.** Generator je urađen na grani `feature/complex-dataset`; generisanje
   punog dataseta i treninzi su sledeći korak.

---

## Deo 1: Uklanjanje ograničenja broja studenata

### 1.1 Polazni problem

Model sa diplomskog token pitanja $i$ pravi od kolone odgovora svih studenata,
$x_i \in \{0,1\}^{500}$, preko `Linear(500, d)`:

$$t_i = \sum_{s=1}^{500} x_{si}\, W_s + b$$

Svaki student ima svoju težinu $W_s$. Iz toga slede dva problema:

- **Broj studenata je fiksiran arhitekturom.** Test sa 300 ili 2000 studenata ne može da se
  obradi bez ponovnog treninga.
- **Redosled studenata nosi značenje.** Permutacija studenata menja ulaz, iako je test isti.

### 1.2 Okvir: ulaz su dva skupa

Matrica odgovora je zapis dva skupa: studenti (redovi) i pitanja (kolone). Redosled ni jednih ni
drugih ne nosi informaciju.

- **Po pitanjima** model treba da bude **ekvivarijantan**: permutacija pitanja samo permutuje
  izlaz. To je urađeno ranije izbacivanjem pozicionog kodiranja.
- **Po studentima** model treba da bude **invarijantan**: permutacija studenata ne menja izlaz.

Polazna ideja je DeepSets (Zaheer et al., 2017): funkcija nad skupom ima oblik
$f(X) = \rho\big(\mathrm{agg}_{x \in X}\, \varphi(x)\big)$. Kad je agregacija prosek, rešava i broj
i redosled studenata istom operacijom. Prosek je bolji od zbira, jer zbir raste sa $N$.


### 1.3 `set` enkoder

Ključni uvid: vezu između pitanja treba računati **direktno za svaki par**, preko studenata. Tada
identitet para dolazi iz indeksa matrice $(i, j)$, a ne iz sadržaja tokena.

**2×2 tabela po paru.** Za svaki par: udeo studenata sa odgovorom $a$ na pitanje $i$ i odgovorom
$b$ na pitanje $j$:

$$P_{ij}(a,b) = \frac{1}{N}\,\\#\{\, s : x_{si} = a,\ x_{sj} = b \,\}, \qquad a, b \in \{0, 1\}$$

**Matrica parova po attention glavi** $h$:

$$S^{(h)}_{ij} = \sum_{a,b \in \{0,1\}} P_{ij}(a,b)\; c^{(h)}_{ab}$$

$c^{(h)}$ je naučena 2×2 matrica: koliko „vredi" svaka kombinacija odgovora. Primer: glava koja
nauči $c_{01} = -1$, a ostalo 0, računa $S_{ij} = -P_{ij}(0,1)$, odnosno stopu IITA kontraprimera.

**Tokeni pitanja** su prosek embedding-a odgovora, u suštini težina pitanja:

$$t_i = \sum_{a} P_i(a)\, E[a]$$

**$S$ ide eksplicitno dalje, na dva mesta:**

- kao **bias na attention logite** postojećeg encodera:
  $\mathrm{softmax}_j\!\big(q_i^\top k_j / \sqrt{d_k} + S_{ij}\big)$;
- kao **ulaz u glavu nad parovima**: $[\,h_i,\ h_j,\ S_{ij}\,]$.

Oba mesta su potrebna. U simetričnim slučajevima (test X) svi tokeni su isti, pa samo
preraspoređivanje pažnje ne pomaže. Razliku tada nosi $S_{ij}$ u glavi.

**Veza sa literaturom.**

- **MSA Transformer** (Rao et al., 2021), provereno u bioRxiv preprintu. Ulaz je matrica sekvenci
  i pozicija, što odgovara našim studentima i pitanjima. *Tied row attention* sabira attention
  logite za par pozicija preko svih sekvenci, i „information about the contact pattern emerges
  directly in the tied row attention maps". Kada ćelija zna samo svoj bit,
  $c_{ab} = q(a)^\top k(b)/\sqrt{d_k}$, pa je naš $S$ tačno taj logit u prvom sloju.
- **AlphaFold2 Evoformer**, iz memorije, nije provereno. Operacija *outer product mean* i
  *pair bias* u attention-u su ista ideja kao naše $P_{ij}$ i bias.

### 1.5 Implementacija

- **`src/kst/model.py`**: izbor `input_encoder` sa vrednostima `"linear"` (legacy, nepromenjen;
  stari checkpointi se učitavaju) i `"set"`. Encoder i glava su isti, osim dodatka $S$.
  - $P_{ij}$ se računa preko one-hot + `einsum`, direktno iz $X$. Nema tenzora (B, N, n, d), pa je
    trening skoro iste brzine kao legacy (oko 31 min za 200 epoha).
  - Ulazni deo ima $2 \cdot 256 + 8 \cdot 4 = 544$ parametra, naspram $500 \cdot 256 = 128\,000$
    kod legacy modela.
  - **Bug u PyTorch 2.6:** „fast path" za `TransformerEncoderLayer` (eval režim bez gradijenata)
    vraća NaN za aditivnu float masku koja nije sva nula. Provereno eksperimentom; slično je
    prijavljeno u pytorch issues #161500 i #107084. Fast path se isključuje samo oko poziva
    encodera u `set` grani.
- **Testovi** (`tests/test_model.py`):
  - invarijantnost na permutaciju studenata;
  - ekvivarijantnost po pitanjima uz padding;
  - dupliranje studenata ne menja izlaz;
  - padding ne menja izlaz;
  - kontraprimer X: tokeni A i C su isti, a izlazi za (A, B) i (A, C) se razlikuju;
  - validacija argumenata.
- **Usput:**
  - `train.py` i `random_search.py` imaju `--input-encoder`;
  - `eval_iita.py` je sveden na jedan režim, poređenje sa kvazi-poretkom (punim zatvorenjem);
  - `demo.py` ima `--students` za `set` model.

### 1.6 Rezultati

Dataset sa diplomskog: 80k uzoraka, 2–10 pitanja, 500 studenata, cilj je puno tranzitivno
zatvorenje. Jedno seme.

**Krive učenja (MLflow):**

| model | train loss (ep. 200) | najbolji val loss |
|---|---|---|
| legacy, učeno poziciono kodiranje | 0.0104 | 0.0625 (ep. 62) |
| legacy, bez pozicionog kodiranja | 0.0113 | 0.0802 (ep. 39) |
| **`set`** | **0.0086** | **0.0065** (ep. 194) |

Legacy modeli overfituju: val loss stane na 0.06–0.09 posle 40–60 epoha, a train loss nastavlja
da pada. `set` nema razmak između train i val, i u 200 epoha još nije dostigao plato.
*Hipoteza:* težine po poziciji studenta omogućavaju učenje napamet, a `set` je invarijantan i
odmah dobija 2×2 tabele.

**Evaluacija na 8000 test uzoraka** (poređenje sa kvazi-poretkom, makro prosek):

| metod | F1 | Hamming | vreme (s, CPU) |
|---|---|---|---|
| IITA (v=1) | 0.911 | 0.032 | ~190 |
| IITA (v=2) | 0.908 | 0.033 | ~177 |
| IITA (v=3) | 0.734 | 0.144 | ~160 |
| legacy, učeno PE | 0.971 | 0.013 | 3.5 |
| legacy, bez PE | 0.964 | 0.016 | 3.7 |
| **`set`** | **0.998** | **0.001** | **3.0** |

IITA rezultati su do decimale isti u svim evaluacijama, dakle reč je o istim uzorcima.

**Demo: generalizacija na veći test i drugi broj studenata.** Graf je stablo sa račvanjem na 15
pitanja (lanac 0–7, pa grane 7→8→9→10 i 7→11→12→13→14). Model je treniran na 2–10 pitanja i
tačno 500 studenata. Greške se broje prema punom zatvorenju (93 para):

| studenata | greške | granični parovi (verovatnoća) |
|---|---|---|
| 100 | 1, $(4,3)$: 3 i 4 proglašeni ekvivalentnim | $(3,4)$ 0.87 |
| 600 | 0–1 zavisno od pokretanja, $(8,14)$ | $(8,14)$ 0.51 / 0.22, $(6,5)$ 0.50 |
| 2000 | 0 | nema |

Tačnost blago raste sa $N$, a model je nesigurniji kad ima manje studenata. *Hipoteza za granične
parove:* stanja se biraju ravnomerno ($p(K) = 1/|K|$, DAKS), a par poput $8 \to 14$ opovrgava
samo jedno od 28 stanja. Sa malo studenata kontraprimera je malo i uporedivo sa šumom. Svaki red
je jedno pokretanje na jednom grafu.

### 1.7 Ograničenja i šta je ostalo

- Jedno seme; bez pretrage hiperparametara (svesno).
- Model je treniran samo na 500 studenata. Za tvrdnju „radi za svako $N$" potreban je dataset sa
  promenljivim $N$, i tada **maska po studentima** u modelu (sada svi uzorci imaju 500).
- Dataset sa diplomskog je verovatno pun lanaca, a u lancu zbir reda određuje stanje znanja. To
  može da sakrije slabosti modela; odatle Deo 2.

---

## Deo 2: Složeniji dataset

### 2.1 Polazni problem

Generator sa diplomskog:

- bira slučajan topološki redosled i uzima parove samo „unapred", pa pravi samo parcijalne
  poretke (DAG), bez ekvivalentnih pitanja;
- **broj izabranih parova je uniforman od 0 do svih mogućih**;
- `Y` je Hasse dijagram;
- `ce` i `lg` se izvlače po uzorku iz $\mathcal{N}(0.1, 0.03)$ i $\mathcal{N}(0.05, 0.02)$, bez
  izvora u literaturi.

Brz uvid je pokazao da je veliki deo uzoraka (skoro) lanac. Detaljna analiza starog dataseta nije
rađena.

### 2.2 Šta kaže literatura

**DAKS** (Ünlü & Sargin; lokalno u `notes/`):

- BLIM dozvoljava posebne stope $\beta_q, \eta_q$ za svako pitanje, ali `simu` koristi iste za
  sva pitanja; stanja su jednako verovatna.
- Za slučajan graf: svaki par ulazi sa verovatnoćom `delta`, pa tranzitivno zatvorenje. Rezultat
  su kvazi-poreci, moguća su ekvivalentna pitanja.
- Python `simu(imp=None)` ne računa eksplicitno zatvorenje, pa ga nismo koristili.

**Sarginova disertacija** (Augsburg 2009, mentor Ünlü; sadrži rad Sargin & Ünlü 2009; fajl
`notes/learning spaces learning material/35097045.pdf`), sek. 3.4.1, str. 32–34:

- 9 pitanja; jedna $\delta$ za ceo graf, svaki od 72 para ulazi sa verovatnoćom $\delta$, pa
  zatvorenje.
- **Normalna šema (2009a):** „The δ values are drawn from a normal distribution with μ = 0.16 and
  σ = 0.06. Values less than 0 or greater than 0.3 are set to 0 or 0.3." Motivacija: uniformno
  $\delta$ (Schrepp, 2003) daje uglavnom skoro potpune relacije.
- **Šum** (str. 33): iz skupa svih stanja izvlači se stanje, pa se svaki unos menja „with a same
  prespecified error probability τ". To znači $ce = lg = \tau$, isto za sva pitanja, sa
  $\tau \in \{0.03, 0.05, 0.08, 0.10, 0.15, 0.20\}$ i brojem ispitanika iz
  $\{50, 100, 200, 400, 800, 1600, 6400\}$.
- **Druga studija (2009b)** (str. 48–51): $\delta = 0.285$, umanjuje se za 0.08 posle svakih 19
  dodatih implikacija. Autori je ocenjuju kao reprezentativniju; za 2009a kažu „quasi orders with
  many implications are overrepresented". Prag 19 važi samo za 9 pitanja.

**Kasniji radovi istih autora** (Frontiers in Psychology):

- Schrepp & Ünlü (2015): normalna šema je „ad-hoc" i „still produced non-representative samples".
  Predlažu induktivni algoritam koji uniformno uzorkuje sve kvazi-poretke, izvodljiv do oko
  15–20 pitanja.
- Ünlü & Schrepp (2016), sudeći po abstraktu: „close to representative" i do 50 pitanja.

### 2.3 Odluke

| tema | odluka | razlog |
|---|---|---|
| šema grafa | normalna šema 2009a | definisana po paru, pa se prenosi na 2–10 pitanja; 2009b je vezana za 9 pitanja |
| tip relacije | kvazi-poreci (ekvivalentna pitanja dozvoljena) | tako rade DAKS i IITA |
| `Y` | puno tranzitivno zatvorenje | Hasse nije jednoznačan kad ima ekvivalentnih pitanja; trening i eval već rade na zatvorenju |
| šum | $ce = lg = \tau$, po uzorku uniformno iz skupa vrednosti iz studije | kao u studiji; uniforman izbor je naš, jer je u studiji $\tau$ faktor eksperimenta |
| simulacija odgovora | i dalje `simu` iz biblioteke | provereno da daje tačna stanja i za kvazi-poretke |
| ostalo | kao stari dataset: 2–10 pitanja (weighted), 500 studenata, seme 42 | da se stari i novi model porede na istom novom datasetu |
| stari generator | ostaje u kodu kao legacy, ne koristi se | |

Poznato odstupanje od studije: `simu` za istu stavku prvo primeni `ce`, pa `lg`, pa stavka koju je
`ce` oborila može da se vrati na 1. Efektivni careless error zato nije tačno $\tau$.

Šta je to „kompleksnost" grafa (kandidat: broj stanja $|K|$) i kako dataset rasporediti po njoj,
ostaje otvoreno.

### 2.4 Implementacija

- **`src/util/generate_dataset_util.py`:**
  - nova funkcija `random_quasi_order`: $\delta \sim \mathcal{N}(0.16, 0.06)$ odsečeno na
    $[0, 0.3]$, svaki par $(i, j)$, $i \neq j$, ulazi sa verovatnoćom $\delta$, pa zatvorenje;
  - novi `generate_dataset`: $\tau$ po uzorku, `Y` = zatvorenje, vraća i `error_rates`;
  - parametri su konstante sa izvorom (strana disertacije);
  - `random_implications` i `generate_dataset_legacy` su nepromenjeni i označeni kao legacy.
- **`scripts/generate_dataset.py`:**
  - izbačeni `--ce`, `--lg`, `--ce-std` i `--lg-std`;
  - `.npz` čuva i `error_rates`;
  - podrazumevano ime ima sufiks `_daks`;
  - kartica dataseta opisuje generator i daje statistiku grafova po broju pitanja.
- **Testovi:** zatvorenost i odsustvo refleksivnih parova, uticaj $\delta$ (0 daje praznu
  relaciju, 1 potpunu), pojava ekvivalentnih pitanja, `Y` = zatvorenje uz ekvivalentna pitanja,
  $ce = lg = \tau$ iz skupa vrednosti. Ukupno 64 testa u projektu, svi prolaze.
- **Configi za trening na novom datasetu:** `configs/closure_no_positional_encoding_daks.toml`
  (legacy `linear` bez pozicionog kodiranja) i `configs/closure_set_encoder_daks.toml`.

### 2.5 Prva zapažanja

*Probni uzorak od samo 300 grafova, orijentaciono.*

| pitanja | prosek implikacija / max | prazna relacija | sa ekvivalentnim pitanjima |
|---|---|---|---|
| 2 | 0.4 / 2 | 60% | 0% |
| 5 | 4.5 / 20 | 9% | 13% |
| 7 | 15.2 / 42 | 0% | 52% |
| 9 | 37.2 / 72 | 0% | 84% |
| 10 | 41.2 / 90 | 2% | 71% |

- **Kod većih testova ekvivalentna pitanja su vrlo česta**, ponekad su sva pitanja međusobno
  ekvivalentna (prostor sa samo 2 stanja). Ovo se slaže sa ocenom autora da šema 2009a precenjuje
  velike kvazi-poretke.
- **Mali testovi su često prazni**, jer je šema kalibrisana za 9 pitanja.

Ovo je osobina izabrane šeme, ne greška implementacije.
