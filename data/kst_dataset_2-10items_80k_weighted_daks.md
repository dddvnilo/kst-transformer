---
dataset_name: kst_dataset_2-10items_80k_weighted_daks
generated_by: scripts/generate_dataset.py
generated_at: '2026-11-22 17:03:24'
generator_args:
  num_samples: 80000
  max_items: 10
  min_items: 2
  size: 500
  output: kst_dataset_2-10items_80k_weighted_daks.npz
  seed: 42
  weighted: true
git_commit: 3cded7265a2e45e657bfe69a236e3c315838f7fe
---

# kst_dataset_2-10items_80k_weighted_daks

## Summary
Sinteticki dataset za KST Transformer - simulirani odgovori studenata, generisani sa `scripts/generate_dataset.py`.

## Generator
- Grafovi: normalna sema (Sargin & Unlu, 2009a) - delta ~ N(0.16, 0.06), odseceno na [0, 0.3]; svaki par (i, j), i != j, ulazi sa verovatnocom delta; pa tranzitivno zatvorenje. Rezultat je kvazi-poredak (moguca ekvivalentna pitanja).
- Sum: careless error = lucky guess = tau, tau uniformno iz [0.03, 0.05, 0.08, 0.1, 0.15, 0.2] po uzorku.
- Izvor: Sargin (2009), disertacija, sek. 3.4.1 (`notes/learning spaces learning material/35097045.pdf`).

## Structure
- `X`: (N, 500, 10) float32 - binarna response matrica, paddovana nulama
- `Y`: (N, 10, 10) float32 - PUNO TRANZITIVNO ZATVORENJE (kvazi-poredak; ekvivalentna pitanja imaju veze u oba smera), dijagonala = 0
- `item_counts`: (N,) int64 - stvarni broj pitanja po uzorku
- `error_rates`: (N,) float32 - stopa greske tau po uzorku

## Distribucija broja pitanja
| items | count | % |
|---|---|---|
| 2 | 1708 | 2.1% |
| 3 | 3556 | 4.4% |
| 4 | 5380 | 6.7% |
| 5 | 7303 | 9.1% |
| 6 | 8979 | 11.2% |
| 7 | 10443 | 13.1% |
| 8 | 12470 | 15.6% |
| 9 | 14091 | 17.6% |
| 10 | 16070 | 20.1% |

## Distribucija stope greske tau
| tau | count | % |
|---|---|---|
| 0.03 | 13420 | 16.8% |
| 0.05 | 13469 | 16.8% |
| 0.08 | 13404 | 16.8% |
| 0.10 | 13153 | 16.4% |
| 0.15 | 13390 | 16.7% |
| 0.20 | 13164 | 16.5% |

## Grafovi po broju pitanja
| items | prosecno implikacija / max | prazna relacija | sa ekvivalentnim pitanjima |
|---|---|---|---|
| 2 | 0.3 / 2 | 72.2% | 2.9% |
| 3 | 1.1 / 6 | 37.7% | 9.6% |
| 4 | 2.6 / 12 | 15.9% | 18.0% |
| 5 | 4.9 / 20 | 7.6% | 29.0% |
| 6 | 8.5 / 30 | 3.5% | 40.3% |
| 7 | 13.7 / 42 | 1.9% | 51.3% |
| 8 | 20.7 / 56 | 1.3% | 60.5% |
| 9 | 30.4 / 72 | 0.8% | 69.0% |
| 10 | 42.4 / 90 | 0.7% | 76.0% |
