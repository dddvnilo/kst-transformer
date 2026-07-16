---
dataset_name: kst_dataset_2-5items_20k_weighted
generated_by: scripts/generate_dataset.py
generated_at: unknown
generator_args:
  num_samples: 20000
  max_items: 5
  min_items: 2
  size: 500
  ce: 0.1
  lg: 0.05
  ce_std: 0.03
  lg_std: 0.02
  output: kst_dataset_2-5items_20k_weighted.npz
  seed: 42
  weighted: true
git_commit: unknown
---

# kst_dataset_2-5items_20k_weighted

## Summary
Sinteticki dataset za KST Transformer - simulirani odgovori studenata, generisani sa `scripts/generate_dataset.py`.

Napomena: ova kartica je rekonstruisana naknadno (fajl je nastao pre uvodjenja automatskog
kartica-writing-a), pa `generated_at` i `git_commit` nisu pouzdano poznati i namerno su
ostavljeni kao `unknown`. `generator_args` su rekonstruisani iz ispisa generisanja
(broj uzoraka, studenata/uzorak, max items) i default vrednosti skripte za ostale parametre
(ce, lg, ce_std, lg_std, seed) - nisu direktno potvrdjeni.

## Structure
- `X`: (N, 500, 5) float32 - binarna response matrica, paddovana nulama
- `Y`: (N, 5, 5) float32 - Hasse dijagram (tranzitivna redukcija)
- `item_counts`: (N,) int64 - stvarni broj pitanja po uzorku

## Distribucija broja pitanja
| items | count | % |
|---|---|---|
| 2 | 2064 | 10.3% |
| 3 | 3930 | 19.6% |
| 4 | 5953 | 29.8% |
| 5 | 8053 | 40.3% |
