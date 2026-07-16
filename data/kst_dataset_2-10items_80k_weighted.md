---
dataset_name: kst_dataset_2-10items_80k_weighted
generated_by: scripts/generate_dataset.py
generated_at: '2026-07-16 15:49:46'
generator_args:
  num_samples: 80000
  max_items: 10
  min_items: 2
  size: 500
  ce: 0.1
  lg: 0.05
  ce_std: 0.03
  lg_std: 0.02
  output: D:\skola\diplomski\kst-transformer\scripts\..\data\kst_dataset_2-10items_80k_weighted.npz
  seed: 42
  weighted: true
git_commit: 7eb1dd949c0df5c148473e02eea1f0d5fe0027f1
---

# kst_dataset_2-10items_80k_weighted

## Summary
Sinteticki dataset za KST Transformer - simulirani odgovori studenata, generisani sa `scripts/generate_dataset.py`.

## Structure
- `X`: (N, 500, 10) float32 - binarna response matrica, paddovana nulama
- `Y`: (N, 10, 10) float32 - Hasse dijagram (tranzitivna redukcija)
- `item_counts`: (N,) int64 - stvarni broj pitanja po uzorku

## Distribucija broja pitanja
| items | count | % |
|---|---|---|
| 2 | 1837 | 2.3% |
| 3 | 3561 | 4.5% |
| 4 | 5319 | 6.6% |
| 5 | 7189 | 9.0% |
| 6 | 8776 | 11.0% |
| 7 | 10603 | 13.3% |
| 8 | 12263 | 15.3% |
| 9 | 14393 | 18.0% |
| 10 | 16059 | 20.1% |
