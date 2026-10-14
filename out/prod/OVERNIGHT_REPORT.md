# Overnight results: production-style (Mixtral-style) cube MoE

## 1. Training and solving, per seed

| seed | next-move acc | greedy solve | beam-256 (training check, 40 cubes) | 500-cube eval: beam 64 / 256 / 1024 |
|---|---|---|---|---|
| 0 | 73.0% | 4.0% | 100% | 492/500 (25.8 mv) / 500/500 (23.8 mv) / 500/500 (22.6 mv) |
| 1 | 72.6% | 4.5% | 100% | 497/500 (25.5 mv) / 500/500 (23.7 mv) / 500/500 (22.6 mv) |
| 2 | 72.0% | 4.5% | 100% | 479/500 (25.8 mv) / 500/500 (24.0 mv) / 500/500 (22.7 mv) |
| 3 | 72.6% | 3.0% | 100% | 483/500 (25.7 mv) / 500/500 (23.8 mv) / 500/500 (22.7 mv) |
| 4 | 73.7% | 9.5% | 100% | 499/500 (24.9 mv) / 500/500 (23.5 mv) / 500/500 (22.4 mv) |

## 2. Is it the model or the search? (seed 0)

- Beam search with **no model** (all moves equally likely), width 1024: **0/30** random cubes solved
- Beam search with an **untrained** model, width 256: **0/30**
- **Trained** model, width 256: **60/60**; trained model with no search (greedy): 1/60
- Along the solutions found, the move taken was the model's **own 1st choice 86.3%** of the time, 2nd 7.8%, 3rd 2.9%, lower 3.0% (1428 moves)
- Memorisation check: **0 of 500** evaluation cubes appear among 26,891,547 training positions (incl. rotations)

## 3. Attention (seed 0)

| attention | next-move acc | beam-256 solve (40 cubes) |
|---|---|---|
| normal | 72.7% | 100% |
| zero | 9.0% | 0% |
| uniform | 2.9% | 0% |

Accuracy with attention removed in one layer only: L0 7.3%, L1 23.5%, L2 48.9%, L3 57.4%, L4 60.6%, L5 66.6%

Decision-token attention to centre / edge / corner stickers (even spread = 10.9 / 43.6 / 43.6%): L0: 9.1 / 43.2 / 44.4; L1: 8.9 / 44.2 / 45.6; L2: 9.2 / 48.3 / 40.9; L3: 10.9 / 42.7 / 44.2; L4: 11.6 / 45.4 / 41.0; L5: 9.0 / 46.0 / 42.5

## 4. Abstract cube properties readable from each layer (seed 0)

In-G1 = classification accuracy; others = R². "raw" = a linear readout of the input stickers.

| property | raw | layer 0 | layer 1 | layer 2 | layer 3 | layer 4 | layer 5 |
|---|---|---|---|---|---|---|---|
| in G1 (phase 2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| # misoriented edges | 0.99 | 0.85 | 0.89 | 0.93 | 0.95 | 0.94 | 0.83 |
| # misoriented corners | 1.00 | 0.97 | 0.97 | 0.97 | 0.96 | 0.95 | 0.88 |
| # slice edges out of slice | 1.00 | 0.95 | 0.95 | 0.95 | 0.95 | 0.93 | 0.86 |
| moves to reach G1 | 0.85 | 0.80 | 0.83 | 0.85 | 0.91 | 0.95 | 0.91 |
| moves to solved | 0.77 | 0.82 | 0.86 | 0.89 | 0.91 | 0.92 | 0.85 |

## 5. Experts: pattern matching or abstract properties? (seed 0)

| layer | sticker tokens routed by own (position, colour) | decision token: best single sticker | phase | moves-to-G1 | bad edges | bad corners | next move | decision-routing entropy |
|---|---|---|---|---|---|---|---|---|
| 0 | 92.6% | 47.5% | 98.0% | 98.0% | 68.4% | 97.7% | 41.1% | 0.94 bits |
| 1 | 89.4% | 18.2% | 7.1% | 7.0% | 5.5% | 7.4% | 4.3% | 0.14 bits (too low: unreliable) |
| 2 | 72.1% | 12.8% | 13.6% | 13.5% | 10.9% | 13.5% | 11.6% | 0.81 bits |
| 3 | 64.2% | 40.0% | 37.8% | 37.7% | 28.5% | 37.7% | 20.1% | 1.85 bits |
| 4 | 59.9% | 36.7% | 72.3% | 77.3% | 54.3% | 73.2% | 36.2% | 1.27 bits |
| 5 | 45.2% | 35.9% | 62.8% | 71.0% | 54.3% | 65.8% | 31.4% | 1.25 bits |

Knockouts at layer 3 (drop in probability of the solver's move):

| expert | phase 1 | phase 2 | ph1 edge-flip moves | ph1 corner-twist moves | effective # moves |
|---|---|---|---|---|---|
| e0 | +0.027 | +0.003 | +0.026 | +0.016 | 15.4 |
| e1 | +0.000 | +0.006 | +0.000 | +0.000 | 10.0 |
| e2 | +0.000 | +0.019 | +0.000 | +0.000 | 10.4 |
| e3 | +0.000 | +0.003 | +0.000 | +0.000 | 8.7 |
| e4 | +0.005 | +0.002 | +0.006 | +0.005 | 13.5 |
| e5 | +0.000 | +0.019 | -0.000 | -0.000 | 11.0 |
| e6 | +0.000 | +0.001 | -0.000 | +0.000 | 9.4 |
| e7 | +0.012 | +0.005 | +0.009 | +0.008 | 14.3 |

Knockouts at layer 5 (drop in probability of the solver's move):

| expert | phase 1 | phase 2 | ph1 edge-flip moves | ph1 corner-twist moves | effective # moves |
|---|---|---|---|---|---|
| e0 | +0.086 | -0.000 | +0.096 | +0.133 | 8.8 |
| e1 | -0.000 | +0.000 | +0.000 | -0.000 | 1.7 |
| e2 | +0.000 | +0.302 | -0.000 | +0.001 | 8.1 |
| e3 | +0.000 | -0.000 | +0.000 | -0.000 | 1.4 |
| e4 | -0.000 | +0.000 | -0.000 | +0.000 | 2.4 |
| e5 | +0.000 | +0.000 | +0.000 | +0.000 | 2.0 |
| e6 | +0.000 | -0.000 | +0.000 | -0.000 | 1.0 |
| e7 | +0.099 | +0.000 | +0.191 | +0.103 | 5.3 |

## 6. The big picture: does the router split at the Kociemba phase boundary, and is it used?

| seed | routing explained by phase | shuffled | phase-1 expert: share of ph1 / ph2 positions | KO it: ph1 / ph2 loss | KO typical other: ph1 / ph2 | force ph2 through it / typical other |
|---|---|---|---|---|---|---|
| 0 | 18.6% | 0.0% | e0: 62% / 0.0% | +8.6 / -0.0 pts | +0.0 / +0.0 pts | 19.4% / 34.5% |
| 1 | 21.2% | 0.0% | e6: 81% / 0.4% | +12.4 / -0.2 pts | +0.0 / +0.0 pts | 7.5% / 7.2% |
| 2 | 27.1% | -0.0% | e1: 58% / 0.0% | +12.4 / +0.0 pts | +0.0 / +0.0 pts | 26.2% / 21.1% |
| 3 | 17.8% | 0.0% | e6: 72% / 0.0% | +11.7 / +0.0 pts | +0.0 / +0.0 pts | 14.1% / 4.4% |
| 4 | 24.9% | 0.0% | e1: 78% / 0.0% | +14.7 / +0.0 pts | +0.0 / +0.0 pts | 11.4% / 9.8% |

---
Generated automatically by src/overnight.sh. Exploratory: interpret with the caveats in MEMORY.md.
