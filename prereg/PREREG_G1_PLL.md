# PRE-REGISTRATION — Does a rotation-invariant move restriction get its own expert?

Written 2026-11-05, before any model is trained on the new data. Already seen: everything in `docs/LAB_NOTEBOOK.md`
up to §38, including the post-hoc observation (§7 of the cube write-up) that CFOP's last layer restricts moves
(OLL 11/18 after rotations) without a clean stage split, while Kociemba's phase 2 (10/18) gets one.

## Hypothesis (H9)

Routing gives a stage its own expert when the stage restricts the allowed moves **in a way that is the same under
every training rotation**. Kociemba's G1 move set {U, D, R2, L2, F2, B2} is mapped onto itself by every training
rotation. CFOP's OLL move set is not: its quarter turns move from face to face.

## Manipulation

`data/cfop250k_g1pll.npz` (`src/gen_cfop_g1pll.py`): identical to `data/cfop250k.npz` (same 250,000 scrambles, same
cross, F2L and OLL moves), except that every PLL segment is replaced by the optimal Kociemba phase-2 solution of its
case, which uses only G1 moves (179 cases; PLL length 12.1 moves on average, against 23.2 before). Within one model
this gives two restricted stages:

| stage | allowed moves | under the training rotations |
|---|---|---|
| OLL (unchanged) | 7 moves raw, 11 after rotations | changes from face to face |
| PLL (G1 moves) | U, D, half turns (10) | identical in every orientation |

**Known confound, stated now:** the new PLL is also shorter and comes from an optimal search solver rather than
from 7 fixed algorithms. This test cannot separate "G1-restricted" from "phase-2-solver-like"; a positive result
supports the pair, not the move set alone.

## Models

Four new models, `cfop_g1pll_6L256_s{0..3}`, trained exactly as the CFOP production models
(`modal/prod.py::train_l40s --preset mixtral --data cfop250k_g1pll --rot keepD --budget 90`, 60M examples), with a
hard stop at 3.7 hours each for budget reasons, reported if it binds.

**Competence gate:** a model counts only if beam 256 solves at least 95% of 200 held-out scrambles of the new data.
If fewer than 3 models pass, the study is void.

**Comparison group:** the five existing CFOP models `cfop_mixtral_6L256_s{0..4}` on `cfop250k`, analysed with the
same script.

## Measures (`src/g1pll_analysis.py`, committed with this file)

Per model: final-layer decision-token top-1 expert, 1,200 held-out solves (same solve-level split as always),
S = moves remaining, permutation-corrected share I(E;V|S)/H(E|S).
- y_PLL: share explained by the binary label "position is in PLL".
- y_OLL: the same for "in OLL".
- Knockouts: each final-layer expert removed (router logit −∞), measuring the drop in p(solver's move) per stage.
  A **PLL-dedicated expert** is one whose PLL drop is at least 3 points while every other stage's drop is at most
  0.25 × its PLL drop.

## Predictions (all must hold for H9)

- **P1:** y_PLL(new) ≥ 2 × the mean y_PLL of the five old models, in at least 3 of the passing new models.
- **P2:** y_PLL(new) ≥ 2 × y_OLL(new), in at least 3 of the passing new models (within-model control).
- **P3:** a PLL-dedicated expert exists in at least 3 of the passing new models.

If fewer than 4 models pass the gate, "at least 3" becomes "all passing models".

Per-stage knockout tables and stage routing shares are reported descriptively.

**Amendment A1 (2026-11-05, before any training):** the hard stop is 3.4 hours per model, not 3.7, to keep the four
runs under the $30 budget on thisaccisfortheschool. If it binds, a model sees about 56M rather than 60M examples; the
competence gate still decides inclusion. The new dataset passes all 9 CFOP QA gates (sha256 93f60f1c…).
