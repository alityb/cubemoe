# PRE-REGISTRATION #13: Does CFOP's routing path carry the stage? (replication of #12 Part B on fresh positions)

Written 2026-11-09, before any measure below has been computed on these positions. This replicates an **exploratory**
observation: #12's Part B was void by its own rule (its permutation null fell outside [−0.05, 0.02] on Kociemba
positions, where H(phase | S) is only 0.12 bits), but on CFOP positions, where the null was in bounds (−0.03 to
−0.05), the six-layer routing path of all 8 CFOP models carried 0.41–0.54 of H(stage | S), every single layer ≤ 0.15,
and the paths of the 5 Kociemba models on the same positions 0.25–0.31 (§42). Those numbers were seen and motivate
this test; the independent evidence is a new set of positions plus two controls #12 did not have. The scripts were
smoke-tested on 20 fresh solves, outputs deleted unread (only file structure and exit codes printed); the only
property of the fresh positions inspected is the stage count per class and the moves-remaining range (73,625
positions; all 4 stages). The design was revised once after an independent review (see the end).

## Positions and models

CFOP `cfop250k` held-out solves **1,200–2,399** in the RandomState(0) held-out ordering: the 1,200 held-out solves right
after the ones every earlier analysis used. Training uses `perm[:ntr]` (modal/prod.py); `src/paths_fresh.py` asserts
the fresh solves are disjoint from it. (The training gate's held-out accuracy read the first ~3,300 held-out solves,
which include these; the final checkpoint is saved without checkpoint selection, so nothing was selected on them.)
Label: stage. Halves A/B = even/odd fresh solve. Decision token, all six layers. Only CFOP positions are scored.
Models: C = `cfop_mixtral_6L256_s{0..7}`; controls K = `mixtral_6L256_s{0..4}` (Kociemba-trained routers) and
U = three **untrained** models of the same architecture (`random_s{100,101,102}`, torch seeds 100–102), whose routing
path is a random hash of the cube state.

## Measure

I(R) = [CE(stage | S) − CE(stage | S, R)] / H(stage | S): gradient-boosted classifiers exactly as #12 (200 iterations,
15 leaves, L2 = 1, no early stopping), fitted on one half of the solves and scored on the other, both directions.
R = the sorted top-2 expert pair of the decision token at one layer, at L0–L4, or at all six layers (the path).
**Additive comparator**: the same six layers one-hot encoded, each with its own copy of S, with interaction
constraints that let S interact with each layer but forbid any two layers from interacting; compared with the same
one-hot features unconstrained. (sklearn 1.7.2 does not enforce multi-feature constraints on categorical features, and
a shared S would re-admit cross-layer interactions through the union rule; the encoding used was verified on synthetic
XOR data: unconstrained 1.00 accuracy, constrained 0.51.)
Null: routing rows permuted within S × half, all layers. Also computed, descriptively, on fresh positions whose exact
state never occurs in the training split.

## Predictions (each in both directions)

- **B0 (instrument):** the null's I in [−0.08, 0.02]. The upper bound is the leak detector; the lower bound only flags a
  grossly broken classifier (it was loosened from #12's −0.05, which the CFOP-position nulls of #12 nearly reached;
  set now, before fresh results; the null is never subtracted from I). A C-model null out of bounds voids the test; a
  K or U null out of bounds voids only B2 or B2r.
- **B1 (beyond the final layer):** in ≥ 6/8 C models, I(all) − I(L5) ≥ 0.10.
- **B1b (beyond any single layer):** in ≥ 6/8 C models, I(all) − max over single layers ≥ 0.05.
- **B1c (combinatorial):** in ≥ 6/8 C models, I(one-hot, unconstrained) − I(additive) ≥ 0.05.
- **B2 (beyond Kociemba-trained routers):** in ≥ 6/8 C models, I(all) − max over K of I(all) ≥ 0.10 AND
  I(L0–4) − max over K of I(L0–4) ≥ 0.10 (the second removes K's near-degenerate final layer on CFOP states).
- **B2r (beyond a random hash of the state):** in ≥ 6/8 C models, I(all) − max over U of I(all) ≥ 0.10.

## Reading rules, written before results

- B1, B2 and B2r hold: **replicated**: the CFOP routing path carries stage-predictive information beyond the final
  layer, beyond Kociemba-trained routers and beyond a random hash of the cube state. Qualified by B1b (beyond any single
  layer, or not) and B1c (combinatorial across layers, or additive). Only B1c licenses "combinatorial". Even if all
  hold, this is information in the routing, not evidence that the path does stage-specific work.
- B1 holds, B2 or B2r fails: the path adds stage information, but not beyond what other routers or a random hash of the
  state carry; not CFOP-specific.
- B1 fails (whatever B2/B2r): not replicated; #12's exploratory observation does not hold on fresh positions.
- B0 fails for a C model: void.

Reported descriptively: every layer's I and CE in bits, H(stage | S), I(L0–4), per-direction values, pair entropies,
the fraction of fresh states seen verbatim in training, and every prediction on the non-duplicate subset.

## Known weaknesses, stated now

- Same C models and method as #12; only positions and controls are new.
- Overfitting: B1 and B1b compare six 28-level features with one, biasing them toward failing. B2 and B2r compare six
  with six; a control with lower pair entropy (K's final layer on CFOP states) has a smaller overfitting penalty,
  biasing B2 toward failing. B1c compares identical features with and without constraints.
- C models were trained on CFOP states; many late-solve states recur across solves, so C may have memorised routing
  for exact states K and U never saw. The non-duplicate subset is reported for this reason.
- Not adopted from the review: a k-means (28 clusters per layer) control on the representation and a sticker-feature
  classifier. The untrained-model control is the direct test of "any path is a hash of the state"; stickers determine
  the stage exactly, so a sticker classifier would only show the ceiling (≈ 1).
- Information is correlational; S = moves remaining is conditioned on as in every earlier analysis.

## Review (before running)

An independent review of the first draft found (11 items, 2 must-fix), and this version fixes:
1. "Beyond any single layer" measured the union of layers, not their combination: B1c with an additive-over-layers
   comparator added (the reviewer's suggested constraint form leaked through the categorical/union behaviour above;
   the one-hot, per-layer-S form was used instead and verified).
2. Any deep path is a fine hash of the state, and the K control is not capacity-matched: untrained-model control (B2r)
   added; B2 also required on L0–L4; reading rules reworded to "stage-predictive information in the routing path"
   (no "division of labour").
3. Overfitting biases stated per prediction. 4. Disjointness from the training split asserted; gate overlap and
   duplicate states stated; non-duplicate subset reported. 5. B0 bound rationale stated; null never subtracted.
6. B0 scope specified. 7–11. Report wording aligned with the prereg, all reading-rule cases listed, the report is not
   run if any job fails, bits reported alongside normalised values.
