# PRE-REGISTRATION #12: Is CFOP's stage in what the router reads, and is it carried across layers?

Written 2026-11-09, before any measure below has been computed on any model. Already seen: everything in
`docs/LAB_NOTEBOOK.md` up to §41. The script was validated on synthetic data (below) and smoke-tested end to end on
20 solves for three model/data combinations and the report on placeholder copies; those outputs were deleted unread
(only file structure and exit codes were printed). The design was revised once after an independent review (see
"Review" at the end); only the revised design counts.

## Why

Two gaps in the paper, both raised by 2026 work:
1. *The Myth of Expert Specialization* (arXiv 2604.09780) argues that a linear router reflects the geometry of its
   input: what is spread out in the representation gets spread out across experts. It cannot say when that geometry
   matches a task's real structure, because language has no ground truth. Our two solvers give a case where routing
   follows the decomposition (Kociemba) and one where it does not (CFOP). Does each router read its label **beyond
   what reading the dominant directions of its input would give (selection)**, or only **as much as the dominant
   geometry carries (geometry)**? And is CFOP's stage even present in what the router reads?
2. Every CFOP routing result so far is from the final layer. *Path-Constrained MoE* (arXiv 2603.18297) treats the
   sequence of experts across layers as the unit. Is CFOP's stage carried by the **path**?

Already known (final layer, decision token): routing explains 55–100% of H(phase | S) in Kociemba models (top-1;
pair 72–99%) and 7–14% of H(stage | S) in CFOP models (pair 12–21%) (§40). In Kociemba models the router's
phase-2-minus-phase-1 weight difference aligns with the phase direction of its input (§30, exp2; cosine 0.83 in s0).
Whether phase lies in the dominant (top-variance) directions of the router input is not known.

## Models and positions

K = `mixtral_6L256_s{0..4}` (Kociemba, `mixed250k`, label phase 1/2); C = `cfop_mixtral_6L256_s{0..7}` (CFOP,
`cfop250k`, label stage cross/F2L/OLL/PLL; s5–s7 from #11). Every model is also run on the **other** family's
positions (cross runs; labels from the positions). 1,200 held-out solves, the RandomState(0) solve-level split of
every earlier analysis; halves A/B = even/odd held-out solve; decision token; every layer L0–L5; S = moves remaining.
Script `src/geom_paths.py`, verdicts `src/geom_report.py`, both committed with this file. One machine for all runs.

## Measures

**Part A**, at each layer's router input h (the MoE input, after RMSNorm):
- η²(M) = tr(M B_c Mᵀ) / tr(M T Mᵀ): the label's share of the within-S variance of a linear readout Mh. T = within-S
  scatter; B_c = between-(S, label)-cell scatter minus its mean over 50 within-S label permutations.
- η²(h) (M = identity), with the 99th percentile of its permutation distribution.
- **Ceiling**: the label share of the best single linear readout, cross-fitted (top generalised eigenvector of
  (B_c, T) found on one half, scored on the other; mean of both directions).
- η²(z) for the trained router, M = C W (C centres the 8 logits; the softmax ignores a common shift), and
  **r = η²(z) / ceiling**: the router's label share as a fraction of the best possible linear readout (bounded, so a
  2-class and a 4-class label are on the same scale).
- **Geometry null (primary)**: 1,000 routers with the trained router's singular values that read the top-7
  principal subspace of h (7 = the rank of C W) in a random orientation within it. This is a router that "reflects
  geometry": it reads only the dominant directions. **Reads beyond geometry** = η²(z) above this null's 99th
  percentile and ≥ 1.5 × its median. (A Haar-random orientation over all of ℝ²⁵⁶ is reported descriptively.)

**Part B** (expert paths): I(R) = [CE(label | S) − CE(label | S, R)] / H(label | S), cross-entropies in bits of
gradient-boosted classifiers (sklearn HistGradientBoosting, 200 iterations, 15 leaves, L2 = 1, no early stopping)
fitted on one half of the solves and scored on the other, both directions. R = the sorted top-2 pair of the decision
token at one layer, at L0–L4, or at all six layers (the path). A held-out lower bound on the information.
Null: routing rows permuted within S × half, all layers.

Synthetic validation (`geom_paths.part_a`, d = 64, 7 high-variance directions): label in a top direction and a router
reading only the top directions → does not beat the geometry null (η²(z) 0.011 vs p99 0.049); label in a low-variance
direction and a router reading it → beats it (0.031 vs 0.0007); random router → does not.

## Predictions (Part A at L5, the final layer where every earlier result was measured)

Part A:
- **A1 (Kociemba reads phase beyond geometry):** in ≥ 4/5 K models, in half A and in half B.
- **A2 (stage is present):** in ≥ 6/8 C models, η²(h) exceeds the 99th percentile of its permutation distribution in
  both halves.
- **A3 (CFOP reads stage beyond geometry, and beyond what any trained router does):** in ≥ 6/8 C models, the router
  reads beyond geometry in both halves AND its r exceeds the largest r of any K model's router on the same CFOP
  positions (a Kociemba router also tracks the cube state; this is the cross-model control).
- **A4 (comparably):** median r over C ≥ 0.5 × median r over K.

Part B:
- **B0 (instrument check):** the null's I lies in [−0.05, 0.02] in every run, both directions; else Part B is void.
- **B1 (the path adds stage beyond the final layer):** in ≥ 6/8 C models, both directions, I(all) − I(L5) ≥ 0.10.
- **B1b (secondary; combinatorial):** in ≥ 6/8 C models, both directions, I(all) − max over single layers ≥ 0.05.
- **B2 (CFOP-specific):** in ≥ 6/8 C models, both directions, I(all) exceeds by ≥ 0.10 the largest I(all) of any K
  model on the same CFOP positions. If, at any layer, every K model's pair entropy on CFOP positions is < 0.5 × the C
  median, B2's control is flagged as degenerate at that layer (K routers may collapse on unfamiliar states).
- **B3 (closes the gap):** median over C of I(all) ≥ 0.5 × median over K (mean of directions).

**My expectation, stated now:** A1 uncertain (phase may sit in the dominant directions, which would make even the
Kociemba router "geometry"); A2 passes; A3 fails; A4 fails. B1 uncertain, B2 fails, B3 fails.

## Reading rules, written before results (every combination)

Part A:
- A2 fails: stage is not detectably present at the final router input; the difference lies upstream of the router.
- A1 passes, A3 fails: **SELECTION.** The Kociemba router reads phase beyond the dominant geometry; the CFOP router
  does not read stage beyond it (or no more than a Kociemba router does). "Routing reflects geometry" describes the
  CFOP router but not the Kociemba one.
- A1 and A3 pass, A4 passes: **both selective, comparably.** The low CFOP routing information then reflects how
  readable stage is (compare ceilings), not the router.
- A1 and A3 pass, A4 fails: **MIXED.** Both read their label selectively; CFOP much less so.
- A1 and A3 fail: **GEOMETRY.** Neither router reads its label beyond the dominant directions; the difference is
  whether the label lies in those directions (reported: the geometry null's r = how much the dominant directions
  carry).
- A1 fails, A3 passes: the CFOP router is more selective than the Kociemba router; this contradicts the premise and is
  reported as such.

Part B (only if B0 passes):
- B1 and B2: CFOP's stage is carried by the path beyond the final layer and beyond state tracking; "no stage split"
  must be limited to the final layer. B1b says whether this is combinatorial, B3 whether it closes the gap.
- B1 only: the path adds stage information, but no more than a Kociemba router's path carries.
- B2 only: CFOP routing carries more stage than a Kociemba router, but already at the final layer.
- Neither: the no-split result extends across layers.

Reported descriptively: every layer's quantities for both families and for the cross runs; the unconditioned
(S constant) Part A; per-half values; the Haar null; pair entropies.

## Known weaknesses, stated now

- η² is a variance share of linear readouts; the router is linear, but top-2 selection depends on the leading logits,
  not on all centred logits. Label variance in logits of experts that never win counts toward η²(z).
- The width of the geometry null depends on how many directions the label occupies; each family is judged against
  its own null, and the 1.5 × median bar adds an effect-size floor.
- Conditioning on S (which the model never sees) is the project convention and is primary; §35 showed it can
  understate phase. The unconditioned Part A is reported.
- Part B's classifier is a lower bound and fits 6 features with 28 levels for the path vs 1 for L5, so overfitting
  biases B1 and B2 toward failing. The K reference is near its ceiling at L5, biasing B3 toward passing. Normalised
  units differ in bits (0.10 = 0.054 bits for stage, 0.012 for phase). No bootstrap intervals.
- B2's and A3's controls take the max over 5 K models (conservative). K models never trained on CFOP states.
- Decision token only; 5 K and 8 C seeds; one architecture (8 experts, top-2).

## Review (before running)

An independent review of the first draft found 12 problems (4 must-fix); this version fixes:
1. The null (Haar rotation over all of ℝ²⁵⁶) was not the "routing reflects geometry" null: a router reading only the
   top principal directions beats it whenever the label lies there (reviewer's synthetic test: 5.2× the null). The
   primary null now reads the top-7 principal subspace; validated above.
2. The selectivity score was unbounded and undefined near zero, making the 2-class vs 4-class comparison unfair.
   Replaced by r = η²(z) / cross-fitted ceiling.
3. The percentile bar depends on label rank: an effect-size floor (≥ 1.5 × null median) was added and the dependence
   is stated.
4. Part A had no cross-model control: A3 now also requires beating every K router on CFOP positions.
5. A2 ("> 0") was near-automatic: now > the permutation distribution's 99th percentile, both halves.
6. "Geometry" was claimed without checking the label lies in the dominant directions: the geometry reading now rests
   on the geometry null, and the null's r is reported.
7. Missing reading-rule combinations: all are now listed.
8. B1 credited the path for a single earlier layer: B1b (beyond the best single layer) added.
9. B2's control could be degenerate on unfamiliar states: pair-entropy check added; bit-scale differences stated.
10. Early stopping used a position-level split: removed (fixed iterations, regularised); the null now permutes within
    S × half; B0 also has a lower bound (−0.05) to catch a broken classifier.
11. Docstring/name inconsistency fixed.
12. Not adopted: a planted-direction power check on real h. The synthetic validation and the K reference
    (A1) serve as the power check; if A1 fails, Part A's "geometry" reading is reported with that caveat.
