# PRE-REGISTRATION #10: Do the two selected experts back each other up, and does the pair carry the stage?

Written 2026-11-08, before any of the measures below has been computed at full size on any model. Already seen:
everything in `docs/LAB_NOTEBOOK.md` up to §39. Before committing, the scripts were smoke-tested on 15–16 solves
for identities and for running end to end; those outputs were discarded unread (no stage-level number was printed).
The design was revised once after an adversarial review (see "Review" at the end); only the revised design counts.

## Why

Every knockout so far removed ONE final-layer expert by setting its router logit to −∞. The next-ranked expert then
takes the free slot and the top-2 weights are renormalised. Two consequences are untested:
1. **Backup.** If two experts can each do a stage's work, removing either alone costs little, because the other fills
   in. Single-expert knockouts would then show no stage-dedicated expert even if a stage-dedicated *pair* exists. This
   could explain part of the Kociemba/CFOP difference (§28–39): CFOP's stage split might be hidden by redundancy
   rather than absent.
2. **Pair code.** Routing shares used the top-1 expert only. With top-2 routing, the stage could be carried by which
   *pair* is selected (RouterInterp, arXiv 2610.11775, proposes routing collections as the unit to interpret).

Literature that shaped the method: arXiv 2611.05804 (results depend on whether removal reroutes, freezes or
renormalises, and on matching the training-time weight rule; all three are reported on the same sets, the primary one
follows the training rule); arXiv 2607.28308 (interactions between co-selected experts must be measured on the output
with the route controlled; geometric overlap is not redundancy); arXiv 2610.00558 (pairwise interaction only exists
where the two experts are co-active, so interactions are decomposed by position type).

## Models and positions

- K: Kociemba, `mixtral_6L256_s{0..4}` (stages: phase 1, phase 2), held-out positions of `data/mixed250k.npz`.
- C: CFOP, `cfop_mixtral_6L256_s{0..4}` (cross, F2L, OLL, PLL), `data/cfop250k.npz`.
- G: CFOP with G1-move PLL, `cfop_g1pll_6L256_s{0..3}`, `data/cfop250k_g1pll.npz`. Descriptive, plus a control.

Final layer, decision token only (the final layer's other tokens never reach the output). 1,200 held-out solves, the
same RandomState(0) solve-level split as every earlier analysis; halves A/B = even/odd held-out solve. All 14 models
and the controls run on the same device (Modal CUDA, fp32).

## Measures (`src/pairs_analysis.py`; verdicts applied by `src/pairs_report.py`; both committed with this file)

Removal of expert set Z (every single expert and all 28 pairs) in three modes: **reroute** (router logits of Z = −∞,
next-ranked experts take the slots, weights renormalised; the training rule and every earlier knockout; PRIMARY),
**drop** (routing kept, Z's outputs removed, survivor keeps its weight) and **renorm** (routing kept, survivor weight
1). Drop and renorm are descriptive.

Two damage scales, averaged per stage, per half: **Δ** = p(teacher move) intact − removed, and **M** = margin intact −
removed, margin = teacher-move logit − largest other logit. M has no softmax saturation.

- **Dedicated pair** for stage s: Δ_ij(s) ≥ 0.05 and Δ_ij(s') ≤ 0.25 × Δ_ij(s) for every other stage s'.
- **Hidden pair** (i, j, s): in half A and in half B separately, (i, j) is a dedicated pair for s, AND
  max(Δ_i(s), Δ_j(s)) ≤ 0.5 × Δ_ij(s), AND M_ij(s) > 0 with max(M_i(s), M_j(s)) ≤ 0.5 × M_ij(s).
  (Single knockouts understate the pair by at least half, on both scales, replicated across solve halves.)
- **Backup index** BI = Σ_pairs (M_ij − M_i − M_j) / Σ_pairs M_ij over all positions (signed; pooled over all
  positions, so it does not depend on how many stages a family has). Undefined if the denominator ≤ 0.
- **Pair increment** = [I(stage; ordered top-2 pair | S) − I(stage; top-1 | S)] / [H(stage | S) − I(stage; top-1 | S)],
  S = moves remaining, each I permutation-corrected with 50 within-S permutations. The share of the stage information
  left unexplained by the top-1 expert that the second expert adds. Undefined if the denominator ≤ 0.

Descriptive: the reroute pair damage split into an **exclusive** part (positions where one of the two is selected:
the extra loss when the partner is unavailable as backup) and a **co-selected** part; who substitutes for whom; stage
selection rates; drop/renorm tables.

## Controls (must pass, or H10a is reported as untestable by this method, not as rejected)

Run on K s0 and on G s0 (a 4-stage model like C). e = the single-dedicated expert (as in #9) with the largest damage;
u = the least-selected other expert. u is made a dormant backup: it gets e's router row and is masked out unless e is
knocked out, so it fills exactly e's slot when e is removed.
- **Positive:** u also gets e's expert weights (a perfect backup). Pass if (e, u) is a hidden pair for e's
  most-damaged stage and BI of the planted model exceeds BI of the unmodified model.
- **Negative:** u keeps its own weights (it takes e's slot but cannot do e's work). Pass if (e, u) is not a hidden pair.

## Predictions

H10a (backup hides CFOP's stage split):
- **P1:** at least 4 of 5 C models have at least one hidden pair.
- **P2:** at least 4 of 5 C models have BI greater than the largest K model's BI, in half A and in half B.

H10b (the pair carries CFOP's stage):
- **P3:** at least 4 of 5 C models have pair increment ≥ max(1.5 × the largest K model's increment, 0.02), in half A
  and in half B.

H10a needs P1 and P2 (and passing controls); H10b needs P3. A model whose quantity is undefined fails that prediction.
Judged separately.

**My expectation, stated now:** H10a fails (CFOP experts that matter tend to hurt several stages at once, so a cleanly
dedicated pair is unlikely), and H10b is uncertain.

## Reading rules, written before results

- H10a holds, H10b fails: single-expert knockouts understated CFOP's stage split because experts back each other up.
  Earlier "no split" statements must be qualified as "no split among single experts".
- H10b holds, H10a fails: the stage is readable from the selected pair, but removing experts does not isolate it.
  Routing encodes stage combinatorially without a stage-dedicated unit of computation.
- Both hold: the stage split exists but is distributed over pairs, in what is selected and in what is needed.
- Neither holds: the absence of a CFOP stage split survives the pair-level analysis; it is not an artifact of
  single-expert, top-1 measurement.

## Limits stated now

Final layer only; cooperation across layers is not tested. The margin scale removes softmax saturation but not the
nonlinearity of the final RMSNorm or of the max over other moves. Hidden pairs are searched over 28 pairs × the
stages of each model; the split-half rule guards against sampling noise, not structural artifacts, which is what the
negative control is for. The 0.25 rule is stricter with 4 stages than with 2; P1 is applied to C only, and K is
reported descriptively. Thresholds (0.05, 0.25, 0.5, 1.5, 0.02) are fixed here and not tuned.

## Review (before running)

An independent review of the first draft found, and the design above fixes:
1. The first null for backup (pair interaction under drop subtracted from that under reroute) was miscalibrated:
   dropping both selected experts zeroes the whole final expert output, so saturation alone could produce "backup".
   Replaced by within-reroute super-additivity required on both the p and the margin scale, plus planted controls.
2. The first pair measure (pair information ÷ top-1 information) had a ceiling set by the weak CFOP top-1 code, so it
   would have re-measured a known fact. Replaced by the increment over top-1, with 50 permutations, and a floor.
3. The first backup index took positive parts per stage, which favours the family with more stages. Replaced by a
   signed index pooled over all positions.
4. No positive control: added planted perfect / non-functional backups on a 2-stage and a 4-stage model.
5. Added: split-half requirement for every prediction, a rule for undefined quantities, one device for all models.
