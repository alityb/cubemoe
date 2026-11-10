# PRE-REGISTRATION #14: Does the stage carried by the CFOP routing path change the model's moves? (path intervention)

Written 2026-11-09. #13 found that the CFOP routing path *carries* stage information (0.40-0.52 of H(stage | S), any
single layer <= 0.15), though not beyond Kociemba-trained routers by the pre-registered margin. This asks the causal
question: if the decision token is routed along a path taken from a *different* stage, do its moves change more than
if it is routed along an equally different path taken from the *same* stage? Thresholds are set from a calibration on
the already-analysed held-out solves 0-1,199 (all 13 models; results below); the test is on solves never used. The
design was revised once after an independent review (end); the calibration was re-run after the revision.

## Positions and models

Confirmation: held-out solves **3,600-4,799** of the RandomState(0) held-out ordering of `cfop250k` (CFOP models; label
stage) and of `mixed250k` (Kociemba models; label phase), asserted disjoint from the training split. (The training gate
read the first ~3,300 held-out solves; these are past it.) Models: C = `cfop_mixtral_6L256_s{0..7}` on CFOP positions;
K = `mixtral_6L256_s{0..4}` on Kociemba positions as the **positive control** (their final-layer routing splits by phase
in every seed): it shows the test can detect a stage-specific route effect when one is known to exist. It is *not* a
scale reference: Kociemba's phase 2 restricts the legal move set, so a phase swap changes moves almost by construction,
while CFOP stages share the move vocabulary.

## Intervention (src/path_swap.py)

20,000 random recipients per model; for each layer set G in {all = L0-L5, L0-4, L5}, two donors from other solves with
the same moves remaining S: **cross** (different label) and **same** (same label), **matched on d = the number of donor
experts not in x's pair, summed over the layers of G** (d >= 1, the same d for both, drawn at random among the values
both pools offer, up to 64 candidates per pool; unmatched recipients are dropped and counted, by label). At every layer
of G the decision token's two experts are replaced by the donor's clean pair, weighted by x's own router probabilities
over them, renormalised; other layers and tokens route normally. Asserted in code: the patched model with no swap gives
the original model's logits exactly; > 99% of swaps change the logits.
Outcomes: **changed** = the argmax move differs from the clean argmax; KL(clean || swapped); drop in log2 p of the
solver's move; **probe shift** = change in P(cross donor's label) under a linear probe of the final decision-token
representation (fitted on clean positions of the other half), measured for both swaps on the cross donor's label.
Contrast D = mean(cross) − mean(same), paired by recipient; 95% CI by cluster bootstrap over recipient solves (1,000).
**Router-mass gap:** x's own router probability on the imposed pair (mean over G) is lower for cross than for same
donors; **mass-adjusted D** = the intercept of a regression of the per-recipient contrast on the mass gap (D at equal
mass), with a cluster-bootstrap CI.

## Calibration (solves 0-1,199; seen before writing this; corrected matching)

| set 'all' | D changed [CI > 0] | mass cross / same | mass-adjusted D | D L0-4 | D L5 | probe shift D |
|---|---|---|---|---|---|---|
| K s0-s4 | 0.204-0.401 (5/5) | 0.35-0.46 / 0.48-0.52 | 0.044-0.275 | 0.156-0.234 | 0.147-0.487 | +0.034 to +0.427 |
| C s0-s7 | 0.002-0.025 (7/8) | 0.465-0.548 / 0.470-0.554 | -0.001 to 0.014 | -0.011 to 0.026 | -0.009 to 0.057 | +0.001 to +0.026 |

Same-label swaps alone change the move in 0.20-0.31 of recipients (C), i.e. routing matters a great deal regardless of
stage. Clean move accuracy C 0.875-0.880, K 0.72-0.74; probe accuracy C 0.92-0.94, K 0.99. Matched recipients: C
~16,900, K 3,672-7,254. (A first calibration with layer-count matching, superseded, is kept in
out/path_swap_calib_v1_layer_matching.log.) The mass-adjusted CI was not computed in calibration.

## Predictions (on the confirmation solves)

- **S0 (positive control):** in >= 4/5 K models, D changed (all) >= 0.15 with CI > 0, and probe shift D (all) >= 0.03.
  (Lowered from 0.05 after the re-run calibration showed K s1 at 0.034; set before any confirmation data.)
- **S1 (a stage-specific causal effect exists):** in >= 6/8 C models, D changed (all) has CI lower bound > 0.
- **S2 (it is substantial):** in >= 6/8 C models, D changed (all) >= 0.05. Calibration predicts S2 FAILS (max 0.025).
- **S3 (beyond the final layer):** in >= 6/8 C models, D changed (L0-4) has CI lower bound > 0.
- **S4 (toward the donor's stage):** in >= 6/8 C models, probe shift D (all) has CI lower bound > 0.
- **S5 (beyond the router-mass gap):** in >= 6/8 C models, the mass-adjusted D (all) has CI lower bound > 0.
  Calibration point estimates (-0.001 to 0.014) suggest S5 may fail.

## Reading rules, written before results

- S0 fails: the test is insensitive; nothing about C is read.
- S1 fails: no detectable stage-specific causal effect of the CFOP routing path on moves.
- S1 holds, S2 fails: **a real but small effect**: routing the CFOP decision token along another stage's path changes
  its moves only slightly more than an equally different same-stage path does. S1 and S2 hold: a substantial effect.
- In either case S3 (whether the effect survives without the final layer), S4 (whether the representation moves
  toward the donor's stage) and S5 qualify it. S5 fails: the effect is accounted for by the recipient's router
  preferring same-stage donors' experts (imposed experts the router dislikes hurt more), not shown to depend on the
  stage beyond that. S5 holds: the effect survives at equal router mass.
- The Kociemba effect is not used as a yardstick for the size of the CFOP effect (see Positions and models).
Reported descriptively: per-set values for every outcome, per-half D, n matched (by label), mean d, per-layer pair
difference rates for cross and same donors, mass on donor pairs, and the same-label swap's own effect.

## Known weaknesses, stated now

- Thresholds are set from a calibration on positions that already showed the effect; the confirmation tests whether it
  holds on new solves. S2's 0.05 is a judgement (twice the largest calibration value).
- S0 shows the test detects an effect of ~0.2-0.4; it does not show sensitivity to effects of ~0.02.
- Swaps only the decision token; the 54 sticker tokens route normally, and the decision token's route affects them
  through attention.
- Donor pairs are taken from the donor's clean run; at later layers the recipient's own routing input has changed.
- Matching equalises the number of differing experts, not which layers they are in.
- The router-mass gap may be part of the mechanism (the router's preference is itself stage-related), so the
  mass adjustment may over-control; S5 is a qualifier, not the primary test. The intercept is an extrapolation to
  zero gap.
- The bootstrap resamples recipient solves only; donors are reused across recipients, so CIs are slightly optimistic.
- Matched recipients come from S values where labels overlap (few for K), not a random sample of positions.
- One probe per half; a linear probe of the final representation is a proxy for "acting like the donor's stage".

## Review (before running)

An independent review (3 must-fix for this prereg) found, and this version fixes: (1) matching on differing *layers*
left the cross donor with slightly more differing experts and lower router mass, enough to produce a D of ~0.006: now
matched on differing experts, with the mass gap reported, adjusted for (S5) and the calibration re-run; (2) the "swap
took effect" assert was a tautology and the clean-equality check was not in the code: both are now real asserts;
(3) the Kociemba effect is not comparable in scale (phase 2 restricts moves): the scale comparison was removed from
the reading rules. Also adopted: matched counts by label, readings for S3/S4 whether or not S2 holds, the bootstrap
caveat.
