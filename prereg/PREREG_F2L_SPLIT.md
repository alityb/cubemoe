# PRE-REGISTRATION #11: Do CFOP models run the first two layers on a standing expert team? (replication)

Written 2026-11-08. This is a **replication** of an exploratory observation, on models that did not exist when the
observation was made. The independent evidence comes from the new seeds; nothing about these seeds has been measured.

## Where the claim comes from

§40 (prereg #10, exploratory part): in all 5 CFOP models, one final-layer expert pair is co-selected on most cross
and F2L positions, and its knockout hurts both; the last layer (OLL + PLL) has no comparably dominant pair.

**Calibration** on those same 5 models with the committed `src/f2l_split.py` (outputs committed in
`out/f2l/calibration/`; exploratory, it is where the claim and the thresholds come from). Values are the lower of
the two cross-fitting directions:

| model | team | co-selected: cross / FL / F2L | team FL ÷ best LL pair's LL | team KO: FL points, FL÷LL, F2L÷LL, margin FL÷LL | LL pair = team? | best two-way cut (routing) |
|---|---|---|---|---|---|---|
| s0 | (5,7) | 0.76 / 0.49 / 0.40 | 1.99 | 20 / 2.39 / 2.51 / 2.26 | no | OLL\|PLL |
| s1 | (1,5) | 0.78 / 0.75 / 0.73 | 1.24 | 25 / 2.45 / 2.81 / 2.25 | yes | OLL\|PLL |
| s2 | (3,7) | 0.93 / 0.76 / 0.70 | 2.47 | 23 / 16.3 / 16.4 / 4.81 | yes | F2L\|LL |
| s3 | (4,7) | 0.98 / 0.72 / 0.64 | 1.67 | 20 / 3.50 / 3.57 / 3.08 | no | F2L\|LL |
| s4 | (3,6) | 0.79 / 0.67 / 0.63 | 1.98 | 22 / 4.51 / 5.08 / 3.68 | no | OLL\|PLL (0.150 vs F2L\|LL 0.147) |

In all 10 picks the same team was chosen in both halves.

What the calibration does NOT support, stated now:
- that routing information is organised at the F2L/last-layer boundary: the best two-way cut is OLL|PLL in 3 of 5
  models. Which cut routing follows is reported descriptively only;
- a last-layer unit: the most co-selected last-layer pair hurts the first two layers MORE than the last layer in every
  calibration model (LL ÷ FL damage 0.06–0.86). There is no double dissociation, so none is predicted.

## Models

Three new CFOP models, `cfop_mixtral_6L256_s{5,6,7}`, trained exactly as `cfop_mixtral_6L256_s{0..4}`
(`modal/prod.py::train_l40s --preset mixtral --data cfop250k --rot keepD --budget 90`, 60M examples), seeds 5–7,
with a 4.2 h safety stop (budget; reported if it binds). **Training was launched on 2026-11-08 at 16:13 EST, before
this file was committed;** training produces no measure used here, and no measure has been computed on these models.

**Competence gate:** beam 256 solves ≥ 95% of 200 held-out scrambles (`modal/prod.py::evaluate --n 200 --widths 256
--budget 90 --data cfop250k`; the report asserts n = 200). Fewer than 2 passing seeds: void.

## Measures (`src/f2l_split.py`; verdicts by `src/f2l_report.py`; both committed with this file)

Final layer, decision token, the same 1,200 held-out solves as every earlier analysis; halves A/B = even/odd solve.
FL = cross + F2L positions, LL = OLL + PLL. The **team** is the most co-selected unordered pair on FL positions;
the **best LL pair** is the most co-selected pair on LL positions. Both are chosen on one half and measured on the
other, in both directions. (With 28 candidate pairs and tens of thousands of positions this guards only against
selection on the same data; it is hygiene, not the replication.) Knockout = reroute (router logits −∞); damage in
p(teacher move) and in margin (teacher logit − best other logit).

## Predictions (each must hold in both cross-fitting directions of a seed)

Thresholds = 0.8 × the calibration minimum, one rule for every quantity.
- **P1 (a standing team):** the same pair is chosen as team in both halves, and it is co-selected on ≥ 0.61 of
  cross, ≥ 0.39 of FL and ≥ 0.32 of F2L positions (F2L alone, so cross cannot carry it).
- **P2 (the last layer has no more dominant pair):** team co-selection on FL ≥ 0.99 × the best LL pair's
  co-selection on LL.
- **P3 (the team is a first-two-layers unit):** team knockout costs ≥ 16 points on FL, and FL ÷ LL damage ≥ 1.91,
  F2L ÷ LL ≥ 2.01, and margin FL ÷ LL ≥ 1.80.

Per prediction: holds in all passing seeds and at least 3 seeds pass → **replicated**; holds in at least 2 →
**partial**; otherwise **not replicated**.

Reported descriptively: each two-way cut's share explained by the selected pair and by the top-1 expert; whether the
best LL pair is the team; the best LL pair's LL ÷ FL damage; full co-selection and knockout tables.

## Known weaknesses, stated now

- P3 largely follows from P1: a pair used mostly on FL will hurt FL more. Its content is that this holds by a
  consistent margin and on the margin scale, not that it could easily have failed.
- P2 is the weakest leg (calibration s1: 1.24, and its best LL pair is the team itself). At 0.99 it only says the
  last layer has no pair more dominant than the first-two-layers team.
- Three seeds; one family (CFOP); final layer only.

## Reading rules, written before results

- P1, P2, P3 replicated: CFOP models reliably run cross and F2L on one standing expert pair, and the last layer has
  no more dominant pair. This replaces "no stage split" as the positive description of CFOP routing (still not
  CFOP's four stages, and not a routing boundary at F2L/LL).
- P1 and P3 replicated, P2 partial or not: a standing first-two-layers team exists; whether the last layer is
  organised differently is not established.
- P1 replicated, P3 not: a standing pair is selected on FL positions but is not specifically needed there.
- P1 partial: the team recurs in some seeds only; report it as seed-dependent, not as a property of CFOP models.
- P1 not replicated: the §40 team does not recur and should not be reported.

## Review (before running)

An independent review of the first draft found, and this version fixes: the gate threshold was hardcoded to n = 200
(now asserted); the committed script had never been run and the calibration came from an older version and lacked
s4 (now rerun on all 5 with the final script and committed); thresholds were set ad hoc (now one rule, 0.8 × the
minimum); FL pooling let cross carry P1 (F2L-only conditions added); no rule for different teams across halves
(added to P1); damage only on the probability scale (margin added); with 2 passing seeds "replicated" was possible
(now needs ≥ 3); a guard that could make P2 fail in its most favourable case (removed); reading rules incomplete
(completed). Its point that cross-fitting does little here is accepted and stated above.
