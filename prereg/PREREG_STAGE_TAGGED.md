# PRE-REGISTRATION #16: If the outputs require the CFOP stage, does routing align with it? (stage-tagged outputs)

Written 2026-11-09, before any model below is trained. Already seen: everything in `docs/LAB_NOTEBOOK.md` up to §45.

## Why

Kociemba's routers split along its phases (#1, #12: the final router takes 21-40% of the best linear phase share,
3.8-15x a dominant-geometry null) and the split drives the moves (#14: a cross-phase route swap changes the move
19-35 points more than an expert-matched same-phase swap). CFOP's routers do not split along its stages, although the
stage is linearly present in what they read (#12: ceiling 0.60-0.67, router r 0.06-0.14), the six-layer path selects
for it (#15), and swapping in another stage's path does not change the moves (#14). One difference between the two
solvers is in the OUTPUT: Kociemba's phase 2 uses a restricted move set, so the next move depends on the phase; CFOP's
stages share the move vocabulary. Earlier tests of output-based explanations were partial: #5 (label-level
conditional MI of the move, "routing follows what changes the output") failed; #9 (PLL replaced by G1-move solutions)
gave PLL dedicated experts under knockout but routing did not single PLL out, with a stated confound (shorter,
optimal-solver PLL).

**Manipulation (one change only):** train on the same CFOP solves with **stage-tagged targets**: the model predicts
(stage, move), 72 classes, class = (stage − 1) × 18 + move. The inputs, the solver, the solutions, their lengths, the
move sets, the rotation augmentation and every hyperparameter are unchanged. The next output now depends on the
stage by construction, maximally (disjoint output vocabularies per stage). Solving and every analysis use the
marginal move policy (logsumexp over stages), so the model is used exactly as the original ones.

**H16:** when the output depends on the latent, routing aligns with it.

## Models

Four new models `cfop_tagged_6L256_s{0..3}`: `modal/prod.py::train_l40s --preset mixtral --data cfop250k --rot keepD
--budget 90 --tag-stage --seed s --max-hours 3.4` (the same command as the CFOP production models plus `--tag-stage`;
the 3.4 h stop is the #9 budget stop, reported if it binds). s0, s1 on Modal `ali-moh-islam-1`, s2, s3 on `dnfcubes`
(about $7 each; November headroom $18.1 and $17.4). Implementation (`src/model_prod.py` n_groups, `modal/prod.py`
tag_stage) committed with this file; existing models load and run unchanged (n_groups = 1).

**Competence gate:** beam 256 solves >= 95% of 200 held-out CFOP scrambles (`evaluate --n 200 --widths 256 --budget 90
--data cfop250k`), as #9. If fewer than 3 pass, the study is void; if fewer than 4 pass, "3/4" below becomes "all
passing models".

**Comparison group (already measured, same scripts and positions):** the untagged CFOP models
`cfop_mixtral_6L256_s{0..4}` (T1, as in #9) and `s{0..7}` (T2, T3, as in #12 and #14). Kociemba models are reported
for scale only.

## Measures (all on held-out solves, final layer unless stated, scripts unchanged)

- **y_stage** (`src/g1pll_analysis.py`, 1,200 held-out solves 0-1,199): permutation-corrected share of the final
  decision-token top-1 expert explained by stage given moves remaining. Old CFOP s0-s4: 2.8, 5.6, 5.4, 5.6, 3.0%
  (mean 4.48%); Kociemba phase ~22% (§35).
- **r and beyond-geometry** (`src/geom_paths.py` Part A, solves 0-1,199, given S): router's stage share / cross-fitted
  best linear readout's share, and whether it exceeds the PCA-subspace null's p99 and 1.5 x its median (#12's rule,
  both halves). Old CFOP r: 0.056-0.135, median 0.084; K 0.21-0.40.
- **D** (`src/path_swap.py`, held-out solves 3,600-4,799, the #14 positions): cross-minus-same route-swap contrast on
  the (marginal) argmax move, all six layers, expert-matched donors, CI by cluster bootstrap. Old CFOP:
  −0.006 to 0.025 (CI > 0 in 5/8); K 0.19-0.35.

## Predictions

- **T1 (routing shares):** in >= 3/4 passing tagged models, y_stage >= 2 × 4.48% = 8.96%.
- **T2 (selection beyond geometry):** in >= 3/4, final-layer r >= 2 × 0.084 = 0.168 AND beyond the PCA null in both
  halves.
- **T3 (the route drives the moves):** in >= 3/4, D (all) >= 0.05 with CI lower bound > 0.
- Descriptive tier: "Kociemba-like" if y_stage >= 15% and r >= 0.21 (K's minimum) in >= 3/4.

## Reading rules, written before training

- T1 and T2 hold: **making the output depend on the stage is sufficient for routing to align with it**; with T3, the
  aligned routing also drives the moves (as in Kociemba). Supports "routing aligns with a decomposition when the
  subtasks need different outputs".
- T1 and T2 fail: **output dependence is not sufficient**: even with disjoint per-stage output vocabularies the
  router does not select for the stage; the Kociemba/CFOP difference lies elsewhere (e.g. in the structure of the
  latent itself or how the representation builds it).
- T1 holds, T2 fails: routing becomes more stage-structured, but no more than the larger stage signal in the
  representation would give a geometry-reading router; not selection.
- T2 holds, T1 fails: selection at the logit level that does not reach top-1 routing shares; reported as such.
- T3 is read on its own: whether swapping a cross-stage route changes the moves beyond a same-stage route.

## Known weaknesses, stated now

- Tagging adds an implicit stage-classification objective, so the representation's stage signal (#12's ceiling) may
  rise; T2 normalises by the ceiling for that reason, T1 does not.
- The stage-tagged output is an extreme manipulation (disjoint vocabularies); a negative result rules out output
  dependence as sufficient, a positive result does not show that milder output differences (Kociemba's restricted
  move set) suffice.
- One architecture, one dataset, four seeds; the comparison group was trained without the 3.4 h stop (full 60M
  examples); if the stop binds, the tagged models see ~90%.
- T3's "move" is the marginal move; the stage tag itself is not scored.
- The data volume copies of `cfop250k.npz` on the two workspaces are the ones the comparison models were trained on;
  not re-hashed here.
