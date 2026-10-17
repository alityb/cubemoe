# PRE-REGISTRATION — does routing follow what changes the output?

Registered in commit `d3a003b` (the test was committed before it was run); result in commit `b3a3d37`.
**Verdict: FAILED (rho 0.18; phase not above stage).** Copied verbatim from the lab notebook (docs/LAB_NOTEBOOK.md), where it was first written.

**PRE-REGISTERED (2026-10-16 ~23:00, before running src/routing_vs_action.py): "routing follows what changes the output"**
Points = (solver, latent variable): Kociemba {phase, moves-to-G1, twisted corners, flipped edges, slice edges out};
CFOP {stage, segment, position-in-algorithm, first/mid/last, which algorithm}. Next move itself reported but EXCLUDED
from the test (x = 1 by definition). For each, mean over 5 seeds, last-layer decision-token top-1 routing, 1200 held-
out solves, conditioning S = moves remaining, permutation-corrected conditional MI (same estimator as before):
  x = I(V; next move | S) / H(next move | S)     (how much V determines the output)   PRIMARY predictor
  y = I(V; expert | S) / H(expert | S)           (routing share)
Prediction: Spearman rho(x, y) > 0.5 across the 10 points, and Kociemba phase above CFOP stage on BOTH axes.
Secondary (exploratory): x2 = I(V; move | S) / H(V | S). If rho <= 0.5 the thesis is not supported; report as such.
RESULT (out/prod/routing_vs_action.{json,log}): PRE-REGISTERED PREDICTION FAILED. Spearman rho(x, y) = 0.18 (p=0.63)
over 10 points (needed > 0.5); phase NOT above stage on x (2.3% vs 3.5%; y 21.9% vs 4.5%). Exploratory rho(x2,y) -0.45.
Exploratory observations (NOT pre-registered): within CFOP, x and y have the same rank order (which-alg 30/44, pos-in-alg
18.5/22, segment 5.6/6.4, first/mid/last 4.5/6.1, stage 3.5/4.5); within Kociemba, all five phase-family variables have
x 2-4% but y 20-25% -> Kociemba routing tracks the G1-distance latent far beyond its move information given moves-
remaining. Caveat for any follow-up: conditioning on moves-remaining removes most of phase's information (H(phase|rem)
0.12 bits) and the model never sees moves-remaining; an unconditioned version would be post hoc.
=> The thesis "routing follows what changes the output" is NOT supported; do not use it as the paper's claim.
