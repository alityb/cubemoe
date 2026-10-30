# PRE-REGISTRATION — does routing follow the plan (future moves)?

Registered in commit `3efba85` (the test was committed before it was run); result in commit `18fb1e9`.
**Verdict: FAILED on both criteria.** Copied verbatim from the lab notebook (docs/LAB_NOTEBOOK.md), where it was first written.

**PRE-REGISTERED (2026-10-16 ~23:25, before running src/routing_vs_plan.py): "routing follows the plan"**
Motivated by the failed test above (so the hypothesis is post hoc; the TEST is fixed here before computing).
Same 10 points, same y (routing share, already saved in out/prod/routing_vs_action.json), same S = moves remaining,
same estimator. For horizon j, use positions with moves remaining > j; m_{t+j} = the solver's move j steps later.
  P1 (primary): x_plan = mean_{j=0..9} I(V; m_{t+j} | S) / H(m_{t+j} | S).
      Prediction: Spearman rho(x_plan, y) > 0.5 over 10 points AND Kociemba phase > CFOP stage on x_plan.
      Known handicap: I(phase; . | S) <= H(phase|S) = 0.12 bits vs stage 0.54 -> P1 is structurally hard for phase.
  S1 (secondary, scale-free): persistence = mean_{j=1..9} I(V; m_{t+j} | S) / I(V; m_t | S).
      Prediction: every Kociemba point's persistence > every CFOP point's, AND rho(persistence, y/x) > 0.5.
If P1 and S1 both fail, the plan thesis is not supported -> write the modest paper.
RESULT (out/prod/routing_vs_plan.{json,log}): BOTH FAIL. P1 rho(x_plan, y) = 0.14, phase x_plan 0.34% < stage 2.65%.
S1: Kociemba persistence is the LOWEST (phase 0.05, fl/tw/so 0.10-0.19; d1 0.96 the exception) vs CFOP 0.69-0.87;
rho(persistence, y/x) = -0.77 (opposite sign). Given moves-remaining, phase's information about the solver's moves
vanishes within ~3 moves, yet routing tracks it at 22%.
=> Neither "routing follows the output" nor "routing follows the plan" (as label-level conditional MI) explains the
Kociemba/CFOP difference. Two pre-registered negatives; report both. Write the MODEST paper: Kociemba phases recovered
(with causal mechanism), CFOP stages not (despite behaviourally following them); what decides it is OPEN.
Methodological lesson (post hoc, flag as such): conditioning on moves-remaining — which the model never sees — makes
phase look output-irrelevant, while the mechanism shows the phase-2 expert causally constrains moves (-4.1 to -4.7
logits on non-G1 moves). Label-level MI given a confound-control variable can misstate what the model uses.
