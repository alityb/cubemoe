# PRE-REGISTRATION — Is the RL gain in the openings? (a cue-style swap inside the cube MoE)

Written 2026-10-29, before any swap rollout. Companion to `PREREG_RL_ROUTING.md` (#7), whose RL checkpoints it reuses.
Already seen: #7's seed-0 results (RL gains: experts-only +44.2, full +46.0 points of sampled pass rate on held-out
scrambles; router-only +0.8). No swap has been run.

## Question

Wang et al. 2026 (2610.06851) found that most of an RL model's gain on MATH-500 is recovered when the RL model writes
only the opening tokens and the base model writes the rest. Is the cube RL gain likewise concentrated at **segment
openings**, or does RL change how the model executes throughout?

## Definitions

- **Progress score** of a position: cross done (1) + F2L slots done while the cross is done (0–4) + OLL done (1)
  + solved (1). Sticker tests in `src/rl_swap.py`; per-slot tests agree 100% with `probe/cfop.py` on 30,000 positions.
- **Opening step:** the first move of an episode, and every move made from a position whose progress score exceeds
  every earlier score in that episode (the first move of each new segment). This gives roughly 7 per solve.

## Conditions

Each condition is run on the held-out scrambles and bins of #7: 2,000 scrambles, 16 samples each, temperature 1,
sampling seed 2, budget 90.

| condition | who chooses the move |
|---|---|
| BASE | base model at every step (from #7's evaluation) |
| RL | RL model at every step (from #7's evaluation) |
| OPEN | RL model at opening steps, base model otherwise |
| RAND | RL model at randomly chosen steps, each step with probability q = OPEN's realised share of RL steps on that seed; base model otherwise |
| REV | base model at opening steps, RL model otherwise |

Arms: **full** (primary) and **experts** (secondary), at the learning rates of #7. Seeds: 0, then 1 and 2 when their
#7 checkpoints exist (after 2026-11-01).

**Outcome:** recovered fraction R(c) = (pass(c) − pass(BASE)) / (pass(RL) − pass(BASE)), with pass rates over all
2,000 scrambles. Per-bin values (EASY, MID, HARD) are reported but carry no verdict.

## Predictions

- **C1, cue-like openings:** R(OPEN) ≥ 0.5 AND R(OPEN) − R(RAND) ≥ 0.2.
- **C2, the converse:** R(REV) ≥ 0.5 (most of the gain survives with base-model openings).

**Stated expectation** (mine, not a criterion): C1 fails and C2 holds, because the base model's failures are loops in
the middle of F2L, not bad openings.

**Verdict:** C1 and C2 are each supported, for the primary arm, if they hold in at least 2 of 3 seeds AND on the
mean. Seed 0 alone is preliminary.
