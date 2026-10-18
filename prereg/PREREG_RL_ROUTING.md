# PRE-REGISTRATION — When RL improves an MoE policy, is the gain routing or expert computation?

Written 2026-10-18, before any RL training run. The only data seen beforehand is the base-model pilot on the
tuning seed (`out/rl/pilot_cfop_mixtral_6L256_s4.json`): greedy solves 8.5% of held-out scrambles; sampling at
temperature 1 solves 9.9% per sample; 96% of scrambles are solved at least once in 64 samples.

## Question

When reinforcement learning improves a mixture-of-experts policy, does the gain come from changing **where tokens are
routed** or from changing **what the experts compute**, and does the answer depend on problem difficulty? The working
hypothesis (H), from the cue literature (Wang et al. 2026, 2610.06851) and Shen et al. 2026 (2607.16097): gains on
problems the model already tends to solve are routing ("triggering"), and gains on problems it rarely solves need new
expert computation ("learning").

## Models and seeds

The five production CFOP policies (`out/prod/cfop_mixtral_6L256_s{0..4}.pt`; Mixtral-style, 8 experts top-2,
6 layers, d=256). **Seed 4 is the tuning seed** (pilot, learning-rate sweep) and is excluded from all confirmatory
results. **Confirmatory seeds: 0, 1, 2.** Seed 0 runs in October 2026; seeds 1–2 run after Modal budgets reset on
2026-11-01, with this document unchanged. Seed 3 is reserve, used only if a confirmatory run fails for technical
reasons, which will be reported.

## Arms (everything else frozen)

| arm | trainable | parameters |
|---|---|---|
| `router` | the 6 router matrices | 12,288 |
| `lora` | rank-4 LoRA on each layer's attention output projection (parameter-matched control) | 12,288 |
| `experts` | all expert weights; routers frozen | 33.0M |
| `full` | everything | 34.6M |

Note: freezing the router does not freeze routing (expert changes alter later routers' inputs), and with top-2
routing the router receives gradient only through the mixing weights of its selected experts. Both are measured.

## RL procedure (`src/rl_train.py`, defaults as committed)

GRPO, one on-policy update per batch: 64 training-split scrambles × 8 samples at temperature 1.0, reward 1 if solved
within 90 moves, else 0. Advantage = (r − group mean) / (group std + 1e-6). Loss = −Σ A·log π(a|s) / (#steps).
No KL term, no load-balancing loss. AdamW (betas 0.9/0.99, no weight decay), gradient clip 1.0, bf16 autocast,
300 updates. Training scrambles drawn fresh each update from the training split, seeded by RL seed and step.

**Learning rate**, chosen per arm on seed 4 only: grid router {1e-3, 3e-3, 1e-2}, lora {3e-4, 1e-3, 3e-3},
experts {1e-5, 3e-5, 1e-4}, full {1e-5, 3e-5, 1e-4}; 100 updates each; pick the rate with the highest mean training
reward over updates 81–100 (ties to the smaller rate). If the best rate is at the edge of its grid, extend the grid
by one step in that direction once and re-pick.

## Evaluation (`src/rl_eval.py`, as committed)

- **Sets:** 2,000 held-out generator scrambles (test split, budget 90) and 1,000 uniformly random cubes
  (`random_cubes(1000, 2026)`, budget 120).
- **Difficulty bins**, from the BASE model of each seed before any RL: per-scramble pass rate p over 64 samples
  (temperature 1, sampling seed 1). EASY: p ≥ 8/64. HARD: p ≤ 1/64. MID otherwise.
- **Outcome:** sampled pass rate, 16 samples per scramble (temperature 1, sampling seed 2, identical for every model).
  Gain Δ(arm, bin) = outcome(RL model) − outcome(base model re-evaluated by the same procedure). The base is always
  re-measured with fresh samples, so regression to the mean cancels.
- Secondary: greedy solve rate; the share of greedy episodes that reach F2L.

**Primary set:** held-out scrambles. The random cubes are secondary; if the held-out HARD bin has fewer than 50
scrambles in a seed, the random-cube HARD bin is used for H3 in that seed (stated in the report).

## Predictions (H)

All on the primary set, per confirmatory seed.

- **H1, easy gains are routing:** Δ(router, EASY) ≥ 0.7 × Δ(full, EASY).
- **H2, routing is special, not just a small update:** Δ(router, EASY) − Δ(lora, EASY) ≥ 0.25 × Δ(full, EASY).
- **H3, hard gains need expert computation:** Δ(router, HARD) ≤ 0.5 × Δ(full, HARD).

**Undecidable cases**, declared now:
- if Δ(full, EASY) < 3 points, H1 and H2 are undecidable;
- if Δ(full, HARD) < 2 points, H3 is undecidable.

**Verdict:** each of H1–H3 is supported if it holds in at least 2 of the 3 confirmatory seeds AND on the mean over
seeds. H is supported only if H1, H2 and H3 are all supported. Seed 0 alone is reported as preliminary and carries
no verdict.

## Mechanism measures (descriptive, no verdict)

On a fixed set of held-out solver positions, for every arm against its base:
1. Routing change in place: the share of decision tokens whose top-1 expert changes, per layer.
2. Routing change from the router weights alone: new router on base-model router inputs.
3. Expert-function change: base inputs and base routing, new experts; relative change in MoE output.
4. Where the greedy policy improves along CFOP: the share of episodes reaching cross, F2L and OLL.

## Known limitations, stated in advance

- One task (CFOP cube solving), small models; there is no cube analogue of a prompt cue.
- The parameter-matched control is one choice (attention-output LoRA); other 12k-parameter updates might behave
  differently.
- With 300 updates of 512 episodes, slow-learning effects may be missed.
