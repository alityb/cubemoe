# When RL improves a mixture-of-experts policy, does it change the routing or the experts?

**Ali Tayeb** · October 2026 · [code and results](https://github.com/alityb/cubemoe) · **Draft: preliminary, 1 of 3 pre-registered seeds**

> **Status.** This is a pre-registered study (`prereg/PREREG_RL_ROUTING.md`, committed before any RL run). The verdict requires three confirmatory seeds; only seed 0 has run so far. Seeds 1–2 run after 2026-11-01 with the protocol unchanged. Everything below about seed 0 is preliminary and carries no verdict.

> **Summary (seed 0).** I ran GRPO on a mixture-of-experts policy that solves the Rubik's cube, with only one part of the model trainable at a time: the routers, the experts, everything, or a LoRA adapter with exactly as many parameters as the routers. Training only the routers changed routing substantially, with up to 22% of tokens switching their first-choice expert, but improved solving by less than 1 point. Training only the experts improved it by 44 points, about as much as training everything. The gains were the same on easy and hard problems, and they landed where the base model was known to fail. In this setting, RL improved the model by changing what the experts compute, not where tokens go.

## 1. Question

Recent work suggests that much of what reinforcement learning adds to a language model was already reachable before training. Fixing one or two opening tokens of a base model's response (a "cue") recovers most of the RL model's accuracy on MATH-500, and RL mostly raises the probability of those openings ([Wang et al., 2026](https://arxiv.org/abs/2610.06851)). In chess, RL mostly amplifies moves the model already favoured on easy puzzles and surfaces rare correct moves on hard ones ([Shen et al., 2026](https://arxiv.org/abs/2607.16097)). Both studies use dense models.

In a mixture-of-experts (MoE) model, a cue-like change can be implemented in two ways. One is to send tokens to different experts, "triggering" computation the model already has. The other is to change what the experts compute, which is learning in the usual sense. The hypothesis tested here (H) is that RL's gains on problems a model already tends to solve come from routing, while gains on problems it rarely solves need new expert computation.

I test this on cube-solving MoE policies because the ground truth is known. From earlier work on these models, I know exactly where they fail without search: they loop during F2L, the second stage of the CFOP method.

## 2. Setup

**Models.** The five CFOP-trained policies from the companion study: Mixtral-style MoE transformers with 6 layers, width 256 and 8 experts with top-2 routing. They read the current cube and predict the next move. Seed 4 is used only for tuning; seeds 0, 1 and 2 are confirmatory.

**Arms.** In each arm everything else is frozen.

| arm | trainable | parameters |
|---|---|---|
| router | the six router matrices | 12,288 |
| LoRA (control) | rank-4 LoRA on each attention output projection | 12,288 |
| experts | all expert weights | 33.0M |
| full | everything | 34.6M |

The LoRA arm has exactly as many parameters as the router arm. It separates "routing is special" from "a small update is enough".

**RL.** GRPO with one on-policy update per batch: 64 scrambles × 8 sampled episodes at temperature 1, reward 1 for solving within 90 moves, group-normalised advantages, no KL term and no load-balancing loss. 300 updates. Each arm's learning rate was chosen on the tuning seed by a pre-registered rule.

**Evaluation.** I binned 2,000 held-out scrambles by the base model's own pass rate p over 64 samples: EASY p ≥ 8/64, HARD p ≤ 1/64, MID otherwise. The outcome is the pass rate over 16 fresh samples, and every gain is measured against the base model re-evaluated the same way. Binning on one noisy measurement and comparing with another would otherwise make hard problems look improved and easy ones worse with no training at all. All models are evaluated with the same random numbers.

**Pre-registered predictions.**
- H1 (easy gains are routing): router gain ≥ 0.7 × full gain on EASY.
- H2 (routing is special): router gain − LoRA gain ≥ 0.25 × full gain on EASY.
- H3 (hard gains need experts): router gain ≤ 0.5 × full gain on HARD.

## 3. Results, seed 0 (preliminary)

Base model: greedy solves 10.4% of held-out scrambles and a single sample solves 10.1%. Yet 96% of scrambles are solved at least once in 64 samples (tuning seed), so the base model almost always *can* solve; it fails by default.

Gain in sampled pass rate over the re-evaluated base, in points:

| arm | EASY (n=494) | MID (n=1,308) | HARD (n=198) | all | greedy | random cubes |
|---|---|---|---|---|---|---|
| router | +1.5 | +0.6 | +0.8 | +0.8 | +0.3 | +2.0 |
| LoRA (12k params) | +8.7 | +10.2 | +10.6 | +9.8 | +4.2 | +25.3 |
| experts | +42.0 | +45.0 | +44.1 | +44.2 | +29.0 | +67.3 |
| full | +43.1 | +47.2 | +45.9 | +46.0 | +30.1 | +69.8 |

- **H1 fails.** On easy problems, router-only RL gains 1.5 points against 43.1 for full RL.
- **H2 fails.** Router-only does worse than the LoRA control with the same number of parameters, so the problem is not that 12,288 parameters is too few.
- **H3 holds, trivially.** Router-only gains nothing on hard problems either.

Two further observations were not predicted.
- **The gains are flat across difficulty.** Experts-only RL improves easy, mid and hard scrambles by 42–45 points alike.
- **The gains land on the known weakness.** The share of greedy episodes that complete F2L rises from 12% to 46% (experts) and 49% (full), against 13% for router-only.

## 4. What changed inside the model

I measured three things against the base model, on 300 held-out solves:
- **Routing change in place:** the share of tokens whose top-1 expert changes.
- **Routing change from the router weights alone:** the new router applied to the base model's router inputs.
- **Expert-output change:** the relative change in a layer's MoE output, with the base model's inputs and routing held fixed.

| arm | routing changed in place, last layer | from router weights alone | expert output change, max over layers |
|---|---|---|---|
| router | 21.9% | 18.7% | 0% |
| LoRA | 14.2% | 0% | 0% |
| experts | 12.3% | 0% | 15.3% |
| full | 15.6% | 3.1% (max 3.9%) | 14.2% |

- **Re-routing did not help.** Router-only RL moved the routing more than any other arm and gained almost nothing. One might expect the router to be stuck, since with top-2 routing which experts get selected receives no gradient, only their mixing weights do. That is not what happened: the routing moved, and it simply was not what limited the model.
- **A frozen router does not mean frozen routing.** With routers frozen, experts-only RL still changed 12–14% of first-choice experts in the last layers, because the routers' inputs changed. So comparing arms by which weights were frozen is not the same as comparing them by whether routing changed. This is why routing has to be measured directly.
- **Given the choice, RL changes the experts.** In the full arm the router weights barely moved (at most 3.9% of first choices change from the weights alone), while the experts' outputs changed by up to 14%.

## 5. What this does and does not show

**What it shows (if seeds 1–2 agree).** In a setting with known ground truth, RL's large improvement came from changing expert computation. Re-routing, even substantial re-routing, did not reproduce it. The improvement fixed the specific competence the model lacked.

**Why it may not transfer to language models.**
- **Different failure mode.** The cube models' failures are loops in F2L; the problem is executing a plan, not reaching knowledge the model has. A language model whose correct behaviour sits behind an opening cue may be the case where routing suffices, and this study cannot rule that out.
- **No cue analogue.** The cube policy reads only the cube state and has no prompt, so the "base + cue" comparison has no counterpart here.
- **Learning rates.** The router arm's rate was chosen on the tuning seed. Router-only RL showed no gain at any rate tried (3e-4 to 1e-2), and at 1e-2 solving collapsed to 0%, so a better rate within this range seems unlikely.
- **Scale.** Small models and 300 updates.

**What it suggests for the language-model experiment.** Router-only RL is cheap: OLMoE-1B-7B's routers are about 2M parameters. It is therefore a fast first test of whether the cue paper's gains have a routing account. If router-only RL on OLMoE also changes routing without changing accuracy, routing is not the mechanism behind cue-like gains either. If it matches a cue on MATH-500, the cube and language settings differ in exactly the way described above.

## 6. Next

Seeds 1 and 2 run after 2026-11-01 under the same pre-registration. The verdict applies the pre-registered rules mechanically (`src/rl_report.py`). Each prediction needs to hold in at least 2 of 3 seeds and on the mean.

Code: `src/rl_train.py` (arms, GRPO), `src/rl_eval.py` (bins, evaluation), `src/rl_mech.py` (routing and expert change), `src/rl_report.py` (verdict). Results: `out/rl/`.
