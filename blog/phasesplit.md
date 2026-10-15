# Mixture-of-experts layers in a learned Rubik's cube solver split along the solver's phases

**Ali Tayeb** · October 2026 · [code, data and checkpoints](https://github.com/alityb/cubemoe)

> **Summary.** I trained small Mixtral-style mixture-of-experts (MoE) transformers to imitate a two-phase Rubik's cube solver, without ever showing them which phase a position belongs to. With beam search, all five models solve 500 out of 500 uniformly random cubes, in solutions about as long as the solver they imitate. In all five, the experts in the final layer that matter divide cleanly by phase: removing a phase-1 expert costs 3–15 points of accuracy on phase-1 positions and at most 1.8 on phase-2 positions, and the reverse holds for phase-2 experts. The same models show that this kind of evidence is easy to get wrong: earlier versions of this project produced convincing routing results from models that could not solve the cube at all. A pre-registered test on OLMoE-1B-7B, a 7B-parameter MoE language model, finds routing patterns consistent with the cube results, but the absolute effects there are small. ⟨PENDING: one sentence on CFOP once the rerun is analysed.⟩ I read this as evidence that MoE routers can recover a decomposition that is latent in their training data, in a setting where the decomposition is known exactly, and not as evidence about what experts in large language models specialise in.

---

## 1. Why a Rubik's cube

Mixture-of-experts layers route each token to a few of many feed-forward "experts." Whether those experts specialise in anything meaningful is an open question, and recent work has made it more open, not less. Routers carry token-position information ([Bershatsky & Oseledets, 2025](https://arxiv.org/abs/2504.04444)), observational routing statistics do not predict which experts matter when removed ([Engmann et al., 2026](https://arxiv.org/abs/2606.10703)), and a pre-registered causal study of a 218B-parameter MoE found that most apparent expert "families" are not selective once the measurement is controlled ([Salomone et al., 2026](https://arxiv.org/abs/2606.25092)).

All of these studies share a limitation: in natural language there is no ground truth for what the experts *should* specialise in. Labels like "math" or "Arabic" are reasonable guesses, not facts about the model's training distribution. When a study finds weak specialisation, it cannot tell whether the specialisation is absent or whether it is organised along lines nobody thought to test.

Cube solvers remove that ambiguity. Kociemba's two-phase algorithm ([Kociemba](https://kociemba.org/math/twophase.htm)) splits every solve at an exactly defined point. Phase 1 brings the cube into the subgroup

$$G_1 = \langle U, D, R2, L2, F2, B2 \rangle,$$

the positions solvable with only those moves, and phase 2 finishes inside $G_1$. Whether a position is in $G_1$ is a deterministic function of the cube, and phase-2 solutions never use quarter turns of R, L, F or B. A model trained on such solutions has a known decomposition in its data. The question is whether its router finds it.

**Contributions.**

- A test bed with exact ground truth for MoE specialisation: cube-solving data with a known two-phase decomposition, released with code and 10 trained checkpoints.
- Evidence that production-style MoE models which genuinely solve the cube divide their final-layer experts along the phase boundary, replicated across 5 training seeds, with causal (knockout) evidence and controls for the main confounds.
- A record of how this kind of evidence goes wrong, including a full set of earlier results that came from models that could not solve the cube.
- A pre-registered test of three routing predictions on OLMoE-1B-7B.

## 2. Setup

**Data.** I use a reimplementation of the two-phase solver that finds an optimal phase-1 solution followed by an optimal phase-2 solution, checked against the `kociemba` Python package. It generated 250,000 solves (5.1M position–move pairs), half from uniformly random cubes and half from cubes 1–12 random moves from solved. On 100 random cubes the solver's solutions average **23.2 moves** (sd 1.2). This is longer than the full Kociemba algorithm, which keeps searching for shorter solutions.

Every dataset passes automated checks before training: solutions replay to solved, labels match a recomputation from the cube state, there are no duplicated scrambles, and the solver's move choices show no repetitive patterns. These checks caught two data bugs in earlier versions, each of which would have produced a convincing but wrong result.

**Augmentation.** Rotating the whole cube about its up–down axis, or turning it upside down, maps $G_1$ onto itself. These 8 rotations therefore map a two-phase solution onto another valid two-phase solution with the phase boundary at the same step. I verified this on 1,600 rotated solutions: all still solve the rotated cube and all keep the boundary at the same step.

**Model.** A Mixtral-style MoE transformer ([Jiang et al., 2024](https://arxiv.org/abs/2401.04088)): 6 layers, width 256, 4 attention heads, RMSNorm, rotary position embeddings, QK-norm, and SwiGLU experts. Every layer has 8 experts with top-2 routing, a load-balancing loss and a router z-loss, about 35M parameters in total. The input is the current cube (54 sticker colours plus a decision token) with bidirectional attention, and the output is the next move. The model is never given the phase.

**Training.** 60M position–move examples per model (about 12 passes over the data), AdamW, cosine schedule, bfloat16, on one L40S GPU (about 3.6 hours). Five seeds.

**Solving.** Beam search over the model's move probabilities. At each step it keeps the 64, 256 or 1,024 most probable move sequences, removes duplicate positions, and stops at the first solved cube or after 40 moves. Test cubes are 500 freshly sampled uniformly random cubes.

## 3. The models solve the cube, and the model does the work

| | seeds 0–4 |
|---|---|
| next-move accuracy on held-out solver positions | 72.8% ± 0.6 |
| solved, no search (always take the most likely move) | 5.1% ± 2.5 |
| solved, beam 64 | 479–499 of 500 |
| **solved, beam 256** | **500 of 500 in every seed**; 23.8 ± 0.2 moves |
| solved, beam 1,024 | 500 of 500 in every seed; 22.6 ± 0.1 moves |
| the solver the models imitate | 23.2 moves |

The model needs search: on its own it solves only about 5% of cubes. To check that the search is not doing the solving, I ran the same beam search with no model (all moves equally likely, width 1,024) and with an untrained model of the same architecture (width 256). Neither solved any of 30 random cubes; the trained model solved 60 of 60. Along the trained model's solutions, the move taken was its own first choice 86% of the time, its second choice 8%, and its third 3%. The search mostly recovers from occasional mistakes. None of the 500 test cubes appears among the 26.9M distinct training positions, counting rotations.

Attention matters. Removing it from every layer drops next-move accuracy from 73% to 9% and the solve rate to 0%. Replacing it with a uniform average over all stickers drops accuracy to 3%. Removing it from the first layer alone drops accuracy to 7%. ⟨PENDING: these are seed 0; seeds 1–4 running.⟩

## 4. Experts in the final layer divide by phase

*Figure 1. ⟨PENDING: one random cube solved by seed 0, move by move: which expert the decision token uses at each step, with the step where the cube enters $G_1$ marked.⟩*

**Routing.** At the final layer, the expert chosen for the decision token carries information about the phase even among positions the same number of moves from solved. Phase explains 21.9% ± 4.0 of the router's choice (range 17.8–27.1%), against 0.0% for a router with its choices shuffled within the same distance. In every seed one expert takes most phase-1 positions (58–81%) and almost no phase-2 positions (0.0–0.4%).

**Knockouts.** Routing that lines up with a concept does not show the model *uses* it. To test use, I removed each final-layer expert in turn by setting its router score to −∞, so tokens go to their next-best experts, and measured the drop in probability of the solver's move.

| seed | phase-1 experts: loss on phase 1 / phase 2 (points) | phase-2 experts: loss on phase 1 / phase 2 |
|---|---|---|
| 0 | e0: 8.6 / 0.0 · e7: 9.9 / 0.0 | e2: 0.0 / **30.2** |
| 1 | e6: 12.4 / −0.2 | e1: 1.8 / **10.2** |
| 2 | e1: 12.4 / 0.0 · e4: 9.3 / 0.0 | e3: 0.0 / **29.9** |
| 3 | e6: 11.7 / 0.0 · e3: 5.5 / 0.6 · e4: 5.1 / 1.0 | e5: 0.0 / **8.1** |
| 4 | e1: 14.7 / 0.0 · e5: 2.7 / 0.4 | e6: 0.0 / **12.4** |

Every expert whose removal costs more than 2 points is dominated by one phase. The largest effect on the other phase is 1.8 points (seed 1, e1), against 2.7–30.2 on its own phase. Each seed has exactly one phase-2 expert and one to three phase-1 experts. Removing any of the remaining experts costs nothing measurable in either phase.

*Figure 2. ⟨PENDING: knockout dissociation, all seeds: loss on phase 1 vs loss on phase 2 for every final-layer expert.⟩*

A second causal test, forcing phase-2 positions through the phase-1 expert, does not work in these models. Forcing the phase-1 expert removes 7.5–26% of the probability on legal moves, depending on the seed, but forcing a typical *other* expert removes 4–35%. Forcing any single expert onto the decision token is disruptive on its own, so this test cannot isolate expert identity. I report knockouts instead. The forcing test worked in the earlier, non-solving models (Section 7), and the contrast is a useful warning.

The phase split is not the only structure. ⟨PENDING after replication: do phase-1 experts split further into edge-orientation and corner experts, as some earlier models did (Thistlethwaite's first two stages)? In seed 0, removing e7 hurts edge-flipping moves about twice as much as corner-twisting moves, and e0 the reverse.⟩

## 5. What the experts respond to

A common worry is that experts are pattern matchers keyed to surface features of the input. Here the answer depends on the layer.

**Experts that process individual stickers start as pattern matchers.** In layer 0, a sticker token's own position and colour explain 93% of which expert it goes to. By layer 5 this is 45%. ⟨PENDING: seeds 1–4.⟩

**The decision token's routing tracks abstract progress, not single stickers.** In the last two layers, its expert choice is explained by phase (63–72%) and by the number of moves to reach $G_1$ (71–77%) about twice as well as by the most informative single sticker (36–37%). Layers whose routing barely varies are excluded from this comparison, because a share of almost no variation is not meaningful.

**The model represents distance to the goal beyond what the stickers state.** A linear readout of the raw stickers recovers several cube properties perfectly: whether the cube is in $G_1$, and how many edges or corners are misoriented (accuracy or $R^2$ 0.99–1.00). These are effectively counts over individual stickers, so finding them inside the model is not evidence of computation. Two properties are not linear in the stickers: the number of moves to reach $G_1$ ($R^2$ 0.85 from raw stickers) and the number of moves to solve ($R^2$ 0.77). Both are read out better from the model's middle layers, reaching 0.95 and 0.92 at layer 4. ⟨PENDING: seeds 1–4.⟩ Linear probes show the information is present, not that the model uses it.

## 6. A second decomposition: CFOP

⟨PENDING: results from 5 production-style CFOP models. Planned content: solve rate on held-out CFOP scrambles; whether final-layer experts divide by CFOP stage under knockouts the way they divide by Kociemba phase; whether they instead own specific moves of memorised algorithms, as earlier non-solving CFOP models did; with the stated limitation that the generator's "cross" stage is not a real cross solve.⟩

## 7. How this goes wrong

This project produced several results that looked right and were not. Each is a mistake other MoE interpretability work can make.

**Interpreting a model that cannot do the task.** For most of the project I studied models that read the starting cube plus the moves made so far. Their next-move accuracy looked reasonable, and their routing showed a clean phase split that passed pre-registered tests. They could not solve a single random cube. One move from solved, where the correct move is fully determined, they picked it 20% of the time. They had learned statistics of the solver's move sequences, not the cube. I found this only when I checked whether the models solve. Every routing result in Sections 4–5 comes from the later models that do.

**Position.** In sequence models, the phase is correlated with how far into the solve a token is. In those early models, a fake router that read only the step index scored inside the real router's range on the standard alignment metric. Every routing number there had to be measured among tokens at the same step.

**Forcing versus removing.** In the earlier models, forcing phase-2 tokens through the phase-1 expert destroyed 29% of legal-move probability, against 0.26% for a typical other expert, which made a clean causal result. In the solving models, the same test is uninformative because forcing any expert is destructive. Knockouts were informative in both.

**Load.** Forcing a set of tokens onto the expert that already handles most of them changes fewer routing decisions, so it does less damage for reasons unrelated to what the expert computes. In the earlier CFOP models this produced an apparent specialisation signal (p = 0.008) that disappeared once the effect was controlled.

**Readouts that prove nothing.** Several cube properties that sound abstract (orientation counts, membership in $G_1$) are linear functions of the input stickers. Probing for them inside a model shows nothing about computation.

**An off-by-one.** For several weeks my analysis read each move's routing one position late. This could only hide alignment, never create it. Fixing it made the scale-replication alignment 31% stronger and changed no conclusions.

## 8. Does this transfer to a language model?

The cube results suggest a simple account: routers follow the current token by default, and experts that matter appear where predicting the next output requires information the current token does not carry. I tested three consequences on OLMoE-1B-7B ([Muennighoff et al., 2024](https://arxiv.org/abs/2409.02060)), the model studied by Engmann et al. The predictions and pass rules were hashed and committed before the model was loaded. I split tokens into *needs-context* tokens, which the model predicts well with full context but not from the last two tokens alone (8,361 tokens), and *local* tokens, which the last two tokens already predict (1,084).

- **A. Routing follows the current token less on needs-context tokens.** Passed in 8 of 8 late layers.
- **B. Removing an expert hurts needs-context tokens more than random noise of the same size does**, compared with local tokens. Passed in 3 of 4 layers (1.75–2.40× vs 0.55–1.34×). It failed in the earliest layer tested, and it was weak in a pilot on a smaller model, Granite-3.1-1B-A400M.
- **C. Memorised text depends most on specific experts.** Passed in 3 of 4 layers, rising to 7.9× the matched noise at the last layer, against about 1.5–2× for prose and code.

The absolute effects are small. Removing one of OLMoE's 64 experts changes the correct token's log-probability by 0.001–0.004 nats on average, because 8 experts are active per token and they substitute for each other. The result concerns where experts matter relative to a matched control, not that any one expert is essential.

## 9. Related work

**Routers and latent structure.** [Dikkala et al. (2023)](https://aclanthology.org/2023.emnlp-main.583/) show that routers recover latent cluster structure in synthetic mixture data. [Nandakumar et al. (2026)](https://arxiv.org/abs/2606.14398) find that on synthetic knowledge tasks, training with the standard losses gives diffuse routing, and clear task–expert alignment appears only with an explicit task-routing loss. Here the phase alignment appears with only the standard load-balancing and z-losses. The difference may come from how sharply phase changes the output (phase-2 solutions use 10 of 18 moves), but I have not tested that.

**Expert specialisation in language models.** [Bershatsky & Oseledets (2025)](https://arxiv.org/abs/2504.04444) show that routers use token position. [Engmann et al. (2026)](https://arxiv.org/abs/2606.10703) find that observational routing statistics do not predict which experts matter when removed. [Salomone et al. (2026)](https://arxiv.org/abs/2606.25092) find that apparent modularity in a 218B MoE mostly dissolves under pre-registered causal tests, and their one surviving module (Arabic) concerns the output vocabulary. The cube setting complements these: it can say what the experts should specialise in, at the cost of being a small, synthetic task.

**Learned cube solvers.** [DeepCubeA (Agostinelli et al., 2019)](https://www.nature.com/articles/s42256-019-0070-z) and [EfficientCube (Takano, 2023)](https://arxiv.org/abs/2106.03157) solve the cube reliably with learned networks and search, with near-optimal solutions; EfficientCube, like this work, uses beam search over a learned move policy. The models here are weaker solvers and are not meant to compete (and solution lengths are not directly comparable, since EfficientCube's main results count a half turn as two moves); solving is used to confirm the models are competent before interpreting them. [Gupta et al. (2025)](https://arxiv.org/abs/2512.03400) train dense transformers on optimal solutions and study world-model probes; they do not study routing.

**Algorithmic interpretability.** Probing for an internal world state follows [Othello-GPT (Li et al., 2023)](https://arxiv.org/abs/2210.13382). Thistlethwaite's four-stage algorithm, whose first two stages make up Kociemba's phase 1, gives a second known decomposition to check against.

## 10. Limitations

- **One task family, small models.** Six layers, width 256, 8 experts. Whether anything here holds in large models is untested apart from the OLMoE predictions, where the effects are small.
- **Mixtral-style routing only.** I also implemented an OLMoE-style configuration (64 experts, 8 active), but on the available hardware it trained about 6× slower, so the cube models use 8 experts with 2 active.
- **The models need search.** Without beam search they solve about 5% of random cubes.
- **The decomposition is in the data by construction.** The models imitate a two-phase solver, so the phase structure is present in the training distribution. The claim is that routers recover it, not that they discover algorithms.
- **The analyses in Sections 4–5 were not pre-registered.** The knockout analysis was written after looking at seed 0 and frozen before seeds 1–4 finished training, then run on them unchanged. The OLMoE test in Section 8 was pre-registered.
- **Knockouts test necessity at one layer.** Middle-layer experts are largely redundant: removing any one costs at most a few points.
- **Probes are linear.** They show information is linearly present, not used.
- ⟨PENDING: CFOP-specific limitation (synthetic cross stage).⟩

## 11. Conclusion

In a setting where the right answer is known, a production-style MoE that genuinely solves the cube divides its final-layer experts along the phase boundary of the algorithm it learned from, and the division is causally used. The result replicates across five seeds. It also took several wrong turns to establish, and the most important control was the simplest: checking that the model can do the task before interpreting how it does it.

Code, data generators, checkpoints, pre-registration files and every result file are at [github.com/alityb/cubemoe](https://github.com/alityb/cubemoe).

---

### Appendix A. Earlier models

⟨To write: the pre-registered results from the earlier sequence models (confirmatory run, 10× scale replication, routing sweep, CFOP H2/H2b/H2c), clearly labelled as coming from models that could not solve the cube, including the top-1 routing result: with top-1 instead of top-2 routing, the phase specialisation disappeared while the task was still learned.⟩

### Appendix B. Training and evaluation details

⟨To write: full hyperparameters, data-quality checks, beam search details, knockout and probe protocols, and cost (about $70 of GPU time for the production models).⟩
