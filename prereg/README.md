# Pre-registrations

Every confirmatory claim in this project comes from a test that was written down and committed before it was run.
Files are unchanged since they were committed (moved here from the repo root on 2026-10-16; content and sha256
are identical, so hashes quoted elsewhere still verify). Details of each run are in `docs/LAB_NOTEBOOK.md`.

| # | File | Registered | Question | Verdict |
|---|---|---|---|---|
| 1 | `PREREGISTRATION.md` | 2026-09-19 | H1: do routers recover Kociemba's G1 phase split? | **Confirmed** (4L/d128) |
| 1a | `PREREG_AMENDMENT_1.md` | 2026-09-19 | Amendment to H1 (layer exemptions, A1.3) | — |
| 1b | `AMENDMENT_2_SCALE_REPLICATION.md` | 2026-09-19 | H1 replication at 6L/d256, 250k solves | **Confirmed** |
| 1c | `AMENDMENT_3_PREFIX_GATE_SCALE.md` | 2026-09-20 | QA prefix gate depends on sample size | — (procedural) |
| 2 | `PREREGISTRATION_SWEEP.md` | 2026-09-20 | H_slack: does routing slack drive specialization? | **Rejected** (premise inverted) |
| 2a | `PREREG_SWEEP_AMENDMENT_4.md` | 2026-09-23 | Amendment to the sweep (concentration floor) | — |
| 3 | `PREREGISTRATION_H2.md` | 2026-09-26 | H2: do routers track CFOP's stages? | **Rejected** (position-confounded) |
| 3a | `PREREG_AMENDMENT_5_POSITION_DECORRELATION.md` | 2026-09-26 | H2b: does decorrelating position rescue it? | **Rejected** on the causal leg |
| 3b | `PREREG_AMENDMENT_6_STAGE_SPECIFIC_READOUT.md` | 2026-09-26 | H2c: stage-specific causal readout | Null, below its own resolution bar |
| 4 | `PREREG_LLM.md` | 2026-09-28 | Does the cube principle show up in OLMoE-1B-7B? | A 8/8, B 3/4, C 3/4 (small effects) |
| 5 | `PREREG_ROUTING_VS_ACTION.md` | 2026-10-16 (`62522ed`) | Production models: does routing follow what changes the output? | **Failed** (rho 0.18) |
| 6 | `PREREG_ROUTING_VS_PLAN.md` | 2026-10-16 (`3efba85`) | Does routing follow the plan (future moves)? | **Failed** on both criteria |
| 7 | `PREREG_RL_ROUTING.md` | 2026-10-18 | When RL improves an MoE policy, is the gain routing or expert computation? | **Not supported**: H1 0/3, H2 0/3 (H3 3/3, trivially) |
| 8 | `PREREG_OPENING_SWAP.md` | 2026-10-29 | Is the RL gain in the segment openings (cue-style swap)? | C1 (cue-like) **not supported** 0/3; C2 **supported** 3/3 |
| 9 | `PREREG_G1_PLL.md` | 2026-11-05 | Does a rotation-invariant move restriction (PLL in G1 moves) get its own expert? | **Not supported**: P1 3/4, P2 0/4, P3 3/4 |
| 10 | `PREREG_EXPERT_PAIRS.md` | 2026-11-08 | Do the two selected experts back each other up, and does the pair carry CFOP's stage? | **Not supported**: P1 0/5, P2 3/5, P3 0/5 (controls pass) |
| 11 | `PREREG_F2L_SPLIT.md` | 2026-11-08 | Replication on 3 new seeds: do CFOP models run cross + F2L on a standing expert pair? | **Partial**: P1 2/3, P2 2/3, P3 3/3 (team seed-dependent; s5 fails on F2L co-selection) |
| 12 | `PREREG_GEOMETRY_PATHS.md` | 2026-11-09 | Does each router read its label beyond the dominant geometry of its input, and is CFOP's stage carried by the expert path across layers? | Part A **Selection** (A1 4/5, A2 8/8, A3 5/8, A4 fails); Part B **void** (null out of bounds on Kociemba positions) |
| 13 | `PREREG_PATHS_FRESH.md` | 2026-11-09 | Replication of #12 Part B on fresh positions, with untrained-model and additive controls: does CFOP's routing path carry the stage? | **Not CFOP-specific** (B1 8/8, B1b 8/8, B1c 8/8, B2 5/8, B2r 8/8) |

"Rejected" / "failed" means the pre-registered criterion was not met. It does not mean the effect was shown to be absent.

**History rewrite, 2026-10-30.** Draft write-ups were removed from the repository's history with `git filter-repo`.
This changed every commit hash; hashes quoted in this folder, the notebook and the scripts were updated to the new
ones. Commit dates and file contents of every pre-registration are unchanged, and a bundle of the original history is
kept offline by the author.
