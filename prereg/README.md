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
| 5 | `PREREG_ROUTING_VS_ACTION.md` | 2026-10-16 (`d3a003b`) | Production models: does routing follow what changes the output? | **Failed** (rho 0.18) |
| 6 | `PREREG_ROUTING_VS_PLAN.md` | 2026-10-16 (`bc95fd7`) | Does routing follow the plan (future moves)? | **Failed** on both criteria |

"Rejected" / "failed" means the pre-registered criterion was not met. It does not mean the effect was shown to be absent.
