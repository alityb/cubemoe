# PhaseSplit: do MoE routers recover the structure hidden in their training data?

We train mixture-of-experts policies to solve the Rubik's cube by imitating two solvers whose solutions have a known,
exact decomposition. The model is never given stage labels.

- **Kociemba two-phase:** phase 1 reaches the subgroup G1 = ⟨U, D, R2, L2, F2, B2⟩; phase 2 solves within it.
- **CFOP:** cross → F2L → OLL → PLL.

The question is whether the experts divide the work along that decomposition.

**TL;DR.** In cube solvers that genuinely solve the cube, the final-layer experts split cleanly along Kociemba's
phase boundary in 5/5 seeds: each expert that matters hurts one phase only (figure below). The router finds the phase
along a single linear direction, and steering that direction changes the moves. The same setup trained on CFOP does
*not* split by stage, even though the model follows CFOP's stages; why is still open.

![Removing one final-layer expert: every expert that matters hurts phase 1 or phase 2, not both](blog/figures/prod_fig2_knockout.png)

## Where things stand (2026-10-29)

The models are production-style MoE transformers (`src/model_prod.py`): Mixtral-style, 8 experts with top-2 routing,
6 layers, d=256, RMSNorm, RoPE, QK-norm and SwiGLU experts. Each one reads the 54 stickers of the current cube and
predicts the next move. There are 5 seeds per solver.

| | Kociemba models | CFOP models |
|---|---|---|
| Solve rate, beam 256 | 500/500 random cubes, 23.8 moves | 200/200 random cubes, 51–52 moves |
| Model, not the search, does the work | 86% of solution moves are the model's first choice; 0/30 for the same search with no model | 98–99% of last-layer moves are the model's first choice |
| Experts split along the decomposition? | **Yes**: knockouts dissociate phase 1 from phase 2 in 5/5 seeds | **No**: stage explains 4.5% of routing; which algorithm is running explains 44% |
| Mechanism | The router reads a linear phase direction (cos 0.82–0.86 with the router-weight gap, 5/5 seeds). Steering along it changes the moves (router-only steering in 4/5 seeds, full-layer steering in 5/5); a random direction has no effect | The model follows CFOP stage order and reuses the teacher's 7 last-layer algorithms. It chooses which side to perform them from, which makes its solutions 12–13 moves shorter than its teacher's on the same scrambles (47–49 vs 60.5) |

**Open question:** why does routing recover one decomposition and not the other? Two pre-registered explanations
both failed: "routing follows what changes the output" and "routing follows the plan" (`prereg/README.md`).

**RL follow-up (preliminary, 1 of 3 pre-registered seeds):** when GRPO improves the CFOP policy, training only the
routers changes routing (up to 22% of tokens switch expert) but not solving (+0.8 points); training only the experts
gives +44, about as much as training everything. Seeds 1–2 run in November. `blog/rl_routing.md`,
`prereg/PREREG_RL_ROUTING.md`.

**Write-ups:** `blog/phasesplit.md` (cube paper, draft) and `blog/rl_routing.md` (RL study, draft).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | All code. Key files:<br>• `model_prod.py`: the model<br>• `cube_fast.py`: vectorized cube<br>• `solve_fast.py`: beam search<br>• `prod_analysis.py`: analysis toolkit<br>• `mech.py`, `mech4.py`: mechanism experiments<br>• `prod_cfop.py`, `cfop_*.py`: CFOP analyses<br>• `routing_vs_*.py`: the two pre-registered tests<br>Earlier-generation scripts (sequence models, sweeps, H1/H2 analyses) are kept as they were run |
| `modal/prod.py` | Training, evaluation and path saving on Modal GPUs |
| `probe/` | Reference cube implementation, Kociemba and CFOP solvers, CFOP case tables |
| `llm/` | Follow-up on a real MoE language model (OLMoE-1B-7B, Granite pilot) |
| `prereg/` | Every pre-registration, with an index of verdicts |
| `docs/LAB_NOTEBOOK.md` | The full chronological record, including corrections and dead ends (§30–35 cover the latest work) |
| `docs/ARTIFACTS.md` | Every gitignored checkpoint and dataset, with sha256, origin and how to regenerate it |
| `out/` | Results (JSON) and logs. Checkpoints (`*.pt`) and datasets (`*.npz`) are gitignored |
| `blog/` | The write-up and its figures. `blog/archive/` holds the superseded first draft |

## Reproducing

```bash
# train (Modal, L40S): Kociemba, then CFOP
modal run --detach modal/prod.py::train_l40s --tag mixtral_6L256_s0 --preset mixtral --seed 0
modal run --detach modal/prod.py::train_l40s --tag cfop_mixtral_6L256_s0 --preset mixtral --seed 0 \
    --data cfop250k --rot keepD --budget 90
# solve tests
modal run modal/prod.py::evaluate --tag mixtral_6L256_s0 --n 500 --widths 64,256,1024
# analyses (local; need the checkpoint in out/prod/)
python src/prod_stage_a.py mixtral_6L256_s0              # model vs search, memorization
python src/mech.py mixtral_6L256_s0 both                  # experiments 1-2 (exp3 = steering)
python src/mech4.py mixtral_6L256_s0                      # where the phase direction is built
python src/prod_cfop.py cfop_mixtral_6L256_s0             # CFOP routing and knockouts
python src/cfop_verify.py 100                             # does the CFOP model solve the CFOP way?
```

Training data: `data/mixed250k.npz` (Kociemba) and `data/cfop250k.npz` (CFOP, `src/gen_cfop.py`). Both are gitignored;
their hashes are in `docs/ARTIFACTS.md`.
