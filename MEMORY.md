# PhaseSplit — project memory

Everything learned, decided, measured, and corrected. Written 2026-09-17.

**Question.** Do MoE routers recover the decomposition latent in their training data?
Case study: Kociemba two-phase Rubik's cube solutions, where the phase boundary (entry into
G1 = `<U, D, R2, L2, F2, B2>`) is a *theorem*, not a heuristic label.

**Status (2026-09-18, complete).** Iteration 2 + 5-seed replication + pre-registered contrast re-run
+ **causal intervention** + arm C to n=6.
Verdict: **cell C (H1 positive) on the sequence arm; cell D on the position-blind arm** (the latter
only because the change-point instrument structurally does not suit a history-free model — see §8f).

> ⚠ **RETRACTED:** an earlier version of this file reported "cell B2 — a sharpened null", built on
> *dense twin + k-means beats the MoE*. That was a **control mis-elevated to the claim**. The
> pre-registered gate was always router-vs-**nulls** (shuffled / position-only / hash twin /
> within-stratum permutation). Re-run against the correct contrast, the router clears every null
> decisively and the verdict flips from null to positive. See §8c.

---

## 0. The one-paragraph version

Raw `NMI(expert; phase)` looks impressive (0.15–0.37) and is meaningless: a synthetic router that
reads **only the step index** scores 0.259 on the same metric. That debunk is the durable result and
it is a router-vs-null statement, so it survives everything below.

Conditioned on step index, the router clears **every** pre-registered null on the sequence arm:
0.1370 ± 0.0140 vs shuffled 0.0013 (**107×**), vs a synthetic position-only router **0.0000**, vs the
trained token-identity hash twin 0.0609 (beaten 5/5 seeds), permutation p at the 0.005 floor in 20/20
(seed, layer) cells; and the change-point slope is +0.441 with p<0.01 in 5/5 seeds. Probe 0.986.
**That is cell C — H1 positive on the sequence arm.** The position-blind arm clears the same nulls
(111×, 3/3) but fails the change-point leg (1/3) → cell D.

The apparent caveat — k-means on the router's **own** pre-router hidden states scores 0.1389 vs the
router's 0.1370, a tie — was **resolved by the causal test (§8d-f)**, not by another correlation.
Forcing a phase-2 token through the phase-1 expert costs **7.5x more** G1-legality than an
equal-magnitude perturbation that collapses routing just as hard (5/5 seeds, p=0.031). A post-hoc
clustering is a *readout* with no causal role by construction, so the tie never undercut H1. The
effect is depth-specific (ratio 0.3 / 0.9 / 2.1 / 7.5 across layers 0-3, turning on exactly where
expert purity jumps 0.70 -> 0.98) and **reproduces at the same median magnitude (7.5x) on the
position-blind arm**, which rules out a positional explanation.

---

## 1. Hard-won facts about the tooling

| Fact | Value | Why it matters |
|---|---|---|
| `kociemba` (PyPI, muodov, C-backed) | **42.5 solves/s/core**, mean 20.73 moves | ~880 (state,move) pairs/s/core. Data gen was never the bottleneck. |
| `kociemba.solve` on an **already-solved** cube | returns a **13-move** maneuver, not `''` | "empty output" cannot be the solved test. |
| `kociemba.solve` on an illegal cube | raises `ValueError` | Free independent legality checker for QA layer 2. |
| `kociemba.solve` on a state **already in G1** | returns a pure-G1 solution only **6/200** times | You **cannot** compose two-phase data with off-the-shelf kociemba. Must write your own phase-2 solver. |
| `hkociemba/RubiksCube-TwophaseSolver` `phase1_prun` table | **>15 min, timed out, no partial cache** | Abandoned. Don't go down this road. |
| **Our own solver tables** | phase-1 **7.1 s**, phase-2 **16.0 s** | 3 small coordinates beat one huge pruning table by ~100×. |

### Our solver (the thing that made the project cheap)
- `probe/cube.py` — cubie model (cp/co/ep/eo), verified round-trip against `kociemba`.
- `probe/phase1.py` — coords twist(2187) × flip(2048) × slice(495); two BFS pruning tables
  (twist×slice, flip×slice). `lower_bound()`, `solve_phase1(state, L)` (exact depth),
  `solve_phase1_optimal()` (iterative deepening from the lower bound).
- `probe/phase2.py` — coords cornperm(40320) × udedge(40320) × sliceperm(24) over the 10 G1 moves.
  Optimal within G1 by iterative deepening.
- `src/solver2.py` — `solve_full()` (deterministic) + mixed start-state sampler.

Throughput: **8.2 solves/s/core** single-core pilot, but only **22/s wall on 10 workers**
(does *not* scale linearly — memory/table contention). 25k solves ≈ 19 min.

**Gotcha:** `probe/phase2.py` originally used a different face-index mapping than `phase1.py`.
Unified to `'URFDLB'.index(m[0])`. Phase 2 also now accepts an incoming `lastface` so the
no-consecutive-same-face rule holds *across* the phase split.

---

## 2. THE central finding: the position confound is structural

`P(phase | step index)` is near-deterministic because a two-phase solver's phase-1 length is
concentrated. Bayes-optimal phase accuracy from surface features (held-out rows):

| feature | stock (naive) | forced-L | mixed |
|---|---|---|---|
| majority baseline | 0.5292 | 0.525 | 0.663 |
| **position alone** | **0.9813** | 0.905 | 0.870 |
| move token alone | 0.7380 | 0.747 | 0.846 |
| prev move alone | 0.6934 | — | — |
| position + move | 0.9939 | — | — |

**It cannot be engineered away.** Forcing phase-1 length only moves it 0.980 → 0.907 even at
L~U(10,20) (seam sd 3.07); it asymptotes because phase-2 length grows with L, keeping the seam
proportionally central.

**Consequences that drive the whole design:**
1. Raw `I(expert; phase)` is meaningless; it appears in the report only as the thing being debunked.
2. The primary metric is `I(expert; phase | step index)` + stratified purity.
3. The load-bearing experiment is the **change-point test**: does the router's per-solve switch
   point move when the seam moves?
4. Phase is a **STATE** property (G1 membership). Position is a **DATASET** property. Never conflate.

Supporting citation: arXiv **2504.04444** documents that MoE routers can extract block index from
embeddings — i.e. positional routing is the expected null, not a surprise.

---

## 3. Data designs tried (in order)

### A. `naive` — stock kociemba. **KEEP.** This is Result 1.
25k solves, 519,467 steps. seam mean 10.99, **sd 0.65**, mixed step-cells 0.289, phase1 frac 0.529.
Labels are deterministic. Result: raw NMI 0.15–0.30 → **conditional 0.008–0.019**. The debunk.

### B. `forced-L` — phase-1 length forced to L~U(10,20). **DEAD. Do not repeat.**
seam sd 3.15, mixed cells 0.385 — good *spread*, but forcing L makes the next-move label
high-entropy (deep in phase 1, nearly any move continues *some* valid length-(L−t) solution).
All three architectures collapsed onto the same weak accuracy:

| model | acc | bar `maj|(phase,pos)`=0.1153 |
|---|---|---|
| moe_forced | 0.1392 | 1.21× |
| dense_forced | 0.1401 | 1.21× |
| hash_forced | 0.1382 | 1.20× |

**Lesson: padding the solution kills learnability.** Kept as the report appendix.

### C. `mixed` — seam variance from STATE SAMPLING. **CURRENT.**
50% uniform random states; 50% near-G1 (uniform random G1 state, scrambled by j~U(0,12) random
moves with no consecutive same-face). Every state solved **optimally and deterministically**
(single orientation, no inverse, fixed DFS order, iterative deepening from the phase-1 lower bound,
shortest phase 2).

25k solves, 514,586 steps. seam mean 6.94, **sd 3.44**, **mixed step-cells 0.534**, sol_len mean
20.58 (vs stock 20.78 — realistic, no padding).

**Why it's better:** a wide seam *and* a low mean means the seam distribution overlaps the
step-index distribution — **53.4%** of rows sit in step cells containing both phases, vs 38.5%
(forced) and 28.9% (naive). Roughly doubles usable signal for the conditional analysis.

**Known side effect:** phase balance shifts to **0.337 / 0.663** (near-G1 states have short phase 1,
but phase 2 stays ~13 moves). NMI is imbalance-sensitive → this is why anchors must be per-dataset.
Knob if needed: raise `jmax` toward 16–18 to push the seam up, at some cost to sd.

---

## 4. Bugs found — ALL of these would have produced a beautiful, wrong figure

### 4.1 DFS move-order stereotypy, part 1 (unigram)
Fixed-order DFS returned the first solution → phase-1 move marginal skewed **U=0.231, R=0.214**
(uniform ≈ 0.065), entropy 3.62 vs stock 4.15. Fix: shuffle move order per call. **Not sufficient.**

### 4.2 DFS move-order stereotypy, part 2 (cyclic padding) — caught only by the prefix metric
Per-*call* shuffling fixed the unigram marginal and the k=3 prefix rate, but the DFS still burned
excess length as a repeating 2-cycle at the **front** of phase 1:
```
F L2 F L2 F | U B' R F2 L' D' R' B' L
D U' D U' D U' D U' D U' | F B2 L2 D2 U R U' D2 F
```
`P(move[t]==move[t-2])` = **0.315** vs chance 0.056. Repeated-5-prefix rate **0.595** vs stock 0.039.
A router could split on "am I in the repetitive padding" and it reads as phase — and it sits at the
front, compounding the position confound.
**Fix: shuffle the move order at EVERY DFS node.** → lag-2 **0.057**, 5-prefix **0.013**. Also *faster*
(123 phase-1 solves/s).
**Now hard gates:** lag-2 ≤ 0.10, repeated-5-prefix ≤ 0.15.

### 4.3 Seam mislabelled (phase is a STATE property)
Generator set `seam = len(phase1_maneuver) = L`. But a length-L phase-1 path can enter G1 at **L−1**
— its last move being a G1 move mapping G1→G1. **295/25000 (1.18%)**, offset always exactly −1,
reproduced at 1.176% on regeneration (structural, not noise).
**Fix:** relabel `seam` to the true first-G1-entry (`src/repair.py`, now a pipeline stage). The
phase-2 ⊂ G1-move-set invariant survived with 0 violations, independently confirming the diagnosis.
**Now a hard gate.**

### 4.4 Tail leakage
**17.9–25.0%** of held-out states appear **verbatim** in training. Split-by-solve is not enough —
the last few states live in a tiny neighbourhood of solved. Profile (fraction leaked by
moves-remaining): `1–3: 1.00, 4: 0.99, 5: 0.73, 6: 0.22, 7: 0.04, 8: ~0.01`.

| exclude last N | leakage (mixed) | rows kept |
|---|---|---|
| 3 | 0.1223 | 0.85 |
| 6 | **0.0056** | 0.71 |

**Phase 2 *is* the tail**, so uncorrected phase-2 "specialization" is partly memorisation.
**Fix:** every routing metric computed on full held-out AND tail-excluded replicates at N=3 and N=6.

### 4.5 The change-point test was itself confounded
Uniform **random** routing scored naive slope **0.546, p=0.005** — because the fitted change point
sits mid-sequence and sequence length correlates with the seam.
**Fix:** report the **partial slope** of seam controlling for `sol_len`, and use a
**length-stratified** permutation null. Recalibrated: random → 0.260 (p=0.055).

### 4.6 Calibration anchors are dataset-specific — my absolute threshold was wrong
I initially used "cond NMI ≥ 0.05 ≈ phase tracker with ≤30% noise", calibrated on a synthetic
~50/50 seam distribution. **Wrong.** Perfect-tracker ceilings:

| dataset | phase-1 frac | ceiling (cond NMI) |
|---|---|---|
| mixed | 0.338 | **0.2161** |
| naive | 0.529 | **0.0465** |

On naive, 0.05 sat *above* the ceiling — unachievable by construction. (Why: seam sd 0.65 → position
nearly determines phase → almost no conditional entropy survives. The mixed design exists to create
that headroom, and gives 4.6× more.)
**Fix:** anchors computed on each dataset's *actual* (seam, sol_len) distribution;
gates are **anchor-relative** (≥ 30% of ceiling). Report `% of anchor`, never raw cross-dataset NMI.

### 4.7 The change-point test is UNTRUSTWORTHY on the naive set
On naive, a **pure position router** scores partial slope **+0.443, p=0.005** — a false positive,
because with seam sd 0.65 the length-stratified permutation has nothing left to permute.
**Change-point results are interpreted on the mixed set only.**
Also: the calibrated perfect-tracker slope on mixed is **0.523, not 1.0** (the fitter is biased
mid-sequence and mixed has many seams near 0). Figure captions must say "identity reference",
not "perfect phase tracker".

### 4.8 The interpretation matrix originally omitted the decisive null
As first coded, the cell logic compared only against shuffled and position-only routers. It scored
`moe_mixed` as **C (H1 positive)**. The dense twin + post-hoc k-means beats the MoE — so the honest
cell is **B2**. **Fix:** the dense-twin null is now decisive, and cell B2 was added.
*This single change is the difference between reporting a false positive and reporting the result.*

### 4.9 Minor
- `run_analysis.py` cleanup raised `KeyError: 'cp'` — the `*_shuffled` tail variants carry no
  change-point entry. Guarded. (Analysis had actually completed; all numbers were in the log.)
- `nohup ... &` under the Bash tool: the launcher shell exits immediately and reports "completed";
  the real job keeps running. Check with `pgrep`, not the task notification.
- Python `multiprocessing` on macOS uses **spawn**, so worker cmdlines don't contain the script
  name — `pgrep -f gen_data.py` shows only the parent. Don't mistake that for dead workers.
- Redirected Python stdout is block-buffered; use `python3 -u` or logs stay empty until exit.

---

## 5. QA layer (`src/qa.py`) — 16 hard gates, refuses to let training start

Runs on the `.npz` (not JSONL). Emits `out/datacard_{kind}.json` + console card. Exit code gates
training in the pipeline shell.

**L1 — invariants, every solve (25k, full replay):** maneuver replays to solved; every state passes
`verify()` (perm/twist≡0 mod 3/flip≡0 mod 2/parity); stored states == replayed states; phase-2 moves
⊂ the 10 G1 moves; `in_G1` monotone; **stored seam == first G1 entry**.

**L2 — independent verification (don't let the solver grade itself):** `kociemba` accepts every
sampled state; **`kociemba` solves the penultimate state in exactly our last move** (this is the
independent proof the maneuver terminates solved, given the solved-cube quirk); independent
facelet→cubie decoder cross-checks stored states; cubie G1 test == coordinate-table G1 test.

**L2b — label determinism (mixed only):** standalone re-solve **0.9950** (gate ≥0.95); re-solve
**with the `lastface` context 1.0000** (gate ==1.0). The 0.5% gap is *entirely* the
consecutive-same-face rule at the split — `(state, last-move-face) → next move` is exactly
deterministic. Note: model C sees only the state, so it faces that 0.5% as irreducible error.

**L3 — stereotypy:** n-gram entropy (1/2/3), repeated-prefix rate at k=3/4/5, lag-2 repetition,
distinct-maneuver count — all compared **forced/mixed vs stock**. Phase-2 stereotypy recorded too
(coverage was phase-1-only at first; phase 2 checks out: entropy 3.19 vs stock 3.22, lag-2 0.147 vs
0.143 — the above-chance lag-2 is a genuine property of optimal G1 solving, not tie-breaking).

**L4 — tail leakage:** split by solve, flag held-out states present in train, leakage by
moves-remaining and at cuts 3–8 with rows-kept.

**L5 — distribution:** realized seam histogram, seam sd ≥ 3.0, ≥9% mixed position cells, phase
balance, per-phase move histograms, zero duplicate scrambles, **code sha256 + seeds** for
bit-for-bit regeneration.

---

## 6. Analysis machinery

- **Metrics:** purity, NMI, ARI, MI(bits) — raw and stratified by step index (model C: by
  *remaining moves*). Primary = `nmi_strat_wt` (row-count-weighted within-stratum) + `cond_mi_bits`.
- **Purity conditioned on step index is uninformative** — strata are already ~98% phase-pure, so any
  assignment scores ~0.99. Computed, never headlined.
- **Change-point:** one per solve, maximising total-variation distance between the two segments'
  expert histograms; regressed on true seam. Report **partial slope** (controls `sol_len`) with a
  length-stratified permutation null.
- **Nulls (all six):** shuffled router; hash-router twin (trained, fixed token-identity routing);
  **dense twin + post-hoc k-means (DECISIVE)**; per-layer linear probe for phase conditioned on
  position; within-stratum label permutation (200 perms); synthetic position-only router.
- **Router health:** utilization = `Σ min(f_i, 1/E)` and expert entropy (2604.07030 metrics).
- **Calibration** (`src/calibrate.py`): synthetic routers (perfect tracker / +30% noise / +60% noise /
  pure position / random) run through the *same* pipeline on the dataset's real (seam, sol_len)
  distribution. Without this, "cond NMI = 0.02" is uninterpretable.

### Interpretation matrix (cells decided programmatically in `report.py`)
- **A** — probe low → concept never learned; uninformative, scale up first.
- **B** — probe high, routing ≈ null, slope ≈ 0 → *routing is positional/token-based until proven otherwise*.
- **B2** — probe high, routing ≫ shuffled/position nulls, **but ≤ dense twin + k-means** → the
  alignment is representation geometry, not routing. ← **we are here**
- **C** — probe high, cond NMI ≫ null, slope > 0 (p<0.01), beats the dense twin, survives in C → H1 positive.
- **D** — mixed evidence.

---

## 7. Models and training

**Compute:** MPS only (no CUDA), 10 CPU cores. Per spec, models reduced to **4L / d=128**
(from 6L / d=256). 8 experts, top-2, every FFN. Loss on **move tokens only**. Load-balance loss at
**whole-batch scope** (128 sequences), **no capacity cap, no token dropping**
(2604.07030: token-dropping prevents specialization; balancing *scope* matters more than method).

Arms: A = sequence MoE; B = dense twin (FFN width 8d to match top-2 active params); hash-router twin;
C = position-blind MoE (current 54 stickers → next move, bidirectional, CLS readout); C-dense.

**MoE implementation note:** all experts computed densely then combined via a scattered weight matrix
(`einsum('ne,end->nd')`) — wasteful but exactly implements "no dropping". Replacing a per-k gather
loop with this einsum took 3155 → 4976 tok/s.

### Learning gate (mixed). Bar = `majority|(phase,pos)` = **0.1439**, need ≥1.30× = 0.1871.

| model | acc | ratio | tok/s | wall |
|---|---|---|---|---|
| moe_mixed | 0.2017 | 1.40× ✔ rising | ~4.6–9k | ~12 min |
| dense_mixed | 0.2017 | 1.40× | 19373 | 238 s |
| hash_mixed | 0.1997 | 1.39× | 5037 | 915 s |
| **state_mixed (C)** | **0.5395** | 3.7× global majority | 19090 | 4001 s |
| state_dense_mixed | 0.5526 | — | 71390 | 1070 s |
| moe_naive | 0.1570 | vs bar 0.1200 = 1.31× | 7562 | 739 s |

**C fits far better** (0.54 vs 0.20) because it has no state-tracking burden — the sequence models
must infer the current cube state from [initial state + move history].

Other baselines worth keeping: mixed global majority 0.1151, `maj|phase` (oracle) 0.1327,
`maj|position` 0.1290. Naive: global 0.0931, `maj|phase` 0.1074, `maj|position` 0.1173, `maj|(phase,pos)` 0.1200.

---

## 8. FINAL RESULTS (Iteration 2)

Anchor (perfect phase tracker, mixed) = **0.2161**.

| model | cond NMI | % anchor | vs shuffled | partial slope | p |
|---|---|---|---|---|---|
| **moe_mixed** (learned router) | 0.1538 | 71.2% | 220× | +0.227 | 0.030 |
| **dense_mixed** (null: dense + k-means) | **0.1736** | **80.3%** | 134× | **+0.769** | **0.005** |
| hash_mixed (null: token-id router) | 0.0598 | 27.7% | 100× | +0.145 | 0.110 |
| state_mixed (C, learned router) | 0.0843 | 39.0% | 56× | +0.483 | 0.005 |
| state_dense_mixed (null: C dense + k-means) | **0.0920** | 42.6% | 77× | +0.338 | 0.005 |
| moe_naive (Result 1) | 0.0193 | 41.5% | 193× | +0.259 | 0.045 |

Linear probe for phase: **0.959–1.000** at every layer of every model.

**Cell B2 on both arms.** The dense twin beats the MoE on *both* metrics and on *both* arms.
Note `dense_mixed`'s +0.769 exceeds even the calibrated 0.523 perfect-tracker anchor.

> ⚠ **These are single-seed numbers.** See §8b — the ordering replicates 5/5, but the
> magnitudes above were optimistic draws. Use the seed-corrected figures.

---

## 8b. SEED STUDY (added 2026-09-17) — the verdict survives, the magnitudes shrink

Iteration 2 reported the decisive comparison from **one seed each**. Re-run with 5 seeds per arm,
both members evaluated on a **matched** held-out set (the original used 2500 solves for the MoE vs
1200 for the dense twin — nested but not properly paired). `np.random.seed` is now fixed alongside
`torch.manual_seed`.

**Sequence arm, conditional NMI (n=5) — PRIMARY:**

| seed | 0 | 1 | 2 | 3 | 4 | mean ± sd |
|---|---|---|---|---|---|---|
| MoE | 0.1514 | 0.1330 | 0.1154 | 0.1465 | 0.1387 | **0.1370 ± 0.0140** |
| dense twin | 0.1736 | 0.1401 | 0.1647 | 0.1748 | 0.1738 | **0.1654 ± 0.0147** |
| gap | +0.0221 | +0.0070 | +0.0492 | +0.0283 | +0.0350 | **+0.0283 ± 0.0156** |

- dense wins **5/5** → sign test one-sided **p = 0.031**; paired t **p = 0.0154**
- Wilcoxon p = 0.0625 — this is the *floor* for n=5 two-sided, so it is uninformative, not contradictory
- |gap| ÷ within-arm sd = **1.92**

**Change-point leg (n=5): NOT significant.** MoE 0.513 ± 0.129 vs dense 0.783 ± 0.189,
dense wins 4/5, paired **p = 0.081**. It *flipped sign at seed 1* (MoE 0.728 vs dense 0.596).
Report as directionally supportive, never as an independent second leg.

**Seed sd is ≈0.014–0.015 on both arms.** Accuracy, by contrast, is rock stable
(MoE 0.1996–0.2038 across five seeds) — the instability is in the routing metric, not in training.

**Position-blind arm C, conditional NMI (n=3) — DOES NOT REPLICATE, REVERSES:**

| seed | 0 | 1 | 2 | mean ± sd |
|---|---|---|---|---|
| C MoE | 0.0843 | 0.1368 | 0.1796 | **0.1336 ± 0.0477** |
| C dense twin | 0.0920 | 0.1334 | 0.1032 | **0.1096 ± 0.0214** |
| gap | +0.0077 | −0.0035 | **−0.0764** | **−0.0240 ± 0.0457** |

Dense wins only **1/3**; paired t p = **0.46**; |gap| ÷ seed sd = 0.50. The **MoE wins on average**.
Seed sd on the C MoE arm is 0.0477 — **36% of its own mean** — while its accuracy is stable to 0.001
across the same seeds. Nothing is concludable on this arm at n=3; it certainly does not support B2.

**Corrected headline numbers** (use these, not §8's single-seed ones):
- Sequence arm (defensible): MoE **0.137 ± 0.014** (~63% of anchor) vs dense **0.165 ± 0.015** (~77%).
- Position-blind arm C (inconclusive): MoE **0.134 ± 0.048** vs dense **0.110 ± 0.021**.

**What this retracts.** Iteration 2 claimed cell B2 on *both* arms from one seed each, and I used
"arm C agrees" as the argument that the null wasn't just a capacity artifact. That argument is dead:
arm C is the better-fit model (0.54 acc, probe 1.000) and it points the *other* way. The scope of
the claim is now the sequence arm alone.

**Fig 1's punchline:** in raw NMI, the **position-only router scores 0.259** — squarely inside the
MoE's 0.250–0.374 range. Raw NMI cannot distinguish a phase tracker from a position tracker.
Conditioned, position-only → 0.000, shuffled → 0.001, dense twin > MoE at every layer.

---

## 9. Novelty / prior work (verdict: GO-WITH-CHANGES; gap is real)

- **Dikkala et al., EMNLP 2023** (`2023.emnlp-main.583`) — canonical "routers recover latent
  structure"; ground truth = mixtures of Gaussians/subspaces/CIFAR-100 classes; **fails** on their
  dictionary-learning setting. Source of the shuffled-router baseline. *Closest prior work.*
- **MoE Routing Testbed**, Amazon AGI (`2604.07030`) — ground-truth reference router from 8 Wikipedia
  language domains; routing-purity metric; balancing **scope** > method; token-dropping kills
  specialization. *Our methods section.*
- **"The Myth of Expert Specialization in MoEs"** (`2604.09780`) — routers are linear maps, so hidden-state
  similarity explains expert usage; specialization is representation geometry. *Our adversary — and
  our result confirms it against an exactly-known ground truth.*
- **Gupta, Conklin, Leslie & Lee** (`2512.03400`, repo `prakharg55/CubeLM-EMNLP`) — **NOT "CubeLM"**;
  actual title "Better World Models Can Lead to Better Post-Training Performance". Dense GPT-2-style
  16L/16H/d1024, 3.6M **optimal** solutions from cube20.org, probes + steering + GRPO. Solve rate only
  **14.7–29.6%**. No MoE, no routing analysis; optimal solutions have no phase structure.
- **Emergent Compositional Skills in MoE VLAs** (`2607.20771`) — closest *in spirit* (router emergently
  sequences phase-level skills, no labels) but **purely qualitative**, no ground truth, no confound controls.
- Also: `2504.04444` (routers encode block index — the position confound), `2305.18390` (emergent
  modularity), `2210.13382` Othello-GPT, `2402.14083` Searchformer, `2404.03683` Stream of Search.

**Nobody has tested MoE routing against a known ground-truth decomposition of an algorithmic task.**

---

## 10. Repo layout

```
probe/   cube.py  phase1.py  phase2.py          # verified cube + coordinate solvers
         bench.py confound.py mitigate.py       # exploratory confound measurements
src/     solver2.py     # deterministic optimal solver + mixed state sampler
         gen_data.py    # kinds: mixed | forced | naive
         repair.py      # relabel seam -> first G1 entry (pipeline stage)
         qa.py          # STEP 1.5, 16 hard gates, emits data card
         model.py train.py
         extract.py run_analysis.py analyze.py calibrate.py
         figures.py report.py
         pipeline*.sh finish.sh                 # gen -> repair -> QA -> train -> analyse -> report
out/     results.md  results.json  datacard_*.json  calibration.json
         fig0_overlap.png fig1_conditional_nmi.png fig2_changepoint.png
         appendix_forcedL/                      # iteration-1 artifacts
data/    mixed.npz  naive.npz  forced.npz
```

---

## 11. If picking this up again

1. **Do not train on forced-L data.** Padding the solution kills learnability (§3B).
2. **Never report raw NMI** without the conditional number and the position-only router next to it.
3. **Always run the dense twin + k-means null.** It is the one that decides the verdict (§4.8).
4. **Recompute calibration anchors per dataset** — they are not comparable across sets (§4.6).
5. **Change-point only on high-seam-variance data** (sd ≥ ~3); it false-positives at sd 0.65 (§4.7).
6. Storing the *requested* L was never implemented — the card reports the realized seam histogram
   instead. One-line fix for the next regeneration.
7. The big open question the MVP cannot answer: **would a 6L/d256 model on CUDA with 250k solves
   change cell B2 to C?** `moe_mixed` at 1.40× its baseline is learning but not strongly. The
   position-blind arm C (0.54 acc, probe 1.000) is the better-fit arm and it *also* reads B2, which
   makes the null more credible than model size alone would suggest.
7b. **Always multi-seed before reporting any comparison.** Routing-metric seed sd is ≈0.015 while
   accuracy seed sd is ≈0.002 — a single seed looks reproducible on the metric you happen to watch
   and is not on the one that carries the verdict. `src/seedeval.py` + `src/seedstats.py` do this;
   `src/train.py` suffixes tags with `_s{seed}` for seed != 0.
8. H2 (CFOP, 4 stages) untouched. Stage labels are cheap state predicates; expect a *weaker* position
   confound than Kociemba because stage lengths vary more — unverified, ~2h to check.


---

## 8c. PRE-REGISTERED CONTRAST (2026-09-17) — supersedes §8 and §8b

**The error.** §8/§8b tested *router vs dense twin + k-means*. The pre-registered gate was
*router vs its nulls*: shuffled router, synthetic position-only router, the trained hash twin, and a
within-stratum label permutation (200 perms). The dense twin was null (iii) of six, and a
cross-model one at that. Substituting it produced a false null.

**Correct contrast, sequence arm A (n=5 seeds), best-NMI layer:**

| | value |
|---|---|
| router cond NMI | **0.1370 ± 0.0140** |
| shuffled null | 0.0013 → **107x** |
| synthetic position-only router | **0.0000** (a positional router carries no phase info once conditioned on position) |
| hash twin (token-identity routing) | 0.0609 → router beats it **5/5** |
| within-stratum permutation p | **0.005 floor in 20/20** (seed, layer) cells |
| change-point partial slope | **+0.441**, p<0.01 in **5/5** seeds at best layer; **4/5** at the pre-specified last layer (mean +0.384) |
| probe(phase \| position) | 0.986 |

→ **Cell C.** All three conditions pass.

**Arm C (position-blind, n=3):** router 0.1336 ± 0.0477, 111x shuffled, perm p at floor 3/3 — the
null-clearing leg passes. Change-point p<0.01 in only **1/3** (0.005 / 0.159 / 0.025). → **Cell D.**

**Within-model control (reported alongside, NOT the cell decision).** k-means (k=8) on each MoE's own
pre-router hidden states (`ln2` output = exactly the router's input), same layer:
- sequence A: router 0.1370 vs k-means **0.1389** — a **tie**; router wins only 2/5 seeds.
- arm C: router 0.1336 vs k-means 0.0893 — router wins 2/3.

So on arm A the router does partition on phase (vs all nulls) but **no more sharply than clustering
its own input would**. That is the live caveat on the positive result, and it is a question about
*causal use*, not correlation — not settled by any clustering comparison.

**Cross-model sanity only:** dense-model k-means 0.1535 (seq) / 0.0883 (C). Confirms the structure is
not MoE-specific. Not a finding.

**Honest caveats.** Change-point is p<0.01 in 7/20 (seed, layer) cells overall vs 5/5 at the best-NMI
layer; alignment and seam-tracking both concentrate in the deepest layers. The pre-specified
last-layer statistic gives 4/5, which is the number to quote if selection is a concern.

9. **The next experiment is causal, not correlational.** The within-model tie on arm A (router 0.1370
   vs k-means on its own pre-router hidden states 0.1389) cannot be resolved by any further
   clustering comparison. The question is whether expert identity is *used* to compute
   phase-appropriate moves, or merely *correlates* with phase. Test: at the best layer, force
   phase-2 tokens through the phase-1-dominant expert (and vice versa) and measure the drop in
   next-move accuracy / in probability mass on G1-legal moves, against a random-expert swap of equal
   magnitude as control. No training needed — the checkpoints exist.
10. **Never substitute a control for the pre-registered contrast.** Doing exactly this (dense twin +
   k-means in place of router-vs-nulls) produced a false null that survived a 4-hour seed study,
   two document rewrites, and a memory entry. The seed study was methodologically sound and
   answered the wrong question. Check *which comparison the gate names* before spending compute.

---

## 8d. CAUSAL INTERVENTION (2026-09-17) — resolves the §8c caveat

**The question §8c could not answer.** k-means on the router's own pre-router hidden states ties the
router on conditional NMI (0.1389 vs 0.1370). No further correlational metric can distinguish
"the router *reads* phase" from "the router's choice of expert *does* phase-appropriate work".

**Design.** At layer 3, force the top-1 expert at prediction positions; read out **probability mass
on the 10 G1-legal moves** in phase 2 (the correct move there is *always* G1-legal, so this is a
sharp phase-specific behavioural signature). Conditions:
- `self` — force the current top-1 (sanity; must equal baseline)
- `random` — force a random other expert
- `single` — force ONE unrelated expert, **matching swap's collapse of routing diversity**
- `swap` — force the opposite-phase expert

The `single` control is the load-bearing one: without it, `swap`'s damage could be explained by
collapsing 8 experts onto 1 rather than by expert identity.

**Result (n=5 seeds, layer 3, P(G1-legal | phase 2)):**

| condition | mean | mean drop |
|---|---|---|
| baseline | 0.9803 | — |
| self | 0.9803 | **0.0000** (exact in 5/5) |
| random | 0.9665 | 0.0138 ± 0.0088 |
| **single** (matched collapse) | 0.9768 | **0.0035 ± 0.0020** |
| **swap** (opposite-phase expert) | 0.9281 | **0.0522 ± 0.0580** |

- swap > single in **5/5** seeds; swap > random in **5/5**; sign test one-sided **p = 0.031** each
- **median ratio swap/single = 7.5x** (quote the median; the mean of 212x is an artifact of one
  seed's near-zero denominator)
- expert purity at layer 3: P(phase1|e1) = **0.982**, P(phase2|e2) = **0.998**, stable across seeds
  (the expert *indices* differ per seed; the structure does not)
- **Caveat:** the paired t-test is NOT significant (p = 0.134 vs single, 0.159 vs random). Seed 3 is
  a 4-8x magnitude outlier (drop 0.155 vs ~0.02-0.04) that inflates the variance. Direction is
  perfectly consistent; magnitude is highly seed-dependent. Report the sign test AND the t-test.

**Conclusion.** `single` collapses diversity exactly as `swap` does yet costs ~nothing, so the damage
is not from losing diversity nor from perturbation magnitude — it is specific to *which* expert
computes. **The router's partition is causally wired to phase-appropriate computation.** The
correlational tie with k-means and this causal dissociation are consistent and non-contradictory:
phase is linearly available in the residual stream *and* the router's use of it does work. A
post-hoc clustering is a readout with no causal role by construction; that is why the tie never
undercut H1 the way it appeared to.

**This is the strongest single result in the project** and the one a writeup should lead with:
*routing a phase-2 token through the phase-1 expert selectively destroys the model's tendency to
emit a legal phase-2 move, 7.5x more than an equal-magnitude perturbation that collapses routing
just as hard.*

## 8e. CAUSAL SWEEP — layer gradient and cross-model controls (2026-09-17)

| model / layer | P(p1\|e1) | P(p2\|e2) | baseline | Δrandom | Δsingle | **Δswap** | **ratio** |
|---|---|---|---|---|---|---|---|
| moe_mixed L0 | 0.70 | 0.97 | 0.9804 | 0.0167 | 0.0202 | 0.0068 | **0.3x** (reversed) |
| moe_mixed L1 | 0.70 | 0.99 | 0.9804 | 0.0156 | 0.0160 | 0.0138 | **0.9x** (null) |
| moe_mixed L2 | 0.98 | 0.99 | 0.9804 | 0.0119 | 0.0098 | 0.0203 | **2.1x** |
| moe_mixed L3 | 0.98 | 1.00 | 0.9804 | 0.0145 | 0.0049 | 0.0367 | **7.5x** |
| hash twin L3 | 0.97 | 0.89 | 0.9792 | 0.0029 | 0.0033 | 0.0069 | **2.1x** |
| moe_naive L3 | 0.97 | 0.99 | 0.9785 | 0.0051 | 0.0010 | 0.0071 | **7.0x** |

**1. The effect is depth-specific and tracks expert purity.** Purity is flat at 0.70 through layers
0-1, jumps to 0.98 at layer 2, and the causal ratio turns on at exactly that layer (0.3 -> 0.9 ->
2.1 -> 7.5). A perturbation artifact would be roughly constant with depth. Conditional NMI turns on
at the same layer (0.049 / 0.043 / 0.130 / 0.151) — correlation and causation agree on *where*.

**2. The hash twin shows a WEAKER but nonzero effect (2.1x).** Not a clean zero, and the reason is
principled: phase-2 moves are restricted to the 10 G1 moves, so **token identity is itself
phase-informative** and a token-identity router inherits partial phase purity for free. Its
conditional NMI is partial for the same reason (0.0609 vs 0.1370). The honest claim is quantitative:
**learning the router roughly doubles conditional alignment and more than triples the causal effect
over what token identity gives for free** (Δswap 0.0367 vs 0.0069, 5.3x).

**3. THE METHODOLOGICAL FINDING — the causal test beats conditional NMI as an instrument.**
The naive-set model shows a full-strength causal effect (**7.0x**, Δswap 0.0071 vs Δsingle 0.0010)
despite a conditional NMI of only **0.0193**. On that dataset the seam barely moves (sd 0.65), so
position nearly determines phase and the perfect-tracker ceiling is just 0.0465 — the correlational
metric has almost no headroom. The intervention does not pay that conditioning-headroom tax.

> **Consequence for any future work: run the causal intervention FIRST.** The entire `mixed` dataset
> design (near-G1 sampling, seam-variance engineering, the forced-L detour, the per-dataset anchor
> calibration) exists to buy headroom for a metric the causal test does not need. Had the
> intervention been run on the stock set on day one, most of that machinery would have been
> unnecessary. The seam-variance work is still needed for the *change-point* test, but not for
> establishing that routing is causally phase-relevant.

## 8f. ARM C COMPLETE (n=6) — the positional alternative is ruled out (2026-09-18)

| leg | sequence arm A (n=5) | position-blind arm C (n=6) |
|---|---|---|
| probe(phase \| strat) | 0.986 | **1.000** |
| router vs shuffled null | **107x**, 5/5 seeds | **80x**, 6/6 seeds |
| synthetic position-only router | 0.0000 | 0.0000 |
| **causal swap > single** | **5/5, p=0.031, median 7.5x** | **5/6, p=0.109, median 7.5x** |
| change-point p<0.01 | **5/5** | **1/6** |
| **cell** | **C** | **D** |

Arm C causal per seed (Δswap vs Δsingle): 0.042/0.086 (0.5x, the lone reversal), 0.082/0.013 (6.1x),
0.150/0.008 (18.7x), 0.046/0.005 (8.9x), 0.073/0.036 (2.0x), 0.279/0.006 (45.8x).
Mean Δswap 0.1118 ± 0.0904 vs Δsingle 0.0258 ± 0.0318.

**THE IMPORTANT INFERENCE.** On the sequence arm, phase correlates with position at 0.87, so the
"phase-1 expert" is also an "early-position expert" — the causal effect there could in principle be
positional damage, not phase damage. **Arm C sees only the 54 stickers of the current state: no
history, no step index, no positional information of any kind.** It reproduces the causal effect at
*the same median magnitude* (7.5x on both arms). The positional alternative explanation is therefore
ruled out: the effect is about phase, not about position.

This is what arm C was for, and it delivered — on the causal leg, not the correlational one.

**Why arm C still fails the change-point leg (1/6, slopes [0.48, 0.07, 0.27, 0.19, -0.10, 0.16]).**
The change-point test fits ONE switch point to a per-solve *sequence* of routing decisions. Arm C
classifies each state independently with no history, so its per-solve routing sequence has no
temporal smoothing and the TV-maximising change point is correspondingly noisy. This is a limitation
of the change-point instrument on position-blind models, not evidence that arm C fails to track
phase — its router clears the shuffled null 80x in 6/6 seeds and its probe is a perfect 1.000.

**Arm C is cell D by the pre-registered rule** (the change-point leg is part of the gate), but the
substantive reading is: nulls pass decisively, causal leans positive at identical magnitude, and only
the instrument that structurally does not suit a history-free model fails.
