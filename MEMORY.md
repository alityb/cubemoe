# PhaseSplit — project memory

Everything learned, decided, measured, and corrected. Written 2026-09-17.

**Question.** Do MoE routers recover the decomposition latent in their training data?
Case study: Kociemba two-phase Rubik's cube solutions, where the phase boundary (entry into
G1 = `<U, D, R2, L2, F2, B2>`) is a *theorem*, not a heuristic label.

**Status (2026-09-26).** **H1 confirmed at two scales; H_slack REJECTED (§11); H2/CFOP REJECTED (§12);
H2b/position-decorrelation REJECTED on its causal leg (§13); H2c/stage-specific readout finds nothing but
missed its own resolution bar (§15).** All six pre-registered programmes complete. Verdicts come ONLY from pre-registered runs on fresh seeds.

**Headline, in one line:** a router recovered the Kociemba G1 decomposition robustly and causally
(~270x causal effect at scale), and **failed** to recover the CFOP 4-stage decomposition its own
training data was built from — because CFOP stage is 86% determined by token position, so the network
gets it free and the router never encodes it. H2b then **manipulated** position-availability directly (cross widened to 0-25, landing at 57.3%
removed-by-position vs Kociemba's 57.9%). Result: routing **alignment** with stage appeared, 5/5 seeds,
2.4x above its own shuffled null — but **no causal evidence it is used** (is_native coef -0.0014,
p=0.92 after the mandatory artifact control), so H_pos is REJECTED per A5.4. That null is
**underpowered** (observed +0.0094 vs MDE 0.0239) and its causal readout is the design's weak link (§13).

**Therefore: position-decorrelation is sufficient to make routing ALIGN with a decomposition, and is
NOT shown sufficient to make routing causally IMPLEMENT it.** The general claim "MoE routers recover the
decomposition latent in their training data" is **unsupported**: the one demonstrated instance of causal
recovery remains G1 alone. The original H2 framing is not supported, and neither is its converse.

| run | scale | seeds | arm A | arm C |
|---|---|---|---|---|
| confirmatory | 4L/d128, 25k | 10-14 | **CONFIRMED** (L1-L4) | **CONFIRMED** (L1+L2+L4; L3 fails, exempt per A1.3) |
| scale replication | 6L/d256, 250k | 20-24 | **CONFIRMED** | **CONFIRMED** |
| routing sweep (§11) | 6L/d256, 250k | 30-34 | top-1 **collapses** (9% of ceiling) | n/a |

Pre-registration `dc0cf984` + Amendment 1 `78b97ac5` (both filed pre-results),
Amendment 2 (scale protocol, post-results, governs only the replication),
Amendment 3 `9fb968e0` (5-prefix QA gate is n-dependent; estimator subsampled, threshold unchanged).

> ⚠ **RETRACTED:** an earlier version reported "cell B2 — a sharpened null", built on
> *dense twin + k-means beats the MoE*. That was a **control mis-elevated to the claim**; the
> pre-registered contrast was always router-vs-nulls. Everything in §0-§8f below is **EXPLORATORY**
> and superseded by §9.

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


---

## 9. CONFIRMATORY + SCALE REPLICATION (the only claims)

Pre-registered before any confirmatory model existed. Last layer only, no selection, fresh seeds,
gates applied mechanically by `src/confirm_gates.py`.

### Arm A (sequence) — all four legs, both scales

| leg | 4L/d128, 25k, seeds 10-14 | 6L/d256, 250k, seeds 20-24 |
|---|---|---|
| L1 acc ratio / probe | 1.397-1.410 / 0.986 | **2.428-2.481 / 0.999** |
| L2 cond NMI vs shuffled | 0.071-0.192 vs ~0.0012 | **0.168-0.215 vs ~0.0014** |
| L2 vs hash twin | beaten 5/5 | beaten 5/5 |
| L3 b1 | +0.370..+0.657, p<=0.01 5/5 | **+0.234..+0.441, p=0.005 5/5** |
| L4 causal | 5/5, sign p=0.031, t p=0.037 | **5/5, sign p=0.031, t p=0.025** |

### Arm C (position-blind) — L1+L2+L4; L3 reported, not required (A1.3)

| leg | 25k | 250k |
|---|---|---|
| L1 | 3.659-3.719 / probe 1.000 | **4.856-4.895 / 1.000** |
| L2 cond NMI | 0.119-0.170 | 0.064-0.098 |
| L3 b1 | 1/5 at p<0.01 | **0/5** (one negative slope) |
| L4 causal | 5/5, t p=0.011 | **5/5, t p=0.0025** |

### What scale changed
- **The causal effect sharpens ~30x.** Arm A's strongest seed: L(e1)=0.488 vs median-other 0.0018
  (~270x) at scale, versus 0.045 vs 0.005 (~9x) at 25k. Forcing a phase-2 token through the phase-1
  expert destroys ~half of all G1-legal probability mass.
- **Underfitting caveat closed**: probes ~1.000, accuracy 2.4-4.9x its bar.
- **Arm C's L3 failed again** on independent seeds at 10x data — as A1.3 predicted *from the design
  of the test*, filed before results. The change-point instrument fits one switch point to a per-solve
  routing sequence; a history-free model has no temporal smoothing.

### Process notes worth keeping
- `confirm_gates.py` had `SEEDS` hardcoded to 10-14 and silently reported "incomplete: 0 seeds" on the
  scale run. Parameterised `--seeds/--dir`; **verified the original run still reads CONFIRMED before
  trusting the new numbers.** A gate script that reports "incomplete" rather than crashing is a
  silent-failure risk.
- Scale run split across two Modal workspaces for billing only (`ali-moh-islam-1` arm A + hash,
  `dnfcubes` arm C). Dataset generated once and copied byte-identical (sha256 `80ea6303...`) rather
  than regenerated, so both arms share exactly the same file.
- **Cost discipline:** an initial 10-way-concurrent A100 launch burned ~$11 and banked ONE model,
  because all containers died together when stopped. Low concurrency (groups of 2-3) banks completed
  models as they finish. Measure throughput with a ~150-step probe (costs cents) before projecting
  spend — my unmeasured estimates were wrong by 10x in one direction and 3x in the other.


## 10. SCALE CONTROLS + concentration (seeds 20-24, 6L/d256, 250k)

Perfect-tracker reference ceiling on the 250k set = **0.2174** (a reference construction,
not a mathematical maximum — values can exceed 100%).

| arm | cond NMI | % ceiling | conditional purity | within-model k-means | router beats k-means |
|---|---|---|---|---|---|
| A | 0.2009 ± 0.0183 | **92%** | 0.954 | 0.2111 | **1/5** |
| C | 0.079 ± 0.013 | 36% | 0.992 | 0.103 | 0/5 |

### Per-layer causal effect (arm A) — turns on with purity, grows with depth
| layer | purity(e1) | L(e1) | median-other | ratio | concentration |
|---|---|---|---|---|---|
| 0 | 0.689 | 0.0017 | 0.0010 | 2x | 0.269 |
| 1 | 0.736 | 0.0013 | 0.0007 | 3x | 0.344 |
| 2 | 0.805 | 0.0468 | 0.0028 | 26x | 0.630 |
| 3 | 0.899 | 0.0825 | 0.0043 | 26x | 0.578 |
| 4 | 0.995 | 0.3852 | 0.0108 | 68x | 0.714 |
| 5 | 1.000 | 0.2898 | 0.0029 | 127x | 0.630 |

**Concentration index** = share of total causal effect on the single most-affected expert
(even = 0.125). At the last layer: **0.630** — one expert carries ~2/3 of it.

### Hash twin at scale — and a statistic that breaks
Absolute `L(e1)` = **0.0031** vs learned router
**0.2898** — a **94x** separation.
**The RATIO is unusable for near-null arms**: hash median-other is at or below zero per seed
([0.000237, -1.9e-05, 0.000707, -1e-05, 0.001371]), producing -92,380x and +354,077x on effects
indistinguishable from zero. **Report absolute L(e1); quote the ratio only when median-other > 0.001.**
This defect was found before the sweep ran and its P3 endpoint was changed to absolute L(e1).

### The headline this supports
Specialization is **concentrated** (0.63 on one expert), **causally load-bearing** (L(e1) 0.29 vs
hash 0.003), **grows with scale** (cond NMI 63% -> 92% of ceiling; L4 ~9x -> ~127x), and is
**understated by NMI/purity** — both are saturated (purity ~0.95 for *any* assignment; k-means beats
the router on cond NMI 4/5) precisely where the causal test is most discriminating. Correlational
metrics show phase is *recoverable* from hidden states; only intervention shows it is *used*.

---

## 11. ROUTING SWEEP — H_slack REJECTED (2026-09-21..25)

Pre-registered: `PREREGISTRATION_SWEEP.md` `c1b023f4` + `PREREG_SWEEP_AMENDMENT_4.md` `2f9add95`.
6L/d256, 250k, seeds 30-34, last layer, same L1-L4.

| arm | routing / experts / LBL scope | cond NMI | % ceiling | L(e1) | Σ_e L(e) |
|---|---|---|---|---|---|
| baseline | top-2 / 8 / large | 0.2009 | **92%** | +0.2898 | 0.3100 |
| **V6** (arity only) | top-1 / 8 / large | 0.0198 | **9%** | −0.00003 | 0.0002 |
| V1 (confounded) | top-1 / 8 / local | 0.0178 | 8% | +0.00004 | 0.0002 |
| V7 (scope only) | top-2 / 8 / local | **NOT RUN** — budget exhausted | | | |

**Verdict: H_slack REJECTED; its premise is inverted.**
- P1 (concentration, PRIMARY): **UNEVALUABLE** — Σ_e L(e) 0.0001–0.0012 vs the pre-stated 0.05 floor.
  There is no causal effect to concentrate.
- P2 (ceiling-normalised cNMI): **REVERSED** — 0/5 pairs in the predicted direction, p < 0.0001.
- P3 (absolute L(e1)): vacuously true (5/5, p=0.024) only because L(e1) ≈ 0.

**Top-1 routing does not spread specialization — it eliminates it.** Top-2 appears *necessary* for
phase specialization to exist at all at this scale. V6 trains normally (acc ratio 2.36–2.43, matching
baseline 2.43–2.48), so this is a model that learns the task **without routing on phase**.

**Decision matrix (A4.5):** V6 collapses on arity alone → arity is sufficient; V1's collapse needs no
appeal to scope. Whether scope *independently* also suffices is **unresolved** (V7 unrun).

### Process notes
- The A4.3 concentration floor was **pre-registered but not implemented in code**; the first pass
  reported P1 as "inconclusive (4/5, p=0.45)" from noise-derived values. Floor now enforced in
  `sweep_analysis.py`. **A pre-registered rule that lives only in prose will not be applied.**
- Two client-side DNS failures killed `modal run` mid-arm. `--detach` protects already-dispatched
  work but NOT the group loop, which ran on the client — dispatch all seeds in one `starmap`.
- Local-scope LBL was 2.8x slower than large-scope (4,118 vs 11,579 tok/s) because it looped over
  128 sequences x 6 layers in Python. Vectorised, verified identical (1.06878528 vs 1.06878531).
- Cost estimates were wrong three times (10x high, 3x low, 3.5x low). A valid probe is **one complete
  model timed end-to-end**, not a steady-state inner loop.

### Budget
All three Modal workspaces exhausted (~$89 total spent). Credits reset monthly.
Remaining unrun: **V7** (~$6), **H1 hash twin for top-1** (~$6), **H2/CFOP** (~$30-50).

## 12. H2 / CFOP — REJECTED, and the reason is a position confound (2026-09-26)

Governing doc: `PREREGISTRATION_H2.md` (`c5d6b4ab`). 5 MoE seeds 40–44 on CFOP data (acc 0.6723–0.6738
vs `maj|(stage,pos)` = 0.1986, a 3.4x ratio, well over the 1.30x L1 gate) + 5 hash twins (0.6662–0.6739).
All 9 CFOP QA gates passed, including the four segment-boundary predicates at 10,000/10,000.

### Verdict: H2 REJECTED by the pre-registered rule

| trained on | vs CFOP stage | vs G1 phase | stage>G1? |
|---|---|---|---|
| CFOP (s40–44)     | 0.0009 | 0.0485 | **0/5** |
| Kociemba (s20–24) | 0.0130 | 0.1911 | 0/5 |

Rule was "REJECTED if the inequality reverses in >=4/5 seeds": it reversed **5/5**, paired t p=0.0005.

**Ceiling-normalized** (methodology control #3 — the raw numbers are uninterpretable without it,
because the stage ceiling is 8.5x smaller than the G1 ceiling and the raw comparison is structurally
rigged against stage):

| | cNMI | ceiling | % of ceiling | shuffled null |
|---|---|---|---|---|
| CFOP router vs CFOP stage | 0.0007–0.0015 | 0.0657 | **1–2%** | 1–2% |
| CFOP router vs G1 phase   | 0.0397–0.0665 | 0.5564 | 7–12% | 2% |
| Kociemba router vs G1     | 0.1635–0.2133 | 0.5190 | 32–41% | 1–2% |

The CFOP router sits **at its own null** for stage even after normalization. The verdict survives the
normalization that could have rescued it.

### The causal test agrees (`src/h2_causal.py`) — this is what makes the null credible

Correlational cNMI has almost no headroom here, and in H1 a 7x causal effect coexisted with cNMI
0.0193 (control #5), so the intervention decides. Force **only** stage-*s* tokens through expert *e*,
read out position-stratified per-stage accuracy -> damage matrix `L[e,s]`. `self_exact=True` on all
5 seeds (forcing the current routing is a bit-exact no-op, validating the hook).

- If experts specialized by stage, a stage's **own** routed expert would be much less damaging than an
  arbitrary one. It is not: `L_native` = 0.0675 vs `L_median` = 0.0722 over 20 cells (5 seeds x 4
  stages), paired **t=0.576, p=0.57**; native less damaging in only 14/20 (binomial p=0.12).
- **Positive control on the same models**, H1's exact G1-mass readout: also absent — L(e1)=0.088 vs
  median_other=0.061, p=0.60, 3/5 seeds, and **2/5 seeds negative** (s40 −0.052, s41 −0.004).
  Reference: H1 at scale gave L(e1)=0.290 vs 0.0011 (~270x, 5/5).

So the CFOP router carries **neither** decomposition causally. That matters: the rejection is **not**
"routing is a fixed property of the cube" — the CFOP router doesn't partition on G1 either. The weak
correlational G1 signal (7–12% of ceiling) has no causal counterpart.

### One apparent signal, killed by control #6

"Native expert == least-damaging expert in 7/20 cells vs chance 2.5, binomial p=0.008" looks like weak
specialization. It is the **load/diversity artifact**: forcing stage-*s* tokens onto the expert that
already handles most of them perturbs fewer tokens, so it damages less for reasons unrelated to
identity. `argmax(already-routed fraction)` == `argmin(damage)` in **exactly the same 7/20 cells**
(p=0.008), and the two predictors coincide 4/4 in s41 and s43. No residual identity effect.

### WHY G1 but not CFOP stage — verified, and NOT the obvious explanation

The obvious story — "G1 restricts the legal move set and stage doesn't" — is **false**, measured:

| label | distinct moves used | pairwise JS between label move-distributions |
|---|---|---|
| CFOP stages 1/2/3/4 | 18 / 18 / **7** / **9** | up to **0.4367** bits (1 vs 3) |
| G1 phases 1/2 | 18 / **10** (G1-mass exactly 1.0000) | 0.3827 bits |

CFOP stages are *more* sharply separated in output distribution than G1 phases. So that is not it.

The actual mechanism is **positional availability of the label**:

| label | maj(label \| pos) | H(label) | H(label \| pos) | entropy removed by position | seam sd |
|---|---|---|---|---|---|
| CFOP stage | **0.9196** | 1.8660 | 0.2568 | **86.2%** | 1.41 |
| G1 phase (Kociemba) | 0.8689 | 0.9216 | 0.3880 | **57.9%** | **3.45** |

In the CFOP data the stage boundaries are nearly pinned to token position (cross is 4–8 moves and the
F2L/OLL/PLL segments are fixed-length canonical algs, so segment offsets barely vary). The network
gets stage almost free from positional embeddings and the router has no reason to encode it. In
Kociemba the seam genuinely moves (sd 3.45), so phase must be computed from the state — and it is.

**Revised claim: a router encodes a data-latent decomposition only when that decomposition is not
already recoverable from token position.** This is a limitation of the CFOP *data design*, named
precisely and fixable in principle (widen `cross_len` far beyond (4,8) to decorrelate stage from
position). It is **not** evidence that H2 would pass if fixed — that is untested and must not be claimed.

### Two defects found and fixed in the analysis before the verdict was trusted

Both would have produced a "REJECTED" headline off broken code. Caught because the Kociemba row
printed `cNMI(stage)` **bit-identical** to `cNMI(G1)` (0.2072 = 0.2072, all 5 seeds), which is
impossible for two different labelings.

1. **The Kociemba control row was vacuous** — it passed `ex['phase']` (already the G1 phase) as the
   "stage" label, so it measured G1 twice. Fixed by applying `cfop.stage_of` to the states.
2. **`g1_labels()` mis-aligned rows** — it paired `te[:len(uniq)]` against a differing extraction
   order and truncated with `lab[:len(ex['phase'])]`, so labels were silently shifted and the tail was
   cut (per-stage output read `[x, x, nan, nan]`: stages 3–4 dropped entirely, though they are 61% of
   the data). Replaced by `both_labels()`, which replays the **same** `te` order extraction uses and
   **asserts** `len(labels) == len(rows)`.

### Disclosure against the pre-registration

`PREREGISTRATION_H2.md` §1 asserts "both label functions are computable on either dataset ... so are
the CFOP stage predicates." **That premise is false as needed for the 2x2.** The stage predicate is
valid only at segment *boundaries*: within a segment it is non-monotone (a PLL alg moves the DR edge
out mid-maneuver), so per-token it agrees with the construction label only **27.6%** of the time and
reads 74% "stage 1". On Kociemba trajectories it is near-constant (dist [4216, 2, 0, 41]) — Kociemba
paths essentially never have the cross solved until the end. So the Kociemba row's "vs CFOP stage"
cell has almost no label entropy and is **not a real measurement**; the prereg's own §2.1 anticipated
the non-monotonicity but the 2x2 was specified as if it did not matter. The CFOP row — the actual test
— is unaffected, since it uses the construction label. The generator is sound: 100/100 at every
segment boundary, and all 179 PLL / 148 OLL / 231 F2L canonical maneuvers preserve the D edges.

Artefacts: `out/h2_2x2.json`, `out/h2_causal.json`, `src/h2_analysis.py`, `src/h2_causal.py`.

## 13. H2b / POSITION DECORRELATION — H_pos REJECTED on the causal leg (2026-09-26)

Governing doc: `PREREG_AMENDMENT_5_POSITION_DECORRELATION.md` (`d4dd9b54`), filed before data generation.
Data `cfop_dec.npz`: 10k solves, `cross_len=(0,25)`, 677,580 steps. Fresh seeds 50–54, 4L/d128,
maxlen 145. Uses the FIXED `53+t` extraction (see §14).

**A5.3 void condition PASSED:** removed-by-position **57.3%**, inside the declared 50–65% window and
within 0.6 pts of Kociemba's 57.9%. All 9 QA gates passed; state-only conflict 6.26%.

### Verdict: H_pos REJECTED (Leg 2 decisive per A5.4)

| gate | result |
|---|---|
| L1 learning | **PASS** — acc 0.5611–0.5747, 5/5 at 3.30x the 0.1701 bar |
| Leg 1 correlational | **PASS** — stage 9.7% of ceiling (null 3.8%) vs G1 6.2%; stage>G1 **5/5**, p=0.0075 |
| Leg 2 causal (co-primary) | **FAIL** — advantage +0.0094, paired t p=0.2723, 4/5 seeds |
| Leg 2 partial control | **FAIL** — `is_native` coef −0.0014, p=0.9244 |
| Leg 2 secondary (stages 2–4) | FAIL — advantage +0.0118, p=0.3083 |

### The manipulation DID work correlationally — this is the real finding

The identical comparison that failed **0/5** in H2 passes **5/5** here:

| | vs CFOP stage (raw) | vs its own shuffled null | stage>G1 |
|---|---|---|---|
| H2, `cross_len=(4,8)` | 0.0009 | **at null** (1–2% vs 1–2%) | 0/5 |
| H2b, `cross_len=(0,25)` | 0.0974 | **2.4x above null** (9.7% vs 3.8%) | **5/5** |

Also `native expert per stage` is **4/4 distinct in all 5 seeds** (H2: 3–4/4). So decorrelating position
from the label changed what the router encodes. **The ceiling itself moved, 0.0657 -> 1.0000**, which is
not a bug but the manipulation's signature: in H2 stage was so position-determined that within most
position strata the label barely varied, leaving nothing for conditional MI to see. Therefore compare
each run **against its own shuffled null**, never H2's "% of ceiling" against H2b's.

### But there is no causal evidence the stage assignment is USED

`L_native` 0.1104 vs `L_median` 0.1198 over 20 cells, p=0.27. After partialling out the already-routed
fraction, `is_native` is **−0.0014, p=0.92** — indistinguishable from zero.

**The artifact control earned its place again:** `already_routed_frac` is itself significant
(coef −0.0555 p=0.052 primary; **−0.1163 p=0.0078** secondary). Tokens already routed to an expert are
cheaper to force there. That is what produced H2's spurious 7/20 signal and it is operating here too.

### Two caveats that must travel with this verdict

1. **The null is UNDERPOWERED and this was pre-declared.** Observed +0.0094 vs MDE **0.0239** — about
   40% of what n=20 can detect. "REJECTED" means **failed to demonstrate**, NOT demonstrated absent.
2. **The causal readout is the weak link, and that is a design error of mine, not a property of the
   data.** H1 used **G1-legal probability mass** — a signature only the phase-1 expert should damage —
   with `median_other` ~0.001, so the effect stood out ~270x. H2b used **raw per-stage accuracy**, where
   `median_other` ~0.12: forcing any single expert in a top-2 MoE removes half the computation for those
   tokens, so diversity-collapse swamps identity. A proper analogue **exists and was not used**: stages
   3 and 4 use only **7 and 9 of 18 moves**, so a stage-characteristic move-mass readout is well
   defined. **This does NOT rescue the verdict** — H_pos is rejected as pre-registered and that stands.
   It is a named limitation requiring its own pre-registered test, not a re-analysis of this one.

### Where the project's claim now stands

- **H1 (G1/Kociemba): CONFIRMED**, correlationally and causally, at two scales. Unaffected.
- **H2 (CFOP stages, position-confounded): REJECTED.**
- **H2b (CFOP stages, position-decorrelated): routing alignment appears (5/5) but no causal use shown.**

So: position-decorrelation is **sufficient to make routing ALIGN with a decomposition**, and is **not
shown sufficient to make routing causally IMPLEMENT it**. The general claim "MoE routers recover the
decomposition latent in their training data" remains **unsupported**; the one demonstrated instance of
causal recovery is still G1 alone.

Artefacts: `out/h2b_2x2.json`, `out/h2b_causal.json`, `out/h2b_verdict.json`, `src/h2b_report.py`,
`src/h2b_train.sh`, `out/h2b_train.log`, `out/h2b_analysis.log`.

## 14. BUG: off-by-one in `extract.py` — attenuated every correlational number (2026-09-26)

`build_seq` puts the target for move *t* at index **53+t**, so the residual/routing that PREDICTS move
*t* lives at 53+t. `extract.py` read **54+t** — the position predicting move *t+1* — while labelling it
with move *t*'s phase/stage. Because phase/stage is piecewise constant along a solve, this mislabelled
exactly the **segment-boundary** tokens, which carry the most decomposition information, and it
included a final position whose target is −100.

### Full re-run, all 10 arm-A models at the original `max_solves 1200` (RESOLVED)

`src/h1_rerun.sh` -> `out/confirm_fixed/`, `out/confirm_scale_fixed/` (published dirs untouched).

| | cNMI published | cNMI fixed | change | causal `L(e1)` |
|---|---|---|---|---|
| **25k** seeds 10-14 | 0.1407 | 0.1320 | −0.0087, t=−1.46, **p=0.22**, up 1/5 | 0.0377 -> 0.0377, **diff exactly 0.0** |
| **250k** seeds 20-24 | 0.1967 | **0.2580** | **+0.0613, t=+9.06, p=0.0008, up 5/5** | 0.2931 -> 0.2931, max diff 6.9e-07 |

**The effect is SCALE-DEPENDENT, not uniformly attenuating.** An earlier claim in this file that the bug
"attenuated every correlational number, H1 is stronger than published" was an **overgeneralization from
one seed at 200 solves** and is corrected here: it holds for the 250k replication (+31% relative) and
NOT for the confirmatory run (p=0.22, mean shift slightly downward).

**The causal legs are provably untouched** — differences are *exactly zero* on all five 25k seeds and
<=6.9e-07 at scale (MPS float nondeterminism; same selected expert, same baseline to 6 dp). So
`confirm.py:causal_seq` and `h2_causal.py` never used this path, as their 53+t indexing implies, and
H1's headline causal effect is unmoved. `EXTRACT_OFF1=1` reproduces the old behaviour for audit.

**No verdict changes** at either scale: every gate still clears widely (cNMI 0.0598-0.2782 vs shuffled
nulls 0.0011-0.0016, permutation p=0.005 throughout, hash twins 0.0096-0.0154, change-point slope
positive at p<=0.02 on all 10 seeds).

**STILL UNRESOLVED — do not restate H1's "92% of ceiling".** That figure came from a *different*
normalization pipeline (`controls_scale.py`/`calibrate.py`), not from `confirm.py`, and is not
comparable to `h2_analysis.py`'s ceiling (which gives 0.5190 for mixed250k, i.e. the fixed 0.2580 would
be ~50% under THAT convention). The ceiling-normalized H1 figure must be recomputed in its own pipeline
before any % is quoted. Conjecture, untested, for the scale-dependence: at 250k routing is far more
sharply phase-specialized, so a one-step shift at the boundary destroys proportionally more of a crisp
signal than it does in the blurrier 25k models.

## 15. H2c / STAGE-SPECIFIC READOUT — finds nothing, but MISSED its own resolution bar (2026-09-27)

Governing doc: `PREREG_AMENDMENT_6_STAGE_SPECIFIC_READOUT.md` (`1dc2f1cf`), filed before the readout
was implemented. A SEPARATE test, not a re-analysis: **H2b's REJECTED verdict is untouched** and the
A6.1 rule forbids writing "H_pos was supported" under any outcome here.

### Instrument validated (A6.4 gate CLEARED — H2c not void)

`M_s` = moves whose training-split frequency in stage *s* exceeds their frequency elsewhere, frozen +
hashed before any intervention (`out/h2c_msets_*.json`).

**The derivation validated itself blind:** on Kociemba data it recovered **M_2 == exactly the 10 G1
moves** (mass 1.000 in phase 2, 0.456 elsewhere) and M_1 == exactly the 8 non-G1 quarter turns (mass
0.000 in phase 2), without being told what G1 is. On CFOP: |M_3|=5 (0.904 of stage mass), |M_4|=8 (0.979).

**Positive control vs `confirm.py`'s H1 machinery, all 40 cells (5 seeds x 8 experts):**
Pearson **r=0.99978**, slope **0.971**, mean |diff| **0.0023**. Reproduces H1 at full strength
(s20: L=0.4753 vs median_other 0.0025). The one apparent mismatch (s21) was **`argmax` instability in
labelling which expert is "e1"** at 200 vs 1200 solves, not a readout error — the underlying per-expert
vector agrees (0.1188 vs 0.1137 etc.).

### Result: no effect, and the point estimate is in the WRONG direction

| | advantage | seeds | paired t | `is_native` coef | `already_routed_frac` |
|---|---|---|---|---|---|
| primary (4 stages) | +0.0023 | **2/5** | p=0.8168 | **+0.0052**, p=0.7233 | −0.0607, **p=0.0347** |
| secondary (stages 3–4) | +0.0021 | 3/5 | p=0.8624 | **+0.0078**, p=0.6172 | −0.1288, **p=0.0084** |

`is_native` POSITIVE = a stage's own expert damages it slightly *more*. The **load artifact appeared in
all three causal tests**, though not always as a significant predictor: in H2 it fully explained a
spurious 7/20 signal (binomial p=0.008) while its correlation with damage was NOT significant (rho +0.11,
p=0.155); in H2b it was significant only on stages 2-4 (p=0.0078; all stages p=0.052); in H2c p=0.035
(all) and p=0.0084 (stages 3-4). (Corrected 2026-09-27: an earlier line here said "significant for the
third time across three tests", which overstated H2 and H2b-primary.)

OLS verified with three independent solvers (`np.linalg.lstsq`, `scipy.linalg.lstsq`, normal equations):
identical to 5 dp, cond(X)=8.2/12.4, full rank. The macOS Accelerate BLAS RuntimeWarnings are cosmetic.

### THE CAVEAT — H2c missed its own pre-declared resolution bar

A6.4 required the instrument to improve resolution **>= ~3x** over H2b before its null could be read as
evidence about routing. Measured:

| readout | mean \|background\| | mean between-expert spread | pooled spread/background |
|---|---|---|---|
| H2b (raw per-stage accuracy) | 0.1209 | 0.0372 | 0.308 |
| H2c (stage-characteristic mass) | **0.0435** | 0.0391 | **0.898** |

Improvement **2.9x** — *at* the bar, not clearly above it. MDE also unmet (0.0274 vs observed 0.0023).

**Therefore A6.5's anticipated conclusion ("H2b's null is ROBUST") is NOT licensed.** `src/h2b_report.py`
auto-printed that line because the pass rule was coded but the *interpretation* was not gated on the
resolution check — a flaw in the reporting code relative to the amendment. Report per the amendment.

### Honest end state of the causal question

Two independent readouts (accuracy, stage-characteristic move mass), load artifact controlled in both,
find **no causal stage specialization and no directional hint of one**. Neither achieved the resolution
to **exclude** a small effect, and H2c narrowly missed the bar set for claiming it had. The converging
nulls are suggestive, not decisive.

Artefacts: `out/h2c.json`, `out/h2c_poscontrol.json`, `out/h2c_verdict.json`, `out/h2c_msets_*.json`,
`src/h2c_readout.py`, `out/h2c.log`, `out/h2c_poscontrol.log`.
