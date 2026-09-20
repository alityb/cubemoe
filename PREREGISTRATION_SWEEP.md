# PRE-REGISTRATION — ROUTING SWEEP (H_slack)

Written **before any sweep model is trained**. This is a **hypothesis test**, not an exploration:
the direction of every comparison is fixed here, in advance.

Governed by `PREREGISTRATION.md` (`dc0cf984`) and Amendments 1–3. **L1–L4 and their thresholds are
unchanged and are not restated as new criteria** — every variant must clear the same gates.

---

## 1. The hypothesis

**H_slack — top-2 routing plus a load-balancing loss creates *slack* that lets specialization
concentrate in a single expert.**

Mechanism: with top-2 and 8 experts, the LBL constrains the *aggregate* assignment distribution, not
each slot independently. One expert can therefore take **slot 0 for essentially all phase-1 tokens**
while the second slot absorbs the balance requirement — the balance target is met exactly while one
expert monopolises the phase-1 computation. Under **top-1** there is no second slot to absorb it:
the balance constraint binds directly on the only assignment, so specialization must spread across
experts.

### Predictions (directional, fixed here)

Comparing **top-1 / 8 experts / local-scope LBL** against the **top-2 / 8 / large-scope baseline**:

| # | quantity | predicted direction under H_slack |
|---|---|---|
| **P1 (PRIMARY)** | concentration index | **top-1 LOWER** than top-2 |
| P2 (secondary) | ceiling-normalised cond NMI | top-1 **HIGHER** |
| P3 (secondary) | **absolute causal loss `L(e1)`** | top-1 **SMALLER** |

P1 is the primary endpoint. P2 and P3 are secondary and reported regardless; designating one primary
avoids multiplicity across three correlated metrics.

**P3 uses the ABSOLUTE loss `L(e1)`, not the ratio `L(e1)/median_other`.** Revised here, before any
sweep model was trained, after the ratio proved numerically unstable on the scale hash twins: their
`median_other` sits at or below zero (per-seed: 0.000237, −0.000019, 0.000707, −0.000010, 0.001371),
producing ratios of −92,380x and +354,077x on effects that are indistinguishable from zero. The
ratio is only meaningful when the denominator is bounded away from 0. It is still **reported**, but
gated: quoted only when `median_other > 0.001`, and marked `n/a (unstable)` otherwise. The absolute
loss is well-behaved for null and non-null arms alike — learned router `L(e1)=0.2898` vs hash twin
`0.0031` at the last layer, a clean 94x separation that the ratio cannot express.

### Concentration index — exact definition
At the **last layer**, on held-out rows, for every expert `e` in turn force the top-1 slot of **all
phase-2 prediction positions** to `e` and measure `R(e)` = stratum-weighted mean probability mass on
the 10 G1-legal moves. Let `L(e) = max(R_base − R(e), 0)`. Then

`concentration = max_e L(e) / Σ_e L(e)`,  range `[1/8, 1]`.

1/8 = the causal effect is spread evenly across experts; 1 = a single expert carries all of it.
This is a **pre-specified descriptive**, computed for every variant. It is **not** a gate.

### Comparison, pairing, and pass rule
- **Primary (full plan): paired within seeds 30–34.** The top-2/8/large-scope baseline is retrained
  at seeds 30–34 so each variant seed pairs with a baseline seed sharing initialisation and data
  order. Paired per seed, n=5.
- **Core-only plan:** the baseline is the **existing seeds 20–24** scale run (same config: top-2, 8
  experts, large-scope LBL, 6L/d256, 250k). Pairing is then **index-matched** (30↔20, 31↔21, …),
  which is arbitrary and is declared as weaker; the sign test is over the 5 matched pairs.
- **H_slack is SUPPORTED iff** P1 holds in **≥4/5** pairs **AND** the paired t-test on the
  concentration index is **p < 0.05 in the predicted direction**.
- **H_slack is REJECTED** if P1 reverses in ≥4/5 pairs.
- Otherwise: **inconclusive**. P2/P3 are reported per-prediction and cannot rescue a failed P1.
- No variant may be dropped post hoc; a variant failing L1 is reported as failing L1 and excluded
  from P1–P3 with that exclusion stated.

---

## 2. Variants

All at **6 layers / d=256, 250k solves, seeds 30–34**, last layer only, same L1–L4.

| # | routing | experts | LBL scope | role |
|---|---|---|---|---|
| V1 | top-1 | 8 | local | H_slack test arm |
| V2 | top-1 | 2 | local | H_slack, minimal-expert limit |
| V3 | top-2 | 8 | **off** | isolates LBL from top-2 |
| V4 | top-2 | 8 | large-scope | **baseline** (matches the seeds 20–24 config) |
| V5 | expert-choice | 8 | n/a (balanced by construction) | alternative to token-choice |
| H1 | hash twin for V1 | 8 | — | L2 null for top-1/8 |
| H2 | hash twin for V2 | 2 | — | L2 null for top-1/2 |

Each variant is evaluated on **L1, L2, L3, L4** exactly as pre-registered, plus the concentration
index. Arm A (sequence) only — arm C is not part of this sweep.

---

## 3. Costed plan

Measured rate: **$1.57 per arm-A-class model** at 6L/d256/250k on A100-40GB (150-step probe,
17,183 tok/s, 0.75 h/model).

| plan | configs | models | GPU cost | baseline / pairing |
|---|---|---|---|---|
| **FULL** | V1–V5 + H1,H2 (7) | 35 | **~$55** | V4 retrained at seeds 30–34 → **exact pairing** |
| **CORE** | V1, V2 + H1, H2 (4) | 20 | **~$31** | existing seeds 20–24 → **index-matched, weaker** |

Confirm/control passes add ~$2–4. **Both workspaces are at $0 credit**
(`ali-moh-islam-1` $30.72, `dnfcubes` $30.13 of $30 each), so either plan is **out of pocket**.

**Recommendation: FULL.** The $24 difference buys exact per-seed pairing on the primary endpoint.
Index-matching seeds 30–34 against 20–24 is arbitrary — it pairs unrelated runs — and P1's pass rule
depends on a paired test, so CORE materially weakens the one comparison the sweep exists to make.

---

## 4. What would falsify H_slack
If top-1/8 shows **equal or higher** concentration than top-2/8, the slack mechanism is wrong: the
concentration would then be a property of the task or the representation, not of routing capacity.
V3 (top-2, LBL off) discriminates further — if V3 also concentrates, LBL is not the cause and the
mechanism is misattributed.
