# AMENDMENT 4 to PREREGISTRATION_SWEEP.md

Filed **before any V6/V7 model is trained**. Amends the sweep pre-registration
(`c1b023f4`, with addendum) which is **left unmodified**.

---

## A4.1 — V1 is confounded. Recorded, not rescued.

**V1 (top-1 / 8 / local-scope) differs from the baseline (top-2 / 8 / large-scope) on TWO axes:
routing arity AND balancing scope.** I wrote that comparison into the sweep pre-registration.
**P2 and P3 as pre-registered are therefore confounded and cannot attribute any effect to arity.**

V1 results stand as observed, reported as-is, with no reinterpretation:

| seed | concentration | cond NMI | % ceiling | L(e1) |
|---|---|---|---|---|
| 30 | 0.486 | 0.0260 | 12% | −0.0000 |
| 31 | 0.677 | 0.0134 | 6% | 0.0001 |
| 32 | 0.487 | 0.0166 | 8% | −0.0001 |
| 33 | 0.314 | 0.0162 | 7% | 0.0002 |
| 34 | 0.671 | 0.0169 | 8% | −0.0001 |

Baseline (top-2/8/large, seeds 20–24) for contrast: cond NMI **0.2009 (92% ceiling)**,
L(e1) **0.2898**. V1 is **causally null** (all |L(e1)| ≤ 0.0002) and correlationally near the floor.

**P1/P2/P3 verdicts on V1 are NOT claimed**, because the arm is confounded. Specifically: P3
("top-1 smaller L(e1)") is *technically* satisfied but only vacuously — there is no effect to be
smaller. That is recorded as an artefact of a confounded design, not as support for H_slack.

## A4.2 — Two single-axis arms

Both at 6L/d256, 250k, **seeds 30–34**, same L1–L4 + concentration index, last layer only.

| arm | routing | experts | LBL scope | isolates |
|---|---|---|---|---|
| **V6** | top-1 | 8 | **large** | **arity only** — the actual H_slack test |
| **V7** | top-2 | 8 | **local** | **scope only** — 2604.07030's claim, against ground truth |
| **H1** | hash | 8 | — | L2 null for top-1/8 (not yet trained) |

Each differs from the top-2/8/large baseline on exactly one axis.

## A4.3 — Concentration index: floor and companion statistic

`concentration = max_e L(e) / Σ_e L(e)` is meaningless when the denominator approaches zero
(same near-zero-denominator pathology already fixed for the L4 ratio in Amendment 1).

**FLOOR = 0.05 on `Σ_e L(e)`.** Pre-stated and justified from observed noise: the largest
`Σ_e L(e)` on a **null arm** (hash twins, seeds 20–24) is **0.0234**; the smallest on the
**learned baseline** is **0.1012**. The floor sits >2× above the null maximum and <½ the baseline
minimum.

- If `Σ_e L(e) < 0.05` → concentration is reported as **undefined (below floor)**, never as a number.
- **`max_e L(e)` is reported beside it in every case**, since it is well-behaved for null arms.
- On this rule V1's concentration values above are **undefined** — its `Σ_e L(e)` ≈ 0.

## A4.4 — Ceiling normalisation, stated and applied uniformly

`ceiling` = the conditional NMI of a **synthetic perfect phase tracker** built on the *same dataset*,
using the *same stratifier* and the *same* row-count-weighted within-stratum NMI estimator
(`src/calibrate.py`): experts are split into two disjoint halves, phase-1 tokens drawn uniformly from
one half and phase-2 tokens from the other, evaluated on the held-out seam/`sol_len` distribution.

`cNMI_pct_ceiling = cond_NMI(router) / ceiling`

On `mixed250k` the ceiling is **0.2174**. It is a *reference construction, not a mathematical
maximum*, so values may exceed 100%. **The same ceiling and estimator are applied to every variant**,
including V1, V6, V7 and the baseline — no per-variant recalibration.

**Recomputed baseline under this formula** (seeds 20–24, 600 held-out solves, last layer):
per-seed cond NMI 0.2072 / 0.1719 / 0.2001 / 0.2029 / 0.2223 → **mean 0.2009 = 92.4% of ceiling**.
This matches the previously reported 92% figure; no revision needed.

## A4.5 — Three-outcome decision matrix (pre-specified)

Let "collapse" mean: cond NMI ≤ 25% of ceiling **and** `Σ_e L(e)` below the 0.05 floor —
i.e. both correlationally near the floor and causally null, as V1 is.

| V6 (arity only) | V7 (scope only) | conclusion |
|---|---|---|
| collapse | no collapse | **Arity is responsible.** Top-1 destroys specialization; H_slack's premise is inverted — slack is not what concentrates it. |
| no collapse | collapse | **Scope is responsible.** Confirms 2604.07030 against an exactly-known decomposition; arity is irrelevant. V1's collapse is fully explained by local-scope LBL. |
| both collapse | both collapse | **Both matter independently**; V1's collapse is over-determined and the two axes cannot be ranked from this design. |
| neither collapses | neither collapses | **Interaction only** — neither axis alone suffices; the collapse requires top-1 *and* local scope together. |

H_slack (as originally stated — that top-2+LBL slack *concentrates* specialization, so top-1 should
show **lower** concentration with **higher** cNMI) is testable only in the top-left and bottom-right
cells, and only V6 bears on it.

## A4.6 — Cost, and an engineering fix that is not a protocol change

V1 trained at **4,118 tok/s / 3.12 h / $3.43 per model** versus a probed 14,294 tok/s. Cause
identified: the **local-scope LBL looped over 128 sequences × 6 layers in Python every step**. It has
been **vectorised**, verified numerically identical to the loop (1.06878528 vs 1.06878531, |Δ| < 1e-5).
`build_seq` was profiled and is *not* a factor (~1 s per run). Large-scope and `off` paths are
untouched.

This is an implementation speed-up with no effect on the objective, and is declared as such.
Expected cost after the fix: **~$0.99–1.20 per model**. Remaining credit is **~$8.6**, so the run
order is **V6 first** (it alone tests H_slack), then V7, then H1 as budget allows; any arm not
completed at 5 seeds is reported as not run rather than at reduced n.
