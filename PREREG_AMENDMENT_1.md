# AMENDMENT 1 to PREREGISTRATION.md

Filed **before any `confirm.py` run**. `out/confirm/` was empty at filing (verified: 0 files) and the
analysis chain was stopped before writing this. The original `PREREGISTRATION.md`
(sha256 `dc0cf9840c8c00f85661e2160a3fc76ae3aefa2885abe3a488cf95be0879c956`) is **unmodified**.

---

## A1.2 — DISCLOSURE: a confirmatory training log WAS viewed

**Declared: yes.** Specifics, so the reader can judge rather than take my word:
- I ran `tail -2 out/confirm_train.log` twice to verify the chain was alive. On both occasions the
  visible output was only `device=mps ...` and `model=moe params=4.50M ...` — **no accuracy**.
- However, while gathering the exploratory numbers for *this amendment*, `head -6` on the same file
  exposed seed 10's `ep1 test_acc=0.1695` and `ep2 test_acc=0.1775` (2 of 10 epochs, one of five
  arm-A seeds).

**Consequence, per the amendment rule: L1 is LEFT UNCHANGED** at its original threshold
(`acc >= 1.30 x majority|(phase, stratum)` AND `probe >= 0.85`, in >= 4/5 seeds), and
**held-out accuracy is added to the headline table** so it is visible rather than merely gating.

### Exploratory held-out accuracies (deterministic `mixed` data), stated for the record
Bar `majority|(phase, pos)` = **0.1439** (arm A); `majority|(phase, remaining)` = **0.1455** (arm C).

| arm | per-seed accuracy | mean | min | ratio to bar (min) |
|---|---|---|---|---|
| A (moe_mixed, seeds 0–4) | 0.2033, 0.2016, 0.2038, 0.2032, 0.1996 | 0.2023 | 0.1996 | **1.387** |
| C (state_mixed, seeds 0–5) | 0.5395, 0.5406, 0.5393, 0.5389, 0.5304, 0.5379 | 0.5378 | 0.5304 | **3.645** |

The pre-registered 1.30x floor sits below arm A's exploratory minimum of 1.387x — i.e. it is a real
but not generous bar for arm A, and trivially clear for arm C. It is retained as written.

---

## A1.1 — Hash-router twins added to arm A's null set

Train **hash-router twins for arm A, seeds 10–14** (fixed token-identity routing; identical
architecture, data, schedule and seed to the corresponding learned-router model).

**L2 criterion (arm A) becomes** — all of:
1. router cond NMI > shuffled null in >= 4/5 seeds, and
2. within-stratum permutation p <= 0.01 in >= 4/5 seeds, and
3. mean(router cond NMI) >= 5 x mean(shuffled), and
4. **router cond NMI > hash-twin cond NMI (same seed, same layer, matched held-out) in >= 4/5 seeds.**

Rationale, recorded before results: phase-2 moves are restricted to the 10 G1 moves, so **token
identity is itself phase-informative** and a token-identity router inherits partial phase purity for
free. Without this comparison, L2 cannot separate "the router learned the decomposition" from "the
vocabulary leaks it". Arm C is exempt — it has no move-token input, so a token-identity hash router
is not defined there.

---

## A1.3 — Arm C confirmation rule

**Arm C is confirmed on L1 + L2 + L4. L3 (change-point) is reported but NOT required.**

Rationale, recorded before results: L3 fits a single change point to a per-solve *sequence* of
routing decisions, maximising total-variation distance between two segments' expert histograms. Arm
C classifies each cube state independently — no history, no recurrence, no temporal smoothing — so
its per-solve routing sequence is a series of independent draws and the TV-maximising index is
correspondingly high-variance. The instrument presupposes sequential structure the model does not
have. This is a structural mismatch, decided on the design of the test and the model, not on any
confirmatory result. Arm A retains L1 + L2 + L3 + L4.

---

## A1.4 — `e1` and top-2 intervention mechanics, stated exactly

**`e1` selection.** On **baseline (un-intervened)** routing at the **last layer**, over **held-out**
rows only, let `N_e` = number of rows whose router top-1 is expert `e`, and `N_e^{p1}` = number of
those whose phase is 1. Among experts with `N_e >= 50`,
`e1 = argmax_e ( N_e^{p1} / N_e )` — the maximum **phase-1 share**. Ties broken by the lower index.

**Top-2 handling.** The layer routes top-2. Concretely, per token:
`probs = softmax(W_g h)`; `(topv, topi) = topk(probs, 2)`; `topv <- topv / sum(topv)`.
The intervention replaces **slot 0 only**: `topi[0] <- e`. Specifically:
- `topv` is computed from the **original** probabilities and renormalised over the **original**
  top-2 **before** the swap, and is **not** recomputed afterwards — the forced expert inherits the
  original top-1's normalised gate weight.
- **Slot 1 (the second expert and its weight) is left untouched.**
- Total gate mass is therefore preserved at 1.0; only *which* expert computes in slot 0 changes.
- Sanity requirement (unchanged): forcing slot 0 to the current top-1 must reproduce the baseline
  readout **exactly**.

This is the mechanism already implemented in `MoEFFN.forward` via `_override`; it is written down
here so it is a pre-committed choice rather than an implementation detail.

---

## Unchanged from the original
Last layer only; strata (step index for A, remaining moves for C); fresh seeds 10–14; 4L/d128 on
25k solves (no CUDA, declared); L3 definition of `b1`; L4 effect = `L(e1) / median{L(e) : e != e1}`
with 5/5 required and paired t reported; no seed may be dropped post hoc.

---

## A1.5 — SCALE REPLICATION protocol (filed with the wrap-up; no new criteria)

If a CUDA host becomes available, rerun **unchanged**:
`src/confirm_resume.sh` (train) -> `src/confirm.py` -> `src/confirm_gates.py`
with only these substitutions:
- **6 layers / d=256** (the `device_auto()=='cuda'` branch in `train.py` already selects this).
- **250k solves** (`python3 src/gen_data.py --kind mixed --n 250000`), re-run `src/qa.py --kind mixed`
  and require the same 16 hard gates to pass before training.
- **Seeds 20–24** per arm, plus hash twins 20–24 for arm A.

**No criterion changes.** L1–L4 thresholds, the last-layer rule, the strata, the `e1` definition, the
top-2 mechanics and the arm-C L1+L2+L4 rule all carry over verbatim from `PREREGISTRATION.md` and
A1.1–A1.4. Report pass/fail per leg in the same table. A scale replication that fails any leg is
reported as a failure at that scale, not reconciled against the 4L/d128 result.
