# AMENDMENT 3 — the 5-prefix QA gate is sample-size dependent

**This is a post-hoc change to a DATA-ADMISSIBILITY gate, made after that gate failed.** Declaring
it plainly rather than editing a threshold quietly.

## Mitigating fact, stated first
**No scale-replication model has been trained or analysed.** At filing, `/vol/out` contains no
`*_s2[0-4].json` and no `confirm/` outputs for seeds 20–24. The change therefore cannot be motivated
by a desired scientific outcome — it concerns only whether the 250k dataset is admissible. **None of
L1–L4, the layer rule, strata, `e1`, or the intervention mechanics are touched.**

## The failure
`qa.py` gate *"L3 repeated 5-prefix rate low (paths not stereotyped)"* requires `rate <= 0.15`.
On the 250k set it read **0.2216 → FAIL**. All other 15 gates passed (seam sd 3.45, mixed cells
0.535, lag-2 0.052, H1 4.09).

## Diagnosis: the statistic is a function of n, not of stereotypy
`rate = 1 - distinct/total` over 5-move phase-1 prefixes. The prefix space is finite, so with more
draws collisions rise by pigeonhole regardless of data quality. Measured on the same 250k file:

| sample | repeat rate | distinct | total |
|---|---|---|---|
| 25k subsample (seed 0) | **0.0284** | 17,847 | 18,368 |
| 25k subsample (seed 1) | **0.0291** | 17,958 | 18,496 |
| 25k subsample (seed 2) | **0.0265** | 17,980 | 18,470 |
| 50k subsample | 0.0548 | 34,889 | 36,913 |
| 100k subsample | 0.1031 | 66,154 | 73,761 |
| full 250k | 0.2216 | 143,785 | 184,726 |

The **25k reference value on the passing confirmatory set is 0.029**. At matched n the 250k data is
indistinguishable from it (0.0265–0.0291). The threshold 0.15 was calibrated at n=25k and fails
mechanically at larger n for any dataset, including a perfect one.

## Change
The 5-prefix gate is evaluated on a **random subsample of min(25000, n) solves** (fixed seed 0) —
i.e. at the sample size at which its threshold was calibrated. The threshold **stays 0.15**,
unchanged. The full-n value is still computed and reported on the data card as
`repeated_prefix_rate_fulln`, labelled as sample-size dependent and not gated.

The companion lag-2 statistic `P(move[t]==move[t-2])` is **already scale-invariant** (a per-position
rate, not a collision count) and is untouched: it reads 0.052 at 250k vs 0.057 at 25k, against a
chance baseline of 0.056. It remains the primary stereotypy gate.

## What would have been wrong
Raising the threshold to admit 0.2216 would have made the gate unfalsifiable at 250k and would not
have been transferable to any other n. Subsampling fixes the estimator; it does not move the bar.
