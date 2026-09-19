# PhaseSplit — CONFIRMATORY PRE-REGISTRATION

Written **before** any confirmatory model was trained or analysed. Everything in the Exploratory
section of `results.md` is hypothesis-generating and is **not** a claim.

## Scale (deviation, declared)
No CUDA on this machine (`torch.cuda.is_available() == False`, MPS only). Models are therefore
**4 layers / d=128**, 8 experts, top-2, every FFN, 25k solves — *not* the 6L/d256 on 250k solves
the protocol specifies for a CUDA host. All confirmatory claims are scoped to this scale.

## Seeds
**Fresh seeds 10–14** for each arm. Seeds 0–5 were used in exploratory analysis and are excluded
from every confirmatory number. No seed may be dropped after the fact.

## Arms
- **A (sequence)**: decoder-only over `[54 sticker tokens][move tokens]`, loss on move tokens only.
- **C (position-blind)**: 54 stickers of the *current* state → next move. No history, no step index.

## Fixed analysis choices (no post-hoc selection)
- **Layer: the LAST layer only** (index `NL-1` = 3). No best-layer search anywhere.
- **Stratum**: step index for arm A; remaining-moves for arm C.
- **Expert assignment**: router top-1.
- Held-out solves only; matched held-out size within an arm.

## Leg 1 — Learning gate (run first; an arm that fails is excluded from all claims)
- `acc` = held-out teacher-forced next-move accuracy.
- Bar = `majority|(phase, stratum)` on the same held-out rows.
- `probe` = L2-regularised logistic regression, phase from the last layer's pre-router hidden
  state, reported as within-stratum row-count-weighted accuracy.
- **PASS: `acc >= 1.30 x bar` AND `probe >= 0.85`, in >= 4/5 seeds.**

## Leg 2 — Router vs nulls
`cond_NMI` = NMI(top-1 expert; phase) computed within each stratum and row-count-weighted.
Nulls: (a) **shuffled** — expert ids permuted across tokens within the layer; (b) **position-only** —
synthetic router assigning 8 quantile bins of the stratum variable; (c) **permutation** — phase
labels shuffled within strata, 200 draws, `p = (#>=obs + 1)/201`.
- **PASS: router > shuffled in >= 4/5 seeds AND permutation p <= 0.01 in >= 4/5 seeds AND
  mean(router) >= 5 x mean(shuffled).**

## Leg 3 — Change-point (exact definition)
For each held-out solve with `n` moves, take the last-layer top-1 expert sequence `e_0..e_{n-1}`.
For each candidate `c` in `2..n-2` let `P` and `Q` be the 8-bin expert histograms of `e_[0:c]` and
`e_[c:n]`; `TV(c) = 0.5 * sum_e |P_e - Q_e|`; the fitted change point is `c_hat = argmax_c TV(c)`.
Solves with `n < 6` are dropped.
Across solves fit ordinary least squares
`c_hat = b1 * seam + b2 * sol_len + b0`.
**"Partial slope" is `b1`** — the coefficient on the true seam index *controlling for total solution
length*. (The unadjusted slope is confounded: `c_hat` sits mid-sequence and `sol_len` correlates with
`seam`, so random routing scores ~0.55.)
Null: permute `seam` across solves **within integer strata of `sol_len`**, refit, 200 draws;
`p = (#{b1_perm >= b1_obs} + 1)/201`.
- **PASS: `b1 > 0` AND `p < 0.01`, in >= 4/5 seeds.**

## Leg 4 — Causal (control fixed as specified)
At the last layer, for every **prediction position whose target move is in phase 2**, force the
top-1 expert to a specified expert `e` (gate weights unchanged) and measure
`R(e)` = mean probability mass on the 10 G1-legal moves. `R_base` = no intervention.
Loss `L(e) = R_base - R(e)`.
`e1` = the expert maximising `P(phase 1 | expert)` among experts holding >= 50 tokens at that layer.
**Effect = L(e1) / median{ L(e) : e != e1 }** — the phase-1 swap against the **median over every
other expert swap**, not against one arbitrary control.
Computed within each stratum and row-count-weighted. Sanity: forcing the current top-1 must
reproduce baseline exactly.
- **PASS: `L(e1) > median_{e != e1} L(e)` in 5/5 seeds (one-sided sign test p = 0.031).**
  Paired t-test reported alongside but **not** required (it is sensitive to magnitude outliers).
  4/5 with paired t p < 0.05 is recorded as *partial*, not a pass.

## Claim rule
An arm is **confirmed** only if Legs 1–4 all pass on that arm, on the fresh seeds, under the
definitions above. Any other outcome is reported leg-by-leg with no headline claim. The
Confirmatory section supplies the headline; the Exploratory section keeps the full history
including the null→positive reversal.
