# AMENDMENT 6 to PREREGISTRATION_H2.md — a stage-specific causal readout (H2c)

Filed **2026-09-26, before the readout is implemented or run**. Amends `PREREGISTRATION_H2.md`
(`c5d6b4ab`) and follows `PREREG_AMENDMENT_5_POSITION_DECORRELATION.md` (`d4dd9b54`), both of which are
**left unmodified**. All H2 and H2b results stand exactly as reported.

---

## A6.1 — What this amendment is, and what it is forbidden from doing

H2b rejected H_pos on its causal leg. **That verdict is final.** This amendment does not re-analyse
H2b's data, does not re-score H2b's checkpoints against a new rule, and **cannot convert H2b's
REJECTED into SUPPORTED**. It defines a *new, separately-named* test (H2c) with its own pre-committed
rule, on the same checkpoints but with a different measurement instrument.

This distinction is not pedantry. This project has already produced one false verdict by substituting a
different comparison for a pre-registered one and reporting the result as though the original rule had
been met. The reporting rule below is therefore binding:

> If H2c fires, the reportable claim is **"H2b's causal readout lacked power to detect a stage effect
> that a stage-specific readout does detect."** The sentence "H_pos was supported" may **not** be
> written. H2b's rejection is reported alongside, with equal prominence, in every venue.
>
> If H2c also finds nothing, that **strengthens** H2b's null, and the two are reported together as
> converging evidence from two independent readouts.

## A6.2 — Why the H2b readout is suspected to be underpowered

Stated as a measured fact, not a hope. H1's causal test read out **G1-legal probability mass** — a
signature that only the phase-1 expert should damage. H2b's read out **raw per-stage accuracy**.

| run | readout | `median_other` (damage from forcing an ARBITRARY expert) | effect found |
|---|---|---|---|
| H1 (scale) | G1-legal move mass | **0.0011** | L(e1)=0.290, **~270x** |
| H2b | raw per-stage accuracy | **~0.12** | L_native 0.110 vs L_median 0.120, p=0.27 |

In a top-2 MoE, forcing all of a stage's tokens through one expert removes half their computation.
Under a *generic* readout that is catastrophic no matter which expert is chosen, so expert **identity**
contributes a small difference on top of a large common effect. `median_other` ~0.12 against a
between-expert spread of ~0.01 is a signal-to-background ratio of roughly 1:12. **H2b's null is
consistent with both "no stage specialization" and "a real effect below the instrument's resolution,"
and it cannot distinguish them.** H2b's own pre-registered MDE (0.0239 vs observed 0.0094) already
recorded that it was underpowered.

## A6.3 — The instrument: stage-characteristic move mass

Measured on `cfop_dec.npz` **before** this amendment, so the readout is not tuned to a result. Each
stage has a distinct move support:

| stage | distinct moves used (of 18) | pairwise JS vs other stages |
|---|---|---|
| 1 cross | 18 | — |
| 2 F2L | 18 | — |
| 3 OLL | **7** | up to 0.4367 bits |
| 4 PLL | **9** | up to 0.4367 bits |

Define, for each stage *s*, the mass-on-characteristic-moves readout
`R_s(state) = sum of predicted probability over M_s`, where `M_s` is the set of moves whose empirical
frequency in stage *s* exceeds their frequency in the pooled other stages. **`M_s` is computed from the
TRAINING split only**, frozen before any intervention is run, and written to
`out/h2c_msets.json` with a sha256 recorded here on first run.

Damage is then `L_s(e) = R_s(base) - R_s(forced through e)`, position-stratified exactly as in H2b.
This is the direct analogue of H1's G1-mass readout: a behaviour that the stage's *own* expert should
be responsible for producing.

**Declared risk.** Stages 1 and 2 use all 18 moves, so `M_1` and `M_2` are defined by frequency
differences rather than support restrictions and will be weaker instruments than `M_3`/`M_4`.
**Pre-committed:** the primary endpoint uses all four stages; a secondary, declared here in advance,
restricts to **stages 3 and 4**, where the support restriction is genuine. If only the secondary fires,
the reportable claim is limited to OLL/PLL and must say so.

## A6.4 — H2c pass rule, pre-committed

Same checkpoints as H2b (`moe_cfopdec_s50..54`, last layer, top-2/8). No retraining, no new data.

**H2c FIRES iff** forcing stage-*s* tokens through that stage's own routed expert damages `R_s` **less**
than the median expert does: `L_median - L_native > 0` in **>=4/5 seeds** AND paired **t p < 0.05**
across the 20 cells, **AND** the effect survives the same mandatory partial control as A5.4 — the
`is_native` coefficient in
`L ~ stage FE + seed FE + already_routed_frac + is_native` must be **negative with p < 0.05**.

The artifact control is **not optional and not weakened**. In H2 it eliminated an apparent 7/20 signal
(p=0.008) entirely, and in H2b `already_routed_frac` was itself significant (−0.1163, p=0.0078). Any
version of this test without that control is inadmissible.

**Positive control, required to run FIRST and to be reported whatever it shows.** Apply the identical
mass-readout machinery to the H1 Kociemba models (`moe_mixed_s20..24`) with `M` = the G1 move set. It
must reproduce H1's known ~270x effect. **If the positive control fails, the instrument is broken and
H2c is VOID** — no H2c number is reported, because a readout that cannot detect the one effect known to
be present cannot be trusted on an effect that may be absent.

**Power, declared in advance.** H2c is only worth running if it materially improves resolution. On
first run, report `median_other` under the new readout. **If `median_other / between-expert spread` is
not at least ~3x better than H2b's ~1:12, the instrument has not solved the stated problem** and that
is reported as the finding, rather than reading its null as evidence about routing.

## A6.5 — What would falsify H2c

`L_native ≈ L_median` under a readout whose positive control reproduces H1's effect at full strength,
with `median_other` small enough that a real identity effect would have been visible. That is the
outcome that would make H2b's null **robust rather than merely underpowered**, and it is the more
informative result of the two. It will be reported with at least equal prominence to a firing.

## A6.6 — Cost

No training. ~42 forward passes per model over 200 solves, 5 H2c models + 5 positive-control models.
~20-30 min local on MPS, free.
