# AMENDMENT 5 to PREREGISTRATION_H2.md — the position-decorrelation arm (H2b)

Filed **2026-09-26, before any H2b model is trained or any H2b data is generated**. Amends
`PREREGISTRATION_H2.md` (`c5d6b4ab`), which is **left unmodified**. All H2 results stand as reported.

---

## A5.1 — Disclosure: a false premise and two analysis bugs in the H2 run

Recorded here because the disclosure rule (Amendment 1) requires defects to be filed even when the
verdict is unchanged. All three were found and fixed **before** the H2 verdict was accepted.

**(i) `PREREGISTRATION_H2.md` §1's premise is false as the 2x2 needs it.** It asserts "both label
functions are computable on either dataset ... and so are the CFOP stage predicates
(`probe/cfop.py`)." The stage predicate is valid only at segment *boundaries*. Within a segment it is
non-monotone — a PLL alg moves the DR edge out mid-maneuver — so per token it agrees with the
construction label only **27.6%** of the time and reads 74% "stage 1". On Kociemba trajectories it is
near-constant (class counts `[4216, 2, 0, 41]`), because Kociemba paths essentially never have the
cross solved until the end. **Consequence:** the Kociemba row's "vs CFOP stage" cell carries almost no
label entropy and is **not a real measurement**. The prereg's own §2.1 anticipated the non-monotonicity
but the 2x2 was specified as though it did not matter. The CFOP row — the actual test — is unaffected,
because it uses the construction label.

**(ii) The Kociemba control row was vacuous.** `h2_analysis.py` passed `ex['phase']` (already the G1
phase on Kociemba data) as the "stage" label, so that row measured G1 twice. The tell was
`cNMI(stage)` printing **bit-identical** to `cNMI(G1)` on all 5 seeds.

**(iii) `g1_labels()` mis-aligned rows.** It paired `te[:len(uniq)]` against a different extraction
order and truncated with `lab[:len(ex['phase'])]`, shifting labels and silently dropping the tail —
the per-stage readout came back `[x, x, nan, nan]`, i.e. stages 3 and 4 missing, which are **61%** of
the data. Replaced by `both_labels()`, which replays the **same** `te` order the extractor uses and
**asserts** `len(labels) == len(rows)`.

The generator was audited and is sound: 100/100 at every segment boundary, and all 179 PLL / 148 OLL /
231 F2L canonical maneuvers preserve the D edges. The H2 verdict is unchanged after all three fixes.

## A5.2 — H2 as pre-registered is REJECTED. This amendment does not rescue it.

Stage alignment reversed in **5/5** seeds (rule: reversed in >=4/5), paired **t p=0.0005**, and the
CFOP router sits at its own shuffled null even ceiling-normalized (**1–2%** of a 0.0657 ceiling). The
causal test agrees: `L_native` 0.0675 vs `L_median` 0.0722, paired **t=0.576, p=0.57**.

**That verdict is final and is not reinterpreted by anything below.** H2b tests a *different*
hypothesis on *different* data. If H2b passes, the reportable claim is "H2 fails on the original CFOP
data design and passes on a position-decorrelated one" — which supports H_pos (A5.3). It does **not**
retroactively convert H2-as-written into supported, and it must never be written that way.

## A5.3 — H_pos: the hypothesis this arm actually tests

**H_pos — a router encodes a data-latent decomposition only when that decomposition is not already
recoverable from token position.**

Motivation, measured **before** this amendment and not assumed. The obvious explanation for H2's
failure is false: CFOP stages are *more* sharply separated in output distribution than G1 phases
(pairwise JS up to **0.4367** bits vs **0.3827**; stages 3 and 4 use only **7** and **9** of 18 moves).
What differs is positional availability:

| label | maj(label \| pos) | H(label) | H(label \| pos) | removed by position |
|---|---|---|---|---|
| CFOP stage, `cross_len=(4,8)` (H2, rejected) | 0.9196 | 1.8660 | 0.2568 | **86.2%** |
| G1 phase, Kociemba (H1, confirmed) | 0.8689 | 0.9216 | 0.3880 | **57.9%** |

H_pos says the 86.2% is why the router ignored stage. **This is currently a correlation across two
datasets, never manipulated. H2b manipulates it.**

### The manipulation, and its pre-measured calibration

Single change to `probe/cfop_gen2.py`: `cross_len=(4,8)` -> **`cross_len=(0,25)`**. Nothing else about
the generator, the canonical case tables, the label construction, or the QA gates changes.

`(0,25)` is **not** chosen to maximise the effect. It is chosen because it reproduces the Kociemba
positional regime, making H2b a controlled comparison — *same positional availability, different
decomposition*. Measured on 1200 generated solves per setting, before training:

| `cross_len` | maj(stage \| pos) | removed by position | mean sol_len |
|---|---|---|---|
| (4,8) — H2 | 0.919 | 86.2% | 61.5 |
| (0,15) | 0.793 | 68.4% | 63.0 |
| **(0,25) — chosen** | **0.702** | **56.8%** | 67.4 |
| (0,40) | 0.610 | 44.7% | 75.2 |
| *Kociemba target* | *0.869* | ***57.9%*** | — |

**Pre-committed:** if the regenerated dataset does not land in **50–65%** removed-by-position, the arm
is **void** and is re-specified before training, not analysed.

### Required config change

Max `sol_len` rises 74 -> 90, so context must grow: **`maxlen` 129 -> 144**. Declared because it is a
deviation from the H2 models. It does **not** compromise the endpoint: the H2/H2b comparison is
**within-model** (cNMI vs stage against cNMI vs G1 on the same checkpoint), so the context change
applies identically to both sides. Truncating long solves instead is **rejected in advance** as a fix,
because it would preferentially drop long crosses and partly undo the decorrelation being bought.

### Arm specification

Fresh seeds **50–54** (seeds 40–44 are spent and must not be reused). 4L/d128, `maxlen` 144, top-2/8
experts/large scope — otherwise identical to the H2 arm. Plus **5 hash twins** (seeds 50–54) as the L2
null. Last layer only. All 9 CFOP QA gates must pass, with the boundary predicates unchanged.

## A5.4 — Pass rule, pre-committed, both legs required

**Leg 1 — correlational.** Ceiling-normalized `cNMI(expert; stage | pos) / ceiling` exceeds
`cNMI(expert; G1 | pos) / ceiling` in **>=4/5 seeds** AND paired **t p < 0.05** in that direction.
Ceiling normalization is mandatory, not optional: the raw comparison is structurally rigged against
stage whenever the two ceilings differ, and in H2 they differed by 8.5x.

**Leg 2 — causal, co-primary.** Forcing stage-*s* tokens through that stage's **own** routed expert is
less damaging than through the median expert: `L_median - L_native > 0` in **>=4/5 seeds** AND paired
**t p < 0.05** over the 20 cells (5 seeds x 4 stages).

**Leg 2 carries a mandatory artifact control.** In H2, an apparent 7/20 "specialization" signal
(binomial p=0.008) was fully explained by load: forcing stage-*s* tokens onto the expert that already
handles most of them perturbs fewer tokens, so it damages less for reasons unrelated to identity.
`argmax(already-routed fraction) == argmin(damage)` hit **exactly the same 7/20 cells**. So:

> Leg 2 counts as passed **only if** the `L_median - L_native` effect survives after the
> already-routed fraction is regressed out, i.e. the partial effect controlling for
> `fraction of stage-s tokens already routed to e` is still positive at p < 0.05.

**H_pos is SUPPORTED iff both legs pass.** **H_pos is REJECTED if Leg 2 fails**, regardless of Leg 1 —
a correlational-only result is not sufficient, per methodology control #5 (in H1 a 7x causal effect
coexisted with cNMI 0.0193; correlational metrics understate, so they cannot be the decider).

**L1 learning gate, unchanged in form:** `acc >= 1.30 x maj|(stage, pos)` AND `probe >= 0.85` in >=4/5
seeds. The longer random prefix may depress accuracy. **If L1 fails the arm is INCONCLUSIVE, not
evidence for or against H_pos** — an unlearned model routes on nothing and says nothing about routing.

**Minimum detectable effect, declared so a null is interpretable.** From H2's observed cell sd of
**0.0361**, a paired test at n=20, alpha=0.05 two-sided, power 0.8 detects `L_median - L_native` down
to about **0.024** accuracy points. A null below that magnitude is **underpowered, not evidence of
absence**, and must be reported as such.

## A5.5 — Declared costs and confounds of this arm

1. **Widening the cross makes segment 1 *less* CFOP-faithful, not more.** At `(4,8)` the "cross"
   segment is already only "undo 4–8 random moves," not a cross solve; at `(0,25)` it is "undo up to
   25 random moves" and grows to the majority of tokens. **We are trading CFOP fidelity for positional
   decorrelation, and that is a real cost.** Stages 2–4 (F2L/OLL/PLL) remain canonical and genuine.
   **Pre-registered secondary endpoint:** both legs recomputed on **stages 2–4 only**. Declared now, in
   advance, precisely so it cannot be reached for afterwards if the 4-stage result disappoints. The
   4-stage version remains primary.
2. **`maxlen` differs from the H2 arm** (144 vs 129). Mitigated by the within-model endpoint (above).
3. **Solutions are not optimal.** Unchanged from H2 — inherited, not introduced.
4. **A pass would be consistent with H_pos but would not isolate position as the *only* gate.**
   Widening the cross also changes stage class balance and mean solve length. A pass licenses "position
   decorrelation was sufficient here," not "position is the unique mechanism."
5. **Five seeds.** Routing-metric seed sd ran 0.015–0.048 in this project while accuracy sd was
   ~0.002. Five seeds is the established budget, and it is thin; see the MDE above.

## A5.6 — Cost

~45 min total on local MPS (H2 trained at ~7 min/model; longer sequences add a little), 10 models
(5 MoE + 5 hash). **No Modal credit** — all three workspaces are exhausted. Analysis reuses
`src/h2_analysis.py` and `src/h2_causal.py` unchanged apart from the data path and `maxlen`.

## A5.7 — What would falsify H_pos

Stage alignment stays at the shuffled null (Leg 1 fails) **and** `L_native ~= L_median` (Leg 2 fails)
on data whose removed-by-position sits in the Kociemba regime of 50–65%. That outcome would mean
positional availability was **not** what suppressed stage routing in H2, the A5.3 mechanism is wrong,
and the honest conclusion becomes: the router recovered G1 and did not recover CFOP stages, with the
reason **unexplained**. That result will be reported with the same prominence as a pass.
