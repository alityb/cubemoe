# PRE-REGISTRATION #15: On cube states never seen in training, does the CFOP routing path carry the stage?

Written 2026-11-09, before any measure below has been computed on these positions. Motivated by a **descriptive**
result of #13 (§43): on the 32.6% of fresh positions whose exact state never occurs in the training split, the
six-layer routing path of the 8 CFOP models carried 0.20-0.34 of H(stage | S), Kociemba-trained routers' paths
0.012-0.034, and the CFOP path was additive across layers (one-hot minus additive 0.008-0.050). Those numbers were
seen. Before writing this, one further property of the #13 positions was checked: **every OLL and PLL state there
recurs verbatim in training, so the novel positions are cross (100% novel) and F2L (79% novel) only.** The #13
"novel-state" result is therefore a cross-vs-F2L result, and this test is scoped to it. Scripts were smoke-tested on
40 solves of the new positions, outputs deleted unread (file structure and exit codes only); the rotated control was
checked on old solves 0-149 (pair entropies only). The design was revised once after an independent review (end).

## Positions and models

CFOP `cfop250k` held-out solves **4,800-7,199** of the RandomState(0) held-out ordering, never used by any analysis or
by the training gate (which read the first ~3,300 held-out solves); `src/novel_paths.py` asserts disjointness from the
training split `perm[:ntr]`. Scored positions: those whose exact 54-sticker state never occurs in the training split
("novel"); any novel OLL/PLL position is dropped and counted; label cross vs F2L; S = moves remaining; halves A/B = even/odd solve. Decision token, six layers.
Models: C = `cfop_mixtral_6L256_s{0..7}`; controls K = `mixtral_6L256_s{0..4}`, and for each C model its **rotated**
control: at every layer the model's own router input read by W q (trained router W, q Haar-random orthogonal, fixed
seed per layer), i.e. the same trained representation hashed by a router with the trained singular values in random
directions, with 28 per-pair biases fitted (labels unused) so that its pair distribution at each layer matches the
trained router's on the scored positions (on old solves 0-149 this matched pair entropy to within 0.01 bits per layer).
Without the biases the rotated paths have 2-4 bits more pair entropy than the trained ones; that raw version is
descriptive only. (This replaces #13's untrained models, whose routers collapsed onto one expert pair.)

## Measure

I(R) = [CE(stage | S) − CE(stage | S, R)] / H(stage | S), identical to #13 (`src/paths_fresh.info`: gradient-boosted
classifiers, 200 iterations, 15 leaves, L2 = 1, no early stopping; fit on one half, score on the other, both
directions; null = routing rows permuted within S × half; one-hot unconstrained vs additive-over-layers comparator).

## Predictions (each must hold in both directions, as in #13)

- **N0 (instrument):** every null in [−0.08, 0.02]. A C null out of bounds voids the test; a K or rotated null voids
  only N2 or N3.
- **N1 (path carries it):** in ≥ 6/8 C models, I(all) ≥ 0.10 and I(all) − max single layer ≥ 0.05.
- **N2 (beyond Kociemba-trained routers):** in ≥ 6/8 C models, I(all) − max over K of I(all) ≥ 0.10 AND
  I(L0–4) − max over K of I(L0–4) ≥ 0.10.
- **N3 (beyond a random readout of the same representation):** in ≥ 6/8 C models, I(all) − I(all, own pair-matched
  rotated) ≥ 0.10. N3 is evaluable only if the rotated nulls are in bounds and the mean per-layer pair-entropy gap
  between trained and matched rotated paths is ≤ 0.5 bits in ≥ 6/8 C models; otherwise it is descriptive.
- **N4 (how the layers combine), two-sided:** "combinatorial" if in ≥ 6/8 C models one-hot − additive ≥ 0.05;
  "additive" if in ≥ 6/8 it is < 0.05 (as #13's descriptive result); otherwise "mixed".

## Reading rules, written before results

- N0 fails for a C model: void. N1 fails: #13's descriptive result does not replicate (N2, N3 not read).
- If N1 holds, N2 and N3 are read independently (each may also be void/not evaluable as above):
  - N2 holds / fails: the path's stage information on novel states is / is not beyond Kociemba-trained routers.
  - N3 holds: beyond a pair-matched random readout of the CFOP model's own representation, i.e. the trained routing
    *selects* for the stage along the path on novel states. N3 fails: no more than such a readout; the stage is in the
    representation and the routing does not select for it (as #12's Part A found at the final layer).
  - Both hold: the strongest reading; N3 without N2, or N2 without N3, is reported as exactly that combination.
  - Every reading is qualified by N4.
Even if all hold, this is information in the routing, not evidence that the path does stage-specific work (that is
#14's question).

Reported descriptively: every layer's I and CE in bits, H(stage | S), I(L0–4), pair entropies (trained and rotated),
novel stage counts, the novel fraction of F2L positions, the number of dropped novel OLL/PLL positions, the fraction
of novel positions whose state occurs in both halves, and the raw (unmatched) rotated control.

## Known weaknesses, stated now

- Only two stages are testable on novel states; "stage" here means cross vs F2L.
- Cross is entirely novel and F2L mostly novel, so novelty cannot be separated from stage within this design; the
  claim is about states never seen, not about generalisation versus memorisation as such.
- "Novel" is exact-state novelty; states equal to a training state up to a cube symmetry or recolouring count as novel.
- Same C and K models and method as #12 and #13; only positions and the rotated control are new. One rotation per C
  model (fixed seed), not a distribution.
- Overfitting biases as in #13: N1's single-layer comparison is biased toward failing; N2 and N3 compare six with six.
- Correlational; S is conditioned on as in every earlier analysis.
- K routers are near-degenerate at some layers on these states (in #13's novel subset K s0's final-layer pair entropy
  was 0.03 bits), so N2 partly tests "beyond a near-constant layer"; N3's pair-matched control is the stronger test.
- Cross vs F2L given S is close to a simple state feature (is the cross solved?), so "carries the stage" means
  "carries an easy state feature"; I is not compared against a sticker classifier (which would reach ≈ 1).
- The smoke test touched the confirmation solves (outputs unread); in hindsight old solves would have been cleaner.

## Review (before running)

An independent review (3 must-fix for this prereg) found, and this version fixes: (1) a single novel OLL/PLL state
would have crashed every run: such positions are now dropped and counted; (2) the rotated control was not
entropy-matched (2-4 bits higher): per-pair biases now match its pair distribution, with an evaluability rule;
(3) the report's decision chain did not read N3 when N2 failed or was void: N2 and N3 are now read independently.
Also adopted: the K-degeneracy and easy-feature caveats, the within-held-out cross-half duplicate count.
