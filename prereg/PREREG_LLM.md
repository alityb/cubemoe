# PRE-REGISTRATION — the cube principle in a real MoE language model (OLMoE)

Filed **before OLMoE is loaded or run**. The Granite pilot (MEMORY.md §20) was exploratory; every
threshold below was fixed after seeing it, and OLMoE has not been looked at. Pilot numbers are quoted
so the reader can see what was known when these rules were written.

## Model and data (fixed)
- `allenai/OLMoE-1B-7B-0924` (the model audited by Engmann et al., arXiv 2606.10703), bf16, MPS.
- The same `llm/corpus.json` and chunking as the pilot: 16 prose (WikiText-2 test), 16 code (Python
  stdlib), 8 license chunks, 512 tokens each, built by the same `chunks()` code with OLMoE's tokenizer.
- Token groups, same thresholds as the pilot: among tokens with p_full > 0.5, **needs-context** =
  p(next | last 2 tokens) < 0.1; **local** = p(next | last 2 tokens) > 0.5.
- Knockout layers: **3, 7, 11, 15** (of 16). Every one of the 64 experts knocked out singly at each
  layer; each paired with a size-matched random-noise control (per-token noise norm = the knockout's
  change to that layer's MoE output, same tokens), exactly as in `llm/pilot.py`.
- Routing analysis: all 16 layers, top-1 expert, chance-corrected share of routing explained by the
  current token, groups subsampled to equal size (3 draws).

## Predictions and pass rules
**A — routing follows the current token less where context matters.**
PASS iff the token-explained share of routing is lower for needs-context than for local tokens in
**>= 6 of the 8 layers 8-15**. (Pilot: 18/19 layers from layer 5 on.)

**B — experts are specialists for needs-context tokens beyond fragility.**
Expert effect = total knockout damage / total matched-noise damage. PASS iff the expert effect is
larger for needs-context than for local tokens in **>= 3 of 4** knockout layers. (Pilot: 4/6 layers,
small margins. **Expected to be weak; failure is an informative outcome and will be reported.**)

**C — memorized text carries the expert-specific effect (the CFOP analog).**
On needs-context tokens, PASS iff the expert effect on license text exceeds BOTH prose and code in
**>= 3 of 4** knockout layers. (Pilot: license highest in 4/6 layers; last layer 2.29x vs 0.54x/0.57x.)

## Reporting
All three verdicts are reported whatever they are, with the absolute damage sizes (in the pilot these
were ~0.004 nats — tiny). A pass licenses "the pattern replicates in a real 7B-parameter MoE LM", not
"it holds in LLMs generally". Granite results stay labelled exploratory.
