"""Builds out/prod/OVERNIGHT_REPORT.md from the training logs, evaluations and analysis JSONs."""
import json, glob, os, numpy as np
R = '/Users/alityb/projects/cubemoe/out/prod'; L = []
def j(p):
    try: return json.load(open(p))
    except Exception: return None
L.append('# Overnight results: production-style (Mixtral-style) cube MoE\n')
L.append('## 1. Training and solving, per seed\n\n| seed | next-move acc | greedy solve | beam-256 (training check, 40 cubes) | 500-cube eval: beam 64 / 256 / 1024 |\n|---|---|---|---|---|')
for s in range(5):
    lg = j(f'{R}/mixtral_6L256_s{s}.json'); ev = j(f'{R}/mixtral_6L256_s{s}_eval.json')
    if not lg: L.append(f'| {s} | (not finished) | | | |'); continue
    g = lg['gates'][-1] if lg.get('gates') else {}
    evs = ' / '.join(f"{ev[f'beam{w}']['solved']}/{ev['n']} ({ev[f'beam{w}']['avg_moves']:.1f} mv)" if ev and f'beam{w}' in ev and ev[f'beam{w}']['avg_moves'] else '–' for w in (64, 256, 1024))
    L.append(f"| {s} | {g.get('heldout_acc',0)*100:.1f}% | {g.get('greedy_solve_random',0)*100:.1f}% | {g.get('beam256_solve_random',0)*100:.0f}% | {evs} |")
a = j(f'{R}/mixtral_6L256_s0_analysis.json') or {}
if 'A' in a:
    A = a['A']; L.append('\n## 2. Is it the model or the search? (seed 0)\n')
    L.append(f"- Beam search with **no model** (all moves equally likely), width 1024: **{A['no_model_beam1024']['solved']}/{A['no_model_beam1024']['n']}** random cubes solved")
    L.append(f"- Beam search with an **untrained** model, width 256: **{A['untrained_beam256']['solved']}/{A['untrained_beam256']['n']}**")
    L.append(f"- **Trained** model, width 256: **{A['trained_beam256']['solved']}/{A['trained_beam256']['n']}**; trained model with no search (greedy): {A['trained_greedy']['solved']}/{A['trained_greedy']['n']}")
    pr = A['path_rank_share']; L.append(f"- Along the solutions found, the move taken was the model's **own 1st choice {pr['1st']*100:.1f}%** of the time, 2nd {pr['2nd']*100:.1f}%, 3rd {pr['3rd']*100:.1f}%, lower {pr['beyond']*100:.1f}% ({pr['n_moves']} moves)")
    mm = A['memorisation']; L.append(f"- Memorisation check: **{mm['found_in_training']} of {mm['eval_cubes']}** evaluation cubes appear among {mm['distinct_training_positions']:,} training positions (incl. rotations)")
if 'B' in a:
    B = a['B']; L.append('\n## 3. Attention (seed 0)\n\n| attention | next-move acc | beam-256 solve (40 cubes) |\n|---|---|---|')
    for k, v in B['modes'].items(): L.append(f"| {k} | {v['acc']*100:.1f}% | {v['solve']*100:.0f}% |")
    L.append('\nAccuracy with attention removed in one layer only: ' + ', '.join(f"L{k} {v*100:.1f}%" for k, v in B['per_layer_zero'].items()))
    L.append('\nDecision-token attention to centre / edge / corner stickers (even spread = 10.9 / 43.6 / 43.6%): ' + '; '.join(f"L{k}: {v['centre']*100:.1f} / {v['edge']*100:.1f} / {v['corner']*100:.1f}" for k, v in B['att_share'].items()))
if 'C' in a:
    C = a['C']; cols = list(next(iter(C.values())).keys())
    L.append('\n## 4. Abstract cube properties readable from each layer (seed 0)\n\nIn-G1 = classification accuracy; others = R². "raw" = a linear readout of the input stickers.\n')
    L.append('| property | ' + ' | '.join(c.replace('raw stickers (input)', 'raw') for c in cols) + ' |'); L.append('|' + '---|' * (len(cols) + 1))
    for t, row in C.items(): L.append(f'| {t} | ' + ' | '.join(f'{row[c]:.2f}' for c in cols) + ' |')
if 'D_routing' in a:
    L.append('\n## 5. Experts: pattern matching or abstract properties? (seed 0)\n\n| layer | sticker tokens routed by own (position, colour) | decision token: best single sticker | phase | moves-to-G1 | bad edges | bad corners | next move | decision-routing entropy |\n|---|---|---|---|---|---|---|---|---|')
    for li, r in a['D_routing'].items():
        ab = r['cls_abstract']; L.append(f"| {li} | {r['sticker_tokens_by_own_position_colour']*100:.1f}% | {r['cls_best_single_sticker']*100:.1f}% | {ab['phase']*100:.1f}% | {ab['d1']*100:.1f}% | {ab['fl']*100:.1f}% | {ab['tw']*100:.1f}% | {ab['move']*100:.1f}% | {r.get('cls_routing_entropy_bits',0):.2f} bits{'' if r.get('reliable',True) else ' (too low: unreliable)'} |")
if 'D_knockout' in a:
    for li, rows in a['D_knockout'].items():
        L.append(f'\nKnockouts at layer {li} (drop in probability of the solver\'s move):\n\n| expert | phase 1 | phase 2 | ph1 edge-flip moves | ph1 corner-twist moves | effective # moves |\n|---|---|---|---|---|---|')
        for e, r in rows.items(): L.append(f"| e{e} | {r['phase 1']:+.3f} | {r['phase 2']:+.3f} | {r['ph1 F/B (edge flips)']:+.3f} | {r['ph1 R/L (corner twists)']:+.3f} | {r['effective_moves'] or 0:.1f} |")
L.append('\n## 6. The big picture: does the router split at the Kociemba phase boundary, and is it used?\n\n| seed | routing explained by phase | shuffled | phase-1 expert: share of ph1 / ph2 positions | KO it: ph1 / ph2 loss | KO typical other: ph1 / ph2 | force ph2 through it / typical other |\n|---|---|---|---|---|---|---|')
for s in range(5):
    e = (j(f'{R}/mixtral_6L256_s{s}_analysis.json') or {}).get('E')
    if e: L.append(f"| {s} | {e['phase_share']*100:.1f}% | {e['shuffled']*100:.1f}% | e{e['phase1_expert']}: {e['phase1_share']*100:.0f}% / {e['phase2_share_of_e1']*100:.1f}% | "
                   f"{e['ko_e1_phase1']*100:+.1f} / {e['ko_e1_phase2']*100:+.1f} pts | {e['ko_other_phase1']*100:+.1f} / {e['ko_other_phase2']*100:+.1f} pts | {e['loss_e1']*100:.1f}% / {e['median_other']*100:.1f}% |")
    else: L.append(f'| {s} | (not run) | | | | | |')
L.append('\n---\nGenerated automatically by src/overnight.sh. Exploratory: interpret with the caveats in docs/LAB_NOTEBOOK.md.')
open(f'{R}/OVERNIGHT_REPORT.md', 'w').write('\n'.join(L) + '\n'); print('report written')
