"""Report for modal/prod.py::cfop_random_greedy: solve rate, length, CFOP stage milestones and last-layer alphabet
on uniformly random cubes (beam 256/1024, greedy) and greedy on held-out CFOP scrambles."""
import sys, json, numpy as np
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
from cfop_ll import states, strip_auf
from cfop_verify import flags, first
from cfop_alphabet import algs, LIB
R = '/Users/alityb/projects/cubemoe'; out = {}
for run in ('held_greedy', 'random_greedy', 'random_w256', 'random_w1024'):
    rows = []
    for s in range(5):
        tag = f'cfop_mixtral_6L256_s{s}'
        try: P = json.load(open(f'{R}/out/prod/{tag}_{run}_paths.json'))
        except FileNotFoundError: continue
        ok = [p for p in P if p['model']]; ms = dict(rate=len(ok) / len(P), n=float(np.mean([len(p['model']) for p in ok])) if ok else float('nan'))
        if ok:
            mil = []; A = []
            for p in ok:
                S = states(p['start'], p['model']); c, f, o = flags(S); n = len(p['model'])
                mil.append((first(c) / n, first(f) / n, first(o) / n)); A.append(algs(p['start'], p['model']))
            mil = np.array(mil); flat = [a for x in A for a in x]; lab = [LIB.get(strip_auf(a), '') for a in flat]
            ms.update(cross=mil[:, 0].mean(), f2l=mil[:, 1].mean(), oll=mil[:, 2].mean(), algs_per_ll=float(np.mean([len(x) for x in A])),
                      known=float(np.mean([l != '' for l in lab])) if lab else float('nan'),
                      other_side=float(np.mean(['other side' in l for l in lab if l])) if any(lab) else float('nan'))
        rows.append(ms); out[f'{tag}_{run}'] = ms
    if not rows: continue
    g = lambda k: (np.nanmean([r.get(k, np.nan) for r in rows]), np.nanmin([r.get(k, np.nan) for r in rows]), np.nanmax([r.get(k, np.nan) for r in rows]))
    print(f"{run:14s} ({len(rows)} seeds): solved {g('rate')[1]*100:.0f}-{g('rate')[2]*100:.0f}% | moves {g('n')[0]:.1f} | cross/F2L/OLL first reached at "
          f"{g('cross')[0]*100:.0f}/{g('f2l')[0]*100:.0f}/{g('oll')[0]*100:.0f}% | algs per LL {g('algs_per_ll')[0]:.2f} | known algs {g('known')[0]*100:.0f}%, "
          f"other side {g('other_side')[0]*100:.0f}%")
json.dump(out, open(f'{R}/out/prod/cfop_random_greedy.json', 'w'), indent=1, default=float)
