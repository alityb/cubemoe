"""Pre-registered test (MEMORY.md, commit d3a003b): does routing follow what changes the output?
For every (solver, latent variable V): x = I(V; next move | S)/H(move | S), y = I(V; expert | S)/H(expert | S),
S = moves remaining, last-layer decision-token top-1 expert, permutation-corrected, 1200 held-out solves, 5 seeds.
Usage: python src/routing_vs_action.py  -> out/prod/routing_vs_action.json"""
import sys, json, numpy as np
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from prod_cfop import heldout_cfop
from explore_cfop_analyze import corrected, cent
from scipy.stats import spearmanr
R = '/Users/alityb/projects/cubemoe'
VARS = {'kociemba': ['phase', 'd1', 'tw', 'fl', 'so', 'move'], 'cfop': ['stage', 'seg', 'segpos', 'firstmidlast', 'case', 'move']}
data = {'kociemba': A.heldout(1200), 'cfop': heldout_cfop(1200)}
out = {}
for solver, (X, lab) in data.items():
    S = lab['rem']; Mv = lab['move']; rng = np.random.RandomState(0)
    xs = {v: corrected(Mv, lab[v], S, rng) / cent(Mv, S) for v in VARS[solver] if v != 'move'}
    xs['move'] = 1.0
    x2 = {v: corrected(Mv, lab[v], S, rng) / cent(lab[v], S) for v in VARS[solver]}
    ys = {v: [] for v in VARS[solver]}
    for s in range(5):
        tag = f'mixtral_6L256_s{s}' if solver == 'kociemba' else f'cfop_mixtral_6L256_s{s}'
        m, _ = A.load(tag); cap = A.capture(m, X); E = cap['route'][len(m.blocks) - 1][:, 0].numpy(); H = cent(E, S)
        for v in VARS[solver]: ys[v].append(corrected(E, lab[v], S, rng) / H)
        print(f"{tag}: " + ", ".join(f"{v} {ys[v][-1]*100:.1f}%" for v in VARS[solver]), flush=True)
    for v in VARS[solver]:
        out[f'{solver}:{v}'] = dict(x=float(xs[v]), x2=float(x2[v]), y=float(np.mean(ys[v])), y_sd=float(np.std(ys[v])), y_seeds=[float(a) for a in ys[v]])
pts = {k: v for k, v in out.items() if not k.endswith(':move')}
rho, p = spearmanr([v['x'] for v in pts.values()], [v['y'] for v in pts.values()])
rho2, p2 = spearmanr([v['x2'] for v in pts.values()], [v['y'] for v in pts.values()])
print(f"\n{'point':24s} {'x = V explains move':>20s} {'x2':>8s} {'y = routing share':>20s}")
for k, v in sorted(out.items(), key=lambda kv: -kv[1]['x']):
    print(f"{k:24s} {v['x']*100:19.1f}% {v['x2']*100:7.1f}% {v['y']*100:13.1f} ± {v['y_sd']*100:.1f}%")
print(f"\nPRIMARY: Spearman rho(x, y) over {len(pts)} points = {rho:.2f} (p={p:.3f}); prediction rho > 0.5")
print(f"phase vs stage: x {out['kociemba:phase']['x']*100:.1f}% vs {out['cfop:stage']['x']*100:.1f}%, y {out['kociemba:phase']['y']*100:.1f}% vs {out['cfop:stage']['y']*100:.1f}%")
print(f"exploratory: Spearman rho(x2, y) = {rho2:.2f} (p={p2:.3f})")
json.dump(dict(points=out, rho=rho, p=p, rho_x2=rho2, p_x2=p2), open(f'{R}/out/prod/routing_vs_action.json', 'w'), indent=1)
