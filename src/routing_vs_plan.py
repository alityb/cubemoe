"""Pre-registered test (MEMORY.md, commit bc95fd7): does routing follow the plan (future moves), not just the next move?
Labels only; y (routing share) from out/prod/routing_vs_action.json.
Usage: python src/routing_vs_plan.py -> out/prod/routing_vs_plan.json"""
import sys, json, numpy as np
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from prod_cfop import heldout_cfop
from explore_cfop_analyze import corrected
from scipy.stats import spearmanr
R = '/Users/alityb/projects/cubemoe'; J = 10
VARS = {'kociemba': ['phase', 'd1', 'tw', 'fl', 'so'], 'cfop': ['stage', 'seg', 'segpos', 'firstmidlast', 'case']}
Y = json.load(open(f'{R}/out/prod/routing_vs_action.json'))['points']
from explore_cfop_analyze import cent
res = {}
for solver, (X, lab) in {'kociemba': A.heldout(1200), 'cfop': heldout_cfop(1200)}.items():
    S, Mv = lab['rem'], lab['move']; n = len(Mv); rng = np.random.RandomState(0)
    for v in VARS[solver]:
        I = []; xs = []
        for j in range(J):
            ok = np.flatnonzero(S > j); fut = Mv[np.minimum(ok + j, n - 1)]        # rows are contiguous per solve
            i = corrected(fut, lab[v][ok], S[ok], rng); I.append(i); xs.append(i / cent(fut, S[ok]))
        k = f'{solver}:{v}'; y = Y[k]['y']; x0 = Y[k]['x']
        res[k] = dict(x_plan=float(np.mean(xs)), per_step=[float(a) for a in xs], persistence=float(np.mean(I[1:]) / I[0]),
                      y=y, y_over_x=y / x0)
pts = list(res.values())
r1, p1 = spearmanr([p['x_plan'] for p in pts], [p['y'] for p in pts])
r2, p2 = spearmanr([p['persistence'] for p in pts], [p['y_over_x'] for p in pts])
kp = [v['persistence'] for k, v in res.items() if k.startswith('kociemba')]; cp = [v['persistence'] for k, v in res.items() if k.startswith('cfop')]
print(f"{'point':22s} {'x_plan':>8s} {'persistence':>12s} {'y':>7s} {'y/x':>6s}   per-step x (j=0..9)")
for k, v in sorted(res.items(), key=lambda kv: -kv[1]['persistence']):
    print(f"{k:22s} {v['x_plan']*100:7.2f}% {v['persistence']:12.2f} {v['y']*100:6.1f}% {v['y_over_x']:6.1f}   " + " ".join(f"{a*100:.1f}" for a in v['per_step']))
print(f"\nP1: rho(x_plan, y) = {r1:.2f} (p={p1:.3f}); phase x_plan {res['kociemba:phase']['x_plan']*100:.2f}% vs stage {res['cfop:stage']['x_plan']*100:.2f}%"
      f"  -> {'PASS' if r1 > 0.5 and res['kociemba:phase']['x_plan'] > res['cfop:stage']['x_plan'] else 'FAIL'}")
print(f"S1: min Kociemba persistence {min(kp):.2f} vs max CFOP {max(cp):.2f}; rho(persistence, y/x) = {r2:.2f} (p={p2:.3f})"
      f"  -> {'PASS' if min(kp) > max(cp) and r2 > 0.5 else 'FAIL'}")
json.dump(dict(points=res, P1=dict(rho=r1, p=p1), S1=dict(rho=r2, p=p2, min_k=min(kp), max_c=max(cp))), open(f'{R}/out/prod/routing_vs_plan.json', 'w'), indent=1)
