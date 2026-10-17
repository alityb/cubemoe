"""Policy alone (greedy argmax, no search): how far does it get? Batched rollouts; for every cube, did it ever reach
the cross / F2L / OLL / solved, and where does it stall (repeating positions)?
Usage: python src/cfop_greedy_stages.py -> out/prod/cfop_greedy_stages.json"""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A, cube_fast as C
from cfop_verify import flags
R = '/Users/alityb/projects/cubemoe'
MP = torch.from_numpy(C.MOVE_PERM)
d = dict(np.load(f'{R}/data/cfop250k.npz')); N = len(d['sol_len'])
held = d['init_state'][np.random.RandomState(0).permutation(N)[int(N * 0.9):][:500]].astype(np.int64)
rnd = A.random_cubes(500, 2026).numpy().astype(np.int64)
out = {}
for s in range(5):
    tag = f'cfop_mixtral_6L256_s{s}'; m, _ = A.load(tag)
    for name, X0, budget in (('held-out scrambles', held, 90), ('random cubes', rnd, 120)):
        X = torch.from_numpy(X0); n = len(X); reach = np.zeros((n, 4), bool); done_at = np.full(n, -1); seen = [set() for _ in range(n)]; loop = np.zeros(n, bool)
        for t in range(budget + 1):
            c, f, o = flags(X.numpy()); solved = (X.numpy() == C.SOLVED).all(1)
            reach |= np.stack([c, f, o, solved], 1); done_at[(done_at < 0) & solved] = t
            for i in np.flatnonzero(~solved & ~loop):
                h = X[i].numpy().tobytes()
                if h in seen[i]: loop[i] = True
                seen[i].add(h)
            if t == budget: break
            with torch.no_grad(): mv = A.logits(m, X).argmax(-1)
            nxt = torch.gather(X, 1, MP[mv]); X = torch.where(torch.from_numpy(solved)[:, None], X, nxt)
        r = dict(solved=float(reach[:, 3].mean()), cross=float(reach[:, 0].mean()), f2l=float(reach[:, 1].mean()), oll=float(reach[:, 2].mean()),
                 loops_among_unsolved=float(loop[~reach[:, 3]].mean()) if (~reach[:, 3]).any() else 0.0,
                 moves=float(done_at[done_at >= 0].mean()) if (done_at >= 0).any() else None)
        out[f'{tag} {name}'] = r
        print(f"{tag} {name:18s}: policy alone ever reaches cross {r['cross']*100:5.1f}% | F2L {r['f2l']*100:5.1f}% | OLL {r['oll']*100:5.1f}% | "
              f"solved {r['solved']*100:5.1f}% ({r['moves'] or 0:.1f} mv) | unsolved that fell into a loop {r['loops_among_unsolved']*100:.0f}%", flush=True)
json.dump(out, open(f'{R}/out/prod/cfop_greedy_stages.json', 'w'), indent=1)
