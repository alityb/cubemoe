"""Save beam-256 solution paths of one CFOP policy on the cfop_verify scrambles, for algorithm-level analysis.
Usage: python src/cfop_paths.py <tag> [n]  -> out/prod/<tag>_paths.json"""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
R = '/Users/alityb/projects/cubemoe'
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
d = dict(np.load(f'{R}/data/cfop250k.npz')); N = len(d['sol_len'])
te = np.random.RandomState(0).permutation(N)[int(N * 0.9):][:n]; off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
m, _ = A.load(tag); out = []
for i in te:
    p = A.beam_solve_path(m, torch.from_numpy(d['init_state'][i].astype(np.int64)), width=256, budget=90)
    out.append(dict(solve=int(i), start=d['init_state'][i].tolist(), model=p, solver=[int(x) for x in d['moves'][off[i]:off[i + 1]]]))
    print(len(out), len(p) if p else None, flush=True)
json.dump(out, open(f'{R}/out/prod/{tag}_paths.json', 'w'))
