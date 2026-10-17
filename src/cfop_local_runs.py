"""Local version of modal/prod.py::cfop_random_greedy for a chosen run and seeds (same cubes, same output files).
Usage: python src/cfop_local_runs.py <run> <n> <seeds...>   run in held_greedy | random_greedy | random_w256"""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
R = '/Users/alityb/projects/cubemoe'
run, n, seeds = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
W, B = {'held_greedy': (1, 90), 'random_greedy': (1, 120), 'random_w256': (256, 120)}[run]
if run.startswith('held'):
    d = dict(np.load(f'{R}/data/cfop250k.npz')); N = len(d['sol_len'])
    te = np.random.RandomState(0).permutation(N)[int(N * 0.9):][:n]; off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    S = [d['init_state'][i].astype(int).tolist() for i in te]; sol = [[int(x) for x in d['moves'][off[i]:off[i + 1]]] for i in te]
else:
    S = A.random_cubes(200, 2026).tolist()[:n]; sol = [None] * len(S)            # same cubes as the Modal run
for s in seeds:
    tag = f'cfop_mixtral_6L256_s{s}'; m, _ = A.load(tag)
    P = [A.beam_solve_path(m, torch.tensor(x), W, B) for x in S]
    json.dump([dict(start=a, model=p, solver=q) for a, p, q in zip(S, P, sol)], open(f'{R}/out/prod/{tag}_{run}_paths.json', 'w'))
    ok = [len(p) for p in P if p]; print(f"{tag} {run}: solved {len(ok)}/{len(P)}, mean {np.mean(ok) if ok else float('nan'):.1f} moves", flush=True)
