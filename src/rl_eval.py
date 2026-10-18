"""Evaluation for the RL routing study (prereg/PREREG_RL_ROUTING.md).

  python src/rl_eval.py bins <tag>          BASE model only: per-scramble pass rate over 64 samples (temp 1, sampling
                                            seed 1) on 2,000 held-out generator scrambles and 1,000 uniformly random
                                            cubes -> difficulty bins. out/rl/bins_<tag>.json
  python src/rl_eval.py eval <tag> <run>    <run> = 'base' or an RL run name (out/rl/<run>.pt). On the binned scrambles:
                                            sampled pass rate (16 samples, temp 1, sampling seed 2, identical for every
                                            model = common random numbers), greedy solve, furthest CFOP stage reached
                                            under greedy. out/rl/eval_<run or tag_base>.json
Bins (fixed here, before any RL): EASY p >= 8/64, HARD p <= 1/64, MID otherwise, where p is the base pass rate from
the bins step. Gains are always measured against the base model re-evaluated by the eval step (fresh samples), never
against p itself, so regression to the mean cancels.
"""
import sys, os, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from rl_common import rollout, scrambles
R = '/Users/alityb/projects/cubemoe'; OUT = f'{R}/out/rl'

def sets():
    held, idx = scrambles('test', 2000); rnd = A.random_cubes(1000, 2026)
    return {'held': held, 'random': rnd.long()}

def per_scramble_pass(m, X, k, seed, budget=90, chunk=500):
    torch.manual_seed(seed); out = []
    for i in range(0, len(X), chunk):
        r = rollout(m, X[i:i + chunk], budget, temp=1.0, G=k); out.append(r['solved'].view(-1, k).float().mean(1))
    return torch.cat(out).numpy()

def bins(tag):
    m, _ = A.load(tag); res = {}
    for name, X in sets().items():
        budget = 90 if name == 'held' else 120
        p = per_scramble_pass(m, X, 64, seed=1, budget=budget)
        b = np.where(p >= 8 / 64, 'EASY', np.where(p <= 1 / 64, 'HARD', 'MID'))
        res[name] = dict(p=p.tolist(), bin=b.tolist(), counts={k: int((b == k).sum()) for k in ('EASY', 'MID', 'HARD')}, budget=budget)
        print(f"{tag} {name}: base pass rate {p.mean()*100:.1f}% | bins {res[name]['counts']}", flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f'{OUT}/bins_{tag}.json', 'w'))

def load_run(tag, run):
    m, _ = A.load(tag)
    if run != 'base':
        ck = torch.load(f'{OUT}/{run}.pt', map_location='cpu', weights_only=False)
        if ck['arm'] == 'lora':
            from rl_train import setup
            setup(m, 'lora')
        m.load_state_dict(ck['sd']); m.to(A.DEV).eval()
    return m

def evaluate(tag, run):
    m = load_run(tag, run); B = json.load(open(f'{OUT}/bins_{tag}.json')); res = {}
    for name, X in sets().items():
        budget = B[name]['budget']; b = np.array(B[name]['bin'])
        p = per_scramble_pass(m, X, 16, seed=2, budget=budget)
        g = rollout(m, X, budget, temp=0, track=True)
        gs, st = g['solved'].numpy(), g['stage'].numpy()
        res[name] = {k: dict(n=int((b == k).sum()), sampled=float(p[b == k].mean()), greedy=float(gs[b == k].mean()),
                             greedy_reach_F2L=float((st[b == k] >= 2).mean())) for k in ('EASY', 'MID', 'HARD')}
        res[name]['ALL'] = dict(n=len(b), sampled=float(p.mean()), greedy=float(gs.mean()), greedy_reach_F2L=float((st >= 2).mean()))
        res[name]['per_scramble_sampled'] = p.tolist()
        print(f"{run} {name}: " + " | ".join(f"{k} n={v['n']} sampled {v['sampled']*100:.1f}% greedy {v['greedy']*100:.1f}%"
                                             for k, v in res[name].items() if k != 'per_scramble_sampled'), flush=True)
    json.dump(res, open(f"{OUT}/eval_{run if run != 'base' else tag + '_base'}.json", 'w'))

if __name__ == '__main__':
    if sys.argv[1] == 'bins': bins(sys.argv[2])
    else: evaluate(sys.argv[2], sys.argv[3])
