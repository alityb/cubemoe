"""Stage A: is the solving coming from the model, or from the search? Writes key 'A' of <tag>_analysis.json."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, '/Users/alityb/projects/cubemoe/src')
import prod_analysis as A, cube_fast as C
from model_prod import ProdStateModel
import os
SMOKE = os.environ.get('SMOKE') == '1'
tag = sys.argv[1]; R = '/Users/alityb/projects/cubemoe'; path = f'{R}/out/prod/{tag}_analysis{"_smoke" if SMOKE else ""}.json'
try: out = json.load(open(path))
except Exception: out = {}
m, cfg = A.load(tag); X, lab = A.heldout(200); res = {}
res['attention_readout_max_diff'] = A.check_attention(m, X[:512]); print('attention readout diff', res['attention_readout_max_diff'], flush=True)
cubes = A.random_cubes(3 if SMOKE else 60, 777)
NS = 3 if SMOKE else None
def run(model, W, n):
    t0 = time.time(); paths = [A.beam_solve_path(model, c, W, 40) for c in cubes[:n]]; ok = [p for p in paths if p is not None]
    return paths, dict(width=W, n=n, solved=len(ok), avg_moves=float(np.mean([len(p) for p in ok])) if ok else None, seconds=time.time() - t0)
_, res['no_model_beam1024'] = run(A.Uniform(), 1024, NS or 30); print('no model', res['no_model_beam1024'], flush=True)
torch.manual_seed(0); un = ProdStateModel(d=cfg['d'], L=cfg['L'], nh=cfg['nh'], preset=cfg['preset']).to(A.DEV).eval()
_, res['untrained_beam256'] = run(un, 256, NS or 30); print('untrained', res['untrained_beam256'], flush=True)
paths, res['trained_beam256'] = run(m, 256, NS or 60); print('trained', res['trained_beam256'], flush=True)
_, res['trained_greedy'] = run(m, 1, NS or 60); print('trained greedy', res['trained_greedy'], flush=True)
ranks = []
for c, p in zip(cubes, paths):
    if p is None: continue
    s = c[None].numpy()
    for mv in p:
        with torch.no_grad(): lg = m(torch.from_numpy(s).to(A.DEV))[0][0].float().cpu()
        ranks.append(int((lg > lg[mv]).sum())); s = C.apply(s, mv)
ranks = np.array(ranks)
res['path_rank_share'] = {'1st': float(np.mean(ranks == 0)), '2nd': float(np.mean(ranks == 1)), '3rd': float(np.mean(ranks == 2)), 'beyond': float(np.mean(ranks >= 3)), 'n_moves': int(len(ranks))}
print('path ranks', res['path_rank_share'], flush=True)
d = dict(np.load(f'{R}/data/mixed250k.npz')); S = d['states'].astype(np.uint8); train = set()
for r in range(8):
    Sr, _ = C.rotate(S, np.zeros(1, int), r); train.update(map(bytes, Sr))
big = A.random_cubes(20 if SMOKE else 500, 2026).numpy().astype(np.uint8)
res['memorisation'] = dict(eval_cubes=len(big), found_in_training=int(sum(bytes(x) in train for x in big)), distinct_training_positions=len(train))
print('memorisation', res['memorisation'], flush=True)
out['A'] = res; json.dump(out, open(path, 'w'), indent=1, default=float)
