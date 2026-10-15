"""CFOP analysis of a production-style policy (the CFOP counterpart of prod_stages.stage_E + expert analysis).
Usage: python src/prod_cfop.py <tag>   -> out/prod/<tag>_analysis.json, key 'CFOP'

  1. routing: share of the decision token's last-layer routing explained by stage (at fixed moves-remaining),
     vs a shuffled router; plus segment (cross / slot1-4 / OLL / PLL), position inside the algorithm, which
     algorithm, next move, best single sticker
  2. knockouts: remove each last-layer expert -> drop in p(solver's move) per STAGE; does each expert own one
     stage (as Kociemba experts own one phase) and how many experts matter for more than one stage?
  3. move ownership: effective number of target moves each knockout damages
"""
import sys, os, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from explore_cfop_analyze import corrected, cent
R = '/Users/alityb/projects/cubemoe'; STAGES = {1: 'cross', 2: 'F2L', 3: 'OLL', 4: 'PLL'}
SMOKE = os.environ.get('SMOKE') == '1'

def heldout_cfop(n_solves, data='cfop250k'):
    d = dict(np.load(f'{R}/data/{data}.npz')); meta = dict(np.load(f'{R}/out/explore/cfopmeta_{data}.npz'))
    n = len(d['sol_len']); te = np.random.RandomState(0).permutation(n)[int(n * 0.9):][:n_solves]
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in te])
    sl = np.concatenate([np.full(int(d['sol_len'][i]), int(d['sol_len'][i])) for i in te]); t = d['pos'][rows].astype(int)
    b = (np.minimum(meta['segpos'][rows], 1) + (meta['segpos'][rows] == meta['seglen'][rows] - 1) * (meta['seglen'][rows] > 1))
    lab = dict(stage=d['phase'][rows].astype(int), seg=meta['seg'][rows].astype(int), segpos=np.minimum(meta['segpos'][rows], 15),
               firstmidlast=b, case=meta['case'][rows], move=d['moves'][rows].astype(int), rem=sl - t)
    return torch.from_numpy(d['states'][rows].astype(np.int64)), lab

def run(tag):
    m, cfg = A.load(tag); X, lab = heldout_cfop(40 if SMOKE else 1200); out = {}
    cap = A.capture(m, X); rng = np.random.RandomState(0); L = len(m.blocks) - 1
    # 1. routing
    E = cap['route'][L][:, 0].numpy(); strat = lab['rem']; H = cent(E, strat); st = lab['stage']
    share = {k: corrected(E, lab[k], strat, rng) / H for k in ('stage', 'seg', 'segpos', 'firstmidlast', 'case', 'move')}
    Esh = E.copy()
    for s in np.unique(strat):
        k = np.flatnonzero(strat == s); Esh[k] = Esh[rng.permutation(k)]
    share['stage_shuffled'] = corrected(Esh, st, strat, rng) / cent(Esh, strat)
    S0 = np.zeros(len(E), int); H0 = cent(E, S0)
    share['best_single_sticker'] = max(corrected(E, X[:, j].numpy(), S0, rng) / H0 for j in range(54) if j % 9 != 4)
    share['stage_unconditioned'] = corrected(E, st, S0, rng) / H0
    out['routing_last_layer'] = share
    print(f"{tag}: last-layer decision routing explained by STAGE (fixed moves-remaining) {share['stage']*100:.1f}% "
          f"(shuffled {share['stage_shuffled']*100:.1f}%); segment {share['seg']*100:.1f}%, position in algorithm "
          f"{share['segpos']*100:.1f}%, which algorithm {share['case']*100:.1f}%, next move {share['move']*100:.1f}%", flush=True)
    owner = {}
    for e in range(m.blocks[L].ffn.E):
        sel = E == e
        if sel.sum() < 50: continue
        owner[e] = {STAGES[s]: float((st[sel] == s).mean()) for s in STAGES}
    out['expert_stage_mix'] = owner
    # 2-3. knockouts
    Y = torch.from_numpy(lab['move']); p0 = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; ko = {}
    for e in range(m.blocks[L].ffn.E):
        m.blocks[L].ffn._knockout = {e}; pe = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; m.blocks[L].ffn._knockout = set()
        dd = (p0 - pe).numpy(); row = {STAGES[s]: float(dd[st == s].mean()) for s in STAGES}
        bt = np.array([max(dd[lab['move'] == t].mean(), 0) if (lab['move'] == t).any() else 0 for t in range(18)])
        q = bt / max(bt.sum(), 1e-12); q = q[q > 0]
        row['effective_moves'] = float(np.exp(-(q * np.log(q)).sum())) if bt.sum() > 1e-6 else None
        ko[e] = row
    out['knockout'] = ko
    multi = [e for e, r in ko.items() if sum(r[s] > 0.02 for s in STAGES.values()) > 1]
    print("  knockouts (drop in p(solver's move) by stage):")
    for e, r in ko.items():
        print(f"    e{e}: " + "  ".join(f"{s} {r[s]*100:+5.1f}" for s in STAGES.values()) + f"  | effective #moves {r['effective_moves'] or 0:.1f}", flush=True)
    print(f"  experts that matter (>2 pts) for MORE THAN ONE stage: {multi if multi else 'none'}", flush=True)
    out['multi_stage_experts'] = multi
    path = f'{R}/out/prod/{tag}_analysis{"_smoke" if SMOKE else ""}.json'
    try: full = json.load(open(path))
    except Exception: full = {}
    full['CFOP'] = out; json.dump(full, open(path, 'w'), indent=1, default=float)

if __name__ == '__main__':
    run(sys.argv[1])
