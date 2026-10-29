"""Opening-swap test (prereg/PREREG_OPENING_SWAP.md): RL model at segment openings, base model elsewhere, plus a
random-position control and the reverse swap. Same scrambles, samples and seeds as src/rl_eval.py.
Usage: python src/rl_swap.py <tag> <run> [<run> ...]  -> out/rl/swap_<run>.json"""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
import cube_fast as C
from rl_common import amp, _CROSS, _F2L, _OLL, _SOL
from rl_eval import load_run, sets, OUT

_U, _R, _F, _D, _L, _B = (np.arange(9) + 9 * k for k in range(6))
SLOTS = [torch.from_numpy(x) for x in (np.r_[_F[5], _R[3], _D[2], _F[8], _R[6]], np.r_[_F[3], _L[5], _D[0], _F[6], _L[8]],
                                        np.r_[_B[5], _L[3], _D[6], _B[8], _L[6]], np.r_[_B[3], _R[5], _D[8], _B[6], _R[8]])]

def progress(X):
    sol = _SOL.to(X.device); ok = lambda idx: (X[:, idx.to(X.device)] == sol[idx.to(X.device)]).all(1).long()
    cross = ok(_CROSS)
    return cross + cross * sum(ok(s) for s in SLOTS) + ok(_OLL) + (X == sol).all(1).long()

@torch.no_grad()
def mixed(base, rl, starts, budget, mode, G=16, q=None, seed=2, bs=8192):
    """mode: 'open' (rl at openings), 'rev' (base at openings), 'rand' (rl w.p. q). Returns per-scramble pass rate, rl-step share."""
    dev = next(base.parameters()).device; torch.manual_seed(seed)
    MP = torch.from_numpy(C.MOVE_PERM).to(dev); SOLVED = torch.from_numpy(C.SOLVED.astype(np.int64)).to(dev)
    X = starts.to(dev).repeat_interleave(G, 0); n = len(X)
    done = (X == SOLVED).all(1); best = torch.full((n,), -1, device=dev); rl_steps = 0; steps = 0
    for t in range(budget):
        act = torch.nonzero(~done).flatten()
        if len(act) == 0: break
        sc = progress(X[act]); opening = sc > best[act]; best[act] = torch.maximum(best[act], sc)
        if mode == 'open': use_rl = opening
        elif mode == 'rev': use_rl = ~opening
        else: use_rl = torch.rand(len(act), device=dev) < q
        with amp(dev):
            lb = torch.cat([base(X[act[i:i + bs]])[0].float() for i in range(0, len(act), bs)])
            lr = torch.cat([rl(X[act[i:i + bs]])[0].float() for i in range(0, len(act), bs)])
        lg = torch.where(use_rl[:, None], lr, lb)
        mv = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
        X[act] = torch.gather(X[act], 1, MP[mv]); done = done | (X == SOLVED).all(1)
        rl_steps += int(use_rl.sum()); steps += len(act)
    return done.view(-1, G).float().mean(1).cpu().numpy(), rl_steps / steps

def main(tag, runs):
    X = sets()['held']; base = load_run(tag, 'base'); B = np.array(json.load(open(f'{OUT}/bins_{tag}.json'))['held']['bin'])
    pb = json.load(open(f'{OUT}/eval_{tag}_base.json'))['held']
    for run in runs:
        rl = load_run(tag, run); pr = json.load(open(f'{OUT}/eval_{run}.json'))['held']; res = {}
        p_open, q = mixed(base, rl, X, 90, 'open'); res['OPEN'] = (p_open, q)
        res['RAND'] = mixed(base, rl, X, 90, 'rand', q=q); res['REV'] = mixed(base, rl, X, 90, 'rev')
        out = dict(rl_share={k: v[1] for k, v in res.items()})
        for b in ('EASY', 'MID', 'HARD', 'ALL'):
            sel = np.ones(len(B), bool) if b == 'ALL' else B == b
            base_p, rl_p = pb[b]['sampled'], pr[b]['sampled']
            out[b] = {k: dict(pass_rate=float(v[0][sel].mean()), recovered=float((v[0][sel].mean() - base_p) / (rl_p - base_p)))
                      for k, v in res.items()} | dict(base=base_p, rl=rl_p)
        json.dump(out, open(f'{OUT}/swap_{run}.json', 'w'), indent=1)
        a = out['ALL']
        print(f"{run}: base {a['base']*100:.1f}% RL {a['rl']*100:.1f}% | OPEN {a['OPEN']['pass_rate']*100:.1f}% (R={a['OPEN']['recovered']:.2f}, "
              f"RL plays {out['rl_share']['OPEN']*100:.0f}% of moves) | RAND {a['RAND']['pass_rate']*100:.1f}% (R={a['RAND']['recovered']:.2f}) | "
              f"REV {a['REV']['pass_rate']*100:.1f}% (R={a['REV']['recovered']:.2f})", flush=True)
        print(f"   C1 (R_open >= 0.5 and R_open - R_rand >= 0.2): {a['OPEN']['recovered'] >= 0.5 and a['OPEN']['recovered'] - a['RAND']['recovered'] >= 0.2} | "
              f"C2 (R_rev >= 0.5): {a['REV']['recovered'] >= 0.5}", flush=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
