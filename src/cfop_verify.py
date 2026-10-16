"""Do the CFOP-trained policies actually solve the CFOP way?  (closed-loop, beam 256, held-out scrambles)

Every stage predicate implies the earlier ones (F2L-done includes cross-done), so "cross before F2L" is true of
ANY solution. The tests below are the ones a non-CFOP solver would fail:
  1 milestones: when (fraction of the solution) is each stage FIRST reached? CFOP: cross early,
    F2L by ~2/3, then OLL, then PLL. A non-CFOP solver completes everything in the last few moves.
  2 last-layer algorithms: is the move sequence from F2L-done to OLL-done (and OLL-done to solved) an exact
    sequence the CFOP solver used in training? i.e. does it execute known OLL / PLL algorithms.
Reference rows: the CFOP solver's own solutions for the same scrambles, and a Kociemba-trained policy (control).
Usage: python src/cfop_verify.py [n_cubes]   -> out/prod/cfop_verify.json
"""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A, cube_fast as C
R = '/Users/alityb/projects/cubemoe'

_U, _R, _F, _D, _L, _B = (np.arange(9) + 9 * k for k in range(6)); _SOL = np.repeat(np.arange(6), 9)
CROSS = np.r_[_D[[1, 3, 5, 7]], _R[7], _F[7], _L[7], _B[7]]          # sticker tests; agree 100% with probe/cfop.py
F2L = np.r_[_D, _R[3:], _F[3:], _L[3:], _B[3:]]; OLL = np.r_[F2L, _U]  # predicates on 20k CFOP states

def flags(S):
    ok = lambda idx: (S[:, idx] == _SOL[idx]).all(1)
    return ok(CROSS), ok(F2L), ok(OLL)

def first(f): return int(np.argmax(f)) if f.any() else len(f) - 1

def segments(S, moves):
    """S: states s_0..s_n along the solution (s_n solved). Stage milestones = FIRST time reached (CFOP algorithms
    break the cross / F2L temporarily mid-algorithm, so 'stays done' is meaningless)."""
    c, f, o = flags(S); tc, tf, to = first(c), first(f), first(o)
    return tc, tf, to, tuple(moves[tf:to]), tuple(moves[to:])

def path_states(start, moves):
    S = [start]
    for mv in moves: S.append(C.apply(S[-1][None], mv)[0])
    return np.stack(S)

def analyse(start, moves, oll_set, pll_set):
    n = len(moves); tc, tf, to, oll, pll = segments(path_states(start, moves), moves)
    return dict(n=n, cross=tc / n, f2l=tf / n, oll=to / n, oll_len=len(oll), pll_len=len(pll),
                oll_known=oll in oll_set, pll_known=pll in pll_set)

def training_segments(d, n_train):
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))]); tr = np.random.RandomState(0).permutation(len(d['sol_len']))[:n_train]
    oll, pll = set(), set()
    for i in tr:
        S = np.concatenate([d['states'][off[i]:off[i + 1]].astype(np.int64), C.SOLVED[None].astype(np.int64)])
        _, _, _, o, p = segments(S, [int(x) for x in d['moves'][off[i]:off[i + 1]]]); oll.add(o); pll.add(p)
    return oll, pll

def summarise(rows):
    k = ('n', 'cross', 'f2l', 'oll', 'oll_len', 'pll_len', 'oll_known', 'pll_known')
    return {x: float(np.mean([r[x] for r in rows])) for x in k} | dict(count=len(rows))

def main(n=100):
    d = dict(np.load(f'{R}/data/cfop250k.npz')); N = len(d['sol_len'])
    perm = np.random.RandomState(0).permutation(N); te = perm[int(N * 0.9):][:n]
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    oll_set, pll_set = training_segments(d, 50000)
    print(f"training OLL sequences {len(oll_set)}, PLL sequences {len(pll_set)} (from 50k training solves)", flush=True)
    starts = [d['init_state'][i].astype(np.int64) for i in te]; res = {}
    res['cfop_solver'] = summarise([analyse(starts[j], [int(x) for x in d['moves'][off[i]:off[i + 1]]], oll_set, pll_set)
                                    for j, i in enumerate(te)])
    for tag, budget in [(f'cfop_mixtral_6L256_s{s}', 90) for s in range(5)] + [('mixtral_6L256_s0', 40)]:
        m, _ = A.load(tag); rows = []; fail = 0
        for s0 in starts:
            path = A.beam_solve_path(m, torch.from_numpy(s0), width=256, budget=budget)
            if path is None: fail += 1; continue
            rows.append(analyse(s0, path, oll_set, pll_set))
        res[tag] = summarise(rows) | dict(unsolved=fail); del m
        print(tag, {k: round(v, 3) for k, v in res[tag].items()}, flush=True)
    print("\n  (stage milestones = first time reached, as a fraction of the solution)")
    print("                           moves | cross at | F2L at | OLL at | OLL seg len | PLL seg len | known OLL alg | known PLL alg")
    for k, r in res.items():
        print(f"  {k:24s} {r['n']:5.1f} | {r['cross']*100:7.0f}% | {r['f2l']*100:5.0f}% | {r['oll']*100:5.0f}% | {r['oll_len']:11.1f} |"
              f" {r['pll_len']:11.1f} | {r['oll_known']*100:12.0f}% | {r['pll_known']*100:12.0f}%")
    json.dump(res, open(f"{R}/out/prod/cfop_verify{'_smoke' if n < 20 else ''}.json", "w"), indent=1)

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
