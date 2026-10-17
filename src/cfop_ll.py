"""Which last-layer technique do the CFOP policies use?  (on paths saved by src/cfop_paths.py)

The training solver's last layer is a toy: OLL from <= 2 of {Sune, Anti-Sune, F R U R' U' F'}, PLL from <= 2 of
{T, Ua, Aa, J} plus AUFs (mean 14 / 23 moves). Candidate explanations for the model's ~10-move PLL:
  ZBLL / 1LLL   orientation and permutation in one sequence -> no separate OLL-done state before the end
  edge control  (ZBLS-like) LL edges already oriented when F2L finishes
  OLLCP         the OLL step also permutes the LL corners -> only an edge PLL (U/H/Z) remains
  1-look PLL    one algorithm where the solver used two -> no 'corners permuted, edges not' intermediate
  skips         OLL / PLL skips more often than the solver
Usage: python src/cfop_ll.py <tag>   (reads out/prod/<tag>_paths.json)
"""
import sys, json, numpy as np
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import cube_fast as C
from qa import facelets_to_cubie
from cfop_canon import OLL_BASE, PLL_BASE
from cfop_verify import flags, first
R = '/Users/alityb/projects/cubemoe'; MN = [f + s for f in 'URFDLB' for s in ('', "'", '2')]
INV = {m: (m[0] if m.endswith("'") else (m if m.endswith('2') else m + "'")) for m in MN}
ids = lambda alg: tuple(MN.index(x) for x in alg.split())
inv = lambda alg: ' '.join(INV[m] for m in reversed(alg.split()))
U_MOVES = {MN.index(x) for x in ('U', "U'", 'U2')}

def strip_auf(seq):
    seq = list(seq)
    while seq and seq[0] in U_MOVES: seq.pop(0)
    while seq and seq[-1] in U_MOVES: seq.pop()
    return tuple(seq)
ONE_PLL = {ids(a) for a in PLL_BASE} | {ids(inv(a)) for a in PLL_BASE}
ONE_OLL = {ids(a) for a in OLL_BASE} | {ids(inv(a)) for a in OLL_BASE}

def cubie(s): return facelets_to_cubie(''.join('URFDLB'[x] for x in s))
def shift(p):
    """k if these four U-layer pieces are solved up to a U-layer turn by k, else None."""
    return next((k for k in range(4) if list(p) == [(i + k) % 4 for i in range(4)]), None)

def states(start, moves):
    S = [np.array(start, dtype=np.int64)]
    for mv in moves: S.append(C.apply(S[-1][None], mv)[0])
    return np.stack(S)

def one(start, moves):
    S = states(start, moves); n = len(moves); c, f, o = flags(S); tf, to = first(f), first(o)
    cf = cubie(S[tf]); cp, _, ep, _ = cubie(S[to]); cs, es = shift(cp[:4]), shift(ep[:4]); skip = cs is not None and cs == es
    pll = moves[to:]; oll = moves[tf:to]
    two_look = False
    for t in range(to + 1, n):                       # strictly inside the PLL segment
        if o[t]:
            q = cubie(S[t]); cs, es = shift(q[0][:4]), shift(q[2][:4])
            if (cs is not None and es != cs) or (es is not None and cs is None): two_look = True   # CPLL/EPLL midpoint
    return dict(n=n, pll_len=len(pll), oll_len=len(oll),
                ll_only_at_end=to >= n - 1 and not o[tf],                       # ZBLL / 1LLL-like
                edges_oriented_at_f2l=all(x == 0 for x in cf[3][:4]),
                corners_oriented_at_f2l=all(x == 0 for x in cf[1][:4]),
                oll_skip=to == tf, corners_perm_at_oll=cs is not None, edges_perm_at_oll=es is not None,
                pll_skip=skip, pll_two_look=two_look,
                pll_one_base_alg=strip_auf(pll) in ONE_PLL, oll_one_base_alg=strip_auf(oll) in ONE_OLL,
                pll_type='skip' if skip else 'edges only (EPLL)' if cs is not None else
                         'corners only' if es is not None else 'corners+edges')

def main(tag):
    P = json.load(open(f'{R}/out/prod/{tag}_paths.json')); P = [p for p in P if p['model']]
    res = {}
    for who in ('solver', 'model'):
        rows = [one(p['start'], p[who]) for p in P]
        keys = [k for k in rows[0] if k != 'pll_type']
        r = {k: float(np.mean([x[k] for x in rows])) for k in keys}
        types = {}
        for x in rows: types.setdefault(x['pll_type'], []).append(x['pll_len'])
        r['pll_types'] = {t: dict(frac=len(v) / len(rows), mean_len=float(np.mean(v))) for t, v in types.items()}
        res[who] = r
    print(f"{tag}: {len(P)} solves (model solution vs the training solver's solution of the same scramble)")
    lab = [('n', 'solution length', 1), ('oll_len', 'OLL segment length', 1), ('pll_len', 'PLL segment length', 1),
           ('ll_only_at_end', 'ZBLL/1LLL-like: LL oriented only at the very end', 100),
           ('edges_oriented_at_f2l', 'edge control: LL edges oriented when F2L done (chance 1/8)', 100),
           ('corners_oriented_at_f2l', 'LL corners oriented when F2L done (chance 1/27)', 100),
           ('oll_skip', 'OLL skip', 100), ('corners_perm_at_oll', 'OLLCP: LL corners permuted at OLL-done (chance 1/6)', 100),
           ('edges_perm_at_oll', 'LL edges permuted at OLL-done (chance 1/6)', 100), ('pll_skip', 'PLL skip (AUF only)', 100),
           ('pll_two_look', '2-look PLL: passes corners-xor-edges permuted', 100),
           ('oll_one_base_alg', 'OLL = one base alg (+AUF)', 100), ('pll_one_base_alg', 'PLL = one base alg (+AUF)', 100)]
    print(f"  {'':62s} {'solver':>8s} {'model':>8s}")
    for k, name, sc in lab:
        print(f"  {name:62s} {res['solver'][k]*sc:8.1f} {res['model'][k]*sc:8.1f}{'%' if sc == 100 else ''}")
    for who in ('solver', 'model'):
        print(f"  {who} PLL cases: " + ", ".join(f"{t} {v['frac']*100:.0f}% ({v['mean_len']:.1f} mv)" for t, v in sorted(res[who]['pll_types'].items())))
    json.dump(res, open(f'{R}/out/prod/{tag}_ll.json', 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1])
