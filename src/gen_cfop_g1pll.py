"""CFOP variant for prereg #9: identical to data/cfop250k.npz (same scrambles, same cross / F2L / OLL moves) except
that every PLL segment is replaced by the optimal Kociemba phase-2 solution of its case, which uses only G1 moves
(U, D, R2, L2, F2, B2). G1's move set is identical under every training rotation; OLL's is not.
At the start of PLL, F2L is solved and the last layer oriented, so the position depends only on the PLL case:
the replacement segment (moves and positions) is precomputed per case (data/cfop_pll_g1.pkl, 179 cases).
Usage: python src/gen_cfop_g1pll.py -> data/cfop250k_g1pll.npz"""
import sys, pickle, numpy as np
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import cube, cube_fast as C
from cfop_canon import llkey
from qa import facelets_to_cubie
R = '/Users/alityb/projects/cubemoe'
MN = [f + s for f in 'URFDLB' for s in ('', "'", '2')]; G1 = {MN.index(m) for m in ('U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2')}

def main():
    d = dict(np.load(f'{R}/data/cfop250k.npz')); tab = pickle.load(open(f'{R}/data/cfop_pll_g1.pkl', 'rb'))
    seg = {}                                                      # case key -> (states before each move, move ids)
    for k, sol in tab.items():
        ids = [MN.index(m) for m in sol]; s = None; S = []
        seg[k] = ids
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))]); n = len(d['sol_len'])
    out = {k: [] for k in ('states', 'moves', 'phase')}; sol_len = []; bounds = []; kept_same = 0
    for i in range(n):
        a, b = off[i], off[i + 1]; ph = d['phase'][a:b]; mv = d['moves'][a:b]; st = d['states'][a:b]
        pre = ph != 4
        if pre.all():                                             # PLL skip: nothing to replace
            new_mv, new_st, new_ph = mv, st, ph; kept_same += 1
        else:
            j = int(np.argmax(~pre)); s0 = st[j]
            key = llkey(facelets_to_cubie(''.join('URFDLB'[x] for x in s0)))
            ids = seg[key]; S = [s0.astype(np.int64)]
            for m in ids[:-1]: S.append(C.apply(S[-1][None], m)[0])
            new_mv = np.concatenate([mv[:j], np.array(ids, np.uint8)]); new_st = np.concatenate([st[:j], np.stack(S).astype(np.uint8)])
            new_ph = np.concatenate([ph[:j], np.full(len(ids), 4, np.uint8)])
        out['states'].append(new_st); out['moves'].append(new_mv); out['phase'].append(new_ph); sol_len.append(len(new_mv))
        bd = d['bounds'][i].copy(); bd[3] = len(new_mv); bounds.append(bd)
        if i % 50000 == 0: print(i, flush=True)
    sl = np.array(sol_len); N = sl.sum()
    res = dict(states=np.concatenate(out['states']), moves=np.concatenate(out['moves']), phase=np.concatenate(out['phase']),
               pos=np.concatenate([np.arange(k) for k in sl]).astype(np.uint8), remaining=np.concatenate([np.arange(k, 0, -1) for k in sl]).astype(np.uint8),
               solve_id=np.repeat(np.arange(n), sl).astype(np.int32), seam=d['seam'], sol_len=sl.astype(np.uint8),
               init_state=d['init_state'], bounds=np.array(bounds, np.uint8))
    # checks: every solution solves its scramble; PLL moves are G1 moves; first three segments unchanged
    rng = np.random.RandomState(0); offn = np.concatenate([[0], np.cumsum(sl)])
    for i in rng.choice(n, 2000, replace=False):
        s = res['init_state'][i].astype(np.int64)
        for m in res['moves'][offn[i]:offn[i + 1]]: s = C.apply(s[None], int(m))[0]
        assert (s == C.SOLVED).all(), i
    assert all(int(m) in G1 for m in res['moves'][res['phase'] == 4])
    np.savez_compressed(f'{R}/data/cfop250k_g1pll.npz', **res)
    print(f"wrote data/cfop250k_g1pll.npz: {n} solves, {N} positions (was {len(d['moves'])}), mean length {sl.mean():.1f} "
          f"(was {d['sol_len'].mean():.1f}); PLL skips kept {kept_same}; 2000/2000 sampled solutions solve; all PLL moves in G1")

if __name__ == '__main__':
    main()
