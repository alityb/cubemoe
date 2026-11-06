"""Builds data/cfop_pll_g1.pkl for prereg #9: for each of the CFOP PLL cases in data/cfop_canon.pkl, the optimal
Kociemba phase-2 solution (G1 moves only). Run from the repo root before src/gen_cfop_g1pll.py.
(Same code as the one-off run of 2026-11-05 that produced the committed table.)"""
import sys, time, pickle, numpy as np; sys.path[:0] = ['probe']
import cube, cfop, phase2
T = pickle.load(open('data/cfop_canon.pkl', 'rb'))
G1MOVES = {'U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2'}
t = time.time(); lens = []; oldl = []; bad = 0; tab = {}
for k, (setup, solution) in T['pll'].items():
    s = cube.SOLVED
    for m in setup: s = cube.apply_move(s, m)
    assert cube.in_G1(s)
    sol = phase2.solve_phase2(s)
    s2 = s
    for m in sol: s2 = cube.apply_move(s2, m)
    bad += (s2 != cube.SOLVED) or any(m not in G1MOVES for m in sol)
    tab[k] = sol; lens.append(len(sol)); oldl.append(len(solution))
print(f"{len(tab)} PLL cases solved in G1 in {time.time()-t:.1f}s | bad {bad} | G1 PLL length mean {np.mean(lens):.1f} "
      f"(min {min(lens)}, max {max(lens)}) vs original {np.mean(oldl):.1f}")
pickle.dump(tab, open('data/cfop_pll_g1.pkl', 'wb'))
