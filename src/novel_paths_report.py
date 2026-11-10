"""Applies the rules of prereg/PREREG_NOVEL_PATHS.md (#15) mechanically to out/novel_paths/*.json.
Usage: python src/novel_paths_report.py  -> prints tables and the verdict; writes out/novel_paths/report.json"""
import json, os
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('NP_OUT', f'{R}/out/novel_paths')
K = [f'mixtral_6L256_s{s}' for s in range(5)]; C = [f'cfop_mixtral_6L256_s{s}' for s in range(8)]
DIRS = ('fit_A_score_B', 'fit_B_score_A'); LAYERS = [f'L{l}' for l in range(6)]

def main():
    O = {t: json.load(open(f'{OUT}/{t}.json')) for t in K + C}
    assert len({o['n_novel'] for o in O.values()}) == 1
    I = lambda t, k, d, part='novel': O[t][part][d][k]
    inb = lambda ts, part='novel': all(-0.08 <= I(t, 'null_all', d, part) <= 0.02 for t in ts for d in DIRS)
    N0C, N0K, N0R = inb(C), inb(K), inb(C, 'rotated')
    mx = lambda k, d: max(I(t, k, d) for t in K)
    N1 = [t for t in C if all(I(t, 'all', d) >= 0.10 and I(t, 'all', d) - max(I(t, l, d) for l in LAYERS) >= 0.05 for d in DIRS)]
    N2 = [t for t in C if all(I(t, 'all', d) - mx('all', d) >= 0.10 and I(t, 'L0-4', d) - mx('L0-4', d) >= 0.10 for d in DIRS)]
    N3 = [t for t in C if all(I(t, 'all', d) - I(t, 'all', d, 'rotated') >= 0.10 for d in DIRS)]
    comb = [t for t in C if all(I(t, 'all_onehot', d) - I(t, 'additive', d) >= 0.05 for d in DIRS)]
    addv = [t for t in C if all(I(t, 'all_onehot', d) - I(t, 'additive', d) < 0.05 for d in DIRS)]
    ok = lambda v: len(v) >= 6
    N4 = 'combinatorial' if ok(comb) else ('additive' if ok(addv) else 'mixed')
    ent = lambda t, part: O[t][part]['pair_entropy_bits']
    gap = {t: sum(abs(x - y) for x, y in zip(ent(t, 'novel'), ent(t, 'rotated'))) / 6 for t in C}
    N3valid = N0R and sum(g <= 0.5 for g in gap.values()) >= 6           # matched control close enough in entropy
    n2 = ok(N2) if N0K else None; n3 = ok(N3) if N3valid else None
    if not N0C: read = 'VOID (N0: a CFOP-model null outside [-0.08, 0.02])'
    elif not ok(N1): read = 'NOT REPLICATED: the path does not carry cross-vs-F2L on novel states beyond its best layer'
    else:
        p2 = {None: 'N2 void (K null out of bounds)', True: 'beyond Kociemba routers', False: 'NOT beyond Kociemba routers'}[n2]
        p3 = {None: 'N3 not evaluable (rotated null out of bounds or entropy gap > 0.5 bits in > 2 models; descriptive only)',
              True: 'beyond a usage-matched random readout of its own representation: the trained routing SELECTS for the stage',
              False: 'NOT beyond a usage-matched random readout of its own representation: the stage is in the representation, the routing does not select for it'}[n3]
        read = f'N1 holds (the path carries cross-vs-F2L on novel states); {p2}; {p3}; layers combine: {N4}'
    m = lambda t, k, part='novel': (I(t, k, DIRS[0], part) + I(t, k, DIRS[1], part)) / 2; f = lambda v: f'{v:7.3f}'
    print('mean of directions: I(L0..L5) | I(L0-4) I(all) | onehot additive | null | pair entropy L0..L5 (bits)')
    rows = [(t, 'novel') for t in K + C] + [(t, 'rotated') for t in C] + [(t, 'rotated_raw') for t in C]
    for t, part in rows:
        print(f'  {t + {"novel": "", "rotated": " ROT", "rotated_raw": " ROTRAW"}[part]:26s} ' + ' '.join(f(m(t, l, part)) for l in LAYERS) + f' | {f(m(t, "L0-4", part))} {f(m(t, "all", part))} | '
              f'{f(m(t, "all_onehot", part))} {f(m(t, "additive", part))} | {f(m(t, "null_all", part))} | ' + ' '.join(f'{e:.2f}' for e in O[t][part]['pair_entropy_bits']))
    o = O[C[0]]; h = o['novel'][DIRS[0]]
    print(f'  H(stage | S) = {h["H_label_given_S_bits"]:.3f} bits; novel {o["n_novel"]} of {o["n_positions"]}; stage counts {o["novel_stage_counts"]}; '
          f'novel fraction of F2L {o["novel_F2L_fraction_of_F2L"]:.3f}; dropped novel OLL/PLL {o["dropped_novel_OLL_PLL"]}; '
          f'novel positions whose state occurs in both halves {o["novel_state_in_both_halves_fraction"]:.3f}')
    v = dict(N0_C=N0C, N0_K=N0K, N0_rotated=N0R, rotated_entropy_gap_bits={t: round(g, 3) for t, g in gap.items()}, N3_evaluable=N3valid, N1=f'{len(N1)}/8', N2=f'{len(N2)}/8', N3=f'{len(N3)}/8',
             N4=dict(combinatorial=f'{len(comb)}/8', additive=f'{len(addv)}/8', reading=N4), verdict=read)
    print(json.dumps(v, indent=1)); json.dump(v, open(f'{OUT}/report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
