"""Applies the rules of prereg/PREREG_PATHS_FRESH.md (#13) mechanically to out/paths_fresh/*.json.
Usage: python src/paths_fresh_report.py  -> prints the tables and verdicts; writes out/paths_fresh/report.json"""
import json, os
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('PF_OUT', f'{R}/out/paths_fresh')
K = [f'mixtral_6L256_s{s}' for s in range(5)]; C = [f'cfop_mixtral_6L256_s{s}' for s in range(8)]
RND = [f'random_s{s}' for s in (100, 101, 102)]
DIRS = ('fit_A_score_B', 'fit_B_score_A'); LAYERS = [f'L{l}' for l in range(6)]

def verdict(O, part):
    I = lambda t, k, d: O[t][part][d][k]
    inb = lambda ts: all(-0.08 <= I(t, 'null_all', d) <= 0.02 for t in ts for d in DIRS)
    B0C, B0K, B0R = inb(C), inb(K), inb(RND)
    mx = lambda ts, k, d: max(I(t, k, d) for t in ts)
    B1 = [t for t in C if all(I(t, 'all', d) - I(t, 'L5', d) >= 0.10 for d in DIRS)]
    B1b = [t for t in C if all(I(t, 'all', d) - max(I(t, l, d) for l in LAYERS) >= 0.05 for d in DIRS)]
    B1c = [t for t in C if all(I(t, 'all_onehot', d) - I(t, 'additive', d) >= 0.05 for d in DIRS)]
    B2 = [t for t in C if all(I(t, 'all', d) - mx(K, 'all', d) >= 0.10 and I(t, 'L0-4', d) - mx(K, 'L0-4', d) >= 0.10 for d in DIRS)]
    B2r = [t for t in C if all(I(t, 'all', d) - mx(RND, 'all', d) >= 0.10 for d in DIRS)]
    ok = lambda v: len(v) >= 6
    if not B0C: read = 'VOID (B0: a CFOP-model null outside [-0.08, 0.02])'
    else:
        b2 = ok(B2) if B0K else None; b2r = ok(B2r) if B0R else None
        if not ok(B1): read = 'NOT REPLICATED: the path adds little beyond the final layer'
        elif b2 is None or b2r is None: read = 'B1 holds; ' + ('B2 void (K null out of bounds) ' if b2 is None else '') + ('B2r void (random-init null out of bounds)' if b2r is None else '')
        elif b2 and b2r:
            read = ('REPLICATED: the CFOP routing path carries stage-predictive information beyond the final layer, beyond Kociemba-trained '
                    'routers and beyond a random hash of the state' + ('; beyond any single layer' if ok(B1b) else '; NOT beyond the best single layer')
                    + ('; combinatorial (beyond an additive-over-layers model)' if ok(B1c) else '; NOT combinatorial (an additive-over-layers model does as well)'))
        else: read = 'NOT CFOP-SPECIFIC: the path adds stage information, but not beyond ' + ' and '.join(x for x, b in (('Kociemba routers', b2), ('a random hash of the state', b2r)) if not b)
    return dict(B0_C=B0C, B0_K=B0K, B0_random=B0R, B1=f'{len(B1)}/8', B1b=f'{len(B1b)}/8', B1c=f'{len(B1c)}/8', B2=f'{len(B2)}/8', B2r=f'{len(B2r)}/8', verdict=read)

def main():
    O = {t: json.load(open(f'{OUT}/{t}.json')) for t in K + C + RND}
    n = {o['n_positions'] for o in O.values()}; assert len(n) == 1, n
    f = lambda v: f'{v:7.3f}'
    for part in ('all', 'nondup'):
        m = lambda t, k: (O[t][part][DIRS[0]][k] + O[t][part][DIRS[1]][k]) / 2
        print(f'[{part}] mean of directions: I(L0..L5) | I(L0-4) I(all) | onehot additive | null | pair entropy L0..L5 (bits)')
        for t in K + C + RND:
            print(f'  {t:22s} ' + ' '.join(f(m(t, l)) for l in LAYERS) + f' | {f(m(t, "L0-4"))} {f(m(t, "all"))} | {f(m(t, "all_onehot"))} {f(m(t, "additive"))} | {f(m(t, "null_all"))} | '
                  + ' '.join(f'{e:.2f}' for e in O[t][part]['pair_entropy_bits']))
        h = O[C[0]][part][DIRS[0]]; print(f'  H(stage | S) = {h["H_label_given_S_bits"]:.3f} bits; CE(stage | S) = {h["CE_S_bits"]:.3f} bits')
    print(f'fraction of fresh states seen verbatim in training: {O[C[0]]["dup_fraction"]:.3f}')
    v = dict(primary=verdict(O, 'all'), descriptive_nondup=verdict(O, 'nondup'))
    print(json.dumps(v, indent=1)); json.dump(v, open(f'{OUT}/report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
