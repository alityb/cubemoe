"""Applies the rules of prereg/PREREG_GEOMETRY_PATHS.md (#12) mechanically to out/geom/*.json.
Usage: python src/geom_report.py  -> prints per-model tables and verdicts; writes out/geom/report.json"""
import json, os, statistics as st
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('GEOM_OUT', f'{R}/out/geom')
K = [f'mixtral_6L256_s{s}' for s in range(5)]; C = [f'cfop_mixtral_6L256_s{s}' for s in range(8)]
OWN = {**{t: 'mixed250k' for t in K}, **{t: 'cfop250k' for t in C}}; OTHER = {'mixed250k': 'cfop250k', 'cfop250k': 'mixed250k'}
DIRS = ('fit_A_score_B', 'fit_B_score_A'); LAYERS = [f'L{l}' for l in range(6)]

def load(t, data): return json.load(open(f'{OUT}/{t}__{data}.json'))

def main():
    O = {t: load(t, OWN[t]) for t in K + C}; X = {t: load(t, OTHER[OWN[t]]) for t in K + C}
    for t in K + C: assert O[t]['data'] == OWN[t] and X[t]['data'] == OTHER[OWN[t]] and O[t]['tag'] == X[t]['tag'] == t
    a = lambda o, part='given_S', l='L5': o['partA'][l][part]
    r = lambda o, part='given_S', l='L5': a(o, part, l)['r'] or 0.            # undefined (ceiling <= 0) -> 0
    beyond = lambda x: x['eta2_z'] > x['null_pca_p99'] and x['eta2_z'] >= 1.5 * x['null_pca_median']
    halves = lambda o, f: f(a(o, 'given_S_halfA')) and f(a(o, 'given_S_halfB'))
    # ---- Part A (L5, given S)
    A1 = sum(halves(O[t], beyond) for t in K)
    A2 = sum(halves(O[t], lambda x: x['eta2_h'] > x['eta2_h_perm_p99']) for t in C)
    rKx = max(r(X[k]) for k in K)                                               # K routers on CFOP positions
    A3 = sum(halves(O[t], beyond) and r(O[t]) > rKx for t in C)
    rK, rC = st.median(r(O[t]) for t in K), st.median(r(O[t]) for t in C); A4 = rC >= 0.5 * rK
    p1, p2, p3 = A1 >= 4, A2 >= 6, A3 >= 6
    if not p2: readA = 'STAGE NOT DETECTABLY PRESENT at the final router input (A2 failed)'
    elif p1 and not p3: readA = 'SELECTION'
    elif p1 and p3: readA = 'BOTH SELECTIVE, COMPARABLE (gap lies in how readable stage is: compare ceilings)' if A4 else 'MIXED'
    elif not p1 and not p3: readA = 'GEOMETRY (neither router reads its label beyond the dominant directions)'
    else: readA = 'CFOP ROUTER MORE SELECTIVE THAN KOCIEMBA (contradicts the premise)'
    # ---- Part B
    I = lambda o, k, d: o['partB'][d][k]
    m = lambda o, k: (I(o, k, DIRS[0]) + I(o, k, DIRS[1])) / 2
    B0 = all(-0.05 <= I(o, 'null_all', d) <= 0.02 for o in list(O.values()) + list(X.values()) for d in DIRS)
    B1 = sum(all(I(O[t], 'all', d) - I(O[t], 'L5', d) >= 0.10 for d in DIRS) for t in C)
    B1b = sum(all(I(O[t], 'all', d) - max(I(O[t], l, d) for l in LAYERS) >= 0.05 for d in DIRS) for t in C)
    B2 = sum(all(I(O[t], 'all', d) - max(I(X[k], 'all', d) for k in K) >= 0.10 for d in DIRS) for t in C)
    iK, iC = st.median(m(O[t], 'all') for t in K), st.median(m(O[t], 'all') for t in C); B3 = iC >= 0.5 * iK
    entC = [st.median(O[t]['partB']['pair_entropy_bits'][l] for t in C) for l in range(6)]
    degen = [l for l in range(6) if min(X[k]['partB']['pair_entropy_bits'][l] for k in K) < 0.5 * entC[l]]
    q1, q2 = B1 >= 6, B2 >= 6
    if not B0: readB = 'VOID (B0 failed: the permutation null is outside [-0.05, 0.02])'
    elif q1 and q2: readB = ('PATH CARRIES CFOP STAGE beyond the final layer and beyond state tracking'
                             + ('; combinatorial (beyond any single layer)' if B1b >= 6 else '; not beyond the best single layer')
                             + ('; closes the gap to Kociemba' if B3 else '; does not close the gap to Kociemba'))
    elif q1: readB = 'path adds stage information beyond the final layer, but no more than a Kociemba router carries'
    elif q2: readB = 'CFOP routing carries more stage than a Kociemba router, but the path adds little beyond the final layer'
    else: readB = 'path adds little beyond the final layer and no more than a Kociemba router: no-split extends across layers'
    if degen: readB += f' [B2 control degenerate at layers {degen}: K routers collapse on CFOP positions]'
    # ---- tables
    f = lambda v: f'{v:7.3f}'
    print('PART A, L5 given S: eta2(h) [perm p99] | eta2(z) | pca null median / p99 | ceiling | r | pca-null r | haar p99 | [uncond r]')
    for t in K + C:
        x = a(O[t]); g = x['null_pca_median'] / x['ceiling'] if x['ceiling'] > 0 else 0.
        print(f'  {t:22s} {f(x["eta2_h"])} [{f(x["eta2_h_perm_p99"])}] | {f(x["eta2_z"])} | {f(x["null_pca_median"])} / {f(x["null_pca_p99"])} |'
              f' {f(x["ceiling"])} | {f(r(O[t]))} | {f(g)} | {f(x["null_haar_p99"])} | [{f(r(O[t], "uncond"))}]')
    print(f'  max r of K routers on CFOP positions: {rKx:.3f}')
    print('  r by layer (L0..L5), own positions | cross positions:')
    for t in K + C: print(f'  {t:22s} ' + ' '.join(f(r(O[t], l=l)) for l in LAYERS) + ' | ' + ' '.join(f(r(X[t], l=l)) for l in LAYERS))
    print('PART B (mean of directions): I(L0..L5) | I(L0-4) I(all) | cross-run I(all) | null')
    for t in K + C:
        print(f'  {t:22s} ' + ' '.join(f(m(O[t], l)) for l in LAYERS) + f' | {f(m(O[t], "L0-4"))} {f(m(O[t], "all"))} | {f(m(X[t], "all"))} | {f(m(O[t], "null_all"))}')
    v = dict(A1=f'{A1}/5', A2=f'{A2}/8', A3=f'{A3}/8 (K-on-CFOP max r {rKx:.3f})', A4=f'median r C {rC:.3f} vs K {rK:.3f} -> {A4}',
             partA=readA, B0=B0, B1=f'{B1}/8', B1b=f'{B1b}/8', B2=f'{B2}/8', B3=f'median I(all) C {iC:.3f} vs K {iK:.3f} -> {B3}',
             B2_degenerate_layers=degen, partB=readB)
    print(v); json.dump(dict(verdict=v), open(f'{OUT}/report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
