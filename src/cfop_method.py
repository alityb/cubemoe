"""Which named last-layer method is each solution closest to?  (on paths saved by src/cfop_paths.py)

After F2L is done, a last-layer method is a fixed ORDER of four sub-goals (all 'up to a U turn'):
  EO edges oriented   CO corners oriented   CP corners permuted   EP edges permuted
For every solution: the step at which each sub-goal becomes true for good, looking only at checkpoints where F2L is
intact (algorithms break F2L in the middle). Sub-goals completed at the same step were done by one algorithm.
That gives a fingerprint like 'EO+CO > CP+EP' (= full OLL then full PLL), matched to named methods:
"""
import sys, json, numpy as np
from collections import Counter
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
from cfop_ll import states, cubie, shift
from cfop_verify import flags, first
R = '/Users/alityb/projects/cubemoe'

METHODS = {   # fingerprint (groups in order; 'given' = already true when F2L finished) -> method
    'EO+CO > CP+EP': 'standard CFOP (1-look OLL, 1-look PLL)',
    'EO > CO > CP+EP': '2-look OLL, 1-look PLL',
    'EO+CO > CP > EP': '1-look OLL, 2-look PLL (corners first)',
    'EO+CO > EP > CP': '1-look OLL, 2-look PLL (edges first)',
    'EO > CO > CP > EP': 'beginner 4-look last layer',
    'EO+CO+CP > EP': 'OLLCP + EPLL',
    'given EO | CO+CP > EP': 'COLL + EPLL (after edge control)',
    'given EO | CO+CP+EP': 'ZBLL (after edge control)',
    'EO+CO+CP+EP': '1LLL (whole last layer in one look)',
}

def fingerprint(start, moves):
    S = states(start, moves); _, f, _ = flags(S); tf = first(f); n = len(moves)
    chk = [t for t in range(tf, n + 1) if f[t]]                      # checkpoints with F2L intact
    props = {}
    for t in chk:
        cp, co, ep, eo = cubie(S[t]); cs, es = shift(cp[:4]), shift(ep[:4])
        props[t] = dict(EO=all(x == 0 for x in eo[:4]), CO=all(x == 0 for x in co[:4]),
                        CP=cs is not None, EP=es is not None and (cs is None or es == cs))   # edges must line up with placed corners (H-perm != done)
    done = {}
    for g in ('EO', 'CO', 'CP', 'EP'):
        t_g = chk[-1]
        for t in reversed(chk):                                         # last checkpoint where it was false, +1
            if not props[t][g]: break
            t_g = t
        done[g] = t_g
    given = [g for g in done if done[g] == tf]
    later = sorted({done[g] for g in done if done[g] > tf})
    groups = ['+'.join(g for g in ('EO', 'CO', 'CP', 'EP') if done[g] == t) for t in later]
    fp = ' > '.join(groups)
    if given: fp = 'given ' + '+'.join(given) + (' | ' + fp if fp else '')
    steps = [done[g] - tf for g in ('EO', 'CO', 'CP', 'EP')]
    return fp, n - tf, steps

def _parse(fp):
    given, core = (fp[6:].split(' | ') + [''])[:2] if fp.startswith('given ') else ('', fp)
    if fp.startswith('given ') and ' | ' not in fp: given, core = fp[6:], ''
    G = set(given.split('+')) - {''}; groups = [set(g.split('+')) for g in core.split(' > ') if g]
    return G, groups

def name(fp):
    """All named methods consistent with this solution once sub-goals already done at F2L are discounted."""
    G, obs = _parse(fp); hits = []
    for mfp, mname in METHODS.items():
        MG, mg = _parse(mfp)
        if not MG <= G: continue
        red = [g - G for g in mg if g - G]
        if red == obs: hits.append(mname)
    return ' / '.join(hits) if hits else ('last layer skipped' if not obs else 'other: ' + fp)

def main(tag):
    P = [p for p in json.load(open(f'{R}/out/prod/{tag}_paths.json')) if p['model']]; out = {}
    for who in ('solver', 'model'):
        rows = [fingerprint(p['start'], p[who]) for p in P]
        cnt = Counter(r[0] for r in rows); bym = Counter()
        for fp, c in cnt.items(): bym[name(fp)] += c
        compat = Counter()
        for fp, c in cnt.items():
            for mth in name(fp).split(' / '): compat[mth] += c
        out[who] = dict(fingerprints=dict(cnt), methods=dict(bym), compatible=dict(compat), ll_len=float(np.mean([r[1] for r in rows])))
        print(f"\n{tag} — {who}: last layer {out[who]['ll_len']:.1f} moves after F2L; closest named method:")
        for mth, c in bym.most_common(): print(f"   {c / len(rows) * 100:5.1f}%  {mth}")
        print("   share of solves CONSISTENT with each method (overlapping):")
        for mth, c in compat.most_common(): print(f"   {c / len(rows) * 100:5.1f}%  {mth}")
        print("   fingerprints: " + "; ".join(f"{fp} {c}" for fp, c in cnt.most_common(8)))
    json.dump(out, open(f'{R}/out/prod/{tag}_method.json', 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1])
