"""Applies the rules of prereg/PREREG_EXPERT_PAIRS.md (#10) mechanically to out/pairs/*.json.
Usage: python src/pairs_report.py  -> prints controls, per-model table and verdicts; writes out/pairs/report.json"""
import json, itertools, os
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('PAIRS_OUT', f'{R}/out/pairs')   # override only for smoke tests
FAM = {'K': [f'mixtral_6L256_s{s}' for s in range(5)], 'C': [f'cfop_mixtral_6L256_s{s}' for s in range(5)],
       'G': [f'cfop_g1pll_6L256_s{s}' for s in range(4)]}
CTRL = ['mixtral_6L256_s0', 'cfop_g1pll_6L256_s0']
HALVES = ('A', 'B')

def stages(o): return o['stages']
def D(o, Z, part, scale='p'):                    # reroute damage per stage (+ '_all') for expert set Z
    return (o['single'] if len(Z) == 1 else o['pair'])['reroute'][','.join(map(str, Z))][scale][part]

def dedicated(d, s, floor, names):
    return d[s] >= floor and all(d[t] <= 0.25 * d[s] for t in names if t != s)

def hidden(o, i, j, s):
    """Dedicated pair whose members' single knockouts understate it, on p AND margin, in both halves."""
    for h in HALVES:
        dp = D(o, (i, j), h); di, dj = D(o, (i,), h), D(o, (j,), h)
        mp, mi, mj = D(o, (i, j), h, 'margin'), D(o, (i,), h, 'margin'), D(o, (j,), h, 'margin')
        if not dedicated(dp, s, 0.05, stages(o)): return False
        if max(di[s], dj[s]) > 0.5 * dp[s]: return False
        if not (mp[s] > 0 and max(mi[s], mj[s]) <= 0.5 * mp[s]): return False
    return True

def hidden_pairs(o):
    E = len(o['single']['reroute'])
    return [(i, j, s) for i, j in itertools.combinations(range(E), 2) for s in stages(o) if hidden(o, i, j, s)]

def BI(o, part):                                 # margin scale, signed, pooled over all positions (stage-free)
    E = len(o['single']['reroute']); num = den = 0.
    for i, j in itertools.combinations(range(E), 2):
        mij = D(o, (i, j), part, 'margin')['_all']; num += mij - D(o, (i,), part, 'margin')['_all'] - D(o, (j,), part, 'margin')['_all']; den += mij
    return num / den if den > 0 else None

def inc(o, part): return o['stage_info'][part]['increment']

def main():
    load = lambda t: json.load(open(f'{OUT}/{t}.json'))
    O = {t: load(t) for f in FAM.values() for t in f}
    # controls: the method must find a planted backup and must not find a planted non-backup
    ctrl_ok = True; ctrl = {}
    for t in CTRL:
        for kind in ('pos', 'neg'):
            c = load(f'{t}_ctrl_{kind}'); e, u = sorted((c['control']['e'], c['control']['u']))
            s = max(stages(c), key=lambda x: D(c, (e, u), 'all')[x]); hp = hidden(c, e, u, s)
            ok = hp if kind == 'pos' else not hp
            if kind == 'pos':
                b_c, b_o = BI(c, 'all'), BI(O[t], 'all'); ok = ok and b_c is not None and b_o is not None and b_c > b_o
            ctrl[f'{t}_{kind}'] = dict(pair=(e, u), stage=s, hidden=hp, ok=ok); ctrl_ok &= ok
            print(f"control {kind} on {t}: planted pair {(e, u)}, stage {s}, hidden={hp} -> {'PASS' if ok else 'FAIL'}")
    res = {}
    for t, o in O.items():
        res[t] = dict(hidden=hidden_pairs(o), BI={h: BI(o, h) for h in ('all',) + HALVES}, inc={h: inc(o, h) for h in ('all',) + HALVES})
    def kmax(f, h): v = [f(res[t], h) for t in FAM['K']]; return None if any(x is None for x in v) else max(v)
    bk = {h: kmax(lambda r, h: r['BI'][h], h) for h in HALVES}; ik = {h: kmax(lambda r, h: r['inc'][h], h) for h in HALVES}
    def p2(t): return all(res[t]['BI'][h] is not None and bk[h] is not None and res[t]['BI'][h] > bk[h] for h in HALVES)
    def p3(t): return all(res[t]['inc'][h] is not None and ik[h] is not None and res[t]['inc'][h] >= max(1.5 * ik[h], 0.02) for h in HALVES)
    C = FAM['C']; n1 = sum(bool(res[t]['hidden']) for t in C); n2 = sum(p2(t) for t in C); n3 = sum(p3(t) for t in C)
    for fam, tags in FAM.items():
        for t in tags:
            r = res[t]; f = lambda x: 'None' if x is None else f'{x:.3f}'
            print(f"[{fam}] {t:22s} BI all {f(r['BI']['all'])} (A {f(r['BI']['A'])}, B {f(r['BI']['B'])}) | pair increment all "
                  f"{f(r['inc']['all'])} (A {f(r['inc']['A'])}, B {f(r['inc']['B'])}) | hidden pairs {r['hidden']}")
    h10a = 'UNTESTABLE (control failed)' if not ctrl_ok else (n1 >= 4 and n2 >= 4)
    v = dict(controls_pass=ctrl_ok, P1=f'{n1}/5', P2=f'{n2}/5 (K max BI A {bk["A"]}, B {bk["B"]})',
             P3=f'{n3}/5 (K max increment A {ik["A"]}, B {ik["B"]})', H10a=h10a, H10b=n3 >= 4)
    print(v); json.dump(dict(controls=ctrl, results=res, verdict=v), open(f'{OUT}/report.json', 'w'), indent=1, default=str)

if __name__ == '__main__':
    main()
