"""Applies prereg/PREREG_F2L_SPLIT.md (#11) mechanically to out/f2l/<new seeds>.json and the gate results.
Usage: python src/f2l_report.py  -> prints per-seed checks and the verdict; writes out/f2l/report.json"""
import json, os
R = '/Users/alityb/projects/cubemoe'; NEW = [f'cfop_mixtral_6L256_s{s}' for s in (5, 6, 7)]
CUTS = ('cross|rest', 'F2L|LL', 'OLL|PLL')

# thresholds = 0.8 x the minimum over the 5 calibration models x 2 cross-fit directions (out/f2l/calibration/)
T = dict(cross=0.61, FL=0.39, F2L=0.32, asym=0.99, ko_FL=0.16, ko_ratio=1.91, ko_F2L_ratio=2.01, mko_ratio=1.80)

def ratio(a, b): return float('inf') if b <= 0 and a > 0 else (a / b if b > 0 else 0.)

def checks(o):
    res = {}; teams = set()
    for k, r in o['crossfit'].items():                      # both cross-fitting directions must pass
        t, l = r['team'], r['ll_pair']; d, md = t['damage'], t['margin_damage']; teams.add(tuple(t['pair']))
        res[k] = dict(P1=t['rate']['cross'] >= T['cross'] and t['rate']['FL'] >= T['FL'] and t['rate']['F2L'] >= T['F2L'],
                      P2=ratio(t['rate']['FL'], l['rate']['LL']) >= T['asym'],
                      P3=d['FL'] >= T['ko_FL'] and ratio(d['FL'], d['LL']) >= T['ko_ratio'] and ratio(d['F2L'], d['LL']) >= T['ko_F2L_ratio']
                         and ratio(md['FL'], md['LL']) >= T['mko_ratio'],
                      ll_pair_is_team=r['ll_pair_is_team'], ll_pair_LL_over_FL=ratio(l['damage']['LL'], l['damage']['FL']))
    c = {p: all(v[p] for v in res.values()) for p in ('P1', 'P2', 'P3')}
    c['P1'] = c['P1'] and len(teams) == 1                    # the same pair must be chosen in both halves
    return c, res

def main():
    passing = []
    for t in NEW:
        e = json.load(open(f'{R}/out/prod/{t}_eval.json')); assert e['n'] == 200 and e['data'] == 'cfop250k' and e['budget'] == 90
        g = e['beam256']; ok = g['rate'] >= 0.95
        print(f"gate {t}: {g['solved']}/200 -> {'PASS' if ok else 'FAIL'}"); passing += [t] if ok else []
    if len(passing) < 2: print('fewer than 2 seeds pass the gate: study VOID'); return
    need = len(passing); out = {}
    for t in passing:
        o = json.load(open(f'{R}/out/f2l/{t}.json')); c, detail = checks(o)
        best = max(CUTS, key=lambda x: o['cut_info'][x]['all'])
        out[t] = dict(checks=c, detail=detail, best_cut=best, cut_info={x: o['cut_info'][x]['all'] for x in CUTS})
        print(t, c, '| best cut (descriptive):', best, {x: round(o['cut_info'][x]['all'], 3) for x in CUTS})
    v = {}
    for p in ('P1', 'P2', 'P3'):
        n = sum(out[t]['checks'][p] for t in passing)
        v[p] = f'{n}/{need} -> ' + ('REPLICATED' if n == need and need >= 3 else 'PARTIAL' if n >= 2 else 'NOT REPLICATED')
    print(v); json.dump(dict(per_seed=out, verdict=v), open(f'{R}/out/f2l/report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
