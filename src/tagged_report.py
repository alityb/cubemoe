"""Applies the rules of prereg/PREREG_STAGE_TAGGED.md (#16) to the tagged models' outputs.
Inputs: out/tagged/gate_<tag>.json (copied eval), out/g1pll/<tag>.json (y_stage), out/geom/<tag>__cfop250k.json (r),
out/tagged/swap/<tag>__cfop250k.json (D). Comparison values are the pre-registered constants.
Usage: python src/tagged_report.py -> prints and writes out/tagged/report.json"""
import json, os
R = '/Users/alityb/projects/cubemoe'; T = [f'cfop_tagged_6L256_s{s}' for s in range(4)]
Y_OLD, R_OLD = 0.0448, 0.084

def main():
    rows = {}
    for t in T:
        g = json.load(open(f'{R}/out/prod/{t}_eval.json'))['beam256']; ok = g['rate'] >= 0.95
        y = json.load(open(f'{R}/out/g1pll/{t}.json'))['y_stage']
        a = json.load(open(f'{R}/out/geom/{t}__cfop250k.json'))['partA']['L5']
        beyond = lambda x: x['eta2_z'] > x['null_pca_p99'] and x['eta2_z'] >= 1.5 * x['null_pca_median']
        r = a['given_S']['r'] or 0.; bg = beyond(a['given_S_halfA']) and beyond(a['given_S_halfB'])
        sw = json.load(open(f'{R}/out/tagged/swap/{t}__cfop250k.json'))['sets']['all']['changed']
        rows[t] = dict(gate_rate=g['rate'], gate_moves=g['avg_moves'], passes_gate=ok, y_stage=y, r=r, beyond_geometry=bg,
                       ceiling=a['given_S']['ceiling'], D=sw['D'], D_ci=sw['ci'], changed_cross=sw['cross'], changed_same=sw['same'])
        print(f"{t}: gate {g['rate']:.3f} ({g['avg_moves']:.1f} moves) | y_stage {y*100:.1f}% | r {r:.3f} ceiling {a['given_S']['ceiling']:.3f} "
              f"beyond-geometry {bg} | D {sw['D']:+.3f} [{sw['ci'][0]:+.3f} {sw['ci'][1]:+.3f}] (cross {sw['cross']:.3f} / same {sw['same']:.3f})")
    P = [t for t in T if rows[t]['passes_gate']]; need = 3 if len(P) == 4 else len(P)
    T1 = [t for t in P if rows[t]['y_stage'] >= 2 * Y_OLD]
    T2 = [t for t in P if rows[t]['r'] >= 2 * R_OLD and rows[t]['beyond_geometry']]
    T3 = [t for t in P if rows[t]['D'] >= 0.05 and rows[t]['D_ci'][0] > 0]
    KL = [t for t in P if rows[t]['y_stage'] >= 0.15 and rows[t]['r'] >= 0.21]
    if len(P) < 3: read = 'VOID: fewer than 3 models pass the competence gate'
    else:
        t1, t2, t3 = len(T1) >= need, len(T2) >= need, len(T3) >= need
        if t1 and t2: read = 'OUTPUT DEPENDENCE SUFFICIENT: making the output depend on the stage makes routing align with it'
        elif not t1 and not t2: read = 'OUTPUT DEPENDENCE NOT SUFFICIENT: even with stage-tagged outputs the router does not select for the stage'
        elif t1: read = 'MORE STAGE-STRUCTURED ROUTING, NOT SELECTION (T1 holds, T2 fails)'
        else: read = 'SELECTION AT THE LOGIT LEVEL WITHOUT TOP-1 ROUTING SHARES (T2 holds, T1 fails)'
        read += '; route swaps ' + ('DO' if t3 else 'do NOT') + ' change the moves (T3)'
        if len(KL) >= need: read += '; Kociemba-like'
    v = dict(passing=f'{len(P)}/4', need=need, T1=f'{len(T1)}/{len(P)}', T2=f'{len(T2)}/{len(P)}', T3=f'{len(T3)}/{len(P)}',
             kociemba_like=f'{len(KL)}/{len(P)}', verdict=read, rows=rows)
    print(json.dumps({k: x for k, x in v.items() if k != 'rows'}, indent=1)); json.dump(v, open(f'{R}/out/tagged/report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
