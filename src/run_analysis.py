"""Driver: confound table, per-layer metrics vs all nulls, change-point, figures."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, '/Users/alityb/projects/cubemoe/src')
from analyze import (metrics, cp_test, null_shuffled, null_position_router,
                     perm_p_condmi, probe_phase, bayes_acc, mi_bits, E)
from extract import extract_seq, extract_state
from model import device_auto
TAIL_CUTS = (3, 6)   # report routing metrics with and without the last N steps (leakage)
from sklearn.cluster import MiniBatchKMeans
ROOT = '/Users/alityb/projects/cubemoe'
R = {}

def confound_table(ex, name):
    ph = ex['phase']
    t = dict(n=int(len(ph)), majority=float(max(np.mean(ph == 1), np.mean(ph == 2))))
    for k, key in [('position', [ex['pos']]), ('move', [ex['move']]), ('prev_move', [ex['prev']]),
                   ('position+move', [ex['pos'], ex['move']])]:
        t[f'bayes_{k}'] = bayes_acc(key, ph)
        t[f'mi_{k}_bits'] = mi_bits(np.unique(np.stack(key, 1), axis=0, return_inverse=True)[1], ph)
    R.setdefault('confound', {})[name] = t
    return t

def analyse_model(tag, data, dev, is_state=False, max_solves=2500, nulls_from=None):
    t0 = time.time()
    ex = (extract_state if is_state else extract_seq)(tag, data, dev, max_solves)
    strat = ex['rem'] if is_state else ex['pos']          # C conditions on remaining moves
    strat_name = 'remaining' if is_state else 'position'
    ph = ex['phase']
    out = dict(tag=tag, data=data, kind=ex['kind'], strat=strat_name, n_rows=int(len(ph)),
               tail_cuts=list(TAIL_CUTS), layers=[])
    for li in range(ex['NL']):
        lay = {'layer': li}
        if ex['kind'] == 'moe':
            e1 = ex['e1'][li]
        else:
            km = MiniBatchKMeans(E, random_state=0, n_init=5, batch_size=4096)
            e1 = km.fit_predict(ex['fh'][li].astype(np.float32))
        lay['observed'] = metrics(e1, ph, strat)
        lay['observed']['cp'] = cp_test(e1, ex['solve'], ex['seam'], ex['sollen'])
        lay['null_shuffled'] = metrics(null_shuffled(e1), ph, strat)
        lay['null_shuffled']['cp'] = cp_test(null_shuffled(e1), ex['solve'], ex['seam'], ex['sollen'])
        pr = null_position_router(strat)
        lay['null_position_router'] = metrics(pr, ph, strat)
        lay['null_position_router']['cp'] = cp_test(pr, ex['solve'], ex['seam'], ex['sollen'])
        lay['perm_p_cond_mi'] = perm_p_condmi(e1, ph, strat, lay['observed']['cond_mi_bits'])
        # --- tail-excluded replicates (near-solved states leak into train) ---
        for cut in TAIL_CUTS:
            keep = ex['rem'] > cut
            if keep.sum() > 500 and len(np.unique(ph[keep])) > 1:
                k = f'excl_tail{cut}'
                lay[k] = metrics(e1[keep], ph[keep], strat[keep])
                lay[k]['cp'] = cp_test(e1[keep], ex['solve'][keep], ex['seam'][keep], ex['sollen'][keep])
                lay[k+'_shuffled'] = metrics(null_shuffled(e1)[keep], ph[keep], strat[keep])
                lay[k+'_perm_p'] = perm_p_condmi(e1[keep], ph[keep], strat[keep], lay[k]['cond_mi_bits'])
        lay['excl_tail'] = lay.get('excl_tail3'); lay['excl_tail_shuffled'] = lay.get('excl_tail3_shuffled')
        lay['excl_tail_perm_p'] = lay.get('excl_tail3_perm_p')
        lay['probe'] = probe_phase(ex['rs'][li], ph, strat)
        out['layers'].append(lay)
        print(f"  [{tag} L{li}] condNMI={lay['observed']['nmi_strat_wt']:.4f} "
              f"(shuf {lay['null_shuffled']['nmi_strat_wt']:.4f}, posrouter {lay['null_position_router']['nmi_strat_wt']:.4f}) "
              f"rawNMI={lay['observed']['nmi_raw']:.4f} cp_slope={lay['observed']['cp']['slope']:.3f} "
              f"p={lay['perm_p_cond_mi']:.4f} probe={lay['probe']['probe_acc']:.3f} "
              f"| exclTail condNMI={lay.get('excl_tail',{}).get('nmi_strat_wt',float('nan')):.4f}", flush=True)
    out['wall_s'] = time.time() - t0
    return out, ex

if __name__ == '__main__':
    dev = device_auto(); print('device', dev)
    todo = [('moe_mixed', 'mixed', False, 2500), ('moe_naive', 'naive', False, 2500),
            ('hash_mixed', 'mixed', False, 2500), ('dense_mixed', 'mixed', False, 1200),
            ('state_mixed', 'mixed', True, 800), ('state_dense_mixed', 'mixed', True, 800)]
    R['models'] = {}
    first = {}
    for tag, data, isst, ms in todo:
        if not os.path.exists(f'{ROOT}/out/{tag}.pt'):
            print(f'SKIP {tag} (no checkpoint)'); continue
        print(f'--- {tag}', flush=True)
        res, ex = analyse_model(tag, data, dev, isst, ms)
        R['models'][tag] = res
        if data not in first: first[data] = ex; confound_table(ex, data)
        R.setdefault('cp_scatter', {})[tag] = {
            str(i): dict(xs=L['observed']['cp'].get('_xs', []), ys=L['observed']['cp'].get('_ys', []))
            for i, L in enumerate(res['layers'])}
    # figure-0 data: P(phase2 | pos) per dataset
    R['overlap'] = {}
    for dname, ex in first.items():
        ps, fr, ct = [], [], []
        for p in np.unique(ex['pos']):
            m = ex['pos'] == p
            if m.sum() < 20: continue
            ps.append(int(p)); fr.append(float(np.mean(ex['phase'][m] == 2))); ct.append(int(m.sum()))
        R['overlap'][dname] = dict(pos=ps, p_phase2=fr, count=ct)
    for tag in list(R['models']):
        for L in R['models'][tag]['layers']:
            for k in ('observed','null_shuffled','null_position_router','excl_tail','excl_tail_shuffled',
                      'excl_tail3','excl_tail3_shuffled','excl_tail6','excl_tail6_shuffled'):
                if k not in L or L[k] is None or 'cp' not in L[k]: continue
                L[k]['cp'].pop('_xs', None); L[k]['cp'].pop('_ys', None)
    json.dump(R, open(f'{ROOT}/out/results.json', 'w'), indent=1)
    print('wrote out/results.json')
