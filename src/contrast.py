"""Pre-registered contrast: ROUTER vs ITS NULLS, per arm per seed.

Null set (the pre-registered one):
  - shuffled router (permute expert ids across tokens within a layer)
  - synthetic position-only router (8 quantile bins of the stratifier)
  - hash twin (separately trained, fixed token-identity router)
  - within-stratum label permutation, 200 perms -> p

Within-model control (NOT a null for the cell decision, reported alongside):
  - k-means (k=8) on the MoE's OWN pre-router hidden states (the ln2 output the router sees),
    same layer. Answers: does the router beat clustering its own input?
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from analyze import metrics, cp_test, null_shuffled, null_position_router, perm_p_condmi, probe_phase
from extract import extract_seq, extract_state
from model import device_auto
from sklearn.cluster import MiniBatchKMeans
ROOT='/Users/alityb/projects/cubemoe'

def run(tag, data, is_state, max_solves, nperm=200):
    dev=device_auto()
    ex=(extract_state if is_state else extract_seq)(tag, data, dev, max_solves)
    strat = ex['rem'] if is_state else ex['pos']
    strat_name = 'remaining moves' if is_state else 'step index'
    ph=ex['phase']
    out=dict(tag=tag, kind=ex['kind'], strat=strat_name, n_rows=int(len(ph)),
             max_solves=max_solves, layers=[])
    for li in range(ex['NL']):
        rec={'layer':li}
        if ex['kind']=='moe':
            e1=ex['e1'][li]
            m=metrics(e1,ph,strat)
            rec['router_cond_nmi']=m['nmi_strat_wt']; rec['router_raw_nmi']=m['nmi_raw']
            rec['router_cond_mi_bits']=m['cond_mi_bits']; rec['utilization']=m['utilization']
            rec['null_shuffled']=metrics(null_shuffled(e1),ph,strat)['nmi_strat_wt']
            rec['null_position_router']=metrics(null_position_router(strat),ph,strat)['nmi_strat_wt']
            rec['perm_p']=perm_p_condmi(e1,ph,strat,m['cond_mi_bits'],nperm=nperm)
            c=cp_test(e1,ex['solve'],ex['seam'],ex['sollen'],nperm=nperm)
            rec['cp_slope']=c['slope']; rec['cp_partial_slope']=c.get('partial_slope')
            rec['cp_r']=c['r']; rec['cp_partial_p']=c.get('partial_p')
        # within-model control: k-means on the PRE-ROUTER hidden states of THIS model, same layer
        km=MiniBatchKMeans(8,random_state=0,n_init=5,batch_size=4096).fit_predict(ex['rs'][li].astype(np.float32))
        mk=metrics(km,ph,strat)
        rec['kmeans_own_hidden_cond_nmi']=mk['nmi_strat_wt']
        rec['kmeans_own_hidden_raw_nmi']=mk['nmi_raw']
        rec['kmeans_perm_p']=perm_p_condmi(km,ph,strat,mk['cond_mi_bits'],nperm=nperm)
        ck=cp_test(km,ex['solve'],ex['seam'],ex['sollen'],nperm=nperm)
        rec['kmeans_cp_partial_slope']=ck.get('partial_slope')
        rec['probe']=probe_phase(ex['rs'][li],ph,strat)
        out['layers'].append(rec)
        if ex['kind']=='moe':
            print(f"  [{tag} L{li}] router={rec['router_cond_nmi']:.4f} shuf={rec['null_shuffled']:.4f} "
                  f"posr={rec['null_position_router']:.4f} kmeans_own={rec['kmeans_own_hidden_cond_nmi']:.4f} "
                  f"permp={rec['perm_p']:.4f} slope={rec['cp_partial_slope']:+.3f} "
                  f"(p={rec['cp_partial_p']:.3f}) probe={rec['probe']['probe_acc']:.3f}", flush=True)
        else:
            print(f"  [{tag} L{li}] (dense) kmeans_own={rec['kmeans_own_hidden_cond_nmi']:.4f} "
                  f"probe={rec['probe']['probe_acc']:.3f}", flush=True)
    if ex['kind']=='moe':
        best=max(out['layers'], key=lambda r:r['router_cond_nmi'])
        out['best_layer']=best['layer']; out['best_router_cond_nmi']=best['router_cond_nmi']
        out['best_kmeans_own']=max(r['kmeans_own_hidden_cond_nmi'] for r in out['layers'])
        out['best_cp_partial_slope']=max((r['cp_partial_slope'] for r in out['layers']
            if r['cp_partial_slope'] is not None and np.isfinite(r['cp_partial_slope'])), default=float('nan'))
    else:
        out['best_kmeans_own']=max(r['kmeans_own_hidden_cond_nmi'] for r in out['layers'])
    os.makedirs(f'{ROOT}/out/contrast', exist_ok=True)
    json.dump(out, open(f'{ROOT}/out/contrast/{tag}.json','w'), indent=1)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed')
    ap.add_argument('--state',action='store_true'); ap.add_argument('--max_solves',type=int,default=1200)
    ap.add_argument('--nperm',type=int,default=200)
    a=ap.parse_args(); run(a.tag,a.data,a.state,a.max_solves,a.nperm)
