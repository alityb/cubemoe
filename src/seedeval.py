"""Per-seed evaluation: best-layer and per-layer conditional NMI + shuffled null +
change-point partial slope. Deliberately skips probes / perm-tests / k-means nulls that
are not part of the paired comparison, so it runs fast."""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from analyze import metrics, cp_test, null_shuffled
from extract import extract_seq, extract_state
from model import device_auto
from sklearn.cluster import MiniBatchKMeans
ROOT='/Users/alityb/projects/cubemoe'

def evaluate(tag, data, is_state, max_solves):
    dev=device_auto()
    ex=(extract_state if is_state else extract_seq)(tag, data, dev, max_solves)
    strat = ex['rem'] if is_state else ex['pos']
    ph=ex['phase']; out=dict(tag=tag, kind=ex['kind'], n_rows=int(len(ph)), max_solves=max_solves, layers=[])
    for li in range(ex['NL']):
        if ex['kind']=='moe': e1=ex['e1'][li]
        else: e1=MiniBatchKMeans(8,random_state=0,n_init=5,batch_size=4096).fit_predict(ex['fh'][li].astype(np.float32))
        m=metrics(e1,ph,strat); c=cp_test(e1,ex['solve'],ex['seam'],ex['sollen'],nperm=100)
        sh=metrics(null_shuffled(e1),ph,strat)
        keep=ex['rem']>3
        m3=metrics(e1[keep],ph[keep],strat[keep]) if keep.sum()>500 and len(np.unique(ph[keep]))>1 else None
        out['layers'].append(dict(layer=li, cond_nmi=m['nmi_strat_wt'], raw_nmi=m['nmi_raw'],
            shuffled=sh['nmi_strat_wt'], cond_nmi_excl3=(m3['nmi_strat_wt'] if m3 else None),
            partial_slope=c.get('partial_slope'), partial_p=c.get('partial_p'),
            utilization=m.get('utilization')))
    out['best_cond_nmi']=max(l['cond_nmi'] for l in out['layers'])
    out['best_layer']=int(np.argmax([l['cond_nmi'] for l in out['layers']]))
    out['best_partial_slope']=max((l['partial_slope'] for l in out['layers']
                                   if l['partial_slope'] is not None and np.isfinite(l['partial_slope'])), default=float('nan'))
    json.dump(out, open(f'{ROOT}/out/seeds/{tag}_eval.json','w'), indent=1)
    print(f"  {tag}: best L{out['best_layer']} condNMI={out['best_cond_nmi']:.4f} "
          f"slope={out['best_partial_slope']:.3f}", flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed')
    ap.add_argument('--state',action='store_true'); ap.add_argument('--max_solves',type=int,default=1200)
    a=ap.parse_args(); evaluate(a.tag,a.data,a.state,a.max_solves)
