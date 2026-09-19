"""CONTROLS for the confirmatory run (NOT headline, NOT gate inputs).
k-means (k=8) on each model's own pre-router hidden states at the LAST layer, vs router cond NMI."""
import sys, os, json
import numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from analyze import metrics
from extract import extract_seq, extract_state
from model import device_auto
from sklearn.cluster import MiniBatchKMeans
ROOT='/Users/alityb/projects/cubemoe'
out={}
dev=device_auto()
for arm,base,is_state,ms in [('A','moe_mixed_s',False,1200),('C','state_mixed_s',True,800)]:
    rows=[]
    for s in [10,11,12,13,14]:
        tag=f"{base}{s}"
        ex=(extract_state if is_state else extract_seq)(tag,'mixed',dev,ms)
        L=ex['NL']-1
        strat = ex['rem'] if is_state else ex['pos']
        km=MiniBatchKMeans(8,random_state=0,n_init=5,batch_size=4096).fit_predict(ex['rs'][L].astype(np.float32))
        kn=metrics(km,ex['phase'],strat)['nmi_strat_wt']
        c=json.load(open(f'{ROOT}/out/confirm/{tag}.json'))
        rows.append(dict(seed=s,router=c['cond_nmi'],kmeans_own=kn,
                         loss_e1=c['causal']['loss_e1'],median_other=c['causal']['median_other'],
                         ratio=c['causal']['loss_e1']/max(c['causal']['median_other'],1e-9),
                         acc=c['acc'],bar=c['bar'],acc_ratio=c['acc_ratio'],
                         hash_twin=c.get('hash_twin')))
        print(f"  {tag}: router={c['cond_nmi']:.4f} kmeans_own={kn:.4f} L4ratio={rows[-1]['ratio']:.2f}x",flush=True)
    out[arm]=rows
json.dump(out,open(f'{ROOT}/out/confirm/controls.json','w'),indent=2)
print("\nwrote out/confirm/controls.json")
