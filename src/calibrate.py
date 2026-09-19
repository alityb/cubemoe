"""Calibration anchors computed on the ACTUAL (seam, sol_len) distribution of a dataset,
so the numbers are directly comparable to the observed routing metrics (NMI is
sensitive to class balance, which differs between datasets)."""
import sys, json; sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
import numpy as np
from analyze import cp_test, metrics
ROOT='/Users/alityb/projects/cubemoe'

def run(kind='mixed', seed=0, max_solves=2500):
    d=np.load(f'{ROOT}/data/{kind}.npz')
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    seams=d['seam'][te].astype(int); lens=d['sol_len'][te].astype(int)
    rng=np.random.RandomState(seed)
    solve=[];seam=[];pos=[];phase=[];sollen=[]
    for i,(s,L) in enumerate(zip(seams,lens)):
        solve+=[i]*L; seam+=[s]*L; pos+=list(range(L))
        phase+=[1]*s+[2]*(L-s); sollen+=[L]*L
    solve,seam,pos,phase,sollen=map(np.array,(solve,seam,pos,phase,sollen))
    def synth(kind_,nz=0.0):
        if kind_=='phase': e=np.where(phase==1,rng.randint(0,4,len(phase)),rng.randint(4,8,len(phase)))
        elif kind_=='position': e=np.minimum(7,pos//4)
        else: e=rng.randint(0,8,len(phase))
        if nz>0:
            m=rng.rand(len(e))<nz; e[m]=rng.randint(0,8,m.sum())
        return e
    out=[]
    for nm,k,nz in [('perfect phase tracker','phase',0),('phase tracker + 30% noise','phase',.3),
                    ('phase tracker + 60% noise','phase',.6),('pure position router','position',0),
                    ('uniform random routing','random',0)]:
        e=synth(k,nz); c=cp_test(e,solve,seam,sollen); m=metrics(e,phase,pos)
        out.append(dict(name=nm, nmi_raw=m['nmi_raw'], cond_nmi=m['nmi_strat_wt'],
                        slope=c['slope'], partial_slope=c['partial_slope'], partial_p=c['partial_p']))
    anchor=out[0]['cond_nmi']
    for x in out: x['frac_of_anchor']=x['cond_nmi']/anchor if anchor>0 else float('nan')
    return dict(kind=kind, phase1_frac=float(np.mean(phase==1)), anchor_cond_nmi=anchor, rows=out)

if __name__=='__main__':
    res={k:run(k) for k in ('mixed','naive')}
    json.dump(res, open(f'{ROOT}/out/calibration.json','w'), indent=1)
    for k,v in res.items():
        print(f"--- {k}: phase1 frac {v['phase1_frac']:.3f}  anchor(perfect tracker) condNMI={v['anchor_cond_nmi']:.4f}")
        for x in v['rows']:
            print(f"   {x['name']:<26} condNMI={x['cond_nmi']:.4f} ({x['frac_of_anchor']*100:5.1f}% of anchor) "
                  f"partial_slope={x['partial_slope']:+.3f} p={x['partial_p']:.3f}")
