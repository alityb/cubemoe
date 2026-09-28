"""EXPLORATORY: what do CFOP routers track? feat mode: per-token features for a dataset's held-out rows
(identical token order for every model on that dataset). route mode: top-1 expert per layer for a model."""
import sys, argparse, numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from extract import extract_seq, _split
import cube, cfop
from h2_analysis import COLORS, facelets_to_cubie
R='/Users/alityb/projects/cubemoe'; MS=400
ap=argparse.ArgumentParser(); ap.add_argument('--mode'); ap.add_argument('--data'); ap.add_argument('--tag',default='')
a=ap.parse_args()
if a.mode=='feat':
    d,te,off=_split(a.data,MS); meta=dict(np.load(f'{R}/out/explore/cfopmeta_{a.data}.npz'))
    gi=[];pos=[]
    for i in te:
        for t in range(int(d['sol_len'][i])): gi.append(off[i]+t); pos.append(t)
    gi=np.array(gi); pos=np.array(pos)
    cross=np.zeros(len(gi),np.int8); slots=np.zeros(len(gi),np.int8); llo=np.zeros(len(gi),np.int8); g1=np.zeros(len(gi),np.int8)
    for k,g in enumerate(gi):
        st=facelets_to_cubie(''.join(COLORS[c] for c in d['states'][g]))
        cross[k]=cfop.cross_done(st); slots[k]=cfop.n_slots_done(st); llo[k]=cfop.oll_done(st); g1[k]=cube.in_G1(st)
    mv=d['moves'][gi].astype(int); prev=np.where(pos>0,np.roll(mv,1),18)
    np.savez_compressed(f'{R}/out/explore/cfopfeat_{a.data}.npz',gi=gi,pos=pos,stage=d['phase'][gi].astype(int),
        seg=meta['seg'][gi],segpos=meta['segpos'][gi],seglen=meta['seglen'][gi],case=meta['case'][gi],
        move=mv,prev=prev,cross=cross,slots=slots,llo=llo,g1=g1)
    print(a.data,'features:',len(gi),'tokens')
else:
    from model import device_auto
    ex=extract_seq(a.tag,a.data,device_auto(),MS)
    np.savez_compressed(f'{R}/out/explore/cfoproute_{a.tag}.npz',e1=np.stack(ex['e1']),e2=np.stack(ex['e2']),pos=ex['pos'],solve=ex['solve'])
    print(a.tag,'routed',len(ex['pos']))
