"""EXPLORATORY: state-only (no history, no position) models. Routing of the prediction (CLS) token and
single-expert knockouts at the last layer. Features reused from the sequence-model exploration files
(same held-out solves, same token order)."""
import sys, json, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from model import StateModel, device_auto
from extract import _split
R='/Users/alityb/projects/cubemoe'
MN=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
FB=[MN.index(m) for m in ("F","F'","B","B'")]; RL=[MN.index(m) for m in ("R","R'","L","L'")]
ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data'); a=ap.parse_args(); dev=device_auto()
ms=400 if 'cfop' in a.data else 1200
ck=torch.load(f'{R}/out/{a.tag}.pt',map_location=dev,weights_only=False); c=ck['cfg']
m=StateModel(c['D'],c['NL'],4,'moe',8,2).to(dev); m.load_state_dict(ck['sd']); m.eval()
d,te,off=_split(a.data,ms); rows=np.concatenate([np.arange(off[i],off[i+1]) for i in te])
st=torch.from_numpy(d['states'][rows].astype(np.int64)); y=d['moves'][rows].astype(int)
if 'cfop' in a.data:
    f=dict(np.load(f'{R}/out/explore/cfopfeat_{a.data}.npz')); assert (f['gi']==rows).all()
    lab=f['stage']; alg=lab>=3
else:
    f=dict(np.load(f'{R}/out/explore/g1_moe_mixed_s10.npz')); assert len(f['pos'])==len(rows) and (f['move']==y).all()
    lab=f['phase']; alg=lab==1
def run(ko=None):
    m.blocks[-1].ffn._knockout=ko; P=[];E=[]
    with torch.no_grad():
        for b0 in range(0,len(st),512):
            lg,info=m(st[b0:b0+512].to(dev),collect=(ko is None)); P.append(F.softmax(lg.float(),-1).cpu().numpy())
            if ko is None: E.append(info[-1][1].reshape(lg.shape[0],55,-1)[:,0,0].cpu().numpy())
    m.blocks[-1].ffn._knockout=None
    p=np.concatenate(P); return p[np.arange(len(y)),y], (np.concatenate(E) if E else None), p
p0,E,pfull=run()
res={'tag':a.tag,'acc':float((pfull.argmax(1)==y).mean()),'route':E.tolist(),
     'base':{int(l):float(p0[lab==l].mean()) for l in np.unique(lab)},'drop':{},'by_target':{},'by_class':{}}
for e in range(8):
    pe,_,_=run([e]); dd=p0-pe
    res['drop'][e]={int(l):float(dd[lab==l].mean()) for l in np.unique(lab)}
    res['by_target'][e]={int(t):float(dd[alg&(y==t)].mean()) for t in range(18) if (alg&(y==t)).sum()>=100}
    res['by_class'][e]={'F/B':float(dd[alg&np.isin(y,FB)].mean()),'R/L':float(dd[alg&np.isin(y,RL)].mean())}
json.dump(res,open(f'{R}/out/explore/state_{a.tag}.json','w'))
print(f"{a.tag}: acc {res['acc']:.3f}  max single-expert KO loss by label:",
      {l:round(max(res['drop'][e][l] for e in range(8)),3) for l in res['base']},' baseline:',{l:round(v,3) for l,v in res['base'].items()})
