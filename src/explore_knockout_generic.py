"""EXPLORATORY: knock out each expert (router masking, last layer); drop in p(correct next move) by label.
Label = Kociemba phase (seam) or CFOP stage (construction)."""
import sys, json, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from model import SeqModel, device_auto
from train import build_seq
R='/Users/alityb/projects/cubemoe'
ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data'); ap.add_argument('--max_solves',type=int,default=400)
a=ap.parse_args(); dev=device_auto()
ck=torch.load(f'{R}/out/{a.tag}.pt',map_location=dev,weights_only=False); c=ck['cfg']
m=SeqModel(c['D'],c['NL'],4,c['maxlen'],'moe',8,2,'learned').to(dev); m.load_state_dict(ck['sd']); m.eval()
d=dict(np.load(f'{R}/data/{a.data}.npz')); n=len(d['sol_len'])
te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:a.max_solves]
X,Y,P=build_seq(d,te,c['maxlen']); B,T=X.shape
off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))]); LB=np.zeros((B,T),int)
for r,i in enumerate(te):
    for t in range(int(d['sol_len'][i])):
        LB[r,53+t]=int(d['phase'][off[i]+t]) if 'cfop' in a.data else (1 if t<int(d['seam'][i]) else 2)
valid=Y>=0
def run(ko):
    m.blocks[-1].ffn._knockout=ko; out=[]
    for b0 in range(0,B,64):
        x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev)
        with torch.no_grad(): lg,_=m(x,pm)
        out.append(F.softmax(lg.float(),-1).cpu().numpy())
    m.blocks[-1].ffn._knockout=None
    p=np.concatenate(out,0); pt=np.zeros((B,T)); pt[valid]=p[valid,Y[valid]]; return pt
labs=sorted(set(LB[valid].tolist())); p0=run(None)
res={'tag':a.tag,'base':{l:float(p0[valid&(LB==l)].mean()) for l in labs},'drop':{}}
for e in range(8):
    pe=run([e]); res['drop'][e]={l:float((p0-pe)[valid&(LB==l)].mean()) for l in labs}
# per target-move breakdown within the last-layer algorithms (CFOP stages 3-4) / whole solve otherwise
alg=valid&((LB>=3) if 'cfop' in a.data else (LB==1))   # CFOP: OLL+PLL; Kociemba: phase 1
res['by_target']={}
for e in range(8):
    m.blocks[-1].ffn._knockout=None
pe_all={}
for e in range(8):
    pe=run([e]); res['by_target'][e]={int(t):float((p0-pe)[alg&(Y==t)].mean()) for t in range(18) if (alg&(Y==t)).sum()>=100}
res['n_target']={int(t):int((alg&(Y==t)).sum()) for t in range(18)}
json.dump(res,open(f'{R}/out/explore/koG_{a.tag}.json','w'),indent=1)
print(a.tag,' max single-expert drop by label:',{l:round(max(res['drop'][e][l] for e in range(8)),3) for l in labs},
      ' baseline:',{l:round(v,3) for l,v in res['base'].items()})
