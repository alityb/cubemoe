"""EXPLORATORY: knock out each expert (router masking, last layer) and measure the drop in probability
of the CORRECT next move, split by what that move is. Double-dissociation test of the edge/corner experts."""
import sys, json, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from model import SeqModel, device_auto
from train import build_seq
R='/Users/alityb/projects/cubemoe'
MN=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
FB=[MN.index(m) for m in ("F","F'","B","B'")]; RL=[MN.index(m) for m in ("R","R'","L","L'")]
ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed250k')
ap.add_argument('--max_solves',type=int,default=400); a=ap.parse_args(); dev=device_auto()
ck=torch.load(f'{R}/out/{a.tag}.pt',map_location=dev,weights_only=False); c=ck['cfg']
m=SeqModel(c['D'],c['NL'],4,c['maxlen'],'moe',8,2,'learned').to(dev); m.load_state_dict(ck['sd']); m.eval()
d=dict(np.load(f'{R}/data/{a.data}.npz')); n=len(d['sol_len'])
te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:a.max_solves]
X,Y,P=build_seq(d,te,c['maxlen']); B,T=X.shape
PH=np.zeros((B,T),int)
for r,i in enumerate(te):
    for t in range(int(d['sol_len'][i])): PH[r,53+t]=1 if t<int(d['seam'][i]) else 2
tgt=Y.copy(); valid=tgt>=0
def run(ko):
    m.blocks[-1].ffn._knockout=ko; out=[]
    for b0 in range(0,B,64):
        x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev)
        with torch.no_grad(): lg,_=m(x,pm)
        out.append(F.softmax(lg.float(),-1).cpu().numpy())
    m.blocks[-1].ffn._knockout=None
    p=np.concatenate(out,0); pt=np.zeros((B,T)); pt[valid]=p[valid,tgt[valid]]
    return pt, p.argmax(-1)
groups={'ph1: correct move is F/B (edge flip)':valid&(PH==1)&np.isin(tgt,FB),
        'ph1: correct move is R/L (corner twist)':valid&(PH==1)&np.isin(tgt,RL),
        'ph1: correct move is a G1 move':valid&(PH==1)&~np.isin(tgt,FB+RL),
        'phase 2':valid&(PH==2)}
p0,a0=run(None)
res={'tag':a.tag,'n':{g:int(s.sum()) for g,s in groups.items()},
     'base':{g:float(p0[s].mean()) for g,s in groups.items()},'drop':{}}
for e in range(8):
    pe,ae=run([e])
    res['drop'][e]={g:float((p0[s]-pe[s]).mean()) for g,s in groups.items()}
    res['drop'][e].update({f'acc_drop {g}':float((a0[s]==tgt[s]).mean()-(ae[s]==tgt[s]).mean()) for g,s in groups.items()})
json.dump(res,open(f'{R}/out/explore/knockout_{a.tag}.json','w'),indent=1)
print(a.tag,' baseline p(correct):',{g.split(':')[-1].strip()[:16]:round(v,3) for g,v in res['base'].items()})
for e in range(8):
    print(f"  knock out e{e}: drop in p(correct)  "+"  ".join(f"{g.split(':')[-1].strip()[:18]:18s}{res['drop'][e][g]:+.3f}" for g in groups))
