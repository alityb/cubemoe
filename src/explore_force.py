"""EXPLORATORY: what does forcing a token through the 'phase-1 expert' make the model DO?
Prediction if it is an edge-orientation expert: forced tokens gain mass on F/F'/B/B' (edge flips)
specifically, not on R/R'/L/L' (also illegal in phase 2, but they don't flip edges)."""
import sys, argparse, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from model import SeqModel, device_auto
from train import build_seq
from h2_analysis import COLORS, facelets_to_cubie
R='/Users/alityb/projects/cubemoe'
MN=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
CLS={'F/B quarter (flip edges)':[MN.index(m) for m in ("F","F'","B","B'")],
     'R/L quarter (twist, no flip)':[MN.index(m) for m in ("R","R'","L","L'")],
     'G1 moves':[MN.index(m) for m in ('U',"U'",'U2','D',"D'",'D2','R2','L2','F2','B2')]}
ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed250k')
ap.add_argument('--max_solves',type=int,default=400); a=ap.parse_args(); dev=device_auto()
ck=torch.load(f'{R}/out/{a.tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
m=SeqModel(cfg['D'],cfg['NL'],4,cfg['maxlen'],'moe',8,2,'learned').to(dev); m.load_state_dict(ck['sd']); m.eval(); L=cfg['NL']-1
d=dict(np.load(f'{R}/data/{a.data}.npz')); n=len(d['sol_len'])
te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:a.max_solves]
X,Y,P=build_seq(d,te,cfg['maxlen']); B,T=X.shape
off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
PH=np.zeros((B,T),int); EO=np.full((B,T),-1,int); POS=np.full((B,T),-1,int)
for r,i in enumerate(te):
    for t in range(int(d['sol_len'][i])):
        st=facelets_to_cubie(''.join(COLORS[c] for c in d['states'][off[i]+t]))
        PH[r,53+t]=1 if t<int(d['seam'][i]) else 2; POS[r,53+t]=t
        EO[r,53+t]=int(all(x==0 for x in st[3]))
def fwd(ov):
    lgs=[];rts=[]
    for b0 in range(0,B,64):
        x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev)
        for li,blk in enumerate(m.blocks):
            blk.ffn._override=None
            if ov is not None and li==L: blk.ffn._override=torch.from_numpy(ov[b0:b0+64].reshape(-1)).to(dev)
        with torch.no_grad(): lg,info=m(x,pm,collect=(ov is None))
        lgs.append(F.softmax(lg.float(),-1).cpu().numpy())
        if ov is None: rts.append(info[L][1].reshape(x.shape[0],x.shape[1],-1)[:,:,0].cpu().numpy())
    for blk in m.blocks: blk.ffn._override=None
    return np.concatenate(lgs,0),(np.concatenate(rts,0) if ov is None else None)
p0,cur=fwd(None); valid=POS>=0
pr=[((cur==k)&(PH==1)&valid).sum()/max(((cur==k)&valid).sum(),1) for k in range(8)]; k1=int(np.argmax(pr))
sets={'phase 2 (in G1)':(PH==2)&valid,'phase 1, edges ORIENTED':(PH==1)&(EO==1)&valid,'phase 1, edges NOT oriented':(PH==1)&(EO==0)&valid}
out=dict(tag=a.tag,k1=k1,res={})
print(f"{a.tag}: phase-1 expert e{k1}")
for sn,sel in sets.items():
    base={c:float(p0[sel][:,ix].sum(-1).mean()) for c,ix in CLS.items()}
    res={}
    for e in range(8):
        ov=np.full((B,T),-1,np.int64); ov[sel]=e; pf=fwd(ov)[0]
        res[e]={c:float(pf[sel][:,ix].sum(-1).mean())-base[c] for c,ix in CLS.items()}
    out['res'][sn]=dict(n=int(sel.sum()),base=base,delta=res)
    print(f"\n  tokens: {sn}  (n={sel.sum()})   own routing to e{k1}: {(cur[sel]==k1).mean():.3f}")
    print(f"    {'move class':30s}{'baseline':>9}{'force e'+str(k1):>10}{'median other':>14}")
    for c in CLS:
        oth=np.median([res[e][c] for e in range(8) if e!=k1])
        print(f"    {c:30s}{base[c]:9.3f}{res[k1][c]:+10.3f}{oth:+14.3f}")
json.dump(out,open(f'{R}/out/explore/force_{a.tag}.json','w'),indent=1)
