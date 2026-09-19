"""Causal intervention on the POSITION-BLIND arm C.
C sees only the 54 stickers of the current state — no history, no step index — so a causal
effect here cannot be positional by construction. Routing is intervened at the CLS readout token."""
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from model import StateModel, device_auto, N_MOVE
import cube
ROOT='/Users/alityb/projects/cubemoe'
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1_IDX=torch.tensor([i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES])

def run(tag, data='mixed', max_solves=800, layer=None, seed=0):
    dev=device_auto()
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
    m=StateModel(cfg['D'],cfg['NL'],4,'moe',8,2).to(dev); m.load_state_dict(ck['sd']); m.eval()
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    rows=np.concatenate([np.arange(off[i],off[i+1]) for i in te])
    st=d['states'][rows].astype(np.int64); ph=d['phase'][rows].astype(np.int64); Y=d['moves'][rows].astype(np.int64)
    NL=cfg['NL']; layer=NL-1 if layer is None else layer
    g1=G1_IDX.to(dev); N=len(st)
    def forward(ov=None):
        L=[];R=[]
        for b0 in range(0,N,512):
            x=torch.from_numpy(st[b0:b0+512]).to(dev); bb=x.shape[0]
            for li,blk in enumerate(m.blocks):
                blk.ffn._override=None
                if ov is not None and li==layer:
                    o=np.full((bb,55),-1,np.int64); o[:,0]=ov[b0:b0+512]   # CLS token only
                    blk.ffn._override=torch.from_numpy(o.reshape(-1)).to(dev)
            with torch.no_grad(): lg,info=m(x,collect=(ov is None))
            L.append(lg.cpu())
            if ov is None: R.append(info[layer][1].reshape(bb,55,2)[:,0,0].cpu())
        for blk in m.blocks: blk.ffn._override=None
        return torch.cat(L,0), (torch.cat(R,0) if R else None)
    lg,route=forward(None)
    def readout(lg):
        p=F.softmax(lg.float(),-1); r={}
        for v,nm in [(1,'phase1'),(2,'phase2')]:
            s=(torch.from_numpy(ph)==v)
            r[f'{nm}_g1mass']=float(p[s][:,g1.cpu()].sum(-1).mean())
            r[f'{nm}_acc']=float((lg[s].argmax(-1)==torch.from_numpy(Y)[s]).float().mean())
        return r
    base=readout(lg)
    cur=route.numpy()
    ct=np.zeros((8,3))
    for e in range(8):
        me=cur==e; ct[e,1]=int(((ph==1)&me).sum()); ct[e,2]=int(((ph==2)&me).sum())
    tot=ct[:,1]+ct[:,2]; ok=tot>50
    p1=np.where(ok,ct[:,1]/np.maximum(tot,1),-1); p2=np.where(ok,ct[:,2]/np.maximum(tot,1),-1)
    e1=int(np.argmax(p1)); e2=int(np.argmax(p2))
    if e1==e2: e2=int(np.argsort(-p2)[1])
    rng=np.random.RandomState(seed)
    others=[e for e in range(8) if e not in (e1,e2)]
    sA,sB=(rng.choice(others,2,replace=False) if len(others)>=2 else (e1,e2))
    ov_swap=np.where(ph==2,e1,e2).astype(np.int64)
    ov_sing=np.where(ph==2,int(sA),int(sB)).astype(np.int64)
    ov_self=cur.astype(np.int64)
    ov_rand=np.array([rng.choice([e for e in range(8) if e!=c]) for c in cur],np.int64)
    res=dict(tag=tag,layer=layer,e_phase1=e1,e_phase2=e2,
             p_phase1_given_e1=float(p1[e1]),p_phase2_given_e2=float(p2[e2]),baseline=base,
             n=int(N))
    for nm,ov in [('swap',ov_swap),('random',ov_rand),('single',ov_sing),('self',ov_self)]:
        res[nm]=readout(forward(ov)[0])
    os.makedirs(f'{ROOT}/out/causal',exist_ok=True)
    json.dump(res,open(f'{ROOT}/out/causal/{tag}_state_L{layer}.json','w'),indent=1)
    print(f"  {tag} L{layer} e1={e1}({p1[e1]:.2f}) e2={e2}({p2[e2]:.2f}) n={N}")
    for nm in ('baseline','self','random','single','swap'):
        r=res[nm]; print(f"    {nm:<9} p2_G1mass={r['phase2_g1mass']:.4f}  p2_acc={r['phase2_acc']:.4f}  p1_acc={r['phase1_acc']:.4f}")
    return res

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed')
    ap.add_argument('--max_solves',type=int,default=800); ap.add_argument('--layer',type=int,default=None)
    a=ap.parse_args(); run(a.tag,a.data,a.max_solves,a.layer)
