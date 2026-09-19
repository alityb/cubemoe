"""CAUSAL test: is expert identity USED to compute phase-appropriate moves, or does it merely
correlate with phase?

At the target layer we force the top-1 expert for selected prediction positions and measure the
behavioural readout that is sharply phase-specific: probability mass on the 10 G1-legal moves
(in phase 2 the correct move is ALWAYS G1-legal), plus next-move accuracy.

Conditions: baseline | swap (phase-2 positions -> phase-1 expert and vice versa)
            | random (force a random OTHER expert, same tokens) | self (force current top-1; sanity)
"""
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from model import SeqModel, device_auto, N_COLOR, N_MOVE
import cube
ROOT='/Users/alityb/projects/cubemoe'
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1_IDX=torch.tensor([i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES])

def load(tag, dev):
    ck=torch.load(f'{ROOT}/out/{tag}.pt', map_location=dev, weights_only=False); cfg=ck['cfg']
    _router = 'hash' if cfg.get('model')=='hash' else 'learned'
    m=SeqModel(cfg['D'],cfg['NL'],4,cfg['maxlen'],'moe',8,2,_router).to(dev)
    m.load_state_dict(ck['sd']); m.eval(); return m,cfg

def build(data, max_solves):
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    return d, te

def run(tag, data='mixed', max_solves=1200, layer=None, seed=0):
    dev=device_auto(); m,cfg=load(tag,dev); d,te=build(data,max_solves)
    from train import build_seq
    X,Y,P=build_seq(d,te,cfg['maxlen'])
    B,T=X.shape; NL=cfg['NL']
    # per-position phase for PREDICTION positions 53..53+n-1 (label = phase of the move predicted)
    PH=np.zeros((B,T),np.int64)
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    for r,i in enumerate(te):
        n_=int(d['sol_len'][i]); sm=int(d['seam'][i])
        for t in range(n_): PH[r,53+t]=1 if t<sm else 2
    g1=G1_IDX.to(dev)
    def forward(ov=None, tgt_layer=None):
        outs=[]
        for b0 in range(0,B,64):
            x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev)
            bb=x.shape[0]
            for li,blk in enumerate(m.blocks):
                blk.ffn._override = None
                if ov is not None and li==tgt_layer:
                    blk.ffn._override = torch.from_numpy(ov[b0:b0+64].reshape(-1)).to(dev)
            with torch.no_grad(): lg,info=m(x,pm,collect=(ov is None))
            outs.append((lg.cpu(), [ (i[1].reshape(bb,x.shape[1],2).cpu() if ov is None else None) for i in info ] if ov is None else None))
        for blk in m.blocks: blk.ffn._override=None
        return outs
    # --- baseline pass: logits + routing
    base=forward(None,None)
    logits=torch.cat([o[0] for o in base],0)
    route=[torch.cat([o[1][li] for o in base],0) for li in range(NL)]
    mask=torch.from_numpy(PH>0)
    ph=torch.from_numpy(PH)
    def readout(lg):
        p=F.softmax(lg.float(),-1)
        res={}
        for phv,name in [(1,'phase1'),(2,'phase2')]:
            sel=(ph==phv)
            res[f'{name}_g1mass']=float(p[sel][:,g1.cpu()].sum(-1).mean())
            tgt=torch.from_numpy(Y)[sel]
            res[f'{name}_acc']=float((lg[sel].argmax(-1)==tgt).float().mean())
        return res
    base_ro=readout(logits)
    # --- choose the phase-1 / phase-2 dominant experts at the target layer
    if layer is None: layer=NL-1
    e_top=route[layer][:,:,0]
    sel=mask
    ct=np.zeros((8,3))
    for e in range(8):
        me=(e_top==e)&sel
        ct[e,1]=int(((ph==1)&me).sum()); ct[e,2]=int(((ph==2)&me).sum())
    tot=ct[:,1]+ct[:,2]; ok=tot>50
    p1=np.where(ok, ct[:,1]/np.maximum(tot,1), -1); p2=np.where(ok, ct[:,2]/np.maximum(tot,1), -1)
    e1=int(np.argmax(p1)); e2=int(np.argmax(p2))
    if e1==e2: e2=int(np.argsort(-p2)[1])
    # --- interventions
    rng=np.random.RandomState(seed)
    ov_swap=np.full((B,T),-1,np.int64); ov_rand=np.full((B,T),-1,np.int64); ov_self=np.full((B,T),-1,np.int64)
    ov_sing=np.full((B,T),-1,np.int64)
    others=[e for e in range(8) if e not in (e1,e2)]
    sA,sB=rng.choice(others,2,replace=False) if len(others)>=2 else (e1,e2)
    cur=e_top.numpy()
    idx=np.argwhere(PH>0)
    for r,c in idx:
        ov_swap[r,c]= e1 if PH[r,c]==2 else e2
        alt=rng.randint(8)
        while alt==cur[r,c]: alt=rng.randint(8)
        ov_rand[r,c]=alt
        ov_self[r,c]=cur[r,c]
        ov_sing[r,c]= int(sA) if PH[r,c]==2 else int(sB)
    res=dict(tag=tag, layer=layer, e_phase1=e1, e_phase2=e2,
             p_phase1_given_e1=float(p1[e1]), p_phase2_given_e2=float(p2[e2]),
             frac_tokens_changed_swap=float((ov_swap[PH>0]!=cur[PH>0]).mean()),
             frac_tokens_changed_rand=float((ov_rand[PH>0]!=cur[PH>0]).mean()),
             baseline=base_ro)
    res['single_pair']=[int(sA),int(sB)]
    for nm,ov in [('swap',ov_swap),('random',ov_rand),('single',ov_sing),('self',ov_self)]:
        lg=torch.cat([o[0] for o in forward(ov,layer)],0)
        res[nm]=readout(lg)
    os.makedirs(f'{ROOT}/out/causal',exist_ok=True)
    json.dump(res,open(f'{ROOT}/out/causal/{tag}_L{layer}.json','w'),indent=1)
    if layer==cfg['NL']-1: json.dump(res,open(f'{ROOT}/out/causal/{tag}.json','w'),indent=1)
    b=res['baseline']
    print(f"  {tag} L{layer} e1={e1}(P(p1|e)={p1[e1]:.2f}) e2={e2}(P(p2|e)={p2[e2]:.2f}) "
          f"changed swap={res['frac_tokens_changed_swap']:.2f} rand={res['frac_tokens_changed_rand']:.2f}")
    for nm in ('baseline','self','random','single','swap'):
        r=res[nm]
        print(f"    {nm:<9} p2_G1mass={r['phase2_g1mass']:.4f}  p2_acc={r['phase2_acc']:.4f}  "
              f"p1_acc={r['phase1_acc']:.4f}")
    return res

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed')
    ap.add_argument('--max_solves',type=int,default=1200); ap.add_argument('--layer',type=int,default=None)
    a=ap.parse_args(); run(a.tag,a.data,a.max_solves,a.layer)
