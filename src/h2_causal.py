"""H2 causal intervention: does the CFOP-trained router CAUSALLY carry the 4-stage
decomposition?  Correlational cNMI has almost no headroom here (stage ceiling 0.066 vs
G1 ceiling 0.556), and in H1 a 7x causal effect coexisted with cNMI 0.0193 -- so the
intervention is the decisive test, not the correlational one.

Two readouts on the SAME model:
  (a) STAGE: force only stage-s tokens through expert e -> per-stage accuracy damage L[e,s].
      Faithful to H1: only the target tokens are overridden, so context is not contaminated.
  (b) G1:    the exact H1 readout (G1-legal probability mass) on the CFOP model.
If routing tracks the data's decomposition, (a) should show stage-specific damage with a
DIFFERENT native expert per stage. If routing tracks the cube instead, (b) fires and (a) does not.
"""
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from extract import extract_seq
from model import SeqModel, device_auto
from train import build_seq
import cube
from h2_analysis import COLORS, facelets_to_cubie
ROOT='/Users/alityb/projects/cubemoe'
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1IDX=[i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES]

def _wt(vals, ws):
    ws=np.asarray(ws,float)
    if ws.sum()==0: return float('nan')
    ws=ws/ws.sum(); return float((np.asarray(vals)*ws).sum())

def run(tag, data, dev, max_solves, E=8):
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
    m=SeqModel(cfg['D'],cfg['NL'],4,cfg['maxlen'],'moe',E,2,'learned').to(dev)
    m.load_state_dict(ck['sd']); m.eval(); L=cfg['NL']-1
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['sol_len']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    X,Y,P=build_seq(d,te,cfg['maxlen']); B,T=X.shape
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])

    STG=np.zeros((B,T),np.int64); G1=np.zeros((B,T),np.int64); POS=np.full((B,T),-1,np.int64)
    for r,i in enumerate(te):
        nn=int(d['sol_len'][i])
        for t in range(nn):
            gt=off[i]+t
            STG[r,53+t]=int(d['phase'][gt]); POS[r,53+t]=t
            st=facelets_to_cubie(''.join(COLORS[c] for c in d['states'][gt]))
            G1[r,53+t]=2 if cube.in_G1(st) else 1

    def fwd(ov):
        lgs=[]; rts=[]
        for b0 in range(0,B,64):
            x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev); bb=x.shape[0]
            for li,blk in enumerate(m.blocks):
                blk.ffn._override=None
                if ov is not None and li==L:
                    blk.ffn._override=torch.from_numpy(ov[b0:b0+64].reshape(-1)).to(dev)
            with torch.no_grad(): lg,info=m(x,pm,collect=(ov is None))
            lgs.append(lg.cpu())
            if ov is None: rts.append(info[L][1].reshape(bb,x.shape[1],-1)[:,:,0].cpu())
        for blk in m.blocks: blk.ffn._override=None
        return torch.cat(lgs,0), (torch.cat(rts,0) if ov is None else None)

    lg0,route=fwd(None); cur=route.numpy()
    Yt=torch.from_numpy(Y)
    def acc(lg, sel):
        pred=lg.argmax(-1); ok=(pred==Yt).numpy()[sel]; s=POS[sel]
        vals=[];ws=[]
        for u in np.unique(s):
            k=s==u
            if k.sum()<20: continue
            vals.append(ok[k].mean()); ws.append(k.sum())
        return _wt(vals,ws)
    def g1mass(lg, sel):
        p=F.softmax(lg.float(),-1).numpy(); v=p[sel][:,G1IDX].sum(-1); s=POS[sel]
        vals=[];ws=[]
        for u in np.unique(s):
            k=s==u
            if k.sum()<20: continue
            vals.append(v[k].mean()); ws.append(k.sum())
        return _wt(vals,ws)

    out=dict(tag=tag,data=data,layer=L,E=E)
    # ---- sanity: forcing the CURRENT routing must be a no-op (validates the hook) ----
    msel=POS>=0
    ovs=np.full((B,T),-1,np.int64); ovs[msel]=cur[msel]
    out['self_exact']=bool(abs(acc(lg0,msel)-acc(fwd(ovs)[0],msel))<1e-9)

    # ---- (a) STAGE damage matrix: force ONLY stage-s tokens through e ----
    stages=[s for s in (1,2,3,4) if (STG==s).sum()>=200]
    Lm=np.full((E,len(stages)),np.nan)
    base_s={}
    for j,s in enumerate(stages):
        sel=(STG==s); base_s[s]=acc(lg0,sel)
        for e in range(E):
            ov=np.full((B,T),-1,np.int64); ov[sel]=e
            Lm[e,j]=base_s[s]-acc(fwd(ov)[0],sel)
    # routing-derived native expert per stage: argmax_e P(stage=s | e)
    native={}
    for j,s in enumerate(stages):
        sc=[]
        for e in range(E):
            me=(cur==e)&msel; tot=me.sum()
            sc.append(((STG==s)&me).sum()/tot if tot>=50 else -1.0)
        native[s]=int(np.argmax(sc))
    # A5.4 artifact control: fraction of stage-s tokens ALREADY routed to e.
    # Forcing stage-s onto the expert that already handles most of them perturbs fewer
    # tokens and so damages less for reasons unrelated to expert IDENTITY. In H2 this
    # fully explained the apparent 7/20 signal, so Leg 2 must be partialled on it.
    fr=np.zeros((E,len(stages)))
    for j,s_ in enumerate(stages):
        sel_=(STG==s_); ns=sel_.sum()
        for e in range(E): fr[e,j]=((cur==e)&sel_).sum()/max(ns,1)
    out['already_routed_frac']=fr.tolist()
    out['stages']=stages; out['base_stage']={int(k):v for k,v in base_s.items()}
    out['damage']=Lm.tolist(); out['native']={int(k):v for k,v in native.items()}
    out['native_distinct']=len(set(native.values()))
    out['argmin_damage']={int(s):int(np.nanargmin(Lm[:,j])) for j,s in enumerate(stages)}
    out['argmin_distinct']=len(set(out['argmin_damage'].values()))
    out['L_native']={int(s):float(Lm[native[s],j]) for j,s in enumerate(stages)}
    out['L_median']={int(s):float(np.nanmedian(Lm[:,j])) for j,s in enumerate(stages)}
    out['L_max']={int(s):float(np.nanmax(Lm[:,j])) for j,s in enumerate(stages)}

    # ---- (b) G1 readout, exactly H1's: force phase-2 tokens through the phase-1 expert ----
    sel2=(G1==2)&msel
    if sel2.sum()>=200:
        p1=[]
        for e in range(E):
            me=(cur==e)&msel; tot=me.sum()
            p1.append(((G1==1)&me).sum()/tot if tot>=50 else -1.0)
        e1=int(np.argmax(p1))
        b=g1mass(lg0,sel2); loss={}
        for e in range(E):
            ov=np.full((B,T),-1,np.int64); ov[sel2]=e
            loss[e]=b-g1mass(fwd(ov)[0],sel2)
        oth=[loss[e] for e in range(E) if e!=e1]
        out['g1']=dict(e1=e1,baseline=b,loss_e1=float(loss[e1]),
                       median_other=float(np.median(oth)),
                       all_losses={int(k):float(v) for k,v in loss.items()},
                       concentration=float(max(loss.values())/max(sum(max(v,0) for v in loss.values()),1e-9)))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--max_solves',type=int,default=200)
    ap.add_argument('--tags',default=''); ap.add_argument('--data',default='cfop')
    ap.add_argument('--out',default='h2_causal'); a=ap.parse_args(); dev=device_auto()
    tags=a.tags.split(',') if a.tags else [f'moe_cfop_s{s}' for s in range(40,45)]
    res=[]
    for t in tags:
        if not os.path.exists(f'{ROOT}/out/{t}.pt'): continue
        r=run(t,a.data,dev,a.max_solves); res.append(r)
        Lm=np.array(r['damage'])
        print(f"\n{t}  layer={r['layer']}  self_exact={r['self_exact']}")
        print(f"  stage accuracy damage L[e,s]  (rows=expert, cols=stage {r['stages']})")
        for e in range(r['E']):
            mark=' <-native' if e in r['native'].values() else ''
            print("   e%d  "%e+"".join(f"{Lm[e,j]:8.4f}" for j in range(len(r['stages'])))+mark)
        print(f"  native expert per stage = {r['native']}  (distinct={r['native_distinct']}/{len(r['stages'])})")
        print(f"  argmin-damage per stage = {r['argmin_damage']} (distinct={r['argmin_distinct']})")
        print(f"  L_native={ {k:round(v,4) for k,v in r['L_native'].items()} }")
        print(f"  L_median={ {k:round(v,4) for k,v in r['L_median'].items()} }")
        if 'g1' in r:
            g=r['g1']
            print(f"  [G1 readout, H1-style] e1={g['e1']} base={g['baseline']:.4f} "
                  f"L(e1)={g['loss_e1']:.4f} median_other={g['median_other']:.4f} conc={g['concentration']:.3f}")
    json.dump(res,open(f'{ROOT}/out/{a.out}.json','w'),indent=1)
    print(f"\nwrote out/h2_causal.json ({len(res)} models)")
