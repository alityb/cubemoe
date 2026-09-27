"""H2c (AMENDMENT 6, `1dc2f1cf`): stage-specific causal readout.

A SEPARATE test, not a re-analysis of H2b. H2b's REJECTED verdict stands regardless of
what this finds. If H2c fires, the claim is "H2b's readout lacked power" -- never
"H_pos was supported" (A6.1).

Instrument: R_s = predicted probability mass on M_s, the moves characteristic of stage s,
with M_s derived from the TRAINING split only and frozen before any intervention (A6.3).
Damage L_s(e) = R_s(base) - R_s(forced through e), position-stratified, forcing ONLY stage-s tokens.
"""
import sys, os, json, hashlib, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from model import SeqModel, device_auto
from train import build_seq
import cube
ROOT='/Users/alityb/projects/cubemoe'
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1SET=sorted(i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES)

def _wt(vals, ws):
    ws=np.asarray(ws,float)
    if ws.sum()==0: return float('nan')
    return float((np.asarray(vals)*(ws/ws.sum())).sum())

def labels_for(d, data):
    """per-token label: CFOP stage from construction, or Kociemba G1 phase from seam."""
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    if 'cfop' in data: return d['phase'].astype(int), off
    lab=np.zeros(len(d['moves']),int)
    for i in range(len(d['sol_len'])):
        nn=int(d['sol_len'][i]); sm=int(d['seam'][i])
        lab[off[i]:off[i]+nn]=np.where(np.arange(nn)<sm,1,2)
    return lab, off

def derive_msets(data):
    """M_s from the TRAINING split only (A6.3). Frozen + hashed before any intervention."""
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['sol_len']); perm=np.random.RandomState(0).permutation(n)
    tr=perm[:int(n*0.9)]                       # train.py's split; te is the tail
    lab,off=labels_for(d,data); mv=d['moves'].astype(int)
    tok=np.zeros(len(mv),bool)
    for i in tr: tok[off[i]:off[i+1]]=True
    M={}; detail={}
    for s in sorted(set(lab[tok].tolist())):
        a=(lab==s)&tok; b=(lab!=s)&tok
        fa=np.bincount(mv[a],minlength=18)/max(a.sum(),1)
        fb=np.bincount(mv[b],minlength=18)/max(b.sum(),1)
        ms=[int(m) for m in range(18) if fa[m]>fb[m]]
        M[int(s)]=ms
        detail[int(s)]=dict(n_moves=len(ms),moves=[MOVE_NAMES[m] for m in ms],
                            mass_in_stage=float(fa[ms].sum()),mass_elsewhere=float(fb[ms].sum()))
    return M, detail

def run(tag, data, M, dev, max_solves, E=8):
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
    m=SeqModel(cfg['D'],cfg['NL'],4,cfg['maxlen'],'moe',E,2,'learned').to(dev)
    m.load_state_dict(ck['sd']); m.eval(); L=cfg['NL']-1
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['sol_len']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    X,Y,P=build_seq(d,te,cfg['maxlen']); B,T=X.shape
    lab,off=labels_for(d,data)
    LB=np.zeros((B,T),np.int64); POS=np.full((B,T),-1,np.int64)
    for r,i in enumerate(te):
        for t in range(int(d['sol_len'][i])):
            LB[r,53+t]=lab[off[i]+t]; POS[r,53+t]=t      # 53+t: the position that PREDICTS move t
    def fwd(ov):
        lgs=[];rts=[]
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
    def mass(lg, sel, ms):
        p=F.softmax(lg.float(),-1).numpy(); v=p[sel][:,ms].sum(-1); s=POS[sel]
        vals=[];ws=[]
        for u in np.unique(s):
            k=s==u
            if k.sum()<20: continue
            vals.append(v[k].mean()); ws.append(k.sum())
        return _wt(vals,ws)
    msel=POS>=0
    stages=[s for s in sorted(M) if (LB==s).sum()>=200]
    out=dict(tag=tag,data=data,layer=L,E=E,stages=stages,readout='stage_characteristic_move_mass')
    ovs=np.full((B,T),-1,np.int64); ovs[msel]=cur[msel]
    out['self_exact']=bool(abs(mass(lg0,msel,list(range(18)))-mass(fwd(ovs)[0],msel,list(range(18))))<1e-9)
    Lm=np.full((E,len(stages)),np.nan); fr=np.zeros((E,len(stages))); base={}
    for j,s in enumerate(stages):
        sel=(LB==s); ms=M[s]; base[s]=mass(lg0,sel,ms)
        for e in range(E):
            ov=np.full((B,T),-1,np.int64); ov[sel]=e
            Lm[e,j]=base[s]-mass(fwd(ov)[0],sel,ms)
            fr[e,j]=((cur==e)&sel).sum()/max(sel.sum(),1)
    native={}
    for j,s in enumerate(stages):
        sc=[]
        for e in range(E):
            me=(cur==e)&msel; tot=me.sum()
            sc.append(((LB==s)&me).sum()/tot if tot>=50 else -1.0)
        native[s]=int(np.argmax(sc))
    out['base']={int(k):v for k,v in base.items()}; out['damage']=Lm.tolist()
    out['already_routed_frac']=fr.tolist(); out['native']={int(k):v for k,v in native.items()}
    out['L_native']={int(s):float(Lm[native[s],j]) for j,s in enumerate(stages)}
    out['L_median']={int(s):float(np.nanmedian(Lm[:,j])) for j,s in enumerate(stages)}
    out['L_max']={int(s):float(np.nanmax(Lm[:,j])) for j,s in enumerate(stages)}
    # instrument resolution (A6.4): background vs between-expert spread
    out['median_other_mean']=float(np.nanmean([np.nanmedian(Lm[:,j]) for j in range(len(stages))]))
    out['spread_mean']=float(np.nanmean([np.nanstd(Lm[:,j]) for j in range(len(stages))]))
    # H1-style cross-label statistic: force label-2 tokens through the label-1 expert
    if 2 in stages and 1 in stages:
        p1=[]
        for e in range(E):
            me=(cur==e)&msel; tot=me.sum()
            p1.append(((LB==1)&me).sum()/tot if tot>=50 else -1.0)
        e1=int(np.argmax(p1)); j2=stages.index(2)
        oth=[Lm[e,j2] for e in range(E) if e!=e1]
        out['h1_style']=dict(e1=e1,loss_e1=float(Lm[e1,j2]),median_other=float(np.median(oth)),
                             ratio=float(Lm[e1,j2]/max(np.median(oth),1e-9)))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['poscontrol','h2c'],required=True)
    ap.add_argument('--max_solves',type=int,default=200); a=ap.parse_args(); dev=device_auto()
    if a.mode=='poscontrol':
        data='mixed250k'; tags=[f'moe_mixed_s{s}' for s in range(20,25)]; outf='h2c_poscontrol'
    else:
        data='cfop_dec'; tags=[f'moe_cfopdec_s{s}' for s in range(50,55)]; outf='h2c'
    M,detail=derive_msets(data)
    mp=f'{ROOT}/out/h2c_msets_{data}.json'
    json.dump(dict(data=data,msets=M,detail=detail),open(mp,'w'),indent=1)
    h=hashlib.sha256(open(mp,'rb').read()).hexdigest()
    print(f"=== M_s frozen from TRAINING split of {data}  (sha256 {h[:16]})")
    for s in sorted(M):
        dd=detail[s]
        print(f"  stage {s}: |M_s|={dd['n_moves']:2d}/18  mass in stage={dd['mass_in_stage']:.3f}"
              f"  elsewhere={dd['mass_elsewhere']:.3f}  {dd['moves']}")
    if data=='mixed250k':
        ok = M.get(2)==G1SET
        print(f"  [derivation check] M_2 == the 10 G1 moves? {ok}  (expected True: phase 2 uses only G1)")
    res=[]
    for t in tags:
        if not os.path.exists(f'{ROOT}/out/{t}.pt'): continue
        r=run(t,data,M,dev,a.max_solves); res.append(r)
        Lm=np.array(r['damage'])
        print(f"\n{t}  layer={r['layer']} self_exact={r['self_exact']}  stages={r['stages']}")
        for e in range(r['E']):
            mk=' <-native' if e in r['native'].values() else ''
            print("   e%d  "%e+"".join(f"{Lm[e,j]:9.4f}" for j in range(len(r['stages'])))+mk)
        print(f"   native={r['native']}  L_native={ {k:round(v,4) for k,v in r['L_native'].items()} }")
        print(f"   L_median={ {k:round(v,4) for k,v in r['L_median'].items()} }")
        print(f"   resolution: median_other={r['median_other_mean']:.4f} spread={r['spread_mean']:.4f}"
              f"  ratio spread/background={r['spread_mean']/max(r['median_other_mean'],1e-9):.3f}")
        if 'h1_style' in r:
            g=r['h1_style']
            print(f"   [H1-style] e1={g['e1']} L(e1)={g['loss_e1']:.4f} median_other={g['median_other']:.4f} ratio={g['ratio']:.1f}x")
    json.dump(res,open(f'{ROOT}/out/{outf}.json','w'),indent=1)
    print(f"\nwrote out/{outf}.json ({len(res)} models)")
