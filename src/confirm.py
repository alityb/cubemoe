"""CONFIRMATORY analysis — implements PREREGISTRATION.md exactly.
LAST LAYER ONLY. No selection anywhere. Fresh seeds only."""
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from analyze import metrics, cp_test, null_shuffled, null_position_router, perm_p_condmi, probe_phase
from extract import extract_seq, extract_state
from model import SeqModel, StateModel, device_auto
import cube
ROOT='/Users/alityb/projects/cubemoe'
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1=[i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES]

def _wt(vals, ws):
    ws=np.asarray(ws,float); ws=ws/ws.sum(); return float((np.asarray(vals)*ws).sum())

# ---------- Leg 4: causal, all-expert swap, median-over-others, stratified ----------
def causal_seq(tag, data, dev, max_solves):
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
    m=SeqModel(cfg['D'],cfg['NL'],4,cfg['maxlen'],'moe',8,2,'learned').to(dev)
    m.load_state_dict(ck['sd']); m.eval(); L=cfg['NL']-1
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    from train import build_seq
    X,Y,P=build_seq(d,te,cfg['maxlen']); B,T=X.shape
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    PH=np.zeros((B,T),np.int64); ST=np.full((B,T),-1,np.int64)
    for r,i in enumerate(te):
        nn=int(d['sol_len'][i]); sm=int(d['seam'][i])
        for t in range(nn):
            PH[r,53+t]= 1 if t<sm else 2; ST[r,53+t]=t
    def fwd(ov):
        out=[]
        for b0 in range(0,B,64):
            x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev); bb=x.shape[0]
            for li,blk in enumerate(m.blocks):
                blk.ffn._override=None
                if ov is not None and li==L:
                    blk.ffn._override=torch.from_numpy(ov[b0:b0+64].reshape(-1)).to(dev)
            with torch.no_grad(): lg,info=m(x,pm,collect=(ov is None))
            out.append((lg.cpu(), info[L][1].reshape(bb,x.shape[1],-1)[:,:,0].cpu() if ov is None else None))
        for blk in m.blocks: blk.ffn._override=None
        return torch.cat([o[0] for o in out],0), (torch.cat([o[1] for o in out],0) if ov is None else None)
    lg,route=fwd(None)
    sel2=(PH==2)                                   # phase-2 prediction positions
    strat=ST[sel2]
    def g1mass(lg):
        p=F.softmax(lg.float(),-1).numpy()
        v=p[sel2][:,G1].sum(-1)
        us=np.unique(strat); vals=[];ws=[]
        for s in us:
            msk=strat==s
            if msk.sum()<20: continue
            vals.append(v[msk].mean()); ws.append(msk.sum())
        return _wt(vals,ws)
    base=g1mass(lg)
    cur=route.numpy()
    ct=np.zeros((8,3))
    msel=PH>0
    for e in range(8):
        me=(cur==e)&msel
        ct[e,1]=((PH==1)&me).sum(); ct[e,2]=((PH==2)&me).sum()
    tot=ct[:,1]+ct[:,2]; ok=tot>=50
    p1=np.where(ok,ct[:,1]/np.maximum(tot,1),-1.0)
    e1=int(np.argmax(p1))
    losses={}
    for e in range(8):
        ov=np.full((B,T),-1,np.int64); ov[sel2]=e
        losses[e]=base-g1mass(fwd(ov)[0])
    ovs=np.full((B,T),-1,np.int64); ovs[sel2]=cur[sel2]
    self_ok = abs(base-g1mass(fwd(ovs)[0])) < 1e-9
    others=[losses[e] for e in range(8) if e!=e1]
    return dict(tag=tag,layer=L,e1=e1,p_phase1_given_e1=float(p1[e1]),baseline=base,
                loss_e1=float(losses[e1]),median_other=float(np.median(others)),
                all_losses={int(k):float(v) for k,v in losses.items()},
                effect=float(losses[e1]/max(np.median(others),1e-9)), self_exact=bool(self_ok),
                n_phase2_rows=int(sel2.sum()))

def causal_state(tag, data, dev, max_solves):
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
    m=StateModel(cfg['D'],cfg['NL'],4,'moe',8,2).to(dev); m.load_state_dict(ck['sd']); m.eval(); L=cfg['NL']-1
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    rows=np.concatenate([np.arange(off[i],off[i+1]) for i in te])
    st=d['states'][rows].astype(np.int64); ph=d['phase'][rows].astype(np.int64)
    rem=d['remaining'][rows].astype(np.int64); N=len(st)
    def fwd(ov):
        Ls=[];R=[]
        for b0 in range(0,N,512):
            x=torch.from_numpy(st[b0:b0+512]).to(dev); bb=x.shape[0]
            for li,blk in enumerate(m.blocks):
                blk.ffn._override=None
                if ov is not None and li==L:
                    o=np.full((bb,55),-1,np.int64); o[:,0]=ov[b0:b0+512]
                    blk.ffn._override=torch.from_numpy(o.reshape(-1)).to(dev)
            with torch.no_grad(): lg,info=m(x,collect=(ov is None))
            Ls.append(lg.cpu())
            if ov is None: R.append(info[L][1].reshape(bb,55,-1)[:,0,0].cpu())
        for blk in m.blocks: blk.ffn._override=None
        return torch.cat(Ls,0), (torch.cat(R,0) if R else None)
    lg,route=fwd(None); sel2=(ph==2); strat=rem[sel2]
    def g1mass(lg):
        p=F.softmax(lg.float(),-1).numpy(); v=p[sel2][:,G1].sum(-1)
        vals=[];ws=[]
        for s in np.unique(strat):
            msk=strat==s
            if msk.sum()<20: continue
            vals.append(v[msk].mean()); ws.append(msk.sum())
        return _wt(vals,ws)
    base=g1mass(lg); cur=route.numpy()
    ct=np.zeros((8,3))
    for e in range(8):
        me=cur==e; ct[e,1]=((ph==1)&me).sum(); ct[e,2]=((ph==2)&me).sum()
    tot=ct[:,1]+ct[:,2]; ok=tot>=50
    p1=np.where(ok,ct[:,1]/np.maximum(tot,1),-1.0); e1=int(np.argmax(p1))
    losses={}
    for e in range(8):
        ov=np.where(sel2,e,cur).astype(np.int64)
        losses[e]=base-g1mass(fwd(ov)[0])
    self_ok = abs(base-g1mass(fwd(cur.astype(np.int64))[0])) < 1e-9
    others=[losses[e] for e in range(8) if e!=e1]
    return dict(tag=tag,layer=L,e1=e1,p_phase1_given_e1=float(p1[e1]),baseline=base,
                loss_e1=float(losses[e1]),median_other=float(np.median(others)),
                all_losses={int(k):float(v) for k,v in losses.items()},
                effect=float(losses[e1]/max(np.median(others),1e-9)),self_exact=bool(self_ok),
                n_phase2_rows=int(sel2.sum()))

def run(tag, data, is_state, max_solves):
    dev=device_auto()
    ex=(extract_state if is_state else extract_seq)(tag,data,dev,max_solves)
    L=ex['NL']-1
    strat = ex['rem'] if is_state else ex['pos']
    ph=ex['phase']; e1=ex['e1'][L]
    mt=metrics(e1,ph,strat)
    out=dict(tag=tag,arm=('C' if is_state else 'A'),layer=L,
             cond_nmi=mt['nmi_strat_wt'], shuffled=metrics(null_shuffled(e1),ph,strat)['nmi_strat_wt'],
             position_only=metrics(null_position_router(strat),ph,strat)['nmi_strat_wt'],
             perm_p=perm_p_condmi(e1,ph,strat,mt['cond_mi_bits'],nperm=200),
             utilization=mt['utilization'])
    cp=cp_test(e1,ex['solve'],ex['seam'],ex['sollen'],nperm=200)
    out['b1']=cp.get('partial_slope'); out['b1_p']=cp.get('partial_p'); out['cp_n']=cp.get('n')
    out['probe']=probe_phase(ex['rs'][L],ph,strat)['probe_acc_strat_wt']
    tr=json.load(open(f'{ROOT}/out/{tag}.json')); out['acc']=tr['acc']
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    nn=len(d['seam']); te=np.random.RandomState(0).permutation(nn)[int(nn*0.9):]
    msk=np.isin(d['solve_id'],te); mv=d['moves'][msk]
    key=(d['phase'][msk].astype(int)*1000 + (d['remaining'][msk] if is_state else d['pos'][msk]).astype(int))
    out['bar']=float(sum(np.bincount(mv[key==v],minlength=18).max() for v in np.unique(key))/len(mv))
    out['acc_ratio']=out['acc']/out['bar']
    # A1.1: hash-twin cond NMI, same seed / layer / held-out size (arm A only)
    out['hash_twin']=None
    if not is_state:
        import re as _re
        mh=_re.search(r'_s(\d+)$',tag); sfx=f"_s{mh.group(1)}" if mh else ""
        htag=f"hash_mixed{sfx}"
        if os.path.exists(f'{ROOT}/out/{htag}.pt'):
            hx=extract_seq(htag,data,dev,max_solves)
            hL=hx['NL']-1
            out['hash_twin']=metrics(hx['e1'][hL],hx['phase'],hx['pos'])['nmi_strat_wt']
            out['hash_twin_tag']=htag
    out['causal']=(causal_state if is_state else causal_seq)(tag,data,dev,max_solves)
    os.makedirs(f'{ROOT}/out/confirm',exist_ok=True)
    json.dump(out,open(f'{ROOT}/out/confirm/{tag}.json','w'),indent=1)
    c=out['causal']
    print(f"  [{tag}] acc={out['acc']:.4f} (bar {out['bar']:.4f}, {out['acc_ratio']:.2f}x) probe={out['probe']:.3f} | "
          f"cNMI={out['cond_nmi']:.4f} shuf={out['shuffled']:.4f} p={out['perm_p']:.4f} | "
          f"hash={out['hash_twin'] if out['hash_twin'] is None else round(out['hash_twin'],4)} | "
          f"b1={out['b1']:+.3f} p={out['b1_p']:.3f} | causal L(e1)={c['loss_e1']:.4f} "
          f"med_other={c['median_other']:.4f} eff={c['effect']:.2f}x self_ok={c['self_exact']}", flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed')
    ap.add_argument('--state',action='store_true'); ap.add_argument('--max_solves',type=int,default=1200)
    a=ap.parse_args(); run(a.tag,a.data,a.state,a.max_solves)
