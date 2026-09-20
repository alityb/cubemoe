"""CONTROLS + descriptives on the scale checkpoints. Not gate inputs, not headline.
 - ceiling-normalised cNMI (per-dataset perfect-tracker anchor)
 - conditional purity per seed
 - within-model k-means (k=8, pre-router states, LAST layer) vs router
 - per-layer L4 ratio  (L(e1) / median over all other experts)
 - concentration index: share of total causal effect carried by the single most-affected expert
"""
import sys, os, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from analyze import metrics, purity
from extract import extract_seq, extract_state
from model import SeqModel, StateModel, device_auto
from sklearn.cluster import MiniBatchKMeans
import calibrate, cube
ROOT='/Users/alityb/projects/cubemoe'
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1=[i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES]

def cond_purity(expert, label, strat):
    vals=[];ws=[]
    for s in np.unique(strat):
        m=strat==s
        if m.sum()<20: continue
        vals.append(purity(expert[m],label[m])); ws.append(m.sum())
    w=np.array(ws,float); w/=w.sum(); return float((np.array(vals)*w).sum())

def per_layer_causal(tag, data, is_state, max_solves, dev):
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location=dev,weights_only=False); cfg=ck['cfg']
    if is_state:
        m=StateModel(cfg['D'],cfg['NL'],4,'moe',8,2).to(dev)
    else:
        rt='hash' if cfg.get('model')=='hash' else 'learned'
        m=SeqModel(cfg['D'],cfg['NL'],4,cfg['maxlen'],'moe',8,2,rt).to(dev)
    m.load_state_dict(ck['sd']); m.eval()
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    out={}
    if not is_state:
        from train import build_seq
        X,Y,P=build_seq(d,te,cfg['maxlen']); B,T=X.shape
        PH=np.zeros((B,T),np.int64); ST=np.full((B,T),-1,np.int64)
        for r,i in enumerate(te):
            nn=int(d['sol_len'][i]); sm=int(d['seam'][i])
            for t in range(nn): PH[r,53+t]=1 if t<sm else 2; ST[r,53+t]=t
        sel2=(PH==2); strat=ST[sel2]
        def fwd(ov,L):
            res=[]; rt=[]
            for b0 in range(0,B,64):
                x=torch.from_numpy(X[b0:b0+64]).to(dev); pm=torch.from_numpy(P[b0:b0+64]).to(dev); bb=x.shape[0]
                for li,blk in enumerate(m.blocks):
                    blk.ffn._override=None
                    if ov is not None and li==L: blk.ffn._override=torch.from_numpy(ov[b0:b0+64].reshape(-1)).to(dev)
                with torch.no_grad(): lg,info=m(x,pm,collect=(ov is None))
                res.append(lg.cpu())
                if ov is None: rt.append(info[L][1].reshape(bb,x.shape[1],2)[:,:,0].cpu())
            for blk in m.blocks: blk.ffn._override=None
            return torch.cat(res,0),(torch.cat(rt,0) if rt else None)
        def g1m(lg):
            p=F.softmax(lg.float(),-1).numpy(); v=p[sel2][:,G1].sum(-1)
            vals=[];ws=[]
            for s in np.unique(strat):
                k=strat==s
                if k.sum()<20: continue
                vals.append(v[k].mean()); ws.append(k.sum())
            w=np.array(ws,float); w/=w.sum(); return float((np.array(vals)*w).sum())
        for L in range(cfg['NL']):
            lg,route=fwd(None,L); base=g1m(lg); cur=route.numpy()
            ct=np.zeros((8,2)); msel=PH>0
            for e in range(8):
                me=(cur==e)&msel; ct[e,0]=((PH==1)&me).sum(); ct[e,1]=((PH==2)&me).sum()
            tot=ct.sum(1); ok=tot>=50
            p1=np.where(ok,ct[:,0]/np.maximum(tot,1),-1.0); e1=int(np.argmax(p1))
            loss={}
            for e in range(8):
                ov=np.full((B,T),-1,np.int64); ov[sel2]=e
                loss[e]=base-g1m(fwd(ov,L)[0])
            others=[loss[e] for e in range(8) if e!=e1]
            ls=np.array([max(loss[e],0.0) for e in range(8)])
            out[L]=dict(e1=e1, purity_e1=float(p1[e1]), base=base, loss_e1=float(loss[e1]),
                        median_other=float(np.median(others)),
                        ratio=float(loss[e1]/max(np.median(others),1e-9)),
                        concentration=float(ls.max()/max(ls.sum(),1e-9)))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='mixed250k')
    ap.add_argument('--seeds',default='20,21,22,23,24'); ap.add_argument('--max_solves',type=int,default=600)
    ap.add_argument('--out',default='out/controls_scale.json')
    a=ap.parse_args(); dev=device_auto(); SE=[int(x) for x in a.seeds.split(',')]
    cal=calibrate.run(a.data); anchor=cal['anchor_cond_nmi']
    print(f"perfect-tracker anchor (ceiling) on {a.data}: {anchor:.4f}", flush=True)
    res=dict(data=a.data, anchor=anchor, arms={})
    for arm,base,is_state,ms in [('A','moe_mixed_s',False,a.max_solves),('C','state_mixed_s',True,a.max_solves)]:
        rows=[]
        for s in SE:
            tag=f"{base}{s}"
            if not os.path.exists(f'{ROOT}/out/{tag}.pt'): print(f"  missing {tag}"); continue
            t0=time.time()
            ex=(extract_state if is_state else extract_seq)(tag,a.data,dev,ms)
            L=ex['NL']-1; strat = ex['rem'] if is_state else ex['pos']; ph=ex['phase']
            e1=ex['e1'][L]; mt=metrics(e1,ph,strat)
            km=MiniBatchKMeans(8,random_state=0,n_init=5,batch_size=4096).fit_predict(ex['rs'][L].astype(np.float32))
            kmt=metrics(km,ph,strat)
            r=dict(seed=s, router_cnmi=mt['nmi_strat_wt'], router_cnmi_pct_ceiling=mt['nmi_strat_wt']/anchor,
                   cond_purity=cond_purity(e1,ph,strat), kmeans_cnmi=kmt['nmi_strat_wt'],
                   kmeans_pct_ceiling=kmt['nmi_strat_wt']/anchor,
                   cond_purity_kmeans=cond_purity(km,ph,strat))
            if not is_state:
                r['per_layer']=per_layer_causal(tag,a.data,False,ms,dev)
            rows.append(r)
            print(f"  {tag}: cNMI={r['router_cnmi']:.4f} ({r['router_cnmi_pct_ceiling']*100:.1f}% ceiling) "
                  f"condPurity={r['cond_purity']:.4f} kmeans={r['kmeans_cnmi']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        res['arms'][arm]=rows
    json.dump(res,open(f'{ROOT}/{a.out}','w'),indent=1); print("wrote",a.out)
