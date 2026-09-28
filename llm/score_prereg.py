"""Applies llm/PREREG_LLM.md rules mechanically to the OLMoE run."""
import sys, os, json, numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from explore_cfop_analyze import corrected, cent
R='/Users/alityb/projects/cubemoe/llm'; short='OLMoE-1B-7B-0924'; KL=(3,7,11,15)
B=dict(np.load(f'{R}/base_{short}.npz')); pf=np.exp(B['lp']); pl=np.exp(B['lloc']); ok=pf>0.5; dom=B['dom']
G={'needs-context':ok&(pl<0.1),'local':ok&(pl>0.5)}
out={}
# ---- A
rng=np.random.RandomState(0); nmin=min(g.sum() for g in G.values()); route=B['route'][...,0]; tok=B['tok']
A=[]
for L in range(route.shape[0]):
    v={}
    for gn,g in G.items():
        idx=np.flatnonzero(g.ravel()); vals=[]
        for d in range(3):
            s=rng.choice(idx,nmin,replace=False); E=route[L].ravel()[s]; S0=np.zeros(nmin,int); H=cent(E,S0)
            vals.append(100*corrected(E,tok.ravel()[s],S0,rng,nperm=3)/H if H>0 else np.nan)
        v[gn]=float(np.nanmean(vals))
    A.append(v)
late=[A[L] for L in range(8,16)]; nA=sum(1 for v in late if v['needs-context']<v['local'])
out['A']=dict(per_layer=A,late_layers_pass=nA,verdict='PASS' if nA>=6 else 'FAIL')
print("A (routing follows current token less on needs-context tokens), share of top-1 routing explained by current token:")
for L,v in enumerate(A): print(f"   layer {L:2d}: needs-context {v['needs-context']:5.1f}%   local {v['local']:5.1f}%{'   <' if v['needs-context']<v['local'] else ''}")
print(f"   layers 8-15 with needs-context < local: {nA}/8  -> A {out['A']['verdict']} (rule: >= 6/8)\n")
# ---- B and C
if all(os.path.exists(f'{R}/ko_{short}_L{L}.npz') for L in KL):
    Brows=[];Crows=[]
    for L in KL:
        K=dict(np.load(f'{R}/ko_{short}_L{L}.npz')); dko=K['dko']; dno=K['dno']
        eff={gn:float(dko[:,g].sum()/dno[:,g].sum()) for gn,g in G.items()}
        absd={gn:float(dko[:,g].mean()) for gn,g in G.items()}
        by={d:float(dko[:,G['needs-context']&(dom[:,None]==d)].sum()/dno[:,G['needs-context']&(dom[:,None]==d)].sum()) for d in ('prose','code','license')}
        Brows.append((L,eff,absd)); Crows.append((L,by))
    nB=sum(1 for L,e,_ in Brows if e['needs-context']>e['local']); nC=sum(1 for L,b in Crows if b['license']>b['prose'] and b['license']>b['code'])
    out['B']=dict(rows=Brows,pass_layers=nB,verdict='PASS' if nB>=3 else 'FAIL')
    out['C']=dict(rows=Crows,pass_layers=nC,verdict='PASS' if nC>=3 else 'FAIL')
    print("B (expert effect = KO damage / size-matched noise damage):")
    for L,e,ab in Brows: print(f"   layer {L:2d}: needs-context {e['needs-context']:.2f}x   local {e['local']:.2f}x   (abs KO damage {ab['needs-context']:.4f} / {ab['local']:.4f} nats)")
    print(f"   -> B {out['B']['verdict']} ({nB}/4 layers; rule >= 3/4)\n")
    print("C (expert effect on needs-context tokens by domain):")
    for L,b in Crows: print(f"   layer {L:2d}: license {b['license']:.2f}x   prose {b['prose']:.2f}x   code {b['code']:.2f}x")
    print(f"   -> C {out['C']['verdict']} ({nC}/4 layers; rule >= 3/4)")
else:
    print("B, C: knockout files not all present yet")
json.dump(out,open(f'{R}/prereg_verdict_{short}.json','w'),indent=1,default=float)
