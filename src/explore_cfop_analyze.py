"""EXPLORATORY: fraction of the router's (position-conditional) entropy explained by each variable,
chance-corrected by permuting the router within strata. Strata = move number [, previous move]."""
import sys, glob, json, numpy as np
R='/Users/alityb/projects/cubemoe'
def codes(x): return np.unique(x,return_inverse=True)[1]
def cmi(E,V,S):
    E=codes(E);V=codes(V);S=codes(S); nE,nV,nS=E.max()+1,V.max()+1,S.max()+1
    J=np.bincount((S*nE+E)*nV+V,minlength=nS*nE*nV).reshape(nS,nE,nV).astype(float)
    n=J.sum((1,2),keepdims=True); P=J/np.maximum(n,1)
    pe=P.sum(2,keepdims=True); pv=P.sum(1,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'):
        t=np.where(P>0,P*np.log2(P/(pe*pv)),0)
    return float((t.sum((1,2))*n.ravel()).sum()/n.sum())
def cent(E,S):
    E=codes(E);S=codes(S);nE,nS=E.max()+1,S.max()+1
    J=np.bincount(S*nE+E,minlength=nS*nE).reshape(nS,nE).astype(float); n=J.sum(1,keepdims=True); P=J/np.maximum(n,1)
    with np.errstate(divide='ignore',invalid='ignore'): h=-np.where(P>0,P*np.log2(P),0).sum(1)
    return float((h*n.ravel()).sum()/n.sum())
def corrected(E,V,S,rng,nperm=5):
    raw=cmi(E,V,S); Sc=codes(S); null=[]
    for _ in range(nperm):
        Ep=E.copy()
        for s in np.unique(Sc):
            m=np.where(Sc==s)[0]; Ep[m]=Ep[rng.permutation(m)]
        null.append(cmi(Ep,V,S))
    return raw-np.mean(null)
def variables(f):
    b=np.minimum(f['segpos'],1)+ (f['segpos']==f['seglen']-1)*(f['seglen']>1)  # 0 first,1 middle,2 last
    return {'stage (4)':f['stage'],'segment: cross/slot1-4/OLL/PLL (7)':f['seg'],
            'first/middle/last move of algorithm':b,'position inside algorithm':np.minimum(f['segpos'],15),
            'which algorithm case':f['case'],'previous move':f['prev'],'target move':f['move'],
            'cross solved':f['cross'],'# F2L slots solved':f['slots'],'last layer oriented':f['llo'],'in G1':f['g1']}
if __name__=='__main__':
    rng=np.random.RandomState(0); out={}
    for data,pref in (('cfop','cfop'),('cfop_dec','cfopdec')):
        f=dict(np.load(f'{R}/out/explore/cfopfeat_{data}.npz')); V=variables(f)
        S1=f['pos']; S2=f['pos']*100+f['prev']
        for kind in ('moe','hash'):
            rows={}
            for p in sorted(glob.glob(f'{R}/out/explore/cfoproute_{kind}_{pref}_s*.npz')):
                r=dict(np.load(p)); E=r['e1'][-1]
                assert len(E)==len(f['pos']) and (r['pos']==f['pos']).all(), 'token order mismatch'
                H1=cent(E,S1); H2=cent(E,S2)
                for vn,v in V.items():
                    rows.setdefault(vn,[]).append((corrected(E,v,S1,rng)/H1, corrected(E,v,S2,rng)/H2 if vn!='previous move' else np.nan))
            out[(data,kind)]=rows
    for data in ('cfop','cfop_dec'):
        print(f"\n=== {data}  (last layer; % of router entropy explained, chance-corrected; mean over seeds)")
        print(f"{'variable':42s}{'MoE | move#':>13}{'MoE | move#,prev':>18}{'hash | move#':>14}")
        m=out[(data,'moe')]; h=out[(data,'hash')]
        for vn in m:
            a=np.array(m[vn]); b=np.array(h[vn]) if vn in h else np.full((1,2),np.nan)
            print(f"{vn:42s}{100*np.nanmean(a[:,0]):12.1f}%{100*np.nanmean(a[:,1]):17.1f}%{100*np.nanmean(b[:,0]):13.1f}%")
    json.dump({f'{k[0]}|{k[1]}':{vn:[list(map(float,x)) for x in v] for vn,v in rows.items()} for k,rows in out.items()},
              open(f'{R}/out/explore/cfop_whattracks.json','w'),indent=1)
