"""H2 primary endpoint: the 2x2 cross-evaluation.
Each model is scored against BOTH decompositions:
  - CFOP stage (1..4), the construction labels in its own data
  - G1 phase (1/2), recomputed as a STATE predicate on the same states
H2 predicts CFOP-trained routers align with CFOP stage > G1 phase.
Also reports PER-STAGE routing, to separate genuine stage-tracking from
algorithm-identity tracking (cross/F2L use no canonical algorithms)."""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from analyze import metrics, null_shuffled, null_position_router, perm_p_condmi
from extract import extract_seq
from model import device_auto
import cube
from qa import facelets_to_cubie, COLORS
from scipy import stats
ROOT='/Users/alityb/projects/cubemoe'

import cfop as _cfop
def both_labels(data, max_solves):
    """Recompute BOTH labelings on the SAME rows, in exactly extract_seq's row order:
    for i in te (order preserved), for t in range(sol_len[i]).
    Returns (g1, cfop_stage) as a state predicate each — no reuse of stored phase."""
    d=dict(np.load(f'{ROOT}/data/{data}.npz'))
    n=len(d['sol_len'])
    te=np.random.RandomState(0).permutation(n)[int(n*0.9):][:max_solves]   # matches extract._split
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    g1=[]; stg=[]
    for i in te:
        for t in range(off[i],off[i+1]):
            st=facelets_to_cubie(''.join(COLORS[c] for c in d['states'][t]))
            g1.append(2 if cube.in_G1(st) else 1)
            stg.append(_cfop.stage_of(st))
    return np.array(g1), np.array(stg)

def run(tag, data, max_solves, dev):
    ex=extract_seq(tag,data,dev,max_solves)
    L=ex['NL']-1; strat=ex['pos']; e1=ex['e1'][L]
    g1, stage_pred = both_labels(data, max_solves)
    assert len(g1)==len(ex['phase']), f"ROW MISALIGNMENT: {len(g1)} labels vs {len(ex['phase'])} rows"
    # CFOP-trained models: stage = the construction label in its own data.
    # Kociemba-trained models: stage = the CFOP stage PREDICATE on the same states.
    stage = ex['phase'] if data=='cfop' else stage_pred
    n=len(g1); out=dict(tag=tag,data=data,n=int(n),
                        stage_src=('construction' if data=='cfop' else 'predicate'))
    for nm,lab in (('stage',stage),('g1',g1)):
        m=metrics(e1,lab,strat)
        out[f'cnmi_{nm}']=m['nmi_strat_wt']
        out[f'shuf_{nm}']=metrics(null_shuffled(e1),lab,strat)['nmi_strat_wt']
        # CEILING: an oracle router that IS the label. cNMI is dataset-specific
        # (methodology control #3) -- an absolute number is uninterpretable without it.
        out[f'ceil_{nm}']=metrics(lab,lab,strat)['nmi_strat_wt']
        c=out[f'ceil_{nm}']
        out[f'frac_{nm}']=(out[f'cnmi_{nm}']/c) if c>1e-9 else float('nan')
        out[f'shuffrac_{nm}']=(out[f'shuf_{nm}']/c) if c>1e-9 else float('nan')
        out[f'posr_{nm}']=metrics(null_position_router(strat),lab,strat)['nmi_strat_wt']
        out[f'permp_{nm}']=perm_p_condmi(e1,lab,strat,m['cond_mi_bits'],nperm=200)
    # per-stage one-vs-rest, to separate stage-tracking from algorithm-identity tracking
    out['per_stage']={}
    for s in (1,2,3,4):
        if (stage==s).sum()<200:
            out['per_stage'][s]=None; continue
        ov=np.where(stage==s,1,0)
        out['per_stage'][s]=metrics(e1,ov,strat)['nmi_strat_wt']
    out['g1_balance']=float(np.mean(g1==2))
    print(f"  {tag}: stage {out['cnmi_stage']:.4f}/ceil {out['ceil_stage']:.4f}="
          f"{100*out['frac_stage']:.0f}%(null {100*out['shuffrac_stage']:.0f}%) | "
          f"G1 {out['cnmi_g1']:.4f}/ceil {out['ceil_g1']:.4f}={100*out['frac_g1']:.0f}%"
          f"(null {100*out['shuffrac_g1']:.0f}%)")
    print(f"     raw: cNMI(stage)={out['cnmi_stage']:.4f} cNMI(G1)={out['cnmi_g1']:.4f} "
          f"| shuf {out['shuf_stage']:.4f}/{out['shuf_g1']:.4f} "
          f"| per-stage {[(round(v,3) if (v:=out['per_stage'].get(s)) is not None else None) for s in (1,2,3,4)]}", flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--max_solves',type=int,default=600)
    a=ap.parse_args(); dev=device_auto(); res={'cfop':[], 'kociemba':[]}
    for s in range(40,45):
        t=f'moe_cfop_s{s}'
        if os.path.exists(f'{ROOT}/out/{t}.pt'): res['cfop'].append(run(t,'cfop',a.max_solves,dev))
    for s in range(20,25):
        t=f'moe_mixed_s{s}'
        if os.path.exists(f'{ROOT}/out/{t}.pt'): res['kociemba'].append(run(t,'mixed250k',a.max_solves,dev))
    json.dump(res,open(f'{ROOT}/out/h2_2x2.json','w'),indent=1)
    print("\n=== H2 PRIMARY ENDPOINT: 2x2 cross-evaluation ===")
    print(f"{'trained on':<12}{'vs CFOP stage':>15}{'vs G1 phase':>14}{'stage>G1?':>11}")
    for k,rows in res.items():
        if not rows: continue
        st=np.array([r['cnmi_stage'] for r in rows]); g=np.array([r['cnmi_g1'] for r in rows])
        print(f"{k:<12}{st.mean():>15.4f}{g.mean():>14.4f}{int((st>g).sum()):>8}/{len(st)}")
        if k=='cfop' and len(st)==5:
            t_,p=stats.ttest_rel(st,g)
            supported = int((st>g).sum())>=4 and p<0.05
            print(f"\n  H2 pass rule: stage > G1 in >=4/5 AND paired t p<0.05")
            print(f"    {int((st>g).sum())}/5 seeds, paired t p={p:.4f} -> "
                  f"**{'SUPPORTED' if supported else ('REJECTED' if int((st<g).sum())>=4 else 'INCONCLUSIVE')}**")
