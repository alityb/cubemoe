"""H_slack analysis: V1 (top-1/8/local, seeds 30-34) vs baseline (top-2/8/large, seeds 20-24).
Index-matched pairing (30<->20, ...), as declared for the CORE plan.
P1 concentration index (PRIMARY) | P2 ceiling-normalised cond NMI | P3 absolute L(e1).
Architecture is read from the checkpoint, NOT assumed."""
import sys, os, json
import numpy as np, torch
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from scipy import stats
from analyze import metrics
from extract import extract_seq
from model import device_auto
import calibrate
ROOT='/Users/alityb/projects/cubemoe'

def arch_of(tag):
    """n_exp from the weights; topk/lbl_scope from cfg or the training record."""
    ck=torch.load(f'{ROOT}/out/{tag}.pt',map_location='cpu',weights_only=False)
    cfg=ck['cfg']; sd=ck['sd']
    E=int(sd['blocks.0.ffn.w1'].shape[0]) if 'blocks.0.ffn.w1' in sd else cfg.get('experts',8)
    k=cfg.get('topk')
    if k is None:
        j=f'{ROOT}/out/{tag}.json'
        k=json.load(open(j)).get('topk',2) if os.path.exists(j) else 2
    return E,int(k),cfg

def analyse(tag, data, max_solves, dev):
    E,k,cfg=arch_of(tag)
    import extract as _ex
    orig=_ex.SeqModel
    class Patched(orig):                      # force the checkpoint's true architecture
        def __init__(self,D,NL,nh,maxlen,kind,n_exp=8,kk=2,router='learned'):
            super().__init__(D,NL,nh,maxlen,kind,E,k,router)
    _ex.SeqModel=Patched
    try:
        ex=extract_seq(tag,data,dev,max_solves)
    finally:
        _ex.SeqModel=orig
    L=ex['NL']-1; strat=ex['pos']; ph=ex['phase']
    mt=metrics(ex['e1'][L],ph,strat)
    from controls_scale import per_layer_causal
    import controls_scale as _cs
    _cs_orig=_cs.SeqModel; _cs.SeqModel=Patched
    try: pl=per_layer_causal(tag,data,False,max_solves,dev)
    finally: _cs.SeqModel=_cs_orig
    last=pl[max(pl.keys())]
    return dict(tag=tag,experts=E,topk=k,cond_nmi=mt['nmi_strat_wt'],
                concentration=last['concentration'],loss_e1=last['loss_e1'],
                median_other=last['median_other'],purity_e1=last['purity_e1'])

if __name__=='__main__':
    dev=device_auto(); data='mixed250k'; ms=600
    anchor=calibrate.run(data)['anchor_cond_nmi']
    print(f"ceiling (perfect-tracker reference) = {anchor:.4f}\n")
    import os as _o
    ARM=_o.environ.get('SWEEP_ARM','V1_top1_e8_local')
    V1=[f'{ARM}_s{s}' for s in range(30,35)]
    BL=[f'moe_mixed_s{s}' for s in range(20,25)]
    out={'anchor':anchor,'arm':ARM,'V1':[],'baseline':[]}
    for arm,tags in (('V1',V1),('baseline',BL)):
        for t in tags:
            if not os.path.exists(f'{ROOT}/out/{t}.pt'): print(f"  missing {t}"); continue
            r=analyse(t,data,ms,dev); r['cnmi_pct_ceiling']=r['cond_nmi']/anchor
            out[arm].append(r)
            print(f"  {t}: E={r['experts']} k={r['topk']} conc={r['concentration']:.3f} "
                  f"cNMI={r['cond_nmi']:.4f} ({r['cnmi_pct_ceiling']*100:.0f}%) L(e1)={r['loss_e1']:.4f}", flush=True)
    a=out['V1']; b=out['baseline']
    if len(a)==len(b)==5:
        print("\n=== H_slack, index-matched pairs (30<->20, ...) ===")
        for key,lab,pred in [('concentration','P1 concentration index (PRIMARY)','top-1 LOWER'),
                             ('cnmi_pct_ceiling','P2 ceiling-normalised cond NMI','top-1 HIGHER'),
                             ('loss_e1','P3 absolute L(e1)','top-1 SMALLER')]:
            x=np.array([r[key] for r in a]); y=np.array([r[key] for r in b]); d=x-y
            wins=int((d<0).sum()) if 'LOWER' in pred or 'SMALLER' in pred else int((d>0).sum())
            t,p=stats.ttest_rel(x,y)
            print(f"\n{lab}  [predicted: {pred}]")
            print(f"  top-1 {x.mean():.4f}+/-{x.std(ddof=1):.4f}  vs  top-2 {y.mean():.4f}+/-{y.std(ddof=1):.4f}")
            print(f"  per-pair diff {[round(v,4) for v in d]}")
            print(f"  predicted direction in {wins}/5 pairs ; paired t p={p:.4f}")
            if key=='concentration':
                print(f"  PASS RULE: >=4/5 AND paired t p<0.05 -> {'SUPPORTED' if (wins>=4 and p<0.05) else ('REJECTED' if (5-wins)>=4 else 'INCONCLUSIVE')}")
    json.dump(out,open(f'{ROOT}/out/sweep_{ARM}.json','w'),indent=1)
    print("\nwrote out/sweep_h_slack.json")
