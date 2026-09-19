"""Paired analysis of the decisive MoE-vs-dense-twin comparison across seeds."""
import json, os, glob
import numpy as np
ROOT='/Users/alityb/projects/cubemoe'
def load(tag):
    p=f'{ROOT}/out/seeds/{tag}_eval.json'
    return json.load(open(p)) if os.path.exists(p) else None
def paired(a_base,b_base,seeds,label,metric='best_cond_nmi'):
    A=[];B=[];S=[]
    for s in seeds:
        sa=a_base+(f"_s{s}" if s else ""); sb=b_base+(f"_s{s}" if s else "")
        ea,eb=load(sa),load(sb)
        if ea is None or eb is None: continue
        A.append(ea[metric]); B.append(eb[metric]); S.append(s)
    A=np.array(A); B=np.array(B); d=B-A
    if len(A)==0: return None
    res=dict(label=label, metric=metric, seeds=S, moe=A.tolist(), dense=B.tolist(), gap=d.tolist(),
             moe_mean=float(A.mean()), moe_sd=float(A.std(ddof=1)) if len(A)>1 else 0.0,
             dense_mean=float(B.mean()), dense_sd=float(B.std(ddof=1)) if len(B)>1 else 0.0,
             gap_mean=float(d.mean()), gap_sd=float(d.std(ddof=1)) if len(d)>1 else 0.0,
             n=len(A), n_dense_wins=int((d>0).sum()))
    res['sign_test_p_one_sided']=float(0.5**len(d)) if res['n_dense_wins']==len(d) else None
    if len(d)>1:
        t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else float('inf')
        res['paired_t']=float(t)
        try:
            from scipy import stats
            res['paired_t_p_two_sided']=float(stats.ttest_rel(B,A).pvalue)
            if len(d)>=5: res['wilcoxon_p']=float(stats.wilcoxon(B,A).pvalue)
        except Exception: pass
        # seed-variance check: is the observed gap bigger than the within-arm seed spread?
        res['gap_vs_seedspread']=float(abs(d.mean())/max(A.std(ddof=1),B.std(ddof=1),1e-9))
    return res
if __name__=='__main__':
    out={}
    out['sequence_arm']=paired('moe_mixed','dense_mixed',[0,1,2,3,4],'sequence arm (A vs dense twin)')
    out['state_arm']=paired('state_mixed','state_dense_mixed',[0,1,2],'position-blind arm C vs its dense twin')
    out['sequence_arm_slope']=paired('moe_mixed','dense_mixed',[0,1,2,3,4],'sequence arm, change-point','best_partial_slope')
    json.dump(out, open(f'{ROOT}/out/seeds/paired.json','w'), indent=1)
    for k,v in out.items():
        if not v: print(f"{k}: no data"); continue
        print(f"\n=== {v['label']}  [{v['metric']}]  n={v['n']} seeds {v['seeds']}")
        print(f"  MoE   : {v['moe_mean']:.4f} +/- {v['moe_sd']:.4f}   {['%.4f'%x for x in v['moe']]}")
        print(f"  dense : {v['dense_mean']:.4f} +/- {v['dense_sd']:.4f}   {['%.4f'%x for x in v['dense']]}")
        print(f"  gap (dense-MoE): {v['gap_mean']:+.4f} +/- {v['gap_sd']:.4f}   dense wins {v['n_dense_wins']}/{v['n']}")
        if v.get('sign_test_p_one_sided'): print(f"  sign test (one-sided): p={v['sign_test_p_one_sided']:.4f}")
        if v.get('paired_t_p_two_sided') is not None: print(f"  paired t-test: t={v.get('paired_t',float('nan')):.2f} p={v['paired_t_p_two_sided']:.4f}")
        if v.get('wilcoxon_p') is not None: print(f"  wilcoxon: p={v['wilcoxon_p']:.4f}")
        if v.get('gap_vs_seedspread') is not None: print(f"  |gap| / within-arm seed sd = {v['gap_vs_seedspread']:.2f}")
