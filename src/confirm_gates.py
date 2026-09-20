"""Apply the PREREGISTRATION.md pass criteria. No discretion."""
import json, os, glob
import numpy as np
from scipy import stats
ROOT='/Users/alityb/projects/cubemoe'
import argparse as _ap
_p=_ap.ArgumentParser(); _p.add_argument('--seeds',default='10,11,12,13,14'); _p.add_argument('--dir',default=None)
_a,_=_p.parse_known_args()
SEEDS=[int(x) for x in _a.seeds.split(',')]
CONFDIR=_a.dir or f'{ROOT}/out/confirm'
def load(arm):
    base='moe_mixed_s' if arm=='A' else 'state_mixed_s'
    r=[]
    for s in SEEDS:
        p=f'{CONFDIR}/{base}{s}.json'
        if os.path.exists(p): r.append(json.load(open(p)))
    return r
def gate(arm):
    R=load(arm)
    if len(R)<5: return dict(arm=arm,n=len(R),error='incomplete')
    acc=np.array([x['acc_ratio'] for x in R]); pr=np.array([x['probe'] for x in R])
    nmi=np.array([x['cond_nmi'] for x in R]); sh=np.array([x['shuffled'] for x in R])
    pp=np.array([x['perm_p'] for x in R])
    b1=np.array([x['b1'] for x in R]); b1p=np.array([x['b1_p'] for x in R])
    le=np.array([x['causal']['loss_e1'] for x in R]); mo=np.array([x['causal']['median_other'] for x in R])
    eff=np.array([x['causal']['effect'] for x in R])
    L1 = int(((acc>=1.30)&(pr>=0.85)).sum())>=4
    ht=[x.get('hash_twin') for x in R]
    has_hash = all(h is not None for h in ht)
    hash_wins = int(sum(1 for a,b in zip(nmi,ht) if b is not None and a>b))
    L2_base = (int((nmi>sh).sum())>=4) and (int((pp<=0.01).sum())>=4) and (nmi.mean()>=5*sh.mean())
    # A1.1: arm A additionally requires beating its hash twin in >=4/5 seeds. Arm C is exempt.
    L2 = L2_base and (hash_wins>=4 if arm=='A' else True)
    L3 = int(((b1>0)&(b1p<0.01)).sum())>=4
    wins=int((le>mo).sum())
    sign_p = stats.binomtest(wins,5,0.5,alternative='greater').pvalue
    tp = stats.ttest_rel(le,mo).pvalue
    L4 = (wins==5)
    # A1.3: arm C confirmed on L1+L2+L4; L3 reported but not required.
    confirmed = (L1 and L2 and L3 and L4) if arm=='A' else (L1 and L2 and L4)
    return dict(arm=arm,n=5,
        acc_ratio=[round(float(x),3) for x in acc],
        acc_abs=[round(float(x['acc']),4) for x in R], bar=[round(float(x['bar']),4) for x in R], probe=[round(float(x),3) for x in pr],
        cond_nmi=[round(float(x),4) for x in nmi], shuffled=[round(float(x),4) for x in sh],
        perm_p=[round(float(x),4) for x in pp],
        b1=[round(float(x),3) for x in b1], b1_p=[round(float(x),3) for x in b1p],
        loss_e1=[round(float(x),4) for x in le], median_other=[round(float(x),4) for x in mo],
        effect=[round(float(x),2) for x in eff],
        causal_wins=wins, causal_sign_p=float(sign_p), causal_paired_t_p=float(tp),
        L1=bool(L1),L2=bool(L2),L3=bool(L3),L4=bool(L4),
        hash_twin=[None if h is None else round(float(h),4) for h in ht],
        hash_wins=hash_wins, has_hash=bool(has_hash),
        L3_required=(arm=='A'), confirmed=bool(confirmed),
        self_exact=[x['causal']['self_exact'] for x in R])
if __name__=='__main__':
    out={a:gate(a) for a in ('A','C')}
    json.dump(out,open(f'{CONFDIR}/gates.json','w'),indent=1)
    for a,g in out.items():
        print(f"\n=== ARM {a} ===")
        if g.get('error'): print("  incomplete:",g['n'],"seeds"); continue
        print(f"  L1 learning  : acc_ratio {g['acc_ratio']} probe {g['probe']} -> {'PASS' if g['L1'] else 'FAIL'}")
        print(f"  L2 nulls     : cNMI {g['cond_nmi']} vs shuf {g['shuffled']} permp {g['perm_p']}")
        print(f"                 hash twin {g['hash_twin']} beaten {g['hash_wins']}/5 "
              f"{'(required)' if g['arm']=='A' else '(arm C exempt)'} -> {'PASS' if g['L2'] else 'FAIL'}")
        print(f"  L3 changept  : b1 {g['b1']} p {g['b1_p']} -> {'PASS' if g['L3'] else 'FAIL'}"
              f"{'' if g['L3_required'] else '   [REPORTED ONLY - not required for arm C per A1.3]'}")
        print(f"  L4 causal    : L(e1) {g['loss_e1']} vs med_other {g['median_other']} wins {g['causal_wins']}/5 "
              f"sign p={g['causal_sign_p']:.4f} paired t p={g['causal_paired_t_p']:.4f} -> {'PASS' if g['L4'] else 'FAIL'}")
        print(f"  self-control exact: {g['self_exact']}")
        print(f"  ==> {'CONFIRMED' if g['confirmed'] else 'NOT CONFIRMED'}")
