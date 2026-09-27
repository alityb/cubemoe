"""Applies AMENDMENT 5 (A5.4) decision rules to the H2b arm, mechanically.

Leg 1 (correlational, ceiling-normalized) and Leg 2 (causal, co-primary) must BOTH pass.
Leg 2 carries a mandatory artifact control: the L_median-L_native effect must survive
partialling out the already-routed fraction, because in H2 that fully explained an
apparent 7/20 signal. H_pos is REJECTED if Leg 2 fails regardless of Leg 1.
"""
import sys, os, json, argparse
import numpy as np
from scipy import stats
ROOT='/Users/alityb/projects/cubemoe'

def ols_t(y, Xd, names):
    """OLS with t-stats. Returns dict name -> (coef, t, p)."""
    b,_,_,_=np.linalg.lstsq(Xd,y,rcond=None)
    resid=y-Xd@b; dof=len(y)-np.linalg.matrix_rank(Xd)
    s2=(resid@resid)/dof
    XtXi=np.linalg.pinv(Xd.T@Xd); se=np.sqrt(np.maximum(np.diag(XtXi)*s2,0))
    out={}
    for i,nm in enumerate(names):
        t=b[i]/se[i] if se[i]>0 else 0.0
        out[nm]=(float(b[i]),float(t),float(2*stats.t.sf(abs(t),dof)))
    return out,dof

def leg2(R, stages_use, label):
    """L_median - L_native > 0 in >=4/5 AND paired t p<0.05, THEN partial control."""
    nat=[];med=[];pair=[]
    for r in R:
        su=[s for s in r['stages'] if s in stages_use]
        for s in su:
            nat.append(r['L_native'][str(s)]); med.append(r['L_median'][str(s)])
        pair.append((r['tag'],su))
    nat=np.array(nat);med=np.array(med); d=med-nat
    nseed=sum(1 for r in R if any(s in stages_use for s in r['stages']))
    perseed=[]
    for r in R:
        su=[s for s in r['stages'] if s in stages_use]
        if not su: continue
        dd=np.mean([r['L_median'][str(s)]-r['L_native'][str(s)] for s in su])
        perseed.append(dd)
    perseed=np.array(perseed)
    t,p=stats.ttest_rel(med,nat)
    nseed_pos=int((perseed>0).sum())
    print(f"\n  --- Leg 2 ({label}) ---")
    print(f"    cells n={len(d)}  mean L_native={nat.mean():.4f}  mean L_median={med.mean():.4f}")
    print(f"    mean advantage (median-native) = {d.mean():+.4f}  sd={d.std(ddof=1):.4f}")
    print(f"    paired t={t:.3f} p={p:.4f}   seeds with positive advantage: {nseed_pos}/{len(perseed)}")
    # ---- mandatory artifact control ----
    ys=[];rows=[];seed_ix=[];stage_ix=[]
    for si,r in enumerate(R):
        Lm=np.array(r['damage']); fr=np.array(r['already_routed_frac'])
        for j,s in enumerate(r['stages']):
            if s not in stages_use: continue
            for e in range(r['E']):
                ys.append(Lm[e,j]); rows.append((fr[e,j], 1.0 if r['native'][str(s)]==e else 0.0))
                seed_ix.append(si); stage_ix.append(s)
    ys=np.array(ys); rows=np.array(rows)
    seed_ix=np.array(seed_ix); stage_ix=np.array(stage_ix)
    # stage + seed fixed effects (damage magnitude varies strongly by stage)
    cols=[np.ones(len(ys))]; names=['const']
    for s in sorted(set(stage_ix))[1:]:
        cols.append((stage_ix==s).astype(float)); names.append(f'stage{s}')
    for si in sorted(set(seed_ix))[1:]:
        cols.append((seed_ix==si).astype(float)); names.append(f'seed{si}')
    cols.append(rows[:,0]); names.append('already_routed_frac')
    cols.append(rows[:,1]); names.append('is_native')
    Xd=np.column_stack(cols)
    res,dof=ols_t(ys,Xd,names)
    b,t_,p_=res['is_native']; bf,tf,pf=res['already_routed_frac']
    print(f"    partial control (damage ~ stage FE + seed FE + already_routed_frac + is_native), n={len(ys)}, dof={dof}")
    print(f"      already_routed_frac: coef={bf:+.4f} t={tf:+.2f} p={pf:.4f}")
    print(f"      is_native:           coef={b:+.4f} t={t_:+.2f} p={p_:.4f}   (specialization => NEGATIVE)")
    partial_ok = (b<0) and (p_<0.05)
    print(f"      -> partial effect {'SURVIVES' if partial_ok else 'does NOT survive'} the control")
    naive = (nseed_pos>=4) and (p<0.05)
    print(f"    Leg 2 naive rule: {nseed_pos}/{len(perseed)} seeds and p={p:.4f} -> {'pass' if naive else 'FAIL'}")
    passed = naive and partial_ok
    print(f"    LEG 2 {'PASS' if passed else 'FAIL'}  (requires naive rule AND partial control)")
    # MDE
    sd=d.std(ddof=1); mde=2.86*sd/np.sqrt(len(d))
    print(f"    MDE at n={len(d)}, alpha=.05, power .8: ~{mde:.4f}; observed {d.mean():+.4f}"
          f" -> {'UNDERPOWERED null' if (not passed and abs(d.mean())<mde) else 'adequately powered'}")
    return passed, dict(n=len(d),L_native=float(nat.mean()),L_median=float(med.mean()),
                        adv=float(d.mean()),t=float(t),p=float(p),seeds_pos=nseed_pos,
                        is_native_coef=b,is_native_p=p_,frac_coef=bf,frac_p=pf,
                        partial_ok=bool(partial_ok),passed=bool(passed),mde=float(mde))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--twox2',default='h2b_2x2'); ap.add_argument('--causal',default='h2b_causal')
    ap.add_argument('--data',default='cfop_dec'); ap.add_argument('--seeds',default='50,51,52,53,54')
    a=ap.parse_args()
    print("="*78); print("AMENDMENT 5 / H2b — H_pos VERDICT (rules applied mechanically)"); print("="*78)

    # ---------- L1 learning gate ----------
    import numpy as _np
    d=dict(_np.load(f'{ROOT}/data/{a.data}.npz'))
    stg=d['phase'].astype(int); pos=d['pos'].astype(int); mv=d['moves'].astype(int)
    key=stg.astype(np.int64)*1000+pos
    bar=0
    for k in np.unique(key):
        m=key==k; bar+=np.bincount(mv[m]).max()
    bar/=len(mv)
    accs=[]
    for s in a.seeds.split(','):
        f=f'{ROOT}/out/moe_cfopdec_s{s}.json'
        if os.path.exists(f): accs.append(json.load(open(f))['acc'])
    print(f"\n--- L1 learning gate ---")
    print(f"  maj|(stage,pos) bar = {bar:.4f}   gate = 1.30x = {1.30*bar:.4f}")
    if accs:
        print(f"  acc: {['%.4f'%x for x in accs]}  min={min(accs):.4f}  ratio={min(accs)/bar:.2f}x")
        npass=sum(1 for x in accs if x>=1.30*bar)
        l1=npass>=4
        print(f"  {npass}/{len(accs)} seeds over the gate -> L1 {'PASS' if l1 else 'FAIL (arm INCONCLUSIVE per A5.4)'}")
    else:
        print("  no trained models yet"); return
    for fn in (a.twox2,a.causal):
        if not os.path.exists(f'{ROOT}/out/{fn}.json'):
            print(f"\n  missing out/{fn}.json — run the analysis first"); return

    # ---------- Leg 1 ----------
    X=json.load(open(f'{ROOT}/out/{a.twox2}.json'))['cfop']
    fs=np.array([r['frac_stage'] for r in X]); fg=np.array([r['frac_g1'] for r in X])
    rs=np.array([r['cnmi_stage'] for r in X]); rg=np.array([r['cnmi_g1'] for r in X])
    ns=np.array([r['shuffrac_stage'] for r in X])
    t1,p1=stats.ttest_rel(fs,fg)
    print(f"\n--- Leg 1 (correlational, CEILING-NORMALIZED) ---")
    print(f"  ceilings: stage={X[0]['ceil_stage']:.4f}  G1={X[0]['ceil_g1']:.4f}")
    for r in X:
        print(f"    {r['tag']}: stage {100*r['frac_stage']:5.1f}% (null {100*r['shuffrac_stage']:4.1f}%)"
              f"   G1 {100*r['frac_g1']:5.1f}% (null {100*r['shuffrac_g1']:4.1f}%)")
    print(f"  mean stage {100*fs.mean():.1f}% of ceiling (shuffled null {100*ns.mean():.1f}%)"
          f" vs G1 {100*fg.mean():.1f}%")
    leg1=(int((fs>fg).sum())>=4) and (p1<0.05)
    print(f"  rule: stage>G1 in >=4/5 AND paired t p<0.05 -> {int((fs>fg).sum())}/{len(fs)}, p={p1:.4f}"
          f" -> LEG 1 {'PASS' if leg1 else 'FAIL'}")

    # ---------- Leg 2 (primary, all stages) + secondary (stages 2-4) ----------
    R=json.load(open(f'{ROOT}/out/{a.causal}.json'))
    leg2p,s_pri=leg2(R,{1,2,3,4},'PRIMARY, all 4 stages')
    leg2s,s_sec=leg2(R,{2,3,4},'SECONDARY, stages 2-4 only (canonical segments)')

    print("\n"+"="*78)
    if not l1:
        v='INCONCLUSIVE (L1 failed)'
    elif leg1 and leg2p: v='H_pos SUPPORTED'
    elif not leg2p:      v='H_pos REJECTED (Leg 2 failed; causal leg is decisive per A5.4)'
    else:               v='H_pos NOT SUPPORTED'
    print(f"VERDICT: {v}")
    print(f"  Leg 1 (correlational) : {'PASS' if leg1 else 'FAIL'}")
    print(f"  Leg 2 (causal, co-primary, PRIMARY 4-stage): {'PASS' if leg2p else 'FAIL'}")
    print(f"  Leg 2 SECONDARY (stages 2-4): {'PASS' if leg2s else 'FAIL'}  [declared in A5.5, not primary]")
    print("="*78)
    json.dump(dict(verdict=v,leg1=bool(leg1),leg1_p=float(p1),
                   frac_stage=fs.tolist(),frac_g1=fg.tolist(),bar=float(bar),accs=accs,
                   leg2_primary=s_pri,leg2_secondary=s_sec),
              open(f'{ROOT}/out/h2b_verdict.json','w'),indent=1)
    print("wrote out/h2b_verdict.json")
if __name__=='__main__': main()
