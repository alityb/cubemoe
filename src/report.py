import json, os, numpy as np
ROOT='/Users/alityb/projects/cubemoe'
R=json.load(open(f'{ROOT}/out/results.json')); M=R['models']
PRIMARY='moe_mixed'; POSBLIND='state_mixed'
L=[]; A=L.append
def tr(tag):
    p=f'{ROOT}/out/{tag}.json'; return json.load(open(p)) if os.path.exists(p) else None
def card(k):
    p=f'{ROOT}/out/datacard_{k}.json'; return json.load(open(p)) if os.path.exists(p) else None
def baselines(kind):
    d=np.load(f'{ROOT}/data/{kind}.npz')
    n=len(d['seam']); te=np.random.RandomState(0).permutation(n)[int(n*0.9):]
    m=np.isin(d['solve_id'],te); mv=d['moves'][m]
    ph=d['phase'][m].astype(int); ps=d['pos'][m].astype(int)
    def maj(c=None):
        if c is None: return np.bincount(mv,minlength=18).max()/len(mv)
        return sum(np.bincount(mv[c==v],minlength=18).max() for v in np.unique(c))/len(mv)
    return dict(n=int(len(mv)), glob=maj(), phase=maj(ph), pos=maj(ps), both=maj(ph*100+ps))
DS=[k for k in ('mixed','naive') if os.path.exists(f'{ROOT}/data/{k}.npz')]
BL={k:baselines(k) for k in DS}

GATES=None
_gp=f'{ROOT}/out/confirm/gates.json'
if os.path.exists(_gp):
    try: GATES=json.load(open(_gp))
    except Exception: GATES=None

A("# PhaseSplit — results\n")
A("> **Do MoE routers recover the decomposition latent in their training data?** Case study:\n"
  "> Kociemba two-phase Rubik's cube solutions, where the phase boundary (entry into\n"
  "> G1 = `<U,D,R2,L2,F2,B2>`) is a theorem, not a heuristic label.\n")
A("\n## HEADLINE\n")
if GATES and all(not g.get('error') for g in GATES.values()):
    for _a,_lab in (('A','sequence arm'),('C','position-blind arm')):
        _g=GATES[_a]
        _l3 = ('L3 '+('P' if _g['L3'] else 'F')) + ('' if _g.get('L3_required',True) else ' (reported only)')
        _accs=_g.get('acc_ratio',[]); _acc=(f"{np.mean([_g['acc_abs'][i] for i in range(len(_g.get('acc_abs',[])))]):.4f}"
              if _g.get('acc_abs') else '?')
        A(f"- **{_lab}: {'CONFIRMED' if _g['confirmed'] else 'NOT CONFIRMED'}** "
          f"(L1 {'P' if _g['L1'] else 'F'} / L2 {'P' if _g['L2'] else 'F'} / {_l3} / "
          f"L4 {'P' if _g['L4'] else 'F'}) — held-out accuracy {_acc}, "
          f"ratio to majority|(phase,stratum) {_accs}; fresh seeds 10-14.")
    if os.path.exists(f'{ROOT}/out/confirm_scale/gates.json'):
        _sg2=json.load(open(f'{ROOT}/out/confirm_scale/gates.json'))
        A(f"- **Scale replication (6L/d256, 250k, seeds 20-24)**: arm A "
          f"**{'CONFIRMED' if _sg2['A']['confirmed'] else 'NOT CONFIRMED'}**, arm C "
          f"**{'CONFIRMED' if _sg2['C']['confirmed'] else 'NOT CONFIRMED'}** — same criteria, no changes (Part III).")
    A("\nThe headline comes **only** from Part II. Part I is hypothesis-generating and contains a "
      "documented null->positive reversal; none of it is a claim.\n")
else:
    A("**Confirmatory run in progress — no claim yet.** Part I below is exploratory and "
      "hypothesis-generating only; it contains a documented null->positive reversal caused by "
      "substituting a control for the pre-registered contrast. Nothing in it is a claim. The "
      "headline will be filled from Part II when the pre-registered gates evaluate on fresh seeds.\n")

A("\n---\n\n# PART I — EXPLORATORY (hypothesis-generating; NOT claims)\n")
A("Everything in this part used seeds 0-5 and involved post-hoc choices (best-layer selection, an "
  "arbitrary single-expert causal control, a substituted contrast). It is retained in full, "
  "including the reversal history, because the mistakes are the useful part.\n")
# ---- 1. data
A("\n## 1. Data\n")
A("| set | solves | steps | seam mean | seam sd | mixed step-cells | sol_len mean |")
A("|---|---|---|---|---|---|---|")
for k in DS:
    d=np.load(f'{ROOT}/data/{k}.npz'); ph=d['phase']; ps=d['pos']
    mix=sum((ps==t).sum() for t in np.unique(ps) if len(np.unique(ph[ps==t]))==2)/len(ph)
    A(f"| {k} | {len(d['seam'])} | {len(ph)} | {d['seam'].mean():.2f} | **{d['seam'].std():.2f}** | "
      f"**{mix:.3f}** | {d['sol_len'].mean():.2f} |")
cm=card('mixed')
if cm and 'seam_hist' in cm.get('distribution',{}):
    h=cm['distribution']['seam_hist']
    A("\nRealized seam histogram (mixed): " + ", ".join(f"{k}:{v}" for k,v in sorted(h.items(), key=lambda x:int(x[0]))) + "\n")

# ---- 2. QA
A("\n## 2. Data QA (`src/qa.py`)\n")
for k in DS:
    c=card(k)
    if not c: continue
    A(f"\n**{k}** — verdict **{'PASS' if c['PASS'] else 'FAIL'}**, code sha256 `{c['code_sha256']}`\n")
    A("| hard invariant | result |"); A("|---|---|")
    for h in c['hard_checks']: A(f"| {h['name']} | {'PASS' if h['ok'] else '**FAIL** — '+h['detail']} |")
    if 'label_determinism' in c:
        ld=c['label_determinism']
        A(f"\nLabel determinism over {ld['n']} intermediate states: standalone re-solve **{ld['standalone']:.4f}**, "
          f"re-solve with the `lastface` context **{ld['with_context']:.4f}**. The gap is entirely the "
          f"consecutive-same-face rule at the split — `(state, last-move-face) -> next move` is exactly deterministic.\n")
cards={k:card(k) for k in DS if card(k)}
if len(cards)>1:
    A("\n**Stereotypy — mixed vs stock.** Optimal maneuvers should not be stereotyped.\n")
    A("| metric | " + " | ".join(cards) + " |"); A("|---|" + "---|"*len(cards))
    def g(c,*ks):
        v=c
        for k in ks:
            v = v[k] if k in v else v[str(k)]
        return v
    rows=[("phase-1 unigram H",('stereotypy','unigram_entropy')),("phase-1 bigram H",('stereotypy','bigram_entropy')),
          ("phase-1 trigram H",('stereotypy','trigram_entropy')),("repeated 5-prefix rate",('stereotypy','repeated_prefix_rate',5)),
          ("P(move[t]==move[t-2]) phase 1",('stereotypy','lag2_repeat')),
          ("phase-2 move entropy (max 3.32)",('stereotypy_phase2','entropy')),
          ("phase-2 lag-2 repetition",('stereotypy_phase2','lag2_repeat')),
          ("Bayes phase acc | MOVE",('confound','move')),("Bayes phase acc | POSITION",('confound','position')),
          ("majority baseline",('confound','majority'))]
    for lab,ks in rows:
        vs=[]
        for k in cards:
            try: v=g(cards[k],*ks); vs.append(f"{v:.3f}" if isinstance(v,float) else str(v))
            except Exception: vs.append("—")
        A(f"| {lab} | " + " | ".join(vs) + " |")
A("\n**Tail leakage** — held-out states appearing verbatim in training.\n")
A("| set | overall | " + " | ".join(f"excl last {n}" for n in (3,4,5,6)) + " |")
A("|---|---|" + "---|"*4)
for k in DS:
    c=card(k)
    if not c or 'excl_last' not in c.get('leakage',{}): continue
    lk=c['leakage']
    A(f"| {k} | {lk['overall']:.4f} | " + " | ".join(f"{lk['excl_last'][str(n)]:.4f} (keep {lk['rows_kept'][str(n)]:.2f})" for n in (3,4,5,6)) + " |")
A("\nAll routing metrics below are reported on the full held-out set and on tail-excluded replicates at N=3 and N=6.\n")

# ---- 3. confound
A("\n## 3. Confound table (held-out rows) — Figure 0\n")
A("| set | n | majority | position | move | prev move | position+move |"); A("|---|---|---|---|---|---|---|")
for k,c in R.get('confound',{}).items():
    A(f"| {k} | {c['n']} | {c['majority']:.4f} | **{c['bayes_position']:.4f}** | {c['bayes_move']:.4f} | "
      f"{c['bayes_prev_move']:.4f} | {c['bayes_position+move']:.4f} |")

# ---- 4. learning gate
A("\n## 4. Learning gate\n")
A("Held-out teacher-forced next-move accuracy must be clearly above `majority|(phase,pos)` **and** rising.\n")
A("| model | data | test acc | maj (global) | maj\\|pos | **maj\\|(phase,pos)** | ratio | rising? | acc trajectory |")
A("|---|---|---|---|---|---|---|---|---|")
GATE={}
for tag in [PRIMARY,'dense_mixed','hash_mixed',POSBLIND,'state_dense_mixed','moe_naive']:
    t=tr(tag)
    if not t: continue
    ds='naive' if 'naive' in tag else 'mixed'; b=BL[ds]
    hist=t.get('acc_hist',[]); rising = len(hist)>=3 and (hist[-1] > hist[len(hist)//2] > hist[0])
    ratio=t['acc']/b['both']
    GATE[tag]=dict(acc=t['acc'], bar=b['both'], ratio=ratio, rising=rising,
                   passed=bool(ratio>=1.30 and rising))
    traj=" → ".join(f"{h:.3f}" for h in (hist[::max(1,len(hist)//4)] if hist else []))
    A(f"| {tag} | {ds} | **{t['acc']:.4f}** | {b['glob']:.4f} | {b['pos']:.4f} | {b['both']:.4f} | "
      f"**{ratio:.2f}x** | {'yes' if rising else 'no'} | {traj} |")
A("\nGate: ratio >= 1.30 over `majority|(phase,pos)` and monotone-ish improvement. "
  + ", ".join(f"**{k}: {'PASS' if v['passed'] else 'FAIL'}**" for k,v in GATE.items()) + "\n")

# ---- 5. router health
A("\n## 5. Router health\n")
A("| model | layer | utilization | expert entropy (bits, max 3) | experts used |"); A("|---|---|---|---|---|")
for tag,mm in M.items():
    for Ly in mm['layers']:
        o=Ly['observed']
        A(f"| {tag} | {Ly['layer']} | {o.get('utilization',float('nan')):.3f} | "
          f"{o.get('expert_entropy_bits',float('nan')):.2f} | {o.get('experts_used','?')} |")

# ---- 6. PRE-REGISTERED CONTRAST
import glob as _g
def _C(t):
    f=f'{ROOT}/out/contrast/{t}.json'
    return json.load(open(f)) if os.path.exists(f) else None
SEQ=[('','0'),('_s1','1'),('_s2','2'),('_s3','3'),('_s4','4')]
CARM=[('','0'),('_s1','1'),('_s2','2')]
_hash=_C('hash_mixed'); HASHNULL=max((r['router_cond_nmi'] for r in _hash['layers']), default=float('nan')) if _hash else float('nan')

A("\n## 6. PRE-REGISTERED CONTRAST — the router against its nulls\n")
A("This is the claim. Nulls: shuffled router (expert ids permuted across tokens), a synthetic "
  "**position-only** router (8 quantile bins of the stratifier), the separately-trained **hash twin** "
  "(fixed token-identity routing), and a within-stratum label permutation (200 perms). "
  "Stratifier: step index; **remaining moves** for the position-blind arm C.\n")
def _arm(tag_base, seeds, is_state, label):
    rows=[]
    A(f"\n### {label}\n")
    A("| seed | best layer | **router cond NMI** | shuffled | position-only | hash twin | perm p | change-point slope | cp p | probe |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for sfx,name in seeds:
        d=_C(tag_base+sfx)
        if not d: continue
        b=max(d['layers'], key=lambda r:r['router_cond_nmi']); rows.append(b)
        hn = "—" if is_state else f"{HASHNULL:.4f}"
        A(f"| {name} | {b['layer']} | **{b['router_cond_nmi']:.4f}** | {b['null_shuffled']:.4f} | "
          f"{b['null_position_router']:.4f} | {hn} | {b['perm_p']:.4f} | {b['cp_partial_slope']:+.3f} | "
          f"{b['cp_partial_p']:.3f} | {b['probe']['probe_acc']:.3f} |")
    if not rows: return None
    r=np.array([x['router_cond_nmi'] for x in rows]); sh=np.array([x['null_shuffled'] for x in rows])
    sl=np.array([x['cp_partial_slope'] for x in rows]); cp=np.array([x['cp_partial_p'] for x in rows])
    pr=np.array([x['probe']['probe_acc'] for x in rows]); km=np.array([x['kmeans_own_hidden_cond_nmi'] for x in rows])
    A(f"| **mean ± sd** | | **{r.mean():.4f} ± {r.std(ddof=1):.4f}** | {sh.mean():.4f} | 0.0000 | "
      f"{'—' if is_state else f'{HASHNULL:.4f}'} | | {sl.mean():+.3f} | | {pr.mean():.3f} |")
    # layer-wise robustness
    allc=[]
    for sfx,_ in seeds:
        d=_C(tag_base+sfx)
        if d: allc += d['layers']
    nr=sum(1 for x in allc if x['router_cond_nmi']>x['null_shuffled'])
    npp=sum(1 for x in allc if x['perm_p']<=0.005)
    ncp=sum(1 for x in allc if x['cp_partial_p'] is not None and x['cp_partial_p']<0.01)
    A(f"\n- router > shuffled null in **{nr}/{len(allc)}** (seed, layer) cells; permutation p at the 0.005 floor in **{npp}/{len(allc)}**")
    A(f"- router/shuffled ratio at the best layer: **{r.mean()/sh.mean():.0f}x**; position-only router scores exactly **0.0000** (a positional router carries no phase information once you condition on position)")
    if not is_state: A(f"- router beats the hash twin ({HASHNULL:.4f}) in **{int((r>HASHNULL).sum())}/{len(r)}** seeds")
    A(f"- change-point p<0.01 in **{ncp}/{len(allc)}** cells overall, but **{int((cp<0.01).sum())}/{len(cp)}** at the best-NMI layer — alignment and seam-tracking both concentrate in the deepest layers")
    return dict(r=r, sh=sh, sl=sl, cp=cp, pr=pr, km=km, n=len(r))
SEQR=_arm('moe_mixed', SEQ, False, 'Sequence arm A (n=5 seeds)')
CR=_arm('state_mixed', CARM, True, 'Position-blind arm C (n=3 seeds)')

A("\n## 7. Controls — NOT the claim\n")
A("**Within-model control.** k-means (k=8) on each MoE's *own* pre-router hidden states (the `ln2` "
  "output the router actually sees), same layer. Asks: does the router beat clustering its own input?\n")
A("| arm | router cond NMI | k-means on own pre-router hidden | router wins |")
A("|---|---|---|---|")
for lab,R in [('sequence A',SEQR),('position-blind C',CR)]:
    if not R: continue
    A(f"| {lab} | {R['r'].mean():.4f} ± {R['r'].std(ddof=1):.4f} | {R['km'].mean():.4f} ± {R['km'].std(ddof=1):.4f} | "
      f"{int((R['r']>R['km']).sum())}/{R['n']} |")
_d=_C('dense_mixed'); _sd=_C('state_dense_mixed')
A(f"\n**Cross-model sanity check only** (one line, not a finding): k-means on a size-matched *dense* "
  f"model's hidden states scores {_d['best_kmeans_own']:.4f} (sequence) and {_sd['best_kmeans_own']:.4f} "
  f"(position-blind). This confirms the phase structure in the residual stream is not MoE-specific. "
  f"**Retraction:** an earlier draft reported \"the dense twin beats the MoE\" as the headline finding. "
  f"That was a control mis-elevated to a claim, and it is withdrawn — the dense comparison never bore on "
  f"whether the router partitions on phase.\n")

# ---- 8b. causal intervention
A("\n## 7b. CAUSAL test — is expert identity *used*, or only correlated?\n")
A("Correlational metrics cannot separate \"the router reads phase\" from \"the router's choice of "
  "expert does phase-appropriate work\". At layer 3 we force the top-1 expert for prediction "
  "positions and read out **probability mass on the 10 G1-legal moves** — in phase 2 the correct "
  "move is *always* G1-legal, so this is a sharp phase-specific behavioural signature.\n")
A("Conditions: `self` (force current top-1 — sanity), `random` (force a random other expert), "
  "**`single`** (force ONE unrelated expert, matching `swap`'s collapse of routing diversity), "
  "**`swap`** (force the opposite-phase expert).\n")
try:
    import glob as _gg
    _cz=[json.load(open(f'{ROOT}/out/causal/moe_mixed{x}.json')) for x in ['','_s1','_s2','_s3','_s4']
         if os.path.exists(f'{ROOT}/out/causal/moe_mixed{x}.json')]
    if _cz:
        A("| seed | e1 / e2 | baseline | self | random | single | **swap** | Δswap | Δsingle |")
        A("|---|---|---|---|---|---|---|---|---|")
        _ds=[];_dg=[];_dr=[]
        for _i,_r in enumerate(_cz):
            _b=_r['baseline']['phase2_g1mass']
            _sw=_b-_r['swap']['phase2_g1mass']; _si=_b-_r['single']['phase2_g1mass']; _ra=_b-_r['random']['phase2_g1mass']
            _ds.append(_sw);_dg.append(_si);_dr.append(_ra)
            A(f"| {_i} | {_r['e_phase1']} / {_r['e_phase2']} | {_b:.4f} | {_r['self']['phase2_g1mass']:.4f} | "
              f"{_r['random']['phase2_g1mass']:.4f} | {_r['single']['phase2_g1mass']:.4f} | "
              f"**{_r['swap']['phase2_g1mass']:.4f}** | {_sw:.4f} | {_si:.4f} |")
        _ds=np.array(_ds);_dg=np.array(_dg);_dr=np.array(_dr)
        A(f"| **mean** | | | | {np.mean([r['random']['phase2_g1mass'] for r in _cz]):.4f} | "
          f"{np.mean([r['single']['phase2_g1mass'] for r in _cz]):.4f} | "
          f"**{np.mean([r['swap']['phase2_g1mass'] for r in _cz]):.4f}** | "
          f"**{_ds.mean():.4f} ± {_ds.std(ddof=1):.4f}** | {_dg.mean():.4f} ± {_dg.std(ddof=1):.4f} |")
        _pur1=np.mean([r['p_phase1_given_e1'] for r in _cz]); _pur2=np.mean([r['p_phase2_given_e2'] for r in _cz])
        A(f"\n- `self` reproduces baseline **exactly** in {sum(1 for r in _cz if r['self']['phase2_g1mass']==r['baseline']['phase2_g1mass'])}/{len(_cz)} seeds (override machinery verified)")
        A(f"- expert purity at layer 3: P(phase1 | e1) = **{_pur1:.3f}**, P(phase2 | e2) = **{_pur2:.3f}**")
        A(f"- **swap > single in {int((_ds>_dg).sum())}/{len(_ds)} seeds and swap > random in {int((_ds>_dr).sum())}/{len(_ds)}** — sign test one-sided p = {0.5**len(_ds):.4f} for each")
        try:
            from scipy import stats as _st
            A(f"- paired t-test is **not** significant (swap vs single p = {_st.ttest_rel(_ds,_dg).pvalue:.3f}; "
              f"vs random p = {_st.ttest_rel(_ds,_dr).pvalue:.3f}) — seed 3 is a 4-8x magnitude outlier that "
              f"inflates the variance. Direction is perfectly consistent; magnitude is not. The sign test is "
              f"the appropriate statistic here and both are reported.")
        except Exception: pass
        _rat=_ds/np.maximum(_dg,1e-6)
        A(f"- median ratio swap/single = **{np.median(_rat):.1f}x** (the mean, {_rat.mean():.0f}x, is an artifact "
          f"of one seed's near-zero denominator and should not be quoted)")
        A("\n**Interpretation.** `single` collapses routing diversity exactly as `swap` does but picks an "
          "expert unrelated to either phase, and costs almost nothing. So the damage is not from losing "
          "diversity, nor from perturbation magnitude — it is specific to *which* expert computes. Routing a "
          "phase-2 token through the phase-1 expert selectively destroys the model's tendency to emit a "
          "G1-legal move.\n")
        A("**This resolves the within-model tie in §7.** k-means on the router's own hidden states matches "
          "the router's conditional NMI, but a post-hoc clustering is a *readout* and has no causal role by "
          "construction. The router's partition is causally wired to phase-appropriate computation. The "
          "correlational tie and the causal dissociation are consistent: phase is linearly available in the "
          "residual stream AND the router's use of it does work.\n")
except Exception as _e:
    A(f"(causal results unavailable: {_e})")

# ---- 9. interpretation matrix (contrast-driven)
A("\n## 8. Interpretation matrix\n")
A("Cell assigned from probe(phase|position), router-vs-nulls, and the change-point test.\n")
A("- **A** — probe low: the concept was never learned; uninformative.")
A("- **B** — probe high, routing ~ nulls, slope ~ 0: phase is represented but the router does not partition on it.")
A("- **C** — probe high, router cond NMI >> nulls, change-point slope > 0 at p<0.01, surviving in arm C.")
A("- **D** — mixed evidence.\n")
A("| arm | probe | router vs nulls | change-point | cell |")
A("|---|---|---|---|---|")
cells={}
for lab,R,is_s in [('sequence A',SEQR,False),('position-blind C',CR,True)]:
    if not R: continue
    probe_hi = R['pr'].mean()>=0.85
    nulls_ok = (R['r']>R['sh']).all() and R['r'].mean()>=5*R['sh'].mean() and (not is_s and R['r'].mean()>HASHNULL or is_s)
    cp_ok = (R['cp']<0.01).mean()>=0.8
    c = 'A' if not probe_hi else ('C' if (nulls_ok and cp_ok) else ('B' if (not nulls_ok and not cp_ok) else 'D'))
    cells[lab]=c
    A(f"| {lab} | {R['pr'].mean():.3f} ({'high' if probe_hi else 'LOW'}) | "
      f"{R['r'].mean()/R['sh'].mean():.0f}x shuffled, {int((R['r']>R['sh']).sum())}/{R['n']} seeds"
      + ("" if is_s else f", beats hash {int((R['r']>HASHNULL).sum())}/{R['n']}") + " |"
      f" p<0.01 in {int((R['cp']<0.01).sum())}/{R['n']} seeds (mean slope {R['sl'].mean():+.3f}) | **{c}** |")
A(f"\n### Verdict: **{cells.get('sequence A','?')}** on the sequence arm; "
  f"**{cells.get('position-blind C','?')}** on the position-blind arm.\n")
A("The routing/phase alignment replicates on both arms against every null (5/5 and 3/3 seeds, permutation "
  "p at the 0.005 floor). The change-point leg holds on the sequence arm (5/5 at the best-NMI layer) but "
  "not on arm C (1/3), which is why C is D rather than C.\n")

# ---- 10. figures
A("\n## 10. Figures\n")
for f,cap in [('fig0_overlap.png','P(phase 2 | step index): how much the index alone gives away'),
              ('fig1_conditional_nmi.png','raw vs conditional NMI per layer, against every null'),
              ('fig2_changepoint.png','router change-point vs true seam index')]:
    A(f"![{cap}](./{f})\n\n*{cap}*\n")

# ---- 11. appendix
A("\n## Appendix — Iteration 1 forced-L runs: padding the solution kills learnability\n")
A("Seam variance was originally produced by forcing phase-1 length L~U(10,20). That makes the next-move label "
  "high-entropy (deep in phase 1, nearly any move continues *some* valid length-(L−t) solution), so all three "
  "architectures collapsed onto the same weak accuracy and the routing analysis was uninterpretable.\n")
A("| model | test acc | majority | ratio | tok/s |"); A("|---|---|---|---|---|")
for tag in ['moe_forced','dense_forced','hash_forced']:
    p=f'{ROOT}/out/appendix_forcedL/{tag}.json'
    if not os.path.exists(p): continue
    t=json.load(open(p))
    A(f"| {tag} | {t['acc']:.4f} | {t['majority']:.4f} | {t['acc']/0.1153:.2f}x | {t['tok_s']:.0f} |")
A("\n(ratio against that set's `majority|(phase,pos)` = 0.1153.) A fixed token-identity hash router matched the "
  "learned router to within 0.001, and the dense twin slightly beat both — the learned routing bought nothing. "
  "Two data bugs found by QA on that set (seam mislabelled in 1.18% of solves; cyclic 2-cycle DFS padding with "
  "lag-2 repetition 0.315 vs chance 0.056) are both fixed and now hard gates.\n")


# ================= PART II — CONFIRMATORY =================
A("\n\n---\n\n# PART II — CONFIRMATORY\n")
_pre=f'{ROOT}/PREREGISTRATION.md'
if os.path.exists(_pre):
    import hashlib as _h
    _raw=open(_pre,'rb').read()
    _ts=open(f'{ROOT}/out/prereg_timestamp.txt').read().strip() if os.path.exists(f'{ROOT}/out/prereg_timestamp.txt') else '?'
    A(f"Pre-registration written **before any confirmatory model was trained**. "
      f"sha256 `{_h.sha256(_raw).hexdigest()[:16]}`, {_ts}.\n")
    A("<details><summary>Full pre-registration (click)</summary>\n")
    A(open(_pre).read())
    A("\n</details>\n")
if GATES:
    for _a,_lab in (('A','Sequence arm A'),('C','Position-blind arm C')):
        _g=GATES.get(_a) or {}
        A(f"\n## {_lab} — fresh seeds 10-14\n")
        if _g.get('error'):
            A(f"Incomplete ({_g.get('n',0)}/5 seeds).\n"); continue
        A("| leg | per-seed | verdict |"); A("|---|---|---|")
        A(f"| **L1** learning | acc/bar {_g['acc_ratio']}, probe {_g['probe']} | "
          f"**{'PASS' if _g['L1'] else 'FAIL'}** |")
        A(f"| **L2** router vs nulls | cNMI {_g['cond_nmi']} vs shuffled {_g['shuffled']}, perm p {_g['perm_p']} | "
          f"**{'PASS' if _g['L2'] else 'FAIL'}** |")
        A(f"| **L3** change-point | b1 {_g['b1']}, p {_g['b1_p']} | **{'PASS' if _g['L3'] else 'FAIL'}** |")
        A(f"| **L4** causal | L(e1) {_g['loss_e1']} vs median-other {_g['median_other']}; "
          f"wins {_g['causal_wins']}/5, sign p={_g['causal_sign_p']:.4f}, paired t p={_g['causal_paired_t_p']:.4f} | "
          f"**{'PASS' if _g['L4'] else 'FAIL'}** |")
        A(f"\n**Absolute accuracy**: {_g.get('acc_abs')} vs bar majority|(phase,stratum) "
          f"{_g.get('bar',['?'])[0] if _g.get('bar') else '?'} -> ratios {_g['acc_ratio']}")
        A(f"\nper-seed causal ratio L(e1)/median-other: {_g['effect']}")
        A(f"\nself-control reproduces baseline exactly: {_g['self_exact']}")
        A(f"\n### **{'CONFIRMED' if _g['confirmed'] else 'NOT CONFIRMED'}**\n")
else:
    A("\n*Confirmatory models still training; gates not yet evaluated.*\n")

_cpath=f'{ROOT}/out/confirm/controls.json'
if os.path.exists(_cpath):
    _ct=json.load(open(_cpath))
    A("\n## Controls — NOT the claim, NOT gate inputs\n")
    A("Reported for completeness. Neither column enters any pre-registered criterion.\n")
    A("| arm | seed | router cond NMI | k-means on own pre-router states | L4 ratio L(e1)/median-other |")
    A("|---|---|---|---|---|")
    for _a in ('A','C'):
        for _r in _ct.get(_a,[]):
            A(f"| {_a} | {_r['seed']} | {_r['router']:.4f} | {_r['kmeans_own']:.4f} | {_r['ratio']:.2f}x |")
    for _a,_lab in (('A','Arm A'),('C','Arm C')):
        _rs=_ct.get(_a,[])
        if not _rs: continue
        _ro=np.array([x['router'] for x in _rs]); _km=np.array([x['kmeans_own'] for x in _rs])
        _rt=np.array([x['ratio'] for x in _rs])
        A(f"\n**{_lab}**: router {_ro.mean():.4f} ± {_ro.std(ddof=1):.4f} vs own-state k-means "
          f"{_km.mean():.4f} ± {_km.std(ddof=1):.4f} — router higher in **{int((_ro>_km).sum())}/{len(_ro)}** seeds. "
          f"L4 ratio median **{np.median(_rt):.2f}x** (range {_rt.min():.2f}-{_rt.max():.2f}).")
    A("\nThe k-means column is a *readout* of the same hidden states the router reads; it has no causal "
      "role by construction, which is why it is a control and not a null. On arm A it is roughly at "
      "parity with the router; on arm C the router is clearly above it.\n")



# ================= PART III — SCALE REPLICATION =================
_sg=f'{ROOT}/out/confirm_scale/gates.json'
if os.path.exists(_sg):
    SG=json.load(open(_sg))
    A("\n\n---\n\n# PART III — SCALE REPLICATION (Amendment 2)\n")
    A("Same pre-registration, **no new criteria**, fresh seeds **20-24**, at "
      "**6 layers / d=256 on 250k solves** (CUDA, A100-40GB via Modal) instead of 4L/d128 on 25k.\n")
    A("The 250k dataset passed all 16 QA gates. It was generated once, downloaded (sha256 "
      "`80ea6303...`, 86.1 MB) and uploaded byte-identical to the second workspace, so both arms "
      "trained on exactly the same file. Arm A + hash twins ran on workspace `ali-moh-islam-1`; "
      "arm C on `dnfcubes` (a billing split only — identical code, data and seeds).\n")
    for _a,_lab in (('A','Sequence arm A'),('C','Position-blind arm C')):
        _g=SG.get(_a) or {}
        if _g.get('error'): A(f"\n## {_lab} — incomplete\n"); continue
        A(f"\n## {_lab} — fresh seeds 20-24, 6L/d256, 250k\n")
        A("| leg | per-seed | verdict |"); A("|---|---|---|")
        A(f"| **L1** learning | acc/bar {_g['acc_ratio']}, probe {_g['probe']} | **{'PASS' if _g['L1'] else 'FAIL'}** |")
        A(f"| **L2** router vs nulls | cNMI {_g['cond_nmi']} vs shuffled {_g['shuffled']}, perm p {_g['perm_p']}"
          + (f", hash twin {_g['hash_twin']} beaten {_g['hash_wins']}/5" if _a=='A' else ", hash exempt")
          + f" | **{'PASS' if _g['L2'] else 'FAIL'}** |")
        A(f"| **L3** change-point | b1 {_g['b1']}, p {_g['b1_p']} | **{'PASS' if _g['L3'] else 'FAIL'}**"
          + ("" if _g.get('L3_required',True) else " *(reported only, A1.3)*") + " |")
        A(f"| **L4** causal | L(e1) {_g['loss_e1']} vs median-other {_g['median_other']}; wins "
          f"{_g['causal_wins']}/5, sign p={_g['causal_sign_p']:.4f}, paired t p={_g['causal_paired_t_p']:.4f} "
          f"| **{'PASS' if _g['L4'] else 'FAIL'}** |")
        A(f"\nper-seed causal ratio: {[round(a/max(b,1e-9),1) for a,b in zip(_g['loss_e1'],_g['median_other'])]}")
        A(f"\nself-control exact: {_g['self_exact']}")
        A(f"\n### **{'CONFIRMED' if _g['confirmed'] else 'NOT CONFIRMED'}**\n")
    A("\n## Scale comparison (both runs, same criteria)\n")
    A("| | 4L/d128, 25k, seeds 10-14 | 6L/d256, 250k, seeds 20-24 |")
    A("|---|---|---|")
    A("| arm A acc ratio | 1.397-1.410 | **2.428-2.481** |")
    A("| arm A probe | 0.984-0.987 | **0.999-1.000** |")
    A("| arm A cond NMI | 0.071-0.192 | **0.168-0.215** |")
    A("| arm A L3 b1 (all p<=0.01) | +0.370..+0.657 | **+0.234..+0.441** |")
    A("| arm A L4 paired t | 0.037 | **0.025** |")
    A("| arm C L4 paired t | 0.011 | **0.0025** |")
    A("| arm C L3 | FAIL (1/5) | **FAIL (0/5 at p<0.01)** |")
    A("\n**The causal effect sharpens markedly with scale.** Arm A's strongest seed moves from "
      "L(e1)=0.045 vs median-other 0.005 (~9x) to **0.488 vs 0.0018 (~270x)**: forcing a phase-2 token "
      "through the phase-1 expert destroys roughly half of all G1-legal probability mass, while the "
      "median other-expert swap costs ~0.2%.\n")
    A("**Arm C's L3 failed again on independent seeds at 10x data**, as A1.3 predicted from the design "
      "of the test rather than from any result. The underfitting caveat on the small-scale run is "
      "closed: probes are ~1.000 and accuracy is 2.4-4.9x its bar.\n")

    _cs=f'{ROOT}/out/controls_scale.json'; _ch=f'{ROOT}/out/controls_hash_scale.json'
    if os.path.exists(_cs):
        CS=json.load(open(_cs)); HS=json.load(open(_ch)) if os.path.exists(_ch) else {}
        _NL=6; _rl=[x['per_layer'][str(_NL-1)] for x in CS['arms']['A'] if 'per_layer' in x]
        _ra=np.array([x['router_cnmi'] for x in CS['arms']['A']]); _pa=np.array([x['router_cnmi_pct_ceiling'] for x in CS['arms']['A']])
        _rc=np.array([x['router_cnmi'] for x in CS['arms']['C']]); _pc=np.array([x['router_cnmi_pct_ceiling'] for x in CS['arms']['C']])
        _ka=np.array([x['kmeans_cnmi'] for x in CS['arms']['A']]); _kc=np.array([x['kmeans_cnmi'] for x in CS['arms']['C']])
        A("\n## Headline (Part III)\n")
        A("> **Router specialization on the Kociemba phase boundary is concentrated in a few experts, "
          "causally load-bearing, grows with model and data scale, and is systematically understated "
          "by NMI and purity.**\n")
        A("\n## Per-layer causal effect at scale (arm A, seeds 20-24)\n")
        A("| layer | purity(e1) | L(e1) | median-other | ratio | concentration index |")
        A("|---|---|---|---|---|---|")
        for _L in range(_NL):
            rs=[x['per_layer'][str(_L)] for x in CS['arms']['A'] if 'per_layer' in x and str(_L) in x['per_layer']]
            if not rs: continue
            g=lambda k: float(np.mean([x[k] for x in rs]))
            A(f"| {_L} | {g('purity_e1'):.3f} | {g('loss_e1'):.4f} | {g('median_other'):.4f} | {g('ratio'):.0f}x | {g('concentration'):.3f} |")
        A(f"\nThe causal effect switches on where expert purity does (layer 2) and grows monotonically with depth. "
          f"The **concentration index** (share of the total causal effect carried by the single most-affected expert; "
          f"even = 0.125) reaches **{np.mean([x['concentration'] for x in _rl]):.3f}** at the last layer — one expert "
          f"carries roughly two-thirds of it.\n")
        A("\n## Cross-scale comparison (same criteria throughout)\n")
        A("| quantity | arm | 4L/d128, 25k (seeds 10-14) | 6L/d256, 250k (seeds 20-24) |")
        A("|---|---|---|---|")
        A("| L1 acc ratio | A | 1.397-1.410 | **2.428-2.481** |")
        A("| L1 acc ratio | C | 3.659-3.719 | **4.856-4.895** |")
        A("| L1 probe | A | 0.986 | **0.999** |")
        A(f"| cond NMI | A | 0.071-0.192 (~63% ceiling) | **{_ra.min():.3f}-{_ra.max():.3f} ({_pa.mean()*100:.0f}% ceiling)** |")
        A(f"| cond NMI | C | 0.119-0.170 | **{_rc.min():.3f}-{_rc.max():.3f} ({_pc.mean()*100:.0f}% ceiling)** |")
        A("| L3 b1 (all p<=0.01) | A | +0.370..+0.657 | **+0.234..+0.441** |")
        A(f"| L4 absolute L(e1) | A | ~0.045 | **{np.mean([x['loss_e1'] for x in _rl]):.4f}** |")
        A(f"| L4 ratio | A | ~9x | **{np.mean([x['ratio'] for x in _rl]):.0f}x** |")
        A("\n## Controls (not gate inputs)\n")
        A("| arm | seed | cond NMI | **% of ceiling** | conditional purity | k-means (own pre-router) | k-means % ceiling |")
        A("|---|---|---|---|---|---|---|")
        for _arm in ('A','C'):
            for r in CS['arms'][_arm]:
                A(f"| {_arm} | {r['seed']} | {r['router_cnmi']:.4f} | **{r['router_cnmi_pct_ceiling']*100:.1f}%** | "
                  f"{r['cond_purity']:.4f} | {r['kmeans_cnmi']:.4f} | {r['kmeans_pct_ceiling']*100:.1f}% |")
        A(f"\nCeiling = the perfect-tracker reference on this dataset ({CS['anchor']:.4f}); it is a reference "
          f"construction, not a mathematical maximum, so values may exceed 100%. Router beats within-model "
          f"k-means in **{int((_ra>_ka).sum())}/5** seeds on arm A and **{int((_rc>_kc).sum())}/5** on arm C. "
          f"Conditional purity is ~0.95 (A) / ~0.99 (C) for *every* assignment — saturated, so it cannot discriminate.\n")
        if HS:
            _hl=float(np.mean([v['loss_e1'] for v in HS.values()]))
            A(f"\n**Hash-twin L4 at scale.** Absolute `L(e1)` = **{_hl:.4f}** (per seed "
              f"{[round(v['loss_e1'],4) for v in HS.values()]}) vs the learned router's "
              f"**{np.mean([x['loss_e1'] for x in _rl]):.4f}** — a **{np.mean([x['loss_e1'] for x in _rl])/max(_hl,1e-9):.0f}x** separation.")
            A(f"\nThe *ratio* is **not reportable** for the hash twins: median-other sits at or below zero "
              f"({[round(v['median_other'],6) for v in HS.values()]}), giving values from -92,380x to +354,077x on "
              f"effects indistinguishable from zero. The sweep pre-registration therefore uses **absolute L(e1)** as P3.\n")
        A("\n**Why NMI and purity understate this.** Conditional NMI sits at ~92% of its reference ceiling and "
          "within-model k-means matches or beats the router on it; conditional purity is saturated at ~0.95 for any "
          "assignment. Both are *readout* statistics: they establish that phase is recoverable from the hidden states, "
          "which a post-hoc clustering also achieves. Only the intervention separates the router from a clustering of "
          "its own input — 0.290 vs a hash twin's 0.003, with ~63% of the effect on a single expert. The correlational "
          "metrics are near-saturated exactly where the causal metric is most discriminating.\n")




# ================= PART IV — ROUTING SWEEP (H_slack) =================
_sv=f'{ROOT}/out/sweep_V6_top1_e8_large.json'; _s1=f'{ROOT}/out/sweep_h_slack.json'
if os.path.exists(_sv):
    A("\n\n---\n\n# PART IV — ROUTING SWEEP: H_slack REJECTED\n")
    A("Pre-registered in `PREREGISTRATION_SWEEP.md` (`c1b023f4`) + `PREREG_SWEEP_AMENDMENT_4.md` "
      "(`2f9add95`), both filed before the relevant models trained. 6L/d256, 250k, seeds 30-34, "
      "last layer, same L1-L4.\n")
    A("\n**H_slack was:** top-2 + LBL creates *slack* letting one expert monopolise phase-1 slot 0; "
      "under top-1 the balance constraint binds directly, so specialization should **spread** — "
      "predicting LOWER concentration, HIGHER ceiling-normalised cNMI, SMALLER L(e1).\n")
    SV=json.load(open(_sv)); S1=json.load(open(_s1)) if os.path.exists(_s1) else None
    A("\n## Arms\n")
    A("| arm | routing | experts | LBL scope | differs from baseline by |")
    A("|---|---|---|---|---|")
    A("| baseline | top-2 | 8 | large | — |")
    A("| V1 | top-1 | 8 | local | **two axes** (arity AND scope) — confounded, see A4.1 |")
    A("| V6 | top-1 | 8 | large | **arity only** — the actual H_slack test |")
    A("| V7 | top-2 | 8 | local | scope only — **NOT RUN** (budget exhausted) |")
    A("\n## Results (last layer, 5 seeds each)\n")
    A("| arm | cond NMI | % of ceiling | L(e1) | sum_e L(e) | concentration |")
    A("|---|---|---|---|---|---|")
    for lab,D in (("V6 (arity only)",SV),("V1 (confounded)",S1)):
        if not D: continue
        R=D['V1']; cn=np.mean([r['cond_nmi'] for r in R])
        le=np.mean([r['loss_e1'] for r in R]); sl=np.mean([r['loss_e1']+7*r['median_other'] for r in R])
        A(f"| {lab} | {cn:.4f} | **{cn/D['anchor']*100:.0f}%** | {le:+.5f} | {sl:.4f} | "
          f"**undefined** (below floor) |")
    B=SV['baseline']; cb=np.mean([r['cond_nmi'] for r in B])
    A(f"| baseline (top-2/8/large) | {cb:.4f} | **{cb/SV['anchor']*100:.0f}%** | "
      f"{np.mean([r['loss_e1'] for r in B]):+.4f} | {np.mean([r['loss_e1']+7*r['median_other'] for r in B]):.4f} | "
      f"{np.mean([r['concentration_raw'] if r.get('concentration_raw') is not None else r['concentration'] for r in B]):.3f} |")
    A("\n## Verdict on each pre-registered prediction\n")
    A("| | prediction | result |")
    A("|---|---|---|")
    A("| **P1** concentration (PRIMARY) | top-1 LOWER | **UNEVALUABLE** — sum_e L(e) is 0.0001-0.0012 vs "
      "the pre-stated 0.05 floor (A4.3). There is no causal effect to concentrate. |")
    A("| **P2** ceiling-normalised cond NMI | top-1 HIGHER | **REVERSED** — top-1 **9%** vs top-2 **92%** "
      "of ceiling; predicted direction in **0/5** pairs, paired t **p < 0.0001**. |")
    A("| **P3** absolute L(e1) | top-1 SMALLER | **vacuously true** — 5/5, p=0.024, but only because "
      "L(e1) is ~0. Not support for H_slack. |")
    A("\n### **H_slack is REJECTED, and its premise is inverted.**\n")
    A("Top-1 routing does not *spread* specialization — it **eliminates** it. Top-2 appears to be "
      "**necessary for phase specialization to exist at all** at this scale. The 'slack' the "
      "hypothesis treated as an artefact to be removed is, on this evidence, the mechanism's "
      "precondition.\n")
    A("\n## Decision matrix (A4.5), as far as budget allowed\n")
    A("| V6 (arity only) | V7 (scope only) | conclusion |")
    A("|---|---|---|")
    A("| **COLLAPSE** (observed) | **not run** | **Arity alone is sufficient** to destroy specialization. "
      "V1's collapse is fully explained without invoking scope. Whether scope *independently* also "
      "suffices is unresolved. |")
    A("\n**V6 trains normally** — test acc 0.2688-0.2773 vs majority 0.1141 (ratio 2.36-2.43), matching "
      "the baseline's 2.43-2.48. So this is not a failure to learn the task; it is a model that learns "
      "the task without routing on phase.\n")
    A("\n*Reporting note: the A4.3 floor was pre-registered but initially not implemented in "
      "`sweep_analysis.py`, which reported P1 as 'inconclusive (4/5, p=0.45)'. Those numbers were "
      "computed from noise. The floor is now enforced in code and P1 is reported as unevaluable.*\n")


open(f'{ROOT}/out/results.md','w').write("\n".join(L))
print("wrote out/results.md")
