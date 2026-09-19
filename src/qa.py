"""STEP 1.5 - data QA. Five layers. Refuses to pass a file that fails a hard invariant."""
import sys, os, json, hashlib, argparse, random, collections
import numpy as np
sys.path.insert(0, '/Users/alityb/projects/cubemoe/probe')
import cube
ROOT = '/Users/alityb/projects/cubemoe'
MOVE_NAMES = [f+s for f in 'URFDLB' for s in ('',"'",'2')]
COLORS = 'URFDLB'
G1_SET = set(cube.G1_MOVES)
HARD = []          # (name, ok, detail)
SOFT = {}

def hard(name, ok, detail=''):
    HARD.append((name, bool(ok), detail))
    return ok

# ---------- independent facelet -> cubie decoder (reverse of cube.to_facelets) ----------
def facelets_to_cubie(f):
    cp=[0]*8; co=[0]*8; ep=[0]*12; eo=[0]*12
    cC=[tuple(x) for x in cube.cC]; eC=[tuple(x) for x in cube.eC]
    for i in range(8):
        for ori in range(3):
            c0 = COLORS.index(f[cube.cF[i][(0+ori)%3]]); c1 = COLORS.index(f[cube.cF[i][(1+ori)%3]])
            for j,(a,b,c) in enumerate(cC):
                if (a,b)==(c0,c1): cp[i]=j; co[i]=ori; break
            else: continue
            break
        else: raise ValueError(f'bad corner {i}')
    for i in range(12):
        for ori in range(2):
            c0 = COLORS.index(f[cube.eF[i][(0+ori)%2]]); c1 = COLORS.index(f[cube.eF[i][(1+ori)%2]])
            for j,(a,b) in enumerate(eC):
                if (a,b)==(c0,c1): ep[i]=j; eo[i]=ori; break
            else: continue
            break
        else: raise ValueError(f'bad edge {i}')
    return (tuple(cp),tuple(co),tuple(ep),tuple(eo))

def verify_state(s):
    cp,co,ep,eo = s
    if sorted(cp)!=list(range(8)) or sorted(ep)!=list(range(12)): return 'perm'
    if sum(co)%3: return 'twist'
    if sum(eo)%2: return 'flip'
    def par(p):
        n=0
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                if p[i]>p[j]: n^=1
        return n
    if par(list(cp))!=par(list(ep)): return 'parity'
    return None

def ngram_entropy(seqs, n):
    c=collections.Counter()
    for s in seqs:
        for i in range(len(s)-n+1): c[tuple(s[i:i+n])]+=1
    tot=sum(c.values())
    if tot==0: return 0.0
    p=np.array(list(c.values()),float)/tot
    return float(-(p*np.log2(p)).sum())

def bayes(keys,label):
    ks=np.stack(keys,1); _,inv=np.unique(ks,axis=0,return_inverse=True)
    B=label.max()+1; ct=np.bincount(inv*B+label,minlength=(inv.max()+1)*B).reshape(-1,B)
    return float(ct.max(1).sum()/ct.sum())

def main(kind, nsample=2000, seed=0):
    d = dict(np.load(f'{ROOT}/data/{kind}.npz'))
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    nsolve = len(d['seam']); rng = random.Random(seed)
    card = dict(kind=kind, n_solves=int(nsolve), n_steps=int(len(d['moves'])))

    # ---------------- LAYER 1: hard invariants, every solve ----------------
    bad_solved=bad_verify=bad_p2=bad_mono=bad_seam=bad_state=0
    seam_vs_g1=[]
    for i in range(nsolve):
        mv=[MOVE_NAMES[m] for m in d['moves'][off[i]:off[i+1]]]
        s=facelets_to_cubie(''.join(COLORS[c] for c in d['init_state'][i]))
        cur=s; g1=[]; ok_state=True
        for t,m in enumerate(mv):
            if verify_state(cur) is not None: bad_verify+=1
            g1.append(cube.in_G1(cur))
            stored=''.join(COLORS[c] for c in d['states'][off[i]+t])
            if cube.to_facelets(cur)!=stored: ok_state=False
            cur=cube.apply_move(cur,m)
        g1.append(cube.in_G1(cur))
        if not ok_state: bad_state+=1
        if cur!=cube.SOLVED: bad_solved+=1
        sm=int(d['seam'][i])
        if not all(x in G1_SET for x in mv[sm:]): bad_p2+=1
        first=g1.index(True) if True in g1 else -1
        if first<0 or not all(g1[first:]): bad_mono+=1
        if first!=sm: bad_seam+=1
        seam_vs_g1.append(first-sm)
    hard('L1 maneuver replays to solved', bad_solved==0, f'{bad_solved}/{nsolve} fail')
    hard('L1 every state passes verify() (perm/twist/flip/parity)', bad_verify==0, f'{bad_verify} bad states')
    hard('L1 stored states == replayed states', bad_state==0, f'{bad_state}/{nsolve} mismatch')
    hard('L1 phase-2 moves subset of the 10 G1 moves', bad_p2==0, f'{bad_p2}/{nsolve} fail')
    hard('L1 in_G1 monotone (first True stays True)', bad_mono==0, f'{bad_mono}/{nsolve} fail')
    hard('L1 stored seam == first G1 entry (phase is a STATE property)', bad_seam==0,
         f'{bad_seam}/{nsolve} disagree; offset dist {collections.Counter(seam_vs_g1).most_common(5)}')
    card['seam_minus_firstG1'] = dict(collections.Counter(int(x) for x in seam_vs_g1).most_common())

    # ---------------- LAYER 2: independent verification ----------------
    import kociemba
    idx = rng.sample(range(nsolve), min(nsample, nsolve))
    invalid=0; lastmove_bad=0; g1_disagree=0
    sys.path.insert(0, f'{ROOT}/probe'); import phase1 as P1
    for i in idx:
        t = rng.randrange(off[i], off[i+1])
        fs = ''.join(COLORS[c] for c in d['states'][t])
        try: kociemba.solve(fs)
        except Exception: invalid+=1
        st = facelets_to_cubie(fs)
        coord_g1 = (P1._twist_of(st)==0 and P1._flip_of(st)==0 and P1._slice_of(st)==P1._slice_of(cube.SOLVED))
        if coord_g1 != cube.in_G1(st): g1_disagree+=1
        lastf = ''.join(COLORS[c] for c in d['states'][off[i+1]-1])
        sol = kociemba.solve(lastf).split()
        if len(sol)!=1 or sol[0]!=MOVE_NAMES[d['moves'][off[i+1]-1]]: lastmove_bad+=1
    hard('L2 kociemba accepts every sampled state (independent legality)', invalid==0, f'{invalid}/{len(idx)} rejected')
    hard('L2 kociemba solves the penultimate state in exactly our last move', lastmove_bad==0,
         f'{lastmove_bad}/{len(idx)} mismatch')
    hard('L2 cubie G1 test == coordinate-table G1 test', g1_disagree==0, f'{g1_disagree}/{len(idx)} disagree')
    card['L2_sampled'] = len(idx)

    # ---- L2b: LABEL DETERMINISM (mixed set only; stock kociemba is a different solver) ----
    if kind == 'mixed':
        sys.path.insert(0, f'{ROOT}/src'); import solver2, phase2 as P2
        FACE = solver2.FACE
        def first_ctx(st, lf):
            m1 = P1.solve_phase1_optimal(st, lastface=lf)
            if m1: return m1[0]
            mm = P2.solve_phase2(st, lastface=lf); return mm[0] if mm else None
        ok_plain=ok_ctx=ntot=0
        for i in rng.sample(range(nsolve), min(1000, nsolve)):
            mv=[MOVE_NAMES[m] for m in d['moves'][off[i]:off[i+1]]]
            cur=facelets_to_cubie(''.join(COLORS[c] for c in d['init_state'][i])); sts=[]
            for m in mv: sts.append(cur); cur=cube.apply_move(cur,m)
            for t in rng.sample(range(len(mv)), min(2, len(mv))):
                ntot+=1
                rr=solver2.solve_full(sts[t])
                if rr and rr[0][0]==mv[t]: ok_plain+=1
                lf = FACE(mv[t-1]) if t>0 else -1
                if first_ctx(sts[t], lf)==mv[t]: ok_ctx+=1
        card['label_determinism']=dict(n=ntot, standalone=ok_plain/max(ntot,1), with_context=ok_ctx/max(ntot,1))
        hard('L2b label determinism (standalone re-solve) >= 0.95', ok_plain/max(ntot,1) >= 0.95,
             f'{ok_plain/max(ntot,1):.4f}')
        hard('L2b label determinism WITH lastface context == 1.0', ok_ctx==ntot,
             f'{ok_ctx}/{ntot}; gap is the consecutive-same-face rule at the split')

    # ---------------- LAYER 3: DFS stereotypy ----------------
    p1s=[]; 
    for i in range(nsolve):
        sm=int(d['seam'][i]); p1s.append([int(x) for x in d['moves'][off[i]:off[i]+sm]])
    pref={}
    for k in (3,4,5):
        c=collections.Counter(tuple(s[:k]) for s in p1s if len(s)>=k)
        pref[k]=float(1-len(c)/max(sum(c.values()),1))
    lag2_rep=lag2_tot=0
    for sq in p1s:
        for t in range(2,len(sq)): lag2_tot+=1; lag2_rep+=(sq[t]==sq[t-2])
    lag2=lag2_rep/max(lag2_tot,1)
    card['stereotypy']=dict(lag2_repeat=float(lag2),
        unigram_entropy=ngram_entropy(p1s,1), bigram_entropy=ngram_entropy(p1s,2),
        trigram_entropy=ngram_entropy(p1s,3), repeated_prefix_rate=pref,
        n_distinct_phase1=len({tuple(s) for s in p1s}), n_phase1=len(p1s))
    if kind!='naive': hard('L3 phase-1 lag-2 repetition near chance (no cyclic DFS padding)', lag2 <= 0.10,
         f'P(move[t]==move[t-2])={lag2:.3f} (chance 0.056); >0.10 means the forced search is '
         f'burning excess length as a repeating short cycle')
    if kind!='naive': hard('L3 repeated 5-prefix rate low (paths not stereotyped)', pref[5] <= 0.15, f'{pref[5]:.3f}')
    # phase-2 stereotypy (phase-2 DFS uses a FIXED move order; compare forced vs stock)
    p2s=[[int(x) for x in d['moves'][off[i]+int(d['seam'][i]):off[i+1]]] for i in range(nsolve)]
    r2=t2=0
    for sq in p2s:
        for t in range(2,len(sq)): t2+=1; r2+=(sq[t]==sq[t-2])
    c2=np.bincount(np.concatenate([np.array(x,int) for x in p2s if x]),minlength=18).astype(float)
    c2/=max(c2.sum(),1); nz2=c2[c2>0]
    card['stereotypy_phase2']=dict(lag2_repeat=float(r2/max(t2,1)),
        entropy=float(-(nz2*np.log2(nz2)).sum()),
        distinct_5prefix=len({tuple(x[:5]) for x in p2s if len(x)>=5}),
        n=len([x for x in p2s if len(x)>=5]))
    ph=d['phase'].astype(int)-1; mv=d['moves'].astype(int); ps=d['pos'].astype(int)
    card['confound']=dict(majority=float(max(ph.mean(),1-ph.mean())),
                          move=bayes([mv],ph), position=bayes([ps],ph), pos_move=bayes([ps,mv],ph))

    # ---------------- LAYER 4: tail leakage ----------------
    perm=np.random.RandomState(0).permutation(nsolve); tr=set(perm[:int(nsolve*0.9)].tolist())
    trmask=np.isin(d['solve_id'],list(tr)); temask=~trmask
    def keyset(mask):
        a=d['states'][mask]
        return a.astype(np.uint8).tobytes(), a
    trset=set(map(bytes, d['states'][trmask]))
    te_states=d['states'][temask]; te_rem=d['remaining'][temask].astype(int)
    leak=np.array([bytes(x) in trset for x in te_states])
    card['leakage']=dict(overall=float(leak.mean()),
        by_remaining={int(r): float(leak[te_rem==r].mean()) for r in sorted(set(te_rem.tolist()))[:10]},
        excl_last={k: float(leak[te_rem>k].mean()) for k in (3,4,5,6,7,8)},
        rows_kept={k: float((te_rem>k).mean()) for k in (3,4,5,6,7,8)})
    card['leakage']['excl_last3']=card['leakage']['excl_last'][3]
    dup = len(d['init_state']) - len({bytes(x) for x in d['init_state']})
    hard('L5 no duplicate scrambles', dup==0, f'{dup} duplicates')

    # ---------------- LAYER 5: distribution ----------------
    mixed=sum((ps==t).sum() for t in np.unique(ps) if len(np.unique(ph[ps==t]))==2)/len(ph)
    card['distribution']=dict(seam_mean=float(d['seam'].mean()), seam_sd=float(d['seam'].std()),
        seam_hist={int(k):int(v) for k,v in zip(*np.unique(d['seam'],return_counts=True))},
        sollen_mean=float(d['sol_len'].mean()), phase1_frac=float((ph==0).mean()),
        mixed_position_cells=float(mixed),
        move_hist_phase1={MOVE_NAMES[i]:int(c) for i,c in enumerate(np.bincount(mv[ph==0],minlength=18))},
        move_hist_phase2={MOVE_NAMES[i]:int(c) for i,c in enumerate(np.bincount(mv[ph==1],minlength=18))})
    if kind in ('forced','mixed'):
        hard('L5 seam sd >= 3.0', d['seam'].std()>=3.0, f"sd={d['seam'].std():.2f}")
        hard('L5 >=9% of position cells mixed', mixed>=0.09, f'{mixed:.3f}')
    h=hashlib.sha256()
    for f in ['src/gen_data.py','probe/phase1.py','probe/phase2.py','probe/cube.py']:
        h.update(open(f'{ROOT}/{f}','rb').read())
    card['code_sha256']=h.hexdigest()[:16]; card['gen_seeds']='1000..1009 (worker i -> 1000+i)'
    card['hard_checks']=[dict(name=n,ok=o,detail=dt) for n,o,dt in HARD]
    card['PASS']=all(o for _,o,_ in HARD)
    json.dump(card, open(f'{ROOT}/out/datacard_{kind}.json','w'), indent=1)
    print(f"\n=== DATA CARD: {kind} ===")
    for n,o,dt in HARD: print(f"  [{'PASS' if o else 'FAIL'}] {n}" + (f"   ({dt})" if dt and not o else ''))
    print(f"  seam sd={card['distribution']['seam_sd']:.2f} mixed={mixed:.3f} "
          f"phase1={card['distribution']['phase1_frac']:.3f}")
    st=card['stereotypy']
    print(f"  stereotypy: H1={st['unigram_entropy']:.2f} H2={st['bigram_entropy']:.2f} H3={st['trigram_entropy']:.2f} "
          f"distinct phase1={st['n_distinct_phase1']}/{st['n_phase1']} "
          f"prefix-repeat k=5:{st['repeated_prefix_rate'][5]:.3f} lag2={st['lag2_repeat']:.3f}")
    print(f"  confound: move={card['confound']['move']:.3f} pos={card['confound']['position']:.3f} maj={card['confound']['majority']:.3f}")
    print("  leakage: overall=%.4f | "%card['leakage']['overall'] +
          " ".join(f"excl-last-{k}={v:.4f}(keep {card['leakage']['rows_kept'][k]:.2f})"
                   for k,v in card['leakage']['excl_last'].items()))
    print(f"  VERDICT: {'PASS' if card['PASS'] else 'FAIL'}")
    return 0 if card['PASS'] else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',required=True); ap.add_argument('--nsample',type=int,default=2000)
    a=ap.parse_args(); sys.exit(main(a.kind,a.nsample))
