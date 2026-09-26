"""CFOP data QA. Replaces the G1-monotonicity gate with segment-boundary verification,
per PREREGISTRATION_H2.md section 3."""
import sys, json, hashlib, collections
import numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
import cube, cfop
from qa import facelets_to_cubie, verify_state, MOVE_NAMES, COLORS
ROOT='/Users/alityb/projects/cubemoe'
HARD=[]
def hard(n,ok,d=''): HARD.append((n,bool(ok),d)); return ok

def main(kind='cfop', nsample=1500):
    d=dict(np.load(f'{ROOT}/data/{kind}.npz'))
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    n=len(d['sol_len']); card={'kind':kind,'n_solves':int(n),'n_steps':int(len(d['moves']))}
    bad_solve=bad_verify=bad_state=0; bnd_ok={k:0 for k in ('cross','F2L','OLL','PLL')}
    for i in range(n):
        mv=[MOVE_NAMES[m] for m in d['moves'][off[i]:off[i+1]]]
        cur=facelets_to_cubie(''.join(COLORS[c] for c in d['init_state'][i]))
        sts=[]
        for t,m in enumerate(mv):
            if verify_state(cur) is not None: bad_verify+=1
            if cube.to_facelets(cur)!=''.join(COLORS[c] for c in d['states'][off[i]+t]): bad_state+=1
            sts.append(cur); cur=cube.apply_move(cur,m)
        sts.append(cur)
        if cur!=cube.SOLVED: bad_solve+=1
        b=d['bounds'][i]
        bnd_ok['cross']+=cfop.cross_done(sts[b[0]]); bnd_ok['F2L']+=cfop.f2l_done(sts[b[1]])
        bnd_ok['OLL']+=cfop.oll_done(sts[b[2]]);   bnd_ok['PLL']+=cfop.pll_done(sts[b[3]])
    hard('L1 maneuver replays to solved', bad_solve==0, f'{bad_solve}/{n}')
    hard('L1 every state passes verify()', bad_verify==0, f'{bad_verify}')
    hard('L1 stored states == replayed states', bad_state==0, f'{bad_state}/{n}')
    for k in ('cross','F2L','OLL','PLL'):
        hard(f'L1b boundary predicate holds at end of {k} segment', bnd_ok[k]==n, f'{bnd_ok[k]}/{n}')
    dup=n-len({bytes(x) for x in d['init_state']})
    hard('L5 no duplicate scrambles', dup==0, f'{dup}')
    ph=d['phase'].astype(int); ps=d['pos'].astype(int); mv=d['moves'].astype(int)
    mix=sum((ps==t).sum() for t in np.unique(ps) if len(np.unique(ph[ps==t]))>1)/len(ph)
    hard('L5 >=9% of step cells contain >1 stage', mix>=0.09, f'{mix:.3f}')
    # REPORTED, not gated: state-only label conflict (CFOP is not state-Markovian)
    seen={}; conf=tot=0
    for i in range(min(n,3000)):
        cur=facelets_to_cubie(''.join(COLORS[c] for c in d['init_state'][i]))
        for t in range(off[i],off[i+1]):
            k=cube.to_facelets(cur); m=int(d['moves'][t])
            if k in seen:
                tot+=1; conf+= (seen[k]!=m)
            else: seen[k]=m
            cur=cube.apply_move(cur,MOVE_NAMES[m])
    card['state_conflict_rate']=conf/max(tot,1); card['state_conflict_n']=tot
    card['stage_share']={cfop.STAGE_NAMES[k]:float(np.mean(ph==k)) for k in (1,2,3,4)}
    card['mixed_step_cells']=float(mix)
    card['move_hist_by_stage']={cfop.STAGE_NAMES[k]:{MOVE_NAMES[i]:int(c) for i,c in
        enumerate(np.bincount(mv[ph==k],minlength=18))} for k in (1,2,3,4)}
    h=hashlib.sha256()
    for f in ['probe/cfop.py','probe/cfop_gen2.py','probe/cfop_canon.py','src/gen_cfop.py']:
        h.update(open(f'{ROOT}/{f}','rb').read())
    card['code_sha256']=h.hexdigest()[:16]
    card['hard_checks']=[dict(name=a,ok=b,detail=c) for a,b,c in HARD]
    card['PASS']=all(b for _,b,_ in HARD)
    json.dump(card,open(f'{ROOT}/out/datacard_{kind}.json','w'),indent=1)
    print(f"\n=== DATA CARD: {kind} ===")
    for a,b,c in HARD: print(f"  [{'PASS' if b else 'FAIL'}] {a}"+(f"  ({c})" if c and not b else ''))
    print(f"  stage share: "+", ".join(f"{k}={v:.3f}" for k,v in card['stage_share'].items()))
    print(f"  mixed step cells: {mix:.3f}")
    print(f"  state-only conflict (REPORTED, not gated): {card['state_conflict_rate']*100:.2f}% of {tot} repeats")
    print(f"  VERDICT: {'PASS' if card['PASS'] else 'FAIL'}")
    return 0 if card['PASS'] else 1
if __name__=='__main__': sys.exit(main())
