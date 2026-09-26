"""Canonical case -> solving-maneuver tables, so the CFOP policy is deterministic.

For a maneuver A applied to solved giving case C, A^-1 solves C. We enumerate compositions,
keep the FIRST (shortest, then lexicographic) maneuver per case, and store case -> A^-1.
Generation then picks a CASE and uses its canonical solution — never a random algorithm.
"""
import sys, itertools, pickle, os
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube, cfop

MOVES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
INV={m:(m[0] if m.endswith("'") else (m if m.endswith('2') else m+"'")) for m in MOVES}
def inv(seq): return [INV[m] for m in reversed(seq)]

OLL_BASE=["R U R' U R U2 R'", "R U2 R' U' R U' R'", "F R U R' U' F'"]
PLL_BASE=["R U R' U' R' F R2 U' R' U' R U R' F'", "R U' R U R U R U' R' U' R2",
          "R' F R' B2 R F' R' B2 R2", "R U R' F' R U R' U' R' F R2 U' R'"]
AUF=["","U","U'","U2"]

def llkey(s):
    cp,co,ep,eo=s
    return (tuple(cp[:4]),tuple(co[:4]),tuple(ep[:4]),tuple(eo[:4]))

def build_tables(max_depth=2):
    S=cube.SOLVED
    oll={}; pll={}
    for depth in (1,2):
        if depth>max_depth: break
        for combo in itertools.product(OLL_BASE,repeat=depth):
            for us in itertools.product(AUF,repeat=depth):
                mv=[]; s=S
                for a,u in zip(combo,us):
                    seq=(u+" "+a).split(); mv+=seq
                    for m in seq: s=cube.apply_move(s,m)
                if cfop.f2l_done(s) and not cfop.oll_done(s):
                    k=llkey(s)
                    if k not in oll or len(mv)<len(oll[k][0]): oll[k]=(mv,inv(mv))
        for combo in itertools.product(PLL_BASE,repeat=depth):
            for us in itertools.product(AUF,repeat=depth):
                mv=[]; s=S
                for a,u in zip(combo,us):
                    seq=(u+" "+a).split(); mv+=seq
                    for m in seq: s=cube.apply_move(s,m)
                if cfop.oll_done(s) and not cfop.pll_done(s):
                    k=llkey(s)
                    if k not in pll or len(mv)<len(pll[k][0]): pll[k]=(mv,inv(mv))
    return oll, pll

def build_f2l_canon(lib):
    """slot -> {case_key: (extraction, insertion)} keyed by the pair's own configuration."""
    S=cube.SOLVED; out={}
    for k in range(4):
        c,e=cfop.SLOTS[k]; tab={}
        for mv in lib[k]:
            s=S
            for m in mv: s=cube.apply_move(s,m)
            cp,co,ep,eo=s
            key=(cp.index(c),co[cp.index(c)],ep.index(e),eo[ep.index(e)])
            if key not in tab or len(mv)<len(tab[key][0]): tab[key]=(list(mv),inv(list(mv)))
        out[k]=tab
    return out

if __name__=='__main__':
    import time
    t0=time.time(); oll,pll=build_tables()
    lib=pickle.load(open('/Users/alityb/projects/cubemoe/data/cfop_extlib.pkl','rb'))
    f2l=build_f2l_canon(lib)
    print(f"built in {time.time()-t0:.1f}s")
    print(f"  canonical OLL cases: {len(oll)}   mean alg len {sum(len(v[0]) for v in oll.values())/len(oll):.1f}")
    print(f"  canonical PLL cases: {len(pll)}   mean alg len {sum(len(v[0]) for v in pll.values())/len(pll):.1f}")
    for k in range(4): print(f"  F2L slot {k}: {len(f2l[k])} distinct pair-configs")
    pickle.dump(dict(oll=oll,pll=pll,f2l=f2l), open('/Users/alityb/projects/cubemoe/data/cfop_canon.pkl','wb'))
    print("  -> data/cfop_canon.pkl")
