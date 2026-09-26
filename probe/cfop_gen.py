"""CFOP solve generator by REVERSE CONSTRUCTION.

A scramble is built from solved by undoing each stage in reverse order:
    solved -> PLL-case -> OLL-case -> extract pairs 3,2,1,0 -> scramble cross
The forward solution is then known exactly, and so are the stage boundaries.
Everything is verified against probe/cfop.py predicates rather than assumed.
"""
import sys, random, itertools
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube, cfop

MOVES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
INV={m:(m[0] if m.endswith("'") else (m if m.endswith('2') else m+"'")) for m in MOVES}
def inv(seq): return [INV[m] for m in reversed(seq)]

OLL_BASE=["R U R' U R U2 R'", "R U2 R' U' R U' R'", "F R U R' U' F'"]
PLL_BASE=["R U R' U' R' F R2 U' R' U' R U R' F'", "R U' R U R U R U' R' U' R2",
          "R' F R' B2 R F' R' B2 R2", "R U R' F' R U R' U' R' F R2 U' R'"]
AUF=["","U","U'","U2"]

def _extraction_library(maxlen=5):
    """For each slot k: maneuvers that break ONLY slot k, keeping cross and slots <k."""
    lib={k:[] for k in range(4)}
    S=cube.SOLVED
    for k in range(4):
        keep=[j for j in range(4) if j<k]
        for L in range(2, maxlen+1):
            for combo in itertools.product(MOVES, repeat=L):
                last=-1; ok=True
                for m in combo:
                    f='URFDLB'.index(m[0])
                    if f==last: ok=False; break
                    last=f
                if not ok: continue
                s=S
                for m in combo: s=cube.apply_move(s,m)
                if cfop.cross_done(s) and not cfop.slot_done(s,k) and all(cfop.slot_done(s,j) for j in keep):
                    lib[k].append(list(combo))
            if len(lib[k])>=400: break
    return lib

_LIB=None
def library():
    global _LIB
    if _LIB is None: _LIB=_extraction_library()
    return _LIB

def generate(rng):
    """Returns (scrambled_state, solution_moves, stage_labels) with stage_labels[i]
    the stage IN PROGRESS when move i is played (1=cross 2=F2L 3=OLL 4=PLL)."""
    lib=library()
    s=cube.SOLVED
    pll_gen=[]; oll_gen=[]; ext=[[] for _ in range(4)]
    # PLL case
    for _ in range(rng.randint(1,2)):
        a=(rng.choice(AUF)+" "+rng.choice(PLL_BASE)).split()
        pll_gen+=a
        for m in a: s=cube.apply_move(s,m)
    # OLL case
    for _ in range(rng.randint(1,2)):
        a=(rng.choice(AUF)+" "+rng.choice(OLL_BASE)).split()
        oll_gen+=a
        for m in a: s=cube.apply_move(s,m)
    # extract pairs 3,2,1,0
    for k in (3,2,1,0):
        cand=[c for c in lib[k] if True]
        a=list(rng.choice(cand)); ext[k]=a
        for m in a: s=cube.apply_move(s,m)
    # scramble the cross
    cross_scr=[]
    last=-1
    for _ in range(rng.randint(4,8)):
        while True:
            m=rng.choice(MOVES)
            if 'URFDLB'.index(m[0])!=last: break
        cross_scr.append(m); last='URFDLB'.index(m[0]); s=cube.apply_move(s,m)
    # forward solution = inverse of the construction, in reverse order.
    # Stage labels are CONSTRUCTION-DERIVED (which segment a move belongs to), because for CFOP
    # stage membership is NOT a monotone state property: insertions and OLL/PLL algorithms
    # temporarily break earlier stages and restore them. Boundaries are verified below.
    segs=[("cross",inv(cross_scr)),("F2L",inv(ext[0])+inv(ext[1])+inv(ext[2])+inv(ext[3])),
          ("OLL",inv(oll_gen)),("PLL",inv(pll_gen))]
    sol=[]; lab=[]; bounds={}
    code={"cross":1,"F2L":2,"OLL":3,"PLL":4}
    for nm,mv in segs:
        sol+=mv; lab+=[code[nm]]*len(mv); bounds[nm]=len(sol)
    return s, sol, lab, bounds

def verify(state, sol, lab, bounds):
    """Boundary check: the defining predicate must hold at the END of each segment."""
    cur=state; states=[]
    for m in sol: states.append(cur); cur=cube.apply_move(cur,m)
    states.append(cur)
    checks={
      "cross": cfop.cross_done(states[bounds["cross"]]),
      "F2L":   cfop.f2l_done(states[bounds["F2L"]]),
      "OLL":   cfop.oll_done(states[bounds["OLL"]]),
      "PLL":   cfop.pll_done(states[bounds["PLL"]]),
    }
    return checks, cur, states

def labels_along(state, sol):
    """Stage in progress at each move, computed from the STATES (not assumed)."""
    out=[]; cur=state
    for m in sol:
        out.append(cfop.stage_of(cur))
        cur=cube.apply_move(cur,m)
    return out, cur
