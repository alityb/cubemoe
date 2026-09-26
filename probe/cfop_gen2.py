"""Canonical CFOP generator: every case is solved by its ONE canonical maneuver,
so the policy is a deterministic function of (case), not of an rng draw."""
import sys, random, pickle
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube, cfop
from cfop_canon import llkey, inv, MOVES

_T=None
def tables():
    global _T
    if _T is None: _T=pickle.load(open('/Users/alityb/projects/cubemoe/data/cfop_canon.pkl','rb'))
    return _T

def generate(rng, cross_len=(4,8)):
    T=tables(); S=cube.SOLVED; s=S
    # pick a PLL case and an OLL case, apply their SETUP maneuvers (canonical, not random)
    pk=rng.choice(list(T['pll'].keys())); pll_setup=T['pll'][pk][0]
    for m in pll_setup: s=cube.apply_move(s,m)
    ok_=rng.choice(list(T['oll'].keys())); oll_setup=T['oll'][ok_][0]
    for m in oll_setup: s=cube.apply_move(s,m)
    ext={}
    for k in (3,2,1,0):
        ck=rng.choice(list(T['f2l'][k].keys())); mv=T['f2l'][k][ck][0]
        ext[k]=mv
        for m in mv: s=cube.apply_move(s,m)
    cross=[]; last=-1
    for _ in range(rng.randint(*cross_len)):
        while True:
            m=rng.choice(MOVES)
            if 'URFDLB'.index(m[0])!=last: break
        cross.append(m); last='URFDLB'.index(m[0]); s=cube.apply_move(s,m)
    segs=[("cross",inv(cross)),
          ("F2L",inv(ext[0])+inv(ext[1])+inv(ext[2])+inv(ext[3])),
          ("OLL",inv(oll_setup)),("PLL",inv(pll_setup))]
    code={"cross":1,"F2L":2,"OLL":3,"PLL":4}
    sol=[]; lab=[]; bounds={}
    for nm,mv in segs:
        sol+=mv; lab+=[code[nm]]*len(mv); bounds[nm]=len(sol)
    return s, sol, lab, bounds
