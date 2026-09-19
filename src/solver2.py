"""Deterministic optimal-phase-1 + optimal-phase-2 solver, and the mixed start-state sampler."""
import sys
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube, phase1 as P1, phase2 as P2
FACE = lambda m: 'URFDLB'.index(m[0])

def solve_full(state):
    """Returns (maneuver, seam). Deterministic: no rng anywhere."""
    m1 = P1.solve_phase1_optimal(state)
    if m1 is None: return None
    g = state
    for m in m1: g = cube.apply_move(g, m)
    m2 = P2.solve_phase2(g, lastface=FACE(m1[-1]) if m1 else -1)
    if m2 is None: return None
    return m1 + m2, len(m1)

def random_G1_state(rng):
    cp=list(range(8)); rng.shuffle(cp)
    ud=list(range(8)); rng.shuffle(ud)
    sl=list(range(8,12)); rng.shuffle(sl)
    ep=ud+sl
    def par(p):
        n=0
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                if p[i]>p[j]: n^=1
        return n
    if par(cp)!=par(ep): ep[0],ep[1]=ep[1],ep[0]
    return (tuple(cp),(0,)*8,tuple(ep),(0,)*12)

def near_G1_state(rng, jmax=12):
    """Uniform random G1 state scrambled by j~U(0,jmax) moves, no consecutive same-face."""
    s = random_G1_state(rng); j = rng.randint(0, jmax); last=-1
    for _ in range(j):
        while True:
            m = rng.choice(P1.MOVE_NAMES)
            if FACE(m) != last: break
        s = cube.apply_move(s, m); last = FACE(m)
    return s

def sample_state(rng, near_frac=0.5, jmax=12):
    return near_G1_state(rng, jmax) if rng.random() < near_frac else cube.random_state(rng)
