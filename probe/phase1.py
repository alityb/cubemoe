"""Own phase-1 solver with FORCED maneuver length, using only the three small
Kociemba phase-1 coordinates. Avoids hkociemba's huge phase1_prun table."""
import cube, time
from collections import deque

MOVE_NAMES = [f+s for f in 'URFDLB' for s in ('',"'",'2')]   # 18 moves
NM = len(MOVE_NAMES)
N_TWIST, N_FLIP, N_SLICE = 2187, 2048, 495

# ---- combination index for which 4 of 12 edge slots hold UD-slice edges ----
from itertools import combinations
COMBS = list(combinations(range(12), 4))
COMB_IDX = {c: i for i, c in enumerate(COMBS)}

def _state_from_twist(t):
    co = []
    for _ in range(7): co.append(t % 3); t //= 3
    co.append((-sum(co)) % 3)
    return (tuple(range(8)), tuple(co), tuple(range(12)), (0,)*12)
def _twist_of(s): return sum(s[1][i]*3**i for i in range(7))

def _state_from_flip(f):
    eo = []
    for _ in range(11): eo.append(f % 2); f //= 2
    eo.append((-sum(eo)) % 2)
    return (tuple(range(8)), (0,)*8, tuple(range(12)), tuple(eo))
def _flip_of(s): return sum(s[3][i]*2**i for i in range(11))

def _state_from_slice(idx):
    pos = COMBS[idx]; ep = [None]*12
    sl = iter([8,9,10,11]); nonsl = iter([0,1,2,3,4,5,6,7])
    for i in range(12): ep[i] = next(sl) if i in pos else next(nonsl)
    return (tuple(range(8)), (0,)*8, tuple(ep), (0,)*12)
def _slice_of(s): return COMB_IDX[tuple(i for i in range(12) if s[2][i] >= 8)]

def build():
    t0 = time.time()
    mt = [[0]*NM for _ in range(N_TWIST)]
    mf = [[0]*NM for _ in range(N_FLIP)]
    ms = [[0]*NM for _ in range(N_SLICE)]
    for c in range(N_TWIST):
        s = _state_from_twist(c)
        for m, nm in enumerate(MOVE_NAMES): mt[c][m] = _twist_of(cube.apply_move(s, nm))
    for c in range(N_FLIP):
        s = _state_from_flip(c)
        for m, nm in enumerate(MOVE_NAMES): mf[c][m] = _flip_of(cube.apply_move(s, nm))
    for c in range(N_SLICE):
        s = _state_from_slice(c)
        for m, nm in enumerate(MOVE_NAMES): ms[c][m] = _slice_of(cube.apply_move(s, nm))
    print(f"  move tables: {time.time()-t0:.1f}s")

    def bfs(nA, mA, nS, mS):
        n = nA*nS; d = bytearray([255])*n
        goal = 0*nS + _slice_of(cube.SOLVED)
        d[goal] = 0; q = deque([goal])
        while q:
            x = q.popleft(); a, sl = divmod(x, nS); dv = d[x]
            for m in range(NM):
                y = mA[a][m]*nS + mS[sl][m]
                if d[y] == 255: d[y] = dv+1; q.append(y)
        return d
    t1 = time.time(); pt = bfs(N_TWIST, mt, N_SLICE, ms); print(f"  twist-slice prun: {time.time()-t1:.1f}s")
    t1 = time.time(); pf = bfs(N_FLIP,  mf, N_SLICE, ms); print(f"  flip-slice prun:  {time.time()-t1:.1f}s")
    print(f"  TOTAL BUILD: {time.time()-t0:.1f}s")
    return mt, mf, ms, pt, pf

MT, MF, MS, PT, PF = build()

import random as _random
def solve_phase1(state, L, rng=None, lastface0=-1):
    """Find a phase-1 maneuver of EXACTLY length L (or None). Returns list of move names.
    rng: if given, the move order is shuffled per call so the returned maneuver is a
    uniformly tie-broken solution rather than the fixed-move-order-first one."""
    tw, fl, sl = _twist_of(state), _flip_of(state), _slice_of(state)
    base = list(range(NM))
    path = []
    def rec(tw, fl, sl, togo, lastface):
        if togo == 0: return tw == 0 and fl == 0 and sl == _slice_of(cube.SOLVED)
        if PT[tw*N_SLICE+sl] > togo or PF[fl*N_SLICE+sl] > togo: return False
        if rng is None:
            order = base
        else:
            order = base[:]; rng.shuffle(order)      # fresh order at EVERY node
        for m in order:
            if m//3 == lastface: continue
            path.append(m)
            if rec(MT[tw][m], MF[fl][m], MS[sl][m], togo-1, m//3): return True
            path.pop()
        return False
    return [MOVE_NAMES[m] for m in path] if rec(tw, fl, sl, L, lastface0) else None


def lower_bound(state):
    """Admissible lower bound on phase-1 length from the two pruning tables."""
    tw, fl, sl = _twist_of(state), _flip_of(state), _slice_of(state)
    return max(PT[tw*N_SLICE+sl], PF[fl*N_SLICE+sl])

def solve_phase1_optimal(state, lastface=-1, maxextra=6):
    """Shortest phase-1 maneuver, deterministic (fixed DFS move order, no rng)."""
    lb = lower_bound(state)
    for L in range(lb, lb+maxextra+1):
        r = solve_phase1(state, L, rng=None, lastface0=lastface)
        if r is not None: return r
    return None
