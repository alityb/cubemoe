"""CFOP stage predicates on the cubie model.

Indexing (from probe/cube.py):
  corners 0..7 = URF UFL ULB UBR DFR DLF DBL DRB
  edges   0..11 = UR UF UL UB DR DF DL DB FR FL BL BR

Stages:
  1 CROSS : the four D-layer edges (DR,DF,DL,DB = 4,5,6,7) placed and oriented
  2 F2L   : cross + four corner/edge slots  (DFR+FR, DLF+FL, DBL+BL, DRB+BR)
  3 OLL   : F2L + all U-layer pieces ORIENTED (corner twist 0, edge flip 0)
  4 PLL   : OLL + U-layer pieces PERMUTED  == solved
"""
D_EDGES = (4, 5, 6, 7)                    # DR DF DL DB
SLOTS   = ((4, 8), (5, 9), (6, 10), (7, 11))   # (corner, edge): DFR+FR, DLF+FL, DBL+BL, DRB+BR
U_CORNERS = (0, 1, 2, 3)
U_EDGES   = (0, 1, 2, 3)

def cross_done(s):
    cp, co, ep, eo = s
    return all(ep[i] == i and eo[i] == 0 for i in D_EDGES)

def slot_done(s, k):
    cp, co, ep, eo = s
    c, e = SLOTS[k]
    return cp[c] == c and co[c] == 0 and ep[e] == e and eo[e] == 0

def f2l_done(s):
    return cross_done(s) and all(slot_done(s, k) for k in range(4))

def oll_done(s):
    cp, co, ep, eo = s
    return f2l_done(s) and all(co[i] == 0 for i in U_CORNERS) and all(eo[i] == 0 for i in U_EDGES)

def pll_done(s):
    cp, co, ep, eo = s
    return oll_done(s) and all(cp[i] == i for i in U_CORNERS) and all(ep[i] == i for i in U_EDGES)

def stage_of(s):
    """Which stage is IN PROGRESS at this state. 1=cross 2=F2L 3=OLL 4=PLL 5=solved."""
    if not cross_done(s): return 1
    if not f2l_done(s):   return 2
    if not oll_done(s):   return 3
    if not pll_done(s):   return 4
    return 5

def n_slots_done(s):
    return sum(slot_done(s, k) for k in range(4)) if cross_done(s) else 0

STAGE_NAMES = {1: "cross", 2: "F2L", 3: "OLL", 4: "PLL", 5: "solved"}
