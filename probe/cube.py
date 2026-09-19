"""Minimal cubie-level 3x3 model + G1 membership test + facelet export for kociemba."""
import random

# corners: URF UFL ULB UBR DFR DLF DBL DRB ; edges: UR UF UL UB DR DF DL DB FR FL BL BR
MOVES = {
 'U': ([3,0,1,2,4,5,6,7],[0]*8,[3,0,1,2,4,5,6,7,8,9,10,11],[0]*12),
 'R': ([4,1,2,0,7,5,6,3],[2,0,0,1,1,0,0,2],[8,1,2,3,11,5,6,7,4,9,10,0],[0]*12),
 'F': ([1,5,2,3,0,4,6,7],[1,2,0,0,2,1,0,0],[0,9,2,3,4,8,6,7,1,5,10,11],[0,1,0,0,0,1,0,0,1,1,0,0]),
 'D': ([0,1,2,3,5,6,7,4],[0]*8,[0,1,2,3,5,6,7,4,8,9,10,11],[0]*12),
 'L': ([0,2,6,3,4,1,5,7],[0,1,2,0,0,2,1,0],[0,1,10,3,4,5,9,7,8,2,6,11],[0]*12),
 'B': ([0,1,3,7,4,5,2,6],[0,0,1,2,0,0,2,1],[0,1,2,11,4,5,6,10,8,9,3,7],[0,0,0,1,0,0,0,1,0,0,1,1]),
}
SOLVED = (tuple(range(8)), (0,)*8, tuple(range(12)), (0,)*12)

def mul(s, m):
    cp,co,ep,eo = s; mcp,mco,mep,meo = m
    ncp = tuple(cp[mcp[i]] for i in range(8))
    nco = tuple((co[mcp[i]] + mco[i]) % 3 for i in range(8))
    nep = tuple(ep[mep[i]] for i in range(12))
    neo = tuple((eo[mep[i]] + meo[i]) % 2 for i in range(12))
    return (ncp,nco,nep,neo)

def apply_move(s, tok):
    face, n = tok[0], (1 if len(tok)==1 else (2 if tok[1]=='2' else 3))
    for _ in range(n): s = mul(s, MOVES[face])
    return s

def apply_seq(s, seq):
    for t in seq.split(): s = apply_move(s, t)
    return s

def in_G1(s):
    cp,co,ep,eo = s
    return all(c==0 for c in co) and all(e==0 for e in eo) and all(ep[i]>=8 for i in (8,9,10,11))

G1_MOVES = {'U','U2',"U'",'D','D2',"D'",'R2','L2','F2','B2'}

# --- facelet export ---
U,R,F,D,L,B = 0,1,2,3,4,5
cF=[[8,9,20],[6,18,38],[0,36,47],[2,45,11],[29,26,15],[27,44,24],[33,53,42],[35,17,51]]
cC=[[U,R,F],[U,F,L],[U,L,B],[U,B,R],[D,F,R],[D,L,F],[D,B,L],[D,R,B]]
eF=[[5,10],[7,19],[3,37],[1,46],[32,16],[28,25],[30,43],[34,52],[23,12],[21,41],[50,39],[48,14]]
eC=[[U,R],[U,F],[U,L],[U,B],[D,R],[D,F],[D,L],[D,B],[F,R],[F,L],[B,L],[B,R]]

def to_facelets(s):
    cp,co,ep,eo = s
    f = [None]*54
    for i,c in enumerate('URFDLB'): f[i*9+4] = c
    for i in range(8):
        j, o = cp[i], co[i]
        for k in range(3): f[cF[i][(k+o)%3]] = 'URFDLB'[cC[j][k]]
    for i in range(12):
        j, o = ep[i], eo[i]
        for k in range(2): f[eF[i][(k+o)%2]] = 'URFDLB'[eC[j][k]]
    return ''.join(f)

def random_state(rng):
    """Uniform random *state* (not random scramble): random perms with parity + orientation fix."""
    cp = list(range(8)); rng.shuffle(cp)
    ep = list(range(12)); rng.shuffle(ep)
    def parity(p):
        n=0
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                if p[i]>p[j]: n^=1
        return n
    if parity(cp) != parity(ep): ep[0],ep[1] = ep[1],ep[0]
    co = [rng.randrange(3) for _ in range(7)]; co.append((-sum(co))%3)
    eo = [rng.randrange(2) for _ in range(11)]; eo.append((-sum(eo))%2)
    return (tuple(cp),tuple(co),tuple(ep),tuple(eo))
