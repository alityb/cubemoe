"""Phase-2 solver: optimal within G1 = <U,D,R2,L2,F2,B2>."""
import cube, time
from collections import deque
G1 = ['U',"U'",'U2','D',"D'",'D2','R2','L2','F2','B2']
NM = 10
FACE_OF = ['URFDLB'.index(m[0]) for m in G1]   # consistent with phase1's m//3
N_CP, N_UD, N_SL = 40320, 40320, 24

def perm_idx(p):
    n=len(p); idx=0
    for i in range(n):
        c=sum(1 for j in range(i+1,n) if p[j]<p[i]); idx = idx*(n-i)+c
    return idx
def idx_perm(idx,n):
    elems=list(range(n)); p=[]
    for i in range(n):
        f=1
        for k in range(1,n-i): f*=k
        c,idx = divmod(idx,f); p.append(elems.pop(c))
    return p
def cp_of(s): return perm_idx(list(s[0]))
def ud_of(s): return perm_idx(list(s[2][:8]))
def sl_of(s): return perm_idx([x-8 for x in s[2][8:]])

def build():
    t0=time.time()
    mcp=[[0]*NM for _ in range(N_CP)]; mud=[[0]*NM for _ in range(N_UD)]; msl=[[0]*NM for _ in range(N_SL)]
    for c in range(N_CP):
        s=(tuple(idx_perm(c,8)),(0,)*8,tuple(range(12)),(0,)*12)
        for m,nm in enumerate(G1): mcp[c][m]=cp_of(cube.apply_move(s,nm))
    for c in range(N_UD):
        s=(tuple(range(8)),(0,)*8,tuple(idx_perm(c,8))+(8,9,10,11),(0,)*12)
        for m,nm in enumerate(G1): mud[c][m]=ud_of(cube.apply_move(s,nm))
    for c in range(N_SL):
        s=(tuple(range(8)),(0,)*8,tuple(range(8))+tuple(x+8 for x in idx_perm(c,4)),(0,)*12)
        for m,nm in enumerate(G1): msl[c][m]=sl_of(cube.apply_move(s,nm))
    print(f"  p2 move tables: {time.time()-t0:.1f}s")
    def bfs(nA,mA):
        n=nA*N_SL; d=bytearray([255])*n; d[0]=0; q=deque([0])
        while q:
            x=q.popleft(); a,sl=divmod(x,N_SL); dv=d[x]
            for m in range(NM):
                y=mA[a][m]*N_SL+msl[sl][m]
                if d[y]==255: d[y]=dv+1; q.append(y)
        return d
    t1=time.time(); pc=bfs(N_CP,mcp); print(f"  p2 corner prun: {time.time()-t1:.1f}s")
    t1=time.time(); pu=bfs(N_UD,mud); print(f"  p2 udedge prun: {time.time()-t1:.1f}s")
    print(f"  P2 TOTAL BUILD: {time.time()-t0:.1f}s")
    return mcp,mud,msl,pc,pu
MCP,MUD,MSL,PC,PU = build()

def solve_phase2(state, maxd=18, lastface=-1):
    cp,ud,sl = cp_of(state), ud_of(state), sl_of(state)
    path=[]
    def rec(cp,ud,sl,togo,lastface):
        if togo==0: return cp==0 and ud==0 and sl==0
        if PC[cp*N_SL+sl]>togo or PU[ud*N_SL+sl]>togo: return False
        for m in range(NM):
            if FACE_OF[m]==lastface: continue
            path.append(m)
            if rec(MCP[cp][m],MUD[ud][m],MSL[sl][m],togo-1,FACE_OF[m]): return True
            path.pop()
        return False
    for d in range(maxd+1):
        path.clear()
        if rec(cp,ud,sl,d,lastface): return [G1[m] for m in path]
    return None
