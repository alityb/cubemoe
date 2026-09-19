"""Test: does randomising phase-1 length decorrelate the seam from position?
Construction: sample uniform state IN G1, apply L random non-G1 moves (L varied).
The inverse of that prefix is a valid phase-1 maneuver of length ~L."""
import cube, kociemba, random, collections
rng=random.Random(11)
NONG1=['R',"R'",'L',"L'",'F',"F'",'B',"B'"]   # moves that leave G1
ALL=[f+s for f in 'URFDLB' for s in ('',"'",'2')]

def random_G1_state(rng):
    """uniform-ish over G1: corners free perm, U/D edges free perm, slice edges perm within slice, all ori 0"""
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

rows=[]
for trial in range(500):
    L=rng.choice([4,5,6,7,8,9,10,11,12])     # randomised phase-1 length
    g=random_G1_state(rng)
    assert cube.in_G1(g)
    pre=[rng.choice(NONG1) for _ in range(L)]
    s=g
    for m in pre: s=cube.apply_move(s,m)
    if cube.in_G1(s): continue               # accidentally still in G1
    sol=kociemba.solve(cube.to_facelets(s))
    mv=sol.split(); cur=s; along=[cur]
    for m in mv: cur=cube.apply_move(cur,m); along.append(cur)
    gg=[cube.in_G1(x) for x in along]
    if True not in gg: continue
    k=gg.index(True)
    for t,m in enumerate(mv): rows.append((t,m,1 if t<k else 2))

n=len(rows)
def bayes(keyfn,label):
    d=collections.defaultdict(collections.Counter)
    for r in rows: d[keyfn(r)][r[2]]+=1
    print(f"  Bayes phase acc from {label:<30} = {sum(max(c.values()) for c in d.values())/n:.3f}")
maj=max(sum(1 for r in rows if r[2]==p) for p in (1,2))/n
print(f"RANDOMISED-SEAM DATA: {n} examples, majority baseline {maj:.3f}")
bayes(lambda r:r[0],"POSITION alone")
bayes(lambda r:r[1],"OUTPUT move token alone")
bayes(lambda r:(r[0],r[1]),"position + output token")
sd=collections.Counter()
for r in rows:
    if r[2]==2: continue
sl=collections.Counter()
for r in rows: sl[r[0]]+=1
seams=collections.Counter()
print(f"  phase-1 fraction: {sum(1 for r in rows if r[2]==1)/n:.3f}")
