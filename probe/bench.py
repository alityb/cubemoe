import cube, kociemba, random, time, collections
rng = random.Random(42)
N = 300
states = [cube.random_state(rng) for _ in range(N)]
fs = [cube.to_facelets(s) for s in states]

t0=time.time()
sols=[kociemba.solve(f) for f in fs]
dt=time.time()-t0
print(f"kociemba.solve: {N} solves in {dt:.2f}s -> {N/dt:.1f} solves/sec single-core")
lens=[len(s.split()) for s in sols]
print(f"solution length: mean {sum(lens)/len(lens):.2f} min {min(lens)} max {max(lens)}")
print(f"=> (state,move) pairs per second: {N/dt*sum(lens)/len(lens):.0f}")

# --- phase seam recoverability ---
seam_found=0; ambiguous=0; seams=[]; nonmono=0
for s,sol in zip(states,sols):
    mv=sol.split()
    cur=s; states_along=[cur]
    for m in mv:
        cur=cube.apply_move(cur,m); states_along.append(cur)
    # earliest index k such that state_k in G1 AND all moves from k on are in G1 move set
    ks=[k for k in range(len(mv)+1)
        if cube.in_G1(states_along[k]) and all(m in cube.G1_MOVES for m in mv[k:])]
    if ks:
        seam_found+=1; seams.append((ks[0], len(mv)))
        if len(ks)>1: ambiguous+=1
    # monotonic check: is G1 membership "once in, stays in"?
    g=[cube.in_G1(x) for x in states_along]
    first=g.index(True) if True in g else None
    if first is not None and not all(g[first:]): nonmono+=1
print(f"\nseam recoverable (state in G1 + all later moves in G1 set): {seam_found}/{N}")
print(f"  of those, >1 valid seam index (ambiguous): {ambiguous}")
print(f"  G1 membership non-monotone along solution: {nonmono}/{N}")
p1=[a for a,b in seams]; p2=[b-a for a,b in seams]
print(f"  phase1 len mean {sum(p1)/len(p1):.2f} (min {min(p1)} max {max(p1)})")
print(f"  phase2 len mean {sum(p2)/len(p2):.2f} (min {min(p2)} max {max(p2)})")
print(f"  phase2 == 0 moves (i.e. no phase 2 at all): {sum(1 for x in p2 if x==0)}/{len(p2)}")

# --- CONFOUND CHECK: move-type distribution per phase ---
c1=collections.Counter(); c2=collections.Counter()
for s,sol,(k,tot) in zip(states,sols,seams):
    mv=sol.split()
    for m in mv[:k]: c1[m]+=1
    for m in mv[k:]: c2[m]+=1
tot1,tot2=sum(c1.values()),sum(c2.values())
print(f"\nphase1 moves: {tot1}, phase2 moves: {tot2}")
p1set=set(c1); p2set=set(c2)
print(f"phase1 move types: {sorted(p1set)}")
print(f"phase2 move types: {sorted(p2set)}")
overlap=p1set & p2set
share1=sum(c1[m] for m in overlap)/tot1; share2=sum(c2[m] for m in overlap)/tot2
print(f"overlapping move types: {sorted(overlap)}")
print(f"  frac of phase1 moves that are of an overlapping type: {share1:.3f}")
print(f"  frac of phase2 moves that are of an overlapping type: {share2:.3f}")
# how well can you predict phase from move token alone?
correct=0
for m in p1set|p2set:
    correct+=max(c1[m],c2[m])
print(f"  BAYES-OPTIMAL phase accuracy from OUTPUT MOVE TOKEN ALONE: {correct/(tot1+tot2):.3f}")
base=max(tot1,tot2)/(tot1+tot2)
print(f"  majority-class baseline: {base:.3f}")
