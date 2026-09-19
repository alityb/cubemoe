import cube, kociemba, random, time, collections
rng=random.Random(7); N=400
states=[cube.random_state(rng) for _ in range(N)]
sols=[kociemba.solve(cube.to_facelets(s)) for s in states]

rows=[]  # (position_index, total_len, move_token, prev_token, phase)
seam_eq=0
for s,sol in zip(states,sols):
    mv=sol.split(); cur=s; along=[cur]
    for m in mv: cur=cube.apply_move(cur,m); along.append(cur)
    g=[cube.in_G1(x) for x in along]
    k_g1=g.index(True)                              # first entry into G1
    k_joint=min(k for k in range(len(mv)+1) if g[k] and all(m in cube.G1_MOVES for m in mv[k:]))
    if k_g1==k_joint: seam_eq+=1
    for t,m in enumerate(mv):
        rows.append((t,len(mv),m, mv[t-1] if t>0 else '<S>', 1 if t<k_g1 else 2))
print(f"first-G1-entry == joint seam definition: {seam_eq}/{N}  (seam label is unambiguous)")
n=len(rows); maj=max(sum(1 for r in rows if r[4]==p) for p in (1,2))/n
print(f"total (position,move) examples: {n}; majority-class phase baseline: {maj:.3f}\n")

def bayes(keyfn, label):
    d=collections.defaultdict(collections.Counter)
    for r in rows: d[keyfn(r)][r[4]]+=1
    acc=sum(max(c.values()) for c in d.values())/n
    print(f"  Bayes-optimal phase acc from {label:<34} = {acc:.3f}   ({len(d)} distinct keys)")
    return acc
print("HOW MUCH PHASE INFO LEAKS FROM TRIVIAL SURFACE FEATURES:")
bayes(lambda r: r[2], "OUTPUT move token alone")
bayes(lambda r: r[3], "PREVIOUS (input) move token alone")
bayes(lambda r: r[0], "POSITION INDEX alone")
bayes(lambda r: (r[0],r[3]), "position + previous token")
bayes(lambda r: (r[0],r[2]), "position + output token")

# matched-token control set: keep only positions whose OUTPUT move type occurs in both phases
byc=collections.defaultdict(collections.Counter)
for r in rows: byc[r[2]][r[4]]+=1
both={m for m,c in byc.items() if c[1]>0 and c[2]>0}
sub=[r for r in rows if r[2] in both]
print(f"\nMATCHED-MOVE CONTROL SET (output move type appears in both phases): {len(sub)}/{n} rows")
d=collections.defaultdict(collections.Counter)
for r in sub: d[r[2]][r[4]]+=1
print(f"  Bayes acc from output token alone on this subset = {sum(max(c.values()) for c in d.values())/len(sub):.3f}")
mj=max(sum(1 for r in sub if r[4]==p) for p in (1,2))/len(sub)
print(f"  majority baseline on this subset = {mj:.3f}")
d2=collections.defaultdict(collections.Counter)
for r in sub: d2[(r[0],r[2])][r[4]]+=1
print(f"  Bayes acc from (position, output token) on subset = {sum(max(c.values()) for c in d2.values())/len(sub):.3f}")

# position-matched AND token-matched: how many usable pairs near the seam?
near=[r for r in rows if r[2] in both and 8<=r[0]<=13]
d3=collections.defaultdict(collections.Counter)
for r in near: d3[(r[0],r[2])][r[4]]+=1
usable=sum(sum(c.values()) for c in d3.values() if c[1]>0 and c[2]>0)
print(f"\nSEAM-WINDOW (pos 8-13, matched move type): {len(near)} rows, "
      f"{usable} live in (pos,token) cells containing BOTH phases -> the clean test set")
d4=collections.defaultdict(collections.Counter)
for r in near: d4[(r[0],r[2])][r[4]]+=1
print(f"  Bayes acc from (position, token) inside seam window = {sum(max(c.values()) for c in d4.values())/len(near):.3f}")
mj2=max(sum(1 for r in near if r[4]==p) for p in (1,2))/len(near)
print(f"  majority baseline inside seam window = {mj2:.3f}")
