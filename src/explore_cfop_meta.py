"""EXPLORATORY: regenerate CFOP solves from their seed to recover per-token construction metadata the
npz doesn't store (which F2L slot, which OLL/PLL case, position inside each algorithm).
Mirrors cfop_gen2.generate EXACTLY (same rng calls, same order) and asserts the moves match the npz."""
import sys, random, pickle, argparse, numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube
from cfop_canon import inv, MOVES
R='/Users/alityb/projects/cubemoe'
MN=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
T=pickle.load(open(f'{R}/data/cfop_canon.pkl','rb'))
def gen(rng, cross_len):
    S=cube.SOLVED; s=S
    pk=rng.choice(list(T['pll'].keys())); pll=T['pll'][pk][0]
    ok_=rng.choice(list(T['oll'].keys())); oll=T['oll'][ok_][0]
    ext={}; cks={}
    for k in (3,2,1,0):
        ck=rng.choice(list(T['f2l'][k].keys())); ext[k]=T['f2l'][k][ck][0]; cks[k]=ck
    cross=[]; last=-1
    for _ in range(rng.randint(*cross_len)):
        while True:
            m=rng.choice(MOVES)
            if 'URFDLB'.index(m[0])!=last: break
        cross.append(m); last='URFDLB'.index(m[0])
    # segments in SOLUTION order: cross, slot0..slot3, OLL, PLL
    segs=[('cross',inv(cross),None),('slot1',inv(ext[0]),cks[0]),('slot2',inv(ext[1]),cks[1]),
          ('slot3',inv(ext[2]),cks[2]),('slot4',inv(ext[3]),cks[3]),('OLL',inv(oll),ok_),('PLL',inv(pll),pk)]
    return segs
ap=argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--seed',type=int)
ap.add_argument('--lo',type=int); ap.add_argument('--hi',type=int); a=ap.parse_args()
d=dict(np.load(f'{R}/data/{a.data}.npz')); rng=random.Random(a.seed)
N=len(d['moves']); seg=np.zeros(N,np.int8); segpos=np.zeros(N,np.int16); seglen=np.zeros(N,np.int16)
case=np.full(N,-1,np.int32); caseid={}; k=0; mism=0
names=['cross','slot1','slot2','slot3','slot4','OLL','PLL']
for i in range(len(d['sol_len'])):
    segs=gen(rng,(a.lo,a.hi))
    for si,(nm,mv,cid) in enumerate(segs):
        key=(nm if nm in ('OLL','PLL') else 'F2L', repr(cid))
        if cid is not None and key not in caseid: caseid[key]=len(caseid)
        for j,m in enumerate(mv):
            if MN.index(m)!=int(d['moves'][k]): mism+=1
            seg[k]=si; segpos[k]=j; seglen[k]=len(mv); case[k]=caseid[key] if cid is not None else -1; k+=1
assert k==N, (k,N)
print(f"{a.data}: {N} tokens regenerated, move mismatches = {mism}")
assert mism==0, "regeneration does not reproduce the stored moves"
np.savez_compressed(f'{R}/out/explore/cfopmeta_{a.data}.npz',seg=seg,segpos=segpos,seglen=seglen,case=case)
print("  segment token shares:", {names[s]:round(float((seg==s).mean()),3) for s in range(7)})
print("  distinct algorithm cases:", len(caseid))
