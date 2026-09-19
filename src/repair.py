"""Fix: forced seam was set to the requested phase-1 length L, but a length-L phase-1
maneuver can enter G1 at L-1 (its last move being a G1 move that preserves G1).
Phase is defined as a STATE property, so seam must be the first G1 entry. Relabel."""
import sys, numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from qa import facelets_to_cubie, MOVE_NAMES, COLORS, G1_SET
ROOT='/Users/alityb/projects/cubemoe'
kind=sys.argv[1]
d=dict(np.load(f'{ROOT}/data/{kind}.npz'))
off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
n=len(d['seam']); newseam=d['seam'].copy(); phase=d['phase'].copy()
changed=0; bad_p2=0
for i in range(n):
    mv=[MOVE_NAMES[m] for m in d['moves'][off[i]:off[i+1]]]
    cur=facelets_to_cubie(''.join(COLORS[c] for c in d['init_state'][i]))
    g1=[]
    for m in mv: g1.append(cube.in_G1(cur)); cur=cube.apply_move(cur,m)
    g1.append(cube.in_G1(cur))
    first=g1.index(True)
    if first!=int(d['seam'][i]):
        changed+=1; newseam[i]=first
    if not all(x in G1_SET for x in mv[first:]): bad_p2+=1
    phase[off[i]:off[i+1]]=np.where(np.arange(len(mv))<first,1,2)
print(f"relabelled {changed}/{n} solves ({changed/n:.4%})")
print(f"phase-2 subset violations after relabel: {bad_p2}")
assert bad_p2==0, "relabel would break the G1 move-set invariant"
d['seam']=newseam; d['phase']=phase
np.savez_compressed(f'{ROOT}/data/{kind}.npz', **d)
print(f"seam sd {newseam.std():.2f}  phase1 frac {(phase==1).mean():.3f}")
