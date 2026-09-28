"""EXPLORATORY (not pre-registered): what, exactly, does the Kociemba router track?
Discover on one seed, confirm on the others. Writes out/explore/g1_<tag>.npz"""
import sys, argparse, numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from extract import extract_seq, _split
from model import device_auto
import cube
from h2_analysis import COLORS, facelets_to_cubie
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
G1MOVES=set(i for i,m in enumerate(MOVE_NAMES) if m in cube.G1_MOVES)
ap=argparse.ArgumentParser(); ap.add_argument('--tag'); ap.add_argument('--data',default='mixed250k')
ap.add_argument('--max_solves',type=int,default=1200); a=ap.parse_args()
ex=extract_seq(a.tag,a.data,device_auto(),a.max_solves)
d,te,off=_split(a.data,a.max_solves)
NL=ex['NL']; rows=len(ex['pos'])
# per-token cube features, replayed in extraction order
tw=np.zeros(rows,int); fl=np.zeros(rows,int); so=np.zeros(rows,int)
for k,(sidx,t) in enumerate(zip(ex['solve'],ex['pos'])):
    i=te[sidx]; st=facelets_to_cubie(''.join(COLORS[c] for c in d['states'][off[i]+t]))
    cp,co,ep,eo=st
    tw[k]=sum(1 for x in co if x!=0)            # misoriented corners
    fl[k]=sum(1 for x in eo if x!=0)            # misoriented edges
    so[k]=sum(1 for p in range(8,12) if ep[p]<8) # slice positions holding a non-slice edge
d1=np.where(ex['phase']==1, ex['seam']-ex['pos'], 0)   # exact distance to G1 (phase 1 is optimal)
np.savez_compressed(f'/Users/alityb/projects/cubemoe/out/explore/g1_{a.tag}.npz',
    e1=np.stack(ex['e1']), e2=np.stack(ex['e2']), phase=ex['phase'], pos=ex['pos'], prev=ex['prev'],
    move=ex['move'], seam=ex['seam'], sollen=ex['sollen'], solve=ex['solve'], tw=tw, fl=fl, so=so, d1=d1)
print('saved', rows, 'tokens,', NL, 'layers')
