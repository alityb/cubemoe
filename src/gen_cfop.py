"""H2 data: 10k CFOP solves, step-matched to the Kociemba 25k set. Same npz schema."""
import sys, os, time, random, argparse
import numpy as np
sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
import cube, cfop, cfop_gen2
MOVE_NAMES=[f+s for f in 'URFDLB' for s in ('',"'",'2')]
MOVE_ID={m:i for i,m in enumerate(MOVE_NAMES)}
COLOR_ID={c:i for i,c in enumerate('URFDLB')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=9000); ap.add_argument('--out',default='cfop')
    # Amendment 5 (H2b): widen the cross segment to decorrelate stage from token position.
    ap.add_argument('--cross_lo',type=int,default=4); ap.add_argument('--cross_hi',type=int,default=8)
    a=ap.parse_args(); rng=random.Random(a.seed)
    t0=time.time()
    states=[];moves=[];pos=[];rem=[];stage=[];sid=[];seam=[];sollen=[];init=[]
    bounds_all=[]; nbad=0
    for i in range(a.n):
        s,sol,lab,b=cfop_gen2.generate(rng,cross_len=(a.cross_lo,a.cross_hi))
        cur=s; sts=[]
        for m in sol: sts.append([COLOR_ID[c] for c in cube.to_facelets(cur)]); cur=cube.apply_move(cur,m)
        if cur!=cube.SOLVED: nbad+=1; continue
        n=len(sol)
        states.extend(sts); moves.extend(MOVE_ID[m] for m in sol)
        pos.extend(range(n)); rem.extend(range(n,0,-1)); stage.extend(lab)
        sid.extend([len(sollen)]*n); sollen.append(n); init.append(sts[0])
        seam.append(b['cross']); bounds_all.append([b['cross'],b['F2L'],b['OLL'],b['PLL']])
    d=dict(states=np.array(states,np.uint8),moves=np.array(moves,np.uint8),
           pos=np.array(pos,np.uint8),remaining=np.array(rem,np.uint8),
           phase=np.array(stage,np.uint8),            # 1=cross 2=F2L 3=OLL 4=PLL
           solve_id=np.array(sid,np.int32),seam=np.array(seam,np.uint8),
           sol_len=np.array(sollen,np.uint8),init_state=np.array(init,np.uint8),
           bounds=np.array(bounds_all,np.uint8))
    p=f"/Users/alityb/projects/cubemoe/data/{a.out}.npz"; np.savez_compressed(p,**d)
    dt=time.time()-t0
    print(f"[cfop] {len(sollen)} solves ({nbad} rejected) in {dt:.1f}s -> {len(moves)} steps")
    print(f"  sol_len mean {d['sol_len'].mean():.1f}  stage shares: "
          +", ".join(f"{cfop.STAGE_NAMES[k]}={np.mean(d['phase']==k):.3f}" for k in (1,2,3,4)))
    mix=sum((d['pos']==t).sum() for t in np.unique(d['pos']) if len(np.unique(d['phase'][d['pos']==t]))>1)/len(d['phase'])
    print(f"  rows in step cells with >1 stage: {mix:.3f}")
    # A5.3 void condition: removed-by-position must land in 50-65%
    lab_=d['phase'].astype(int); pos_=d['pos'].astype(int)
    acc=0.0; H=0.0
    for t in np.unique(pos_):
        k=pos_==t; c=np.bincount(lab_[k]); acc+=c.max()
        cc=c[c>0]/k.sum(); H+=(k.sum()/len(lab_))*(-(cc*np.log2(cc)).sum())
    acc/=len(lab_); c=np.bincount(lab_); c=c[c>0]/len(lab_); H0=-(c*np.log2(c)).sum()
    rem_pct=100*(1-H/H0)
    print(f"  [A5.3] maj(stage|pos)={acc:.4f}  H(stage)={H0:.4f}  H(stage|pos)={H:.4f}  "
          f"removed by position={rem_pct:.1f}%  -> "
          +("IN RANGE 50-65%, arm proceeds" if 50<=rem_pct<=65 else "OUT OF RANGE -> ARM IS VOID per A5.3"))
    print(f"  max sol_len={d['sol_len'].max()} -> maxlen will be {54+int(d['sol_len'].max())+1}")
    print(f"  saved {p} ({os.path.getsize(p)/1e6:.1f} MB)")
if __name__=='__main__': main()
