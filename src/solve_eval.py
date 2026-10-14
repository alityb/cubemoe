"""Does the model actually SOLVE the cube? Closed-loop rollouts: the model picks a move (greedy),
the move is applied to a real cube, the new state is fed back, until solved or out of budget."""
import sys, json, argparse, random, numpy as np, torch
sys.path.insert(0,'/Users/alityb/projects/cubemoe/src'); sys.path.insert(0,'/Users/alityb/projects/cubemoe/probe')
from model import SeqModel, StateModel, device_auto, N_COLOR
from qa import facelets_to_cubie, COLORS
import cube
R='/Users/alityb/projects/cubemoe'; dev=device_auto()
MN=[f+s for f in 'URFDLB' for s in ('',"'",'2')]; CID={c:i for i,c in enumerate('URFDLB')}
def enc(st): return [CID[c] for c in cube.to_facelets(st)]
def load(tag):
    ck=torch.load(f'{R}/out/{tag}.pt',map_location=dev,weights_only=False); c=ck['cfg']
    if c['model'].startswith('state'): m=StateModel(c['D'],c['NL'],4,'moe',8,2)
    else: m=SeqModel(c['D'],c['NL'],4,c['maxlen'],'moe' if c['model']!='dense' else 'dense',8,2,'hash' if c['model']=='hash' else 'learned')
    m.load_state_dict(ck['sd']); return m.to(dev).eval(), c
def rollout(m,c,states,budget,policy='model',seed=0):
    rng=np.random.RandomState(seed); n=len(states); cur=list(states); solved_at=[0 if s==cube.SOLVED else None for s in states]
    stick=torch.tensor([enc(s) for s in states]); moves=[]
    for t in range(budget):
        if policy=='random': a=rng.randint(0,18,n)
        elif c['model'].startswith('state'):
            with torch.no_grad(): lg,_=m(torch.tensor([enc(s) for s in cur]).to(dev))
            a=lg.argmax(-1).cpu().numpy()
        else:
            ids=torch.cat([stick]+[torch.tensor(mv)[:,None]+N_COLOR for mv in moves],1) if moves else stick
            with torch.no_grad(): lg,_=m(ids.to(dev))
            a=lg[:,-1].argmax(-1).cpu().numpy()
        moves.append(a)
        for i in range(n):
            if solved_at[i] is None:
                cur[i]=cube.apply_move(cur[i],MN[a[i]])
                if cur[i]==cube.SOLVED: solved_at[i]=t+1
    return solved_at
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tags'); ap.add_argument('--n',type=int,default=300); a=ap.parse_args()
    res={}
    for tag in a.tags.split(','):
        m,c=load(tag); data=c['data']; d=dict(np.load(f'{R}/data/{data}.npz'))
        te=np.random.RandomState(0).permutation(len(d['sol_len']))[int(len(d['sol_len'])*0.9):][:a.n]
        held=[facelets_to_cubie(''.join(COLORS[x] for x in d['init_state'][i])) for i in te]; ref=d['sol_len'][te]
        budget=(c['maxlen']-54) if not c['model'].startswith('state') else 40
        sets={'held-out scrambles':(held,ref)}
        if 'mixed' in data:
            rr=random.Random(1); sets['fully random cubes']=([cube.random_state(rr) for _ in range(a.n)],None)
        res[tag]={}
        for sn,(S,refl) in sets.items():
            sa=rollout(m,c,S,budget); ok=np.array([x is not None for x in sa])
            rnd=rollout(None,c,S,budget,policy='random'); okr=np.mean([x is not None for x in rnd])
            extra=''
            if refl is not None and ok.any():
                L=np.array([x for x in sa if x is not None]); extra=f"  moves used {L.mean():.1f} (solver's own: {refl[ok].mean():.1f})"
            res[tag][sn]=dict(solve_rate=float(ok.mean()),random_policy=float(okr),n=len(S),budget=budget)
            print(f"{tag:18s} {sn:20s} SOLVED {ok.mean()*100:5.1f}%  (n={len(S)}, budget {budget} moves; random policy {okr*100:.1f}%){extra}",flush=True)
    json.dump(res,open(f'{R}/out/solve_eval.json','w'),indent=1)

def beam_solve(m, start, width=256, budget=40):
    """Beam search with a state-input policy: keep the `width` most probable move sequences,
    expand each by all 18 moves, dedupe positions; return #moves to solved or None."""
    import torch.nn.functional as F
    if start==cube.SOLVED: return 0
    beam=[(0.0,start)]
    for t in range(budget):
        sts=[s for _,s in beam]
        with torch.no_grad(): lg,_=m(torch.tensor([enc(s) for s in sts]).to(dev))
        lp=F.log_softmax(lg.float(),-1).cpu().numpy()
        cand={}
        for (sc,s),row in zip(beam,lp):
            for a in range(18):
                ns=cube.apply_move(s,MN[a])
                if ns==cube.SOLVED: return t+1
                v=sc+row[a]
                if ns not in cand or v>cand[ns]: cand[ns]=v
        beam=[(v,s) for s,v in sorted(cand.items(),key=lambda kv:-kv[1])[:width]]
    return None
