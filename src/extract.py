"""Routing / residual extraction for seq (A,B,hash) and state (C) models."""
import sys, numpy as np, torch
sys.path.insert(0, '/Users/alityb/projects/cubemoe/src')
from model import SeqModel, StateModel
ROOT = '/Users/alityb/projects/cubemoe'

def _split(data, max_solves):
    d = dict(np.load(f'{ROOT}/data/{data}.npz'))
    n = len(d['seam']); perm = np.random.RandomState(0).permutation(n)
    te = perm[int(n*0.9):][:max_solves]
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    return d, te, off

def extract_seq(tag, data, dev, max_solves=2500):
    ck = torch.load(f'{ROOT}/out/{tag}.pt', map_location=dev, weights_only=False)
    cfg = ck['cfg']; kind = 'dense' if cfg['model'] == 'dense' else 'moe'
    router = 'hash' if cfg['model'] == 'hash' else 'learned'
    m = SeqModel(cfg['D'], cfg['NL'], 4, cfg['maxlen'], kind, 8, 2, router).to(dev)
    m.load_state_dict(ck['sd']); m.eval()
    d, te, off = _split(data, max_solves)
    from train import build_seq
    X, Y, P = build_seq(d, te, cfg['maxlen']); NL = cfg['NL']
    store = {}
    hooks = [b.ln2.register_forward_hook(lambda mo,i,o,li=li: store.__setitem__(('rs',li), o.detach()))
             for li, b in enumerate(m.blocks)]
    if kind == 'dense':
        hooks += [b.ffn[1].register_forward_hook(lambda mo,i,o,li=li: store.__setitem__(('fh',li), o.detach()))
                  for li, b in enumerate(m.blocks)]
    E1=[[] for _ in range(NL)]; E2=[[] for _ in range(NL)]; RS=[[] for _ in range(NL)]; FH=[[] for _ in range(NL)]
    meta=[]
    with torch.no_grad():
        for b0 in range(0, len(X), 64):
            x = torch.from_numpy(X[b0:b0+64]).to(dev); pm = torch.from_numpy(P[b0:b0+64]).to(dev)
            _, info = m(x, pm, collect=(kind=='moe'))
            B, T = x.shape
            for r in range(B):
                i = te[b0+r]; nmv = int(d['sol_len'][i]); sm = int(d['seam'][i])
                mv = d['moves'][off[i]:off[i+1]]; idx = 54 + np.arange(nmv)
                for t in range(nmv):
                    meta.append((b0+r, t, nmv-t, 1 if t<sm else 2, int(mv[t]),
                                 int(mv[t-1]) if t>0 else 18, sm, nmv))
                for li in range(NL):
                    RS[li].append(store[('rs',li)].reshape(B,T,-1)[r][idx].cpu().numpy().astype(np.float16))
                    if kind=='moe':
                        ti = info[li][1].reshape(B,T,2)[r][idx].cpu().numpy()
                        E1[li].append(ti[:,0]); E2[li].append(ti[:,1])
                    else:
                        FH[li].append(store[('fh',li)].reshape(B,T,-1)[r][idx].cpu().numpy().astype(np.float16))
    for h in hooks: h.remove()
    meta = np.array(meta)
    out = dict(solve=meta[:,0], pos=meta[:,1], rem=meta[:,2], phase=meta[:,3], move=meta[:,4],
               prev=meta[:,5], seam=meta[:,6], sollen=meta[:,7], NL=NL, kind=kind, tag=tag,
               rs=[np.concatenate(x) for x in RS])
    if kind=='moe': out['e1']=[np.concatenate(x) for x in E1]; out['e2']=[np.concatenate(x) for x in E2]
    else: out['fh']=[np.concatenate(x) for x in FH]
    return out

def extract_state(tag, data, dev, max_solves=2500):
    ck = torch.load(f'{ROOT}/out/{tag}.pt', map_location=dev, weights_only=False)
    cfg = ck['cfg']; kind = 'dense' if cfg['model']=='state_dense' else 'moe'
    m = StateModel(cfg['D'], cfg['NL'], 4, kind, 8, 2).to(dev); m.load_state_dict(ck['sd']); m.eval()
    d, te, off = _split(data, max_solves)
    rows = np.concatenate([np.arange(off[i], off[i+1]) for i in te])
    st = d['states'][rows].astype(np.int64); NL = cfg['NL']
    store={}
    hooks=[b.ln2.register_forward_hook(lambda mo,i,o,li=li: store.__setitem__(('rs',li),o.detach()))
           for li,b in enumerate(m.blocks)]
    if kind=='dense':
        hooks+=[b.ffn[1].register_forward_hook(lambda mo,i,o,li=li: store.__setitem__(('fh',li),o.detach()))
                for li,b in enumerate(m.blocks)]
    ECLS=[[] for _ in range(NL)]; EAGG=[[] for _ in range(NL)]; RS=[[] for _ in range(NL)]; FH=[[] for _ in range(NL)]
    with torch.no_grad():
        for b0 in range(0, len(st), 512):
            x = torch.from_numpy(st[b0:b0+512]).to(dev)
            _, info = m(x, collect=(kind=='moe')); B = x.shape[0]
            for li in range(NL):
                RS[li].append(store[('rs',li)].reshape(B,55,-1)[:,0].cpu().numpy().astype(np.float16))
                if kind=='moe':
                    ti = info[li][1].reshape(B,55,2)
                    ECLS[li].append(ti[:,0,0].cpu().numpy())
                    stick = ti[:,1:,0].cpu().numpy()
                    EAGG[li].append(np.array([np.bincount(s, minlength=8).argmax() for s in stick]))
                else:
                    FH[li].append(store[('fh',li)].reshape(B,55,-1)[:,0].cpu().numpy().astype(np.float16))
    for h in hooks: h.remove()
    out = dict(solve=d['solve_id'][rows], pos=d['pos'][rows].astype(int), rem=d['remaining'][rows].astype(int),
               phase=d['phase'][rows].astype(int), move=d['moves'][rows].astype(int),
               seam=d['seam'][d['solve_id'][rows]].astype(int), NL=NL, kind=kind, tag=tag,
               rs=[np.concatenate(x) for x in RS])
    out['prev'] = np.where(out['pos']>0, np.roll(out['move'],1), 18)
    out['sollen'] = out['pos'] + out['rem']
    if kind=='moe': out['e1']=[np.concatenate(x) for x in ECLS]; out['e2']=[np.concatenate(x) for x in EAGG]
    else: out['fh']=[np.concatenate(x) for x in FH]
    return out
