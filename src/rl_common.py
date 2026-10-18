"""Shared pieces for the routing-vs-computation RL study on the cube policies (see prereg/PREREG_RL_ROUTING.md).

  rollout(model, starts, budget, temp, G)  batched episodes, G samples per start; returns solved flags, lengths,
                                           and (states, actions, episode index) for every step taken
  scrambles(split, n, seed)                CFOP generator scrambles: 'train' = training split, 'test' = held-out split
  stage_reached(states)                    furthest CFOP milestone reached along an episode (0 none .. 3 OLL, 4 solved)
"""
import sys, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import cube_fast as C
from cfop_verify import flags
R = '/Users/alityb/projects/cubemoe'
_DATA = None

def _data():
    global _DATA
    if _DATA is None: _DATA = dict(np.load(f"{R}/data/cfop250k.npz"))
    return _DATA

def scrambles(split, n, seed=0):
    """Same solve-level split as training and every analysis (RandomState(0) permutation, last 10% held out)."""
    d = _data(); N = len(d['sol_len']); perm = np.random.RandomState(0).permutation(N); cut = int(N * 0.9)
    pool = perm[:cut] if split == 'train' else perm[cut:]
    idx = np.random.RandomState(seed).choice(pool, n, replace=False) if split == 'train' else pool[:n]
    return torch.from_numpy(d['init_state'][idx].astype(np.int64)), idx

_U, _R, _F, _D, _L, _B = (np.arange(9) + 9 * k for k in range(6)); _SOL = torch.from_numpy(np.repeat(np.arange(6), 9))
_CROSS = torch.from_numpy(np.r_[_D[[1, 3, 5, 7]], _R[7], _F[7], _L[7], _B[7]])
_F2L = torch.from_numpy(np.r_[_D, _R[3:], _F[3:], _L[3:], _B[3:]]); _OLL = torch.cat([_F2L, torch.from_numpy(_U)])

def milestone(X):
    """0 nothing, 1 cross, 2 F2L, 3 OLL (F2L + top oriented colours), 4 solved — sticker tests as in src/cfop_verify.py."""
    sol = _SOL.to(X.device); ok = lambda idx: (X[:, idx.to(X.device)] == sol[idx.to(X.device)]).all(1)
    return ok(_CROSS).long() + ok(_F2L).long() + ok(_OLL).long() + (X == sol).all(1).long()

import contextlib
def amp(dev):
    """bf16 autocast on CUDA (as in training and evaluation); full precision elsewhere."""
    return torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(dev).type == 'cuda' else contextlib.nullcontext()

@torch.no_grad()
def rollout(model, starts, budget, temp=1.0, G=1, dev=None, bs=8192, keep=False, track=False):
    """Run len(starts)*G episodes in parallel. temp=0 -> greedy. Episodes stop when solved."""
    dev = dev or next(model.parameters()).device
    MP = torch.from_numpy(C.MOVE_PERM).to(dev); SOLVED = torch.from_numpy(C.SOLVED.astype(np.int64)).to(dev)
    X = starts.to(dev).repeat_interleave(G, 0); n = len(X)
    done = (X == SOLVED).all(1); length = torch.zeros(n, dtype=torch.long, device=dev)
    stage = milestone(X) if track else None
    S, A, E = [], [], []
    for t in range(budget):
        act = torch.nonzero(~done).flatten()
        if len(act) == 0: break
        with amp(X.device):
            lg = torch.cat([model(X[act[i:i + bs]])[0].float() for i in range(0, len(act), bs)])
        mv = lg.argmax(-1) if temp == 0 else torch.multinomial(torch.softmax(lg / temp, -1), 1).squeeze(1)
        if keep: S.append(X[act].cpu()); A.append(mv.cpu()); E.append(act.cpu())
        X[act] = torch.gather(X[act], 1, MP[mv]); length[act] += 1
        done = done | (X == SOLVED).all(1)
        if track: stage = torch.maximum(stage, milestone(X))
    out = dict(solved=done.cpu(), length=length.cpu(), final=X.cpu(), stage=stage.cpu() if track else None)
    if keep: out.update(states=torch.cat(S), actions=torch.cat(A), episode=torch.cat(E))
    return out
