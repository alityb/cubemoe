"""Analysis toolkit for the production-style cube policy (src/model_prod.ProdStateModel).

  load(tag)            model from out/prod/<tag>.pt
  heldout(n)           held-out Kociemba positions (solve-level split identical to training) + labels:
                       phase (1/2), dist to G1, moves remaining, misoriented edges/corners, slice edges out,
                       solver's next move, solve id
  capture(model, X)    per layer: CLS residual after the block, CLS attention row (heads x 55),
                       top-1 expert of every token
  attention_mode(m)    context manager: 'normal' | 'zero' (attention output removed) |
                       'uniform' (every token attends equally to all tokens), optionally per layer
Attention is recomputed explicitly (softmax(QK^T/sqrt(d))V) so weights can be read; verified equal to
F.scaled_dot_product_attention by check_attention().
"""
import sys, contextlib, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import model_prod as MP
from model_prod import ProdStateModel, rope
R = '/Users/alityb/projects/cubemoe'
DEV = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def load(tag):
    s = torch.load(f'{R}/out/prod/{tag}.pt', map_location='cpu', weights_only=False); c = s['log']['cfg']
    m = ProdStateModel(d=c['d'], L=c['L'], nh=c['nh'], preset=c['preset']); m.load_state_dict(s['sd'])
    return m.to(DEV).eval(), c

def heldout(n_solves=1200, data='mixed250k'):
    import cube
    from qa import facelets_to_cubie
    d = dict(np.load(f'{R}/data/{data}.npz')); n = len(d['sol_len'])
    perm = np.random.RandomState(0).permutation(n); te = perm[int(n * 0.9):][:n_solves]
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in te])
    sid = np.concatenate([np.full(int(d['sol_len'][i]), k) for k, i in enumerate(te)])
    t = d['pos'][rows].astype(int); seam = np.concatenate([np.full(int(d['sol_len'][i]), int(d['seam'][i])) for i in te])
    sl = np.concatenate([np.full(int(d['sol_len'][i]), int(d['sol_len'][i])) for i in te])
    fl = np.zeros(len(rows), int); tw = np.zeros(len(rows), int); so = np.zeros(len(rows), int)
    for j, g in enumerate(rows):
        cp, co, ep, eo = facelets_to_cubie(''.join('URFDLB'[c] for c in d['states'][g]))
        tw[j] = sum(x != 0 for x in co); fl[j] = sum(x != 0 for x in eo); so[j] = sum(ep[p] < 8 for p in range(8, 12))
    lab = dict(phase=np.where(t < seam, 1, 2), d1=np.maximum(seam - t, 0), rem=sl - t, fl=fl, tw=tw, so=so,
               move=d['moves'][rows].astype(int), solve=sid)
    return torch.from_numpy(d['states'][rows].astype(np.int64)), lab

_STATE = {'mode': 'normal', 'layers': None, 'capture': False, 'att': []}
_ORIG = MP.Attention.forward

def _attn_forward(self, x, pad_mask=None):
    B, T, d = x.shape
    q = self.q_norm(self.q(x)).view(B, T, self.nh, self.dh).transpose(1, 2)
    k = self.k_norm(self.k(x)).view(B, T, self.nh, self.dh).transpose(1, 2)
    v = self.v(x).view(B, T, self.nh, self.dh).transpose(1, 2)
    pos = torch.arange(T, device=x.device); q, k = rope(q, pos), rope(k, pos)
    li = getattr(self, '_layer', -1); active = _STATE['layers'] is None or li in _STATE['layers']
    mode = _STATE['mode'] if active else 'normal'
    if mode == 'zero': return torch.zeros_like(x)
    att = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / self.dh ** 0.5, -1)
    if self.causal: raise NotImplementedError('analysis targets the bidirectional state model')
    if mode == 'uniform': att = torch.full_like(att, 1.0 / T)
    if _STATE['capture']: _STATE['att'].append(att[:, :, 0, :].detach().cpu())     # CLS row
    y = (att @ v.float()).to(v.dtype)
    return self.o(y.transpose(1, 2).reshape(B, T, d))

def _install(model):
    for i, b in enumerate(model.blocks): b.attn._layer = i
    MP.Attention.forward = _attn_forward

@contextlib.contextmanager
def attention_mode(model, mode='normal', layers=None):
    _install(model); old = (_STATE['mode'], _STATE['layers']); _STATE['mode'], _STATE['layers'] = mode, layers
    try: yield
    finally: _STATE['mode'], _STATE['layers'] = old

def check_attention(model, X):
    """Explicit attention must reproduce the model's own (SDPA) output."""
    MP.Attention.forward = _ORIG
    with torch.no_grad(): a = model(X.to(DEV))[0].float().cpu()
    _install(model)
    with torch.no_grad(): b = model(X.to(DEV))[0].float().cpu()
    return float((a - b).abs().max())

def logits(model, X, bs=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(model(X[i:i + bs].to(DEV))[0].float().cpu())
    return torch.cat(out)

def capture(model, X, bs=1024):
    """Per layer: CLS residual after the block, CLS attention row, top-1 expert for every token."""
    _install(model); res = [[] for _ in model.blocks]; route = [[] for _ in model.blocks]
    hooks = [b.register_forward_hook(lambda mod, i, o, li=li: res[li].append(o[0][:, 0].detach().float().cpu()))
             for li, b in enumerate(model.blocks)]
    _STATE['capture'] = True; _STATE['att'] = []
    try:
        with torch.no_grad():
            for i in range(0, len(X), bs):
                _, info, _ = model(X[i:i + bs].to(DEV), collect=True)
                for li, (p, ti) in enumerate(info): route[li].append(ti.reshape(-1, 55, ti.shape[-1])[:, :, 0].cpu())
    finally:
        _STATE['capture'] = False
        for h in hooks: h.remove()
    L = len(model.blocks); att = _STATE['att']
    att_by_layer = [torch.cat(att[li::L]) for li in range(L)]
    return dict(res=[torch.cat(r) for r in res], att=att_by_layer, route=[torch.cat(r) for r in route])

def beam_solve_path(model, start, width=256, budget=40):
    """Same search as src/solve_fast.beam_solve, but also returns the move sequence found."""
    import cube_fast as C
    MPm = torch.from_numpy(C.MOVE_PERM).to(DEV); SOLVED = torch.from_numpy(C.SOLVED.astype('int64')).to(DEV)
    g = torch.Generator(device='cpu').manual_seed(0); H = torch.randint(1, 2**61, (54,), generator=g).to(DEV)
    beam = start[None].long().to(DEV); score = torch.zeros(1, device=DEV); hist = torch.zeros(1, 0, dtype=torch.long, device=DEV)
    if (beam[0] == SOLVED).all(): return []
    for t in range(budget):
        with torch.no_grad(): lg = model(beam)[0]
        lp = F.log_softmax(lg.float(), -1); cand = beam[:, MPm]
        hit = (cand == SOLVED).all(-1)
        if hit.any():
            w, mv = [int(x) for x in hit.nonzero()[0]]; return hist[w].tolist() + [mv]
        sc = (score[:, None] + lp).reshape(-1); cand = cand.reshape(-1, 54)
        hh = torch.cat([hist.repeat_interleave(18, 0), torch.arange(18, device=DEV).repeat(len(beam))[:, None]], 1)
        h = (cand * H).sum(-1); uniq, inv = torch.unique(h, return_inverse=True)
        best = torch.full((len(uniq),), -1e30, device=DEV).scatter_reduce(0, inv, sc, 'amax')
        keep = torch.nonzero(sc == best[inv]).flatten(); kin = inv[keep]; o = torch.argsort(kin, stable=True)
        kin, keep = kin[o], keep[o]; fm = torch.ones_like(kin, dtype=torch.bool); fm[1:] = kin[1:] != kin[:-1]; keep = keep[fm]
        top = torch.topk(sc[keep], min(width, len(keep))).indices
        beam, score, hist = cand[keep[top]], sc[keep[top]], hh[keep[top]]
    return None

class Uniform(torch.nn.Module):
    """'No model': every move equally likely, so beam search = blind breadth-limited search."""
    def forward(self, x): return torch.zeros(len(x), 18, device=x.device), [], None

def random_cubes(n, seed):
    import random, cube
    rng = random.Random(seed); CID = {c: i for i, c in enumerate('URFDLB')}
    return torch.tensor([[CID[c] for c in cube.to_facelets(cube.random_state(rng))] for _ in range(n)])
