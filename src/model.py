"""MoE / dense / position-blind models. No capacity cap, no token dropping."""
import math, torch, torch.nn as nn, torch.nn.functional as F

N_COLOR, N_MOVE = 6, 18
VOCAB = N_COLOR + N_MOVE + 1          # +1 BOS
BOS = N_COLOR + N_MOVE

def device_auto():
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

class MoEFFN(nn.Module):
    """Top-k routed FFN. All experts computed densely then gathered => no dropping."""
    def __init__(self, d, n_exp=8, k=2, mult=4, router='learned'):
        super().__init__()
        self.E, self.k, self.router = n_exp, k, router
        self.gate = nn.Linear(d, n_exp, bias=False)
        self.w1 = nn.Parameter(torch.empty(n_exp, d, mult * d)); self.w2 = nn.Parameter(torch.empty(n_exp, mult * d, d))
        for w in (self.w1, self.w2): nn.init.normal_(w, std=0.02)
        self.b1 = nn.Parameter(torch.zeros(n_exp, mult * d)); self.b2 = nn.Parameter(torch.zeros(n_exp, d))
        if router == 'hash':
            self.register_buffer('hash_tab', torch.randint(0, n_exp, (4096,)))
        self._override = None   # (N,) long, -1 = leave alone, else force this expert into slot 0

    def forward(self, x, tok_ids=None):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        if self.router == 'hash':
            h = self.hash_tab[(tok_ids.reshape(-1) * 2654435761 % 4096)]
            probs = F.one_hot(h, self.E).float()
            topv, topi = probs.topk(self.k, -1)
            topv = torch.ones_like(topv) / self.k
        else:
            logits = self.gate(flat)
            probs = F.softmax(logits, -1)
            topv, topi = probs.topk(self.k, -1)
            topv = topv / topv.sum(-1, keepdim=True)
        if self._override is not None:
            ov = self._override
            msk = ov >= 0
            topi = topi.clone(); topi[msk, 0] = ov[msk]
        # dense expert compute: (E, N, d)
        hmid = torch.einsum('nd,edh->enh', flat, self.w1) + self.b1[:, None, :]
        hmid = F.gelu(hmid)
        yall = torch.einsum('enh,ehd->end', hmid, self.w2) + self.b2[:, None, :]
        W = torch.zeros(flat.shape[0], self.E, device=x.device, dtype=yall.dtype)
        W.scatter_(1, topi, topv.to(yall.dtype))
        y = torch.einsum('ne,end->nd', W, yall)
        return y.reshape(B, T, d), probs, topi

def _lbl_one(p, ti, E):
    if p.numel() == 0: return p.new_zeros(())
    f = torch.zeros(E, device=p.device)
    f.scatter_add_(0, ti.reshape(-1), torch.ones(ti.numel(), device=p.device))
    f = f / ti.numel()
    return E * (f * p.mean(0)).sum()

def lb_loss(probs, topi, E, mask, scope='large', B=None, T=None):
    """Switch load-balance loss.
    scope='large' : balanced over the WHOLE batch (default, as used in all prior runs)
    scope='local' : balanced within each SEQUENCE independently, then averaged
    scope='off'   : no auxiliary loss
    """
    if scope == 'off': return probs.new_zeros(())
    if scope == 'large' or B is None:
        return _lbl_one(probs[mask], topi[mask], E)
    # vectorised per-sequence balance (identical maths to the per-sequence loop)
    pv = probs.reshape(B, T, -1); tv = topi.reshape(B, T, -1); mv = mask.reshape(B, T).float()
    cnt = mv.sum(1)                                   # (B,) valid tokens per sequence
    live = cnt > 0
    k = tv.shape[-1]
    oh = torch.zeros(B, T, E, device=probs.device, dtype=probs.dtype)
    oh.scatter_(2, tv.clamp(min=0), 1.0)              # (B,T,E) counts of expert picks per token
    f = (oh * mv[:, :, None]).sum(1) / (cnt[:, None] * k).clamp(min=1)   # (B,E) fraction of picks
    P = (pv * mv[:, :, None]).sum(1) / cnt[:, None].clamp(min=1)         # (B,E) mean gate prob
    per_seq = E * (f * P).sum(-1)                     # (B,)
    return (per_seq * live).sum() / live.sum().clamp(min=1)

class Block(nn.Module):
    def __init__(self, d, nh, kind, n_exp=8, k=2, router='learned'):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.kind = kind
        if kind == 'moe': self.ffn = MoEFFN(d, n_exp, k, 4, router)
        else: self.ffn = nn.Sequential(nn.Linear(d, 8 * d), nn.GELU(), nn.Linear(8 * d, d))  # active-param matched

    def forward(self, x, attn_mask=None, pad_mask=None, tok_ids=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        if self.kind == 'moe':
            y, probs, topi = self.ffn(h, tok_ids)
            return x + y, probs, topi, h
        return x + self.ffn(h), None, None, h

class SeqModel(nn.Module):
    """A/B: decoder-only over [54 stickers][moves]. Loss only on move positions."""
    def __init__(self, d=128, L=4, nh=4, maxlen=100, kind='moe', n_exp=8, k=2, router='learned'):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, nh, kind, n_exp, k, router) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, N_MOVE)
        self.maxlen, self.kind = maxlen, kind

    def forward(self, ids, pad_mask=None, collect=False):
        B, T = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        am = torch.triu(torch.ones(T, T, device=ids.device, dtype=torch.bool), 1)
        info = []
        for b in self.blocks:
            x, probs, topi, h = b(x, am, pad_mask, ids)
            if collect: info.append((probs, topi))
        return self.head(self.lnf(x)), info

class StateModel(nn.Module):
    """C: position-blind. 54 stickers of CURRENT state -> next move. No history, no step index."""
    def __init__(self, d=128, L=4, nh=4, kind='moe', n_exp=8, k=2, router='learned'):
        super().__init__()
        self.emb = nn.Embedding(N_COLOR, d); self.pos = nn.Embedding(55, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, nh, kind, n_exp, k, router) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, N_MOVE)
        self.kind = kind

    def forward(self, st, collect=False):
        B = st.shape[0]
        x = self.emb(st) + self.pos(torch.arange(54, device=st.device))[None]
        x = torch.cat([self.cls.expand(B, -1, -1) + self.pos(torch.tensor([54], device=st.device))[None], x], 1)
        ids = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=st.device), st], 1)
        info = []
        for b in self.blocks:
            x, probs, topi, h = b(x, None, None, ids)     # bidirectional, no causal mask
            if collect: info.append((probs, topi))
        return self.head(self.lnf(x[:, 0])), info
