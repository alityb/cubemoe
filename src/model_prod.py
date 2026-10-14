"""Production-style MoE transformer, modelled on OLMoE (allenai/OLMoE-1B-7B), scaled down.

Differences from src/model.py (the original research models), all chosen to match production MoEs:
  RMSNorm (not LayerNorm) | rotary position embeddings, RoPE (not a learned position table)
  QK-norm in attention (OLMoE) | SwiGLU experts (not GELU) | fine-grained experts: 64, top-8 (OLMoE)
  router = bias-free linear layer, softmax -> top-k (OLMoE; Mixtral does top-k -> softmax)
  load-balancing loss (0.01) + router z-loss (0.001), OLMoE's coefficients
  sparse dispatch: only the selected experts run on each token
Presets also cover Mixtral-style (8 experts, top-2, renormalised weights) and DeepSeek-style
(fine-grained experts + an always-on shared expert).

The MoE layer keeps the research hooks used by the analysis code:
  ._override (N,) long: force this expert into slot 0 (-1 = leave alone)
  ._knockout set: router logits of these experts set to -inf before top-k
Model forward returns (logits, info, aux_loss); info[layer] = (router_probs, top_k_index).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

N_COLOR, N_MOVE = 6, 18
VOCAB = N_COLOR + N_MOVE + 1

PRESETS = {
    #            experts top_k expert_hidden(x d) shared norm_topk
    'olmoe':    dict(n_exp=64, k=8, h_mult=0.5, n_shared=0, norm_topk=False),
    'mixtral':  dict(n_exp=8,  k=2, h_mult=3.5, n_shared=0, norm_topk=True),
    'deepseek': dict(n_exp=64, k=6, h_mult=0.5, n_shared=1, norm_topk=True),
}

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w

def rope(x, pos, base=10000.0):
    """x: (B, H, T, Dh). Rotary embedding on the last dim (half-split convention, as in Llama/OLMoE)."""
    Dh = x.shape[-1]; half = Dh // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * inv[None]                      # (T, half)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)

class Attention(nn.Module):
    def __init__(self, d, nh, causal):
        super().__init__()
        self.nh, self.dh, self.causal = nh, d // nh, causal
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(d); self.k_norm = RMSNorm(d)      # QK-norm, as in OLMoE
    def forward(self, x, pad_mask=None):
        B, T, d = x.shape
        q = self.q_norm(self.q(x)).view(B, T, self.nh, self.dh).transpose(1, 2)
        k = self.k_norm(self.k(x)).view(B, T, self.nh, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        pos = torch.arange(T, device=x.device); q, k = rope(q, pos), rope(k, pos)
        mask = None
        if pad_mask is not None:                                # True = padding key
            mask = ~pad_mask[:, None, None, :]
            if self.causal: mask = mask & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.o(y.transpose(1, 2).reshape(B, T, d))

class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False); self.up = nn.Linear(d, h, bias=False); self.down = nn.Linear(h, d, bias=False)
    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))

class MoE(nn.Module):
    def __init__(self, d, n_exp, k, h_mult, n_shared, norm_topk, lb_coef=0.01, z_coef=0.001):
        super().__init__()
        self.E, self.k, self.norm_topk = n_exp, k, norm_topk
        self.router = nn.Linear(d, n_exp, bias=False)
        h = max(8, int(round(h_mult * d))); self.h = h
        # experts stored stacked (E, ., .) so all of them run in one batched matmul, as production kernels do
        self.w_gate = nn.Parameter(torch.randn(n_exp, d, h) * d ** -0.5)
        self.w_up = nn.Parameter(torch.randn(n_exp, d, h) * d ** -0.5)
        self.w_down = nn.Parameter(torch.randn(n_exp, h, d) * h ** -0.5)
        self.shared = nn.ModuleList([SwiGLU(d, h) for _ in range(n_shared)])
        self.lb_coef, self.z_coef = lb_coef, z_coef
        self._override = None; self._knockout = set()
    def forward(self, x, token_mask=None):
        B, T, d = x.shape; flat = x.reshape(-1, d)
        logits = self.router(flat).float()
        if self._knockout:
            logits = logits.clone(); logits[:, list(self._knockout)] = float('-inf')
        probs = F.softmax(logits, -1)
        topv, topi = probs.topk(self.k, -1)
        if self.norm_topk: topv = topv / topv.sum(-1, keepdim=True)
        if self._override is not None:
            msk = self._override >= 0; topi = topi.clone(); topi[msk, 0] = self._override[msk]   # same as src/model.py
        out = self.dispatch(flat, topi, topv)
        for s in self.shared: out = out + s(flat)
        # aux losses over real tokens only
        m = torch.ones(flat.shape[0], dtype=torch.bool, device=x.device) if token_mask is None else token_mask.reshape(-1)
        if m.any() and self.training:
            pm = probs[m]; cnt = torch.zeros(self.E, device=x.device).index_add_(0, topi[m].reshape(-1), torch.ones(topi[m].numel(), device=x.device))
            f = cnt / topi[m].numel(); P = pm.mean(0)
            aux = self.lb_coef * self.E * (f * P).sum() + self.z_coef * torch.logsumexp(logits[m], -1).pow(2).mean()
        else:
            aux = torch.zeros((), device=x.device)
        return out.reshape(B, T, d), probs, topi, aux

    def expert(self, e, x):                                      # one expert applied to x (used by tests/analysis)
        return (F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]
    def dispatch(self, flat, topi, topv):
        """Dropless sparse dispatch: each token goes only to its k selected experts. Token copies are sorted by
        expert so each expert's tokens are one contiguous slice; per-expert counts are read once (one host sync
        per layer), each expert runs on its slice, and outputs are scattered back weighted by the router.
        No padding, so memory scales with the real work even when routing is unbalanced."""
        N, d = flat.shape; k = topi.shape[1]
        e_flat = topi.reshape(-1); w_flat = topv.reshape(-1).to(flat.dtype)
        tok = torch.arange(N, device=flat.device).repeat_interleave(k)
        order = torch.argsort(e_flat, stable=True); tok_s, w_s = tok[order], w_flat[order]
        counts = torch.bincount(e_flat, minlength=self.E).tolist()
        xs = flat[tok_s]; ys = []; start = 0
        for e, c in enumerate(counts):
            if c: ys.append(self.expert(e, xs[start:start + c]))
            start += c
        y = torch.cat(ys, 0)
        return flat.new_zeros(N, d).index_add_(0, tok_s, y * w_s[:, None])

class Block(nn.Module):
    def __init__(self, d, nh, causal, moe_kw):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, nh, causal); self.ffn = MoE(d, **moe_kw)
    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.n1(x), pad_mask)
        y, probs, topi, aux = self.ffn(self.n2(x), None if pad_mask is None else ~pad_mask)
        return x + y, probs, topi, aux

class ProdStateModel(nn.Module):
    """Policy over the CURRENT cube: [CLS][54 stickers] -> next move. Bidirectional attention."""
    def __init__(self, d=256, L=6, nh=4, preset='olmoe'):
        super().__init__()
        p = PRESETS[preset]; self.preset = preset
        self.emb = nn.Embedding(N_COLOR, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, nh, False, p) for _ in range(L)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d, N_MOVE, bias=False)
    def forward(self, st, collect=False):
        x = torch.cat([self.cls.expand(st.shape[0], -1, -1), self.emb(st)], 1)
        info = []; aux = torch.zeros((), device=st.device)
        for b in self.blocks:
            x, probs, topi, a = b(x); aux = aux + a
            if collect: info.append((probs, topi))
        return self.head(self.norm(x[:, 0])), info, aux

class ProdSeqModel(nn.Module):
    """Decoder-only over [54 stickers][moves], causal, RoPE. Same input format as src/model.SeqModel."""
    def __init__(self, d=256, L=6, nh=4, preset='olmoe'):
        super().__init__()
        p = PRESETS[preset]; self.preset = preset
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, nh, True, p) for _ in range(L)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d, N_MOVE, bias=False)
    def forward(self, ids, pad_mask=None, collect=False):
        x = self.emb(ids); info = []; aux = torch.zeros((), device=ids.device)
        for b in self.blocks:
            x, probs, topi, a = b(x, pad_mask); aux = aux + a
            if collect: info.append((probs, topi))
        return self.head(self.norm(x)), info, aux
