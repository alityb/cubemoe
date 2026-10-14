"""Vectorised beam search with a state-input policy (model(x) -> (logits, info, aux)). Shared by
modal/prod.py and local tests. Device-agnostic; bf16 autocast only on CUDA."""
import contextlib

def _ac(dev):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(dev).startswith("cuda") else contextlib.nullcontext()

def beam_solve(model, start, width, budget, dev):
    """GPU beam search over the policy's log-probs; dedupes positions each step. Returns moves used or None."""
    import torch, torch.nn.functional as F, cube_fast as C
    MP = torch.from_numpy(C.MOVE_PERM).to(dev); SOLVED = torch.from_numpy(C.SOLVED.astype("int64")).to(dev)
    g = torch.Generator(device="cpu").manual_seed(0); H = torch.randint(1, 2**61, (54,), generator=g).to(dev)
    beam = start[None].long(); score = torch.zeros(1, device=dev)
    if (beam[0] == SOLVED).all(): return 0
    for t in range(budget):
        with torch.no_grad(), _ac(dev):
            lg, _, _ = model(beam)
        lp = F.log_softmax(lg.float(), -1)                                       # (W, 18)
        cand = beam[:, MP]                                                       # (W, 18, 54)
        if (cand == SOLVED).all(-1).any(): return t + 1
        sc = (score[:, None] + lp).reshape(-1); cand = cand.reshape(-1, 54)
        h = (cand * H).sum(-1)
        uniq, inv = torch.unique(h, return_inverse=True)
        best = torch.full((len(uniq),), -1e30, device=dev).scatter_reduce(0, inv, sc, "amax")
        keep = torch.nonzero(sc == best[inv]).flatten()
        k_inv = inv[keep]; order = torch.argsort(k_inv, stable=True); k_inv = k_inv[order]; keep = keep[order]
        firstmask = torch.ones_like(k_inv, dtype=torch.bool); firstmask[1:] = k_inv[1:] != k_inv[:-1]
        keep = keep[firstmask]
        top = torch.topk(sc[keep], min(width, len(keep))).indices
        beam = cand[keep[top]]; score = sc[keep[top]]
    return None

