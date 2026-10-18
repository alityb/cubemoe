"""Mechanism measures for the RL routing study (prereg/PREREG_RL_ROUTING.md, descriptive, no verdict).
On 300 held-out CFOP solves' positions, RL model vs its base, per layer:
  routing_in_place   share of tokens whose top-1 expert changes (decision token; all tokens)
  routing_weights    same, but new router applied to the BASE model's router inputs (router weights alone)
  expert_change      base router inputs and base top-2 routing, new experts: ||y_new - y_base|| / ||y_base||
Usage: python src/rl_mech.py <tag> <run> [<run> ...]  -> out/rl/mech_<run>.json"""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from prod_cfop import heldout_cfop
from rl_eval import load_run, OUT
from rl_common import amp

@torch.no_grad()
def collect(m, X, bs=1024):
    """Per layer: router input (n2 output) and top-1 expert, for every token."""
    L = len(m.blocks); H = [[] for _ in range(L)]; T = [[] for _ in range(L)]; hooks = []
    for l, b in enumerate(m.blocks):
        hooks.append(b.n2.register_forward_hook(lambda mod, i, o, l=l: H[l].append(o.detach().float().cpu())))
    for i in range(0, len(X), bs):
        with amp(A.DEV): _, info, _ = m(X[i:i + bs].to(A.DEV), collect=True)
        for l in range(L): T[l].append(info[l][1].reshape(-1, 55, m.blocks[l].ffn.k)[:, :, 0].cpu())
    for h in hooks: h.remove()
    return [torch.cat(h) for h in H], [torch.cat(t) for t in T]

@torch.no_grad()
def moe_out(moe, h, topi, topv):
    y = torch.zeros_like(h)
    for e in range(moe.E):
        for slot in range(topi.shape[1]):
            sel = topi[:, slot] == e
            if sel.any(): y[sel] += topv[sel, slot, None] * moe.expert(e, h[sel]).float()
    return y

def main(tag, runs, n_solves=300):
    X, _ = heldout_cfop(n_solves); m0 = load_run(tag, 'base'); H0, T0 = collect(m0, X)
    rng = np.random.RandomState(0); sub = torch.from_numpy(rng.choice(H0[0].shape[0] * 55, 20000, replace=False))
    for run in runs:
        m1 = load_run(tag, run); H1, T1 = collect(m1, X); res = []
        for l, (b0, b1) in enumerate(zip(m0.blocks, m1.blocks)):
            h = H0[l].reshape(-1, H0[l].shape[-1])[sub].to(A.DEV)
            p0 = torch.softmax(b0.ffn.router(h).float(), -1); p1 = torch.softmax(b1.ffn.router(h).float(), -1)
            tv0, ti0 = p0.topk(b0.ffn.k, -1); tv0 = tv0 / tv0.sum(-1, keepdim=True)
            y0 = moe_out(b0.ffn, h, ti0, tv0); y1 = moe_out(b1.ffn, h, ti0, tv0)
            res.append(dict(layer=l, routing_in_place_decision=float((T0[l][:, 0] != T1[l][:, 0]).float().mean()),
                            routing_in_place_all=float((T0[l] != T1[l]).float().mean()),
                            routing_weights_top1=float((p0.argmax(-1) != p1.argmax(-1)).float().mean()),
                            expert_change=float((y1 - y0).norm() / y0.norm())))
        json.dump(res, open(f'{OUT}/mech_{run}.json', 'w'), indent=1)
        print(run); [print(f"  L{r['layer']}: routing changed in place {r['routing_in_place_decision']*100:5.1f}% (decision) "
                           f"{r['routing_in_place_all']*100:5.1f}% (all) | by router weights alone {r['routing_weights_top1']*100:5.1f}% | "
                           f"expert output change {r['expert_change']*100:5.1f}%") for r in res]

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
