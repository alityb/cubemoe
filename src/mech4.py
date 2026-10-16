"""Experiment 4: where is the phase direction built? (production-style Kociemba policy, decision token)

  a  emergence: phase linear-probe accuracy and separation of the CLS residual after every attention / MoE sublayer
  b  direct router attribution (DRA): the final router's phase-2-vs-phase-1 logit gap, read in residual space as
     r = (W_router[e2] - W_router[e1]) * n2.gain with the final pre-router RMS scale frozen, is an exact sum over every
     component that writes to the CLS residual (each attention head, each MoE layer; CLS embedding is constant).
     Report each component's share of the ph2-minus-ph1 gap.
  c  causal: mean-ablate each head (all positions, per-position dataset mean) and each whole layer's attention;
     measure how much of the router gap survives, phase-2 expert top-1 use by phase, accuracy, G1 mass on phase 2
  d  concept: G1 = (no twisted corners) AND (no flipped edges) AND (slice edges in slice). Among phase-1 positions,
     where does the router gap sit for states meeting 0, 1 or 2 of the three conditions? AND-like vs additive.
Usage: python src/mech4.py <tag>   -> out/prod/<tag>_mech.json, key 'exp4'
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from mech import G1
from sklearn.linear_model import LogisticRegression
R = '/Users/alityb/projects/cubemoe'

def phase_experts(m, X, ph, L):
    H = []
    hk = m.blocks[L].n2.register_forward_hook(lambda mod, i, o: H.append(o[:, 0].detach().float().cpu()))
    with torch.no_grad():
        for i in range(0, len(X), 1024): m(X[i:i + 1024].to(A.DEV))
    hk.remove()
    with torch.no_grad(): top1 = m.blocks[L].ffn.router(torch.cat(H).to(A.DEV)).argmax(-1).cpu().numpy()
    E = m.blocks[L].ffn.E
    e1 = max(range(E), key=lambda e: (top1[ph == 1] == e).mean() - (top1[ph == 2] == e).mean())
    e2 = max(range(E), key=lambda e: (top1[ph == 2] == e).mean() - (top1[ph == 1] == e).mean())
    return e1, e2

def capture_components(m, X, L, bs=1024):
    """CLS-row writes of every component, residual after every sublayer, pre-router residual at layer L,
    and per-position means of every attention head's pre-o input (for mean ablation)."""
    nl = len(m.blocks); d = m.cls.shape[-1]; C = {}; sums = [torch.zeros(55, d, device=A.DEV) for _ in range(nl)]; hooks = []
    buf = {}
    for l, b in enumerate(m.blocks):
        def pre_o(mod, inp, l=l):
            y = inp[0].float(); sums[l] += y.sum(0); buf[('o', l)] = y[:, 0].detach().cpu()
        def moe_out(mod, inp, o, l=l): buf[('moe', l)] = o[0][:, 0].detach().float().cpu()
        def mid(mod, inp, l=l): buf[('mid', l)] = inp[0][:, 0].detach().float().cpu()
        def post(mod, inp, o, l=l): buf[('post', l)] = o[0][:, 0].detach().float().cpu()
        hooks += [b.attn.o.register_forward_pre_hook(pre_o), b.ffn.register_forward_hook(moe_out),
                  b.n2.register_forward_pre_hook(mid), b.register_forward_hook(post)]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            m(X[i:i + bs].to(A.DEV))
            for k, v in buf.items(): C.setdefault(k, []).append(v)
    for h in hooks: h.remove()
    return {k: torch.cat(v) for k, v in C.items()}, [s / len(X) for s in sums]

def run(tag, n_solves=1200):
    m, cfg = A.load(tag); X, lab = A.heldout(n_solves); ph = lab['phase']; L = len(m.blocks) - 1
    nh = m.blocks[0].attn.nh; dh = m.blocks[0].attn.dh; nl = len(m.blocks); out = {}
    e1, e2 = phase_experts(m, X, ph, L); C, means = capture_components(m, X, L)
    rng = np.random.RandomState(0); tr = rng.rand(len(ph)) < 0.7; y2 = ph == 2

    # a. emergence (baseline: the same probe on one-hot raw stickers)
    oh = np.eye(6, dtype=np.float32)[X.numpy()].reshape(len(X), -1)
    raw_acc = float((LogisticRegression(max_iter=3000).fit(oh[tr], y2[tr]).predict(oh[~tr]) == y2[~tr]).mean())
    out['raw_sticker_probe_acc'] = raw_acc
    emer = []
    for l in range(nl):
        for kind in ('mid', 'post'):
            Z = C[(kind, l)].numpy(); Z = Z / np.sqrt((Z ** 2).mean(1, keepdims=True) + 1e-6)
            acc = float((LogisticRegression(max_iter=2000).fit(Z[tr], y2[tr]).predict(Z[~tr]) == y2[~tr]).mean())
            dm = Z[tr & y2].mean(0) - Z[tr & ~y2].mean(0); p = Z[~tr] @ dm; a, b = p[y2[~tr]], p[~y2[~tr]]
            sep = float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var())))
            emer.append(dict(layer=l, after='attention' if kind == 'mid' else 'MoE', probe_acc=acc, dprime=sep))
    out['emergence'] = emer
    print(f"EXP4 {tag}  phase-1 expert e{e1}, phase-2 expert e{e2}\n  a. phase readable from the decision token's residual (probe acc / d'); raw one-hot stickers: {raw_acc*100:.1f}%")
    for r in emer: print(f"     L{r['layer']} after {r['after']:9s}: {r['probe_acc']*100:5.1f}%  d'={r['dprime']:5.2f}")

    # b. direct router attribution
    moe = m.blocks[L].ffn; Wr = moe.router.weight.detach().float().cpu()
    rdir = (Wr[e2] - Wr[e1]) * m.blocks[L].n2.w.detach().float().cpu()
    xr = C[('mid', L)]; rms = torch.sqrt(xr.pow(2).mean(-1) + m.blocks[L].n2.eps)
    gap_true = (xr @ rdir) / rms
    comps = {}
    for l in range(nl):
        Wo = m.blocks[l].attn.o.weight.detach().float().cpu(); yo = C[('o', l)]
        for h in range(nh):
            sl = slice(h * dh, (h + 1) * dh); comps[f'L{l}H{h}'] = (yo[:, sl] @ Wo[:, sl].T) @ rdir / rms
        if l < L: comps[f'L{l}MoE'] = (C[('moe', l)] @ rdir) / rms
    cls_c = (m.cls[0, 0].detach().float().cpu() @ rdir) / rms
    recon = sum(comps.values()) + cls_c; err = float((recon - gap_true).abs().max())
    tot = float(gap_true[y2].mean() - gap_true[~y2].mean())
    dra = {k: float(v[y2].mean() - v[~y2].mean()) / tot for k, v in comps.items()}
    dra['CLS(const-norm)'] = float(cls_c[y2].mean() - cls_c[~y2].mean()) / tot
    out['dra'] = dict(share_of_gap=dra, gap=tot, recon_max_err=err)
    top = sorted(dra.items(), key=lambda kv: -abs(kv[1]))[:8]
    print(f"  b. router phase gap = {tot:.2f} logits; exact decomposition (max recon err {err:.1e}). Largest shares:")
    print("     " + ", ".join(f"{k} {v*100:+.0f}%" for k, v in top))
    byl = {f'L{l}': sum(v for k, v in dra.items() if k.startswith(f'L{l}H')) for l in range(nl)}
    byl.update({f'L{l}MoE': dra[f'L{l}MoE'] for l in range(L)})
    print("     attention by layer: " + ", ".join(f"L{l} {byl[f'L{l}']*100:+.0f}%" for l in range(nl))
          + " | MoE by layer: " + ", ".join(f"L{l} {byl[f'L{l}MoE']*100:+.0f}%" for l in range(L)))

    # c. causal mean-ablation
    Y = torch.from_numpy(lab['move']); state = {'abl': None}
    def ablate(mod, inp, l):
        if state['abl'] is None or state['abl'][0] != l: return None
        y = inp[0].clone(); T = y.shape[1]
        for h in state['abl'][1]: y[:, :, h * dh:(h + 1) * dh] = means[l][:T, h * dh:(h + 1) * dh].to(y.dtype)
        return (y,)
    hooks = [m.blocks[l].attn.o.register_forward_pre_hook(lambda mod, inp, l=l: ablate(mod, inp, l)) for l in range(nl)]
    H = []; hk = m.blocks[L].n2.register_forward_hook(lambda mod, i, o: H.append(o[:, 0].detach().float()))
    def measure():
        H.clear(); P = []
        with torch.no_grad():
            for i in range(0, len(X), 1024): P.append(F.softmax(m(X[i:i + 1024].to(A.DEV))[0].float(), -1).cpu())
            h = torch.cat(H); lg = moe.router(h).float().cpu()
        P = torch.cat(P); g = (lg[:, e2] - lg[:, e1]).numpy(); t1 = lg.argmax(-1).numpy()
        return dict(gap=float(g[y2].mean() - g[~y2].mean()), e2_top1_ph1=float((t1[~y2] == e2).mean()),
                    e2_top1_ph2=float((t1[y2] == e2).mean()), acc=float((P.argmax(-1) == Y).float().mean()),
                    g1_mass_ph2=float(P[y2][:, G1].sum(-1).mean()), g1_mass_ph1=float(P[~y2][:, G1].sum(-1).mean()))
    base = measure(); abl = {}
    for l in range(nl):
        for hs, name in [((h,), f'L{l}H{h}') for h in range(nh)] + [(tuple(range(nh)), f'L{l}all')]:
            state['abl'] = (l, hs); r = measure(); state['abl'] = None
            r['gap_retained'] = r['gap'] / base['gap']; abl[name] = r
    for h in hooks: h.remove()
    hk.remove()
    out['ablation'] = dict(base=base, runs=abl)
    print(f"  c. mean-ablation (base: gap {base['gap']:.2f}, e2 top-1 ph1/ph2 {base['e2_top1_ph1']*100:.0f}%/{base['e2_top1_ph2']*100:.0f}%, "
          f"acc {base['acc']*100:.1f}%, G1 mass ph1/ph2 {base['g1_mass_ph1']*100:.0f}%/{base['g1_mass_ph2']*100:.0f}%)")
    for k, r in abl.items():
        if k.endswith('all') or r['gap_retained'] < 0.85:
            print(f"     {k:7s}: gap retained {r['gap_retained']*100:5.1f}% | e2 top-1 ph1/ph2 {r['e2_top1_ph1']*100:3.0f}%/{r['e2_top1_ph2']*100:3.0f}%"
                  f" | acc {r['acc']*100:5.1f}% | G1 mass ph1/ph2 {r['g1_mass_ph1']*100:3.0f}%/{r['g1_mass_ph2']*100:3.0f}%")

    # d. concept: AND of three G1 conditions?
    g = gap_true.numpy(); conds = np.stack([lab['tw'] == 0, lab['fl'] == 0, lab['so'] == 0], 1); nsat = conds.sum(1)
    lo, hi = g[~y2].mean(), g[y2].mean(); concept = {}
    print("  d. router gap among positions by how many G1 conditions hold (0 = phase-1 mean, 100 = phase-2 mean):")
    for k in range(4):
        for pset, pname in ((~y2, 'phase1'), (y2, 'phase2')):
            sel = pset & (nsat == k)
            if sel.sum() < 20: continue
            v = float((g[sel].mean() - lo) / (hi - lo)); concept[f'{pname}_{k}of3'] = dict(n=int(sel.sum()), pos=v)
            print(f"     {pname} meeting {k}/3: n={int(sel.sum()):5d}  position {v*100:6.1f}")
    names = ('corners oriented', 'edges oriented', 'slice edges in slice')
    for j in range(3):
        o = [i for i in range(3) if i != j]; sel = (~y2) & conds[:, o].all(1) & ~conds[:, j]
        if sel.sum() >= 20:
            v = float((g[sel].mean() - lo) / (hi - lo)); concept[f'missing_only_{names[j]}'] = dict(n=int(sel.sum()), pos=v)
            print(f"     phase 1, only '{names[j]}' missing: n={int(sel.sum()):5d}  position {v*100:6.1f}")
    out['concept'] = concept
    out.update(phase1_expert=e1, phase2_expert=e2)
    path = f'{R}/out/prod/{tag}_mech.json'
    try: full = json.load(open(path))
    except Exception: full = {}
    full['exp4'] = out; json.dump(full, open(path, 'w'), indent=1, default=float)

if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1200)
