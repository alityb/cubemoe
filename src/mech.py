"""Mechanistic analysis of the production-style Kociemba policy (final MoE layer, decision token).

  exp1  what each final-layer expert writes: direct logit attribution (DLA) of its output onto the 18 moves,
        using the final RMSNorm's actual per-token scale (frozen), centred across moves. Summary per expert:
        mean push on the 8 non-G1 quarter turns (R R' L L' F F' B B') minus mean push on the 10 G1 moves.
  exp2  does the router read a phase direction? phase direction = difference of mean router inputs
        (phase 2 minus phase 1) and a logistic-probe direction; compare with the router-weight difference
        between the main phase-2 and phase-1 experts; random-direction control.
Usage: python src/mech.py <tag> [exp1|exp2|both]   -> out/prod/<tag>_mech.json
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
R = '/Users/alityb/projects/cubemoe'
MN = [f + s for f in 'URFDLB' for s in ('', "'", '2')]
G1 = [MN.index(x) for x in ('U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2')]
QT = [MN.index(x) for x in ('R', "R'", 'L', "L'", 'F', "F'", 'B', "B'")]

def collect(m, X, bs=1024):
    """Decision-token router input h, block output x_final, at the last layer."""
    L = len(m.blocks) - 1; H = []; XF = []
    hk = m.blocks[L].n2.register_forward_hook(lambda mod, i, o: H.append(o[:, 0].detach().float().cpu()))
    hb = m.blocks[L].register_forward_hook(lambda mod, i, o: XF.append(o[0][:, 0].detach().float().cpu()))
    with torch.no_grad():
        for i in range(0, len(X), bs): m(X[i:i + bs].to(A.DEV))
    hk.remove(); hb.remove()
    return torch.cat(H), torch.cat(XF), L

def exp1(m, X, lab, out):
    h, xf, L = collect(m, X); moe = m.blocks[L].ffn
    with torch.no_grad():
        logits = moe.router(h.to(A.DEV)).float(); probs = F.softmax(logits, -1)
        topv, topi = probs.topk(moe.k, -1)
        if moe.norm_topk: topv = topv / topv.sum(-1, keepdim=True)
        rms = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + m.norm.eps).to(A.DEV)
        W = (m.head.weight * m.norm.w[None]).float()                         # (18, d): head composed with norm gain
        dla = {}
        for e in range(moe.E):
            ye = moe.expert(e, h.to(A.DEV)).float()                            # what expert e would write for every token
            sel = (topi == e).any(-1)
            w = (topv * (topi == e)).sum(-1, keepdim=True)                     # its actual routing weight (0 if not selected)
            contrib = (w * ye / rms) @ W.T                                     # (N, 18) actual logit contribution
            contrib = contrib - contrib.mean(-1, keepdim=True)
            intrinsic = (ye / rms) @ W.T; intrinsic = intrinsic - intrinsic.mean(-1, keepdim=True)
            ph = torch.from_numpy(lab['phase']).to(A.DEV)
            row = {'selected_share_ph1': float(sel[ph == 1].float().mean()), 'selected_share_ph2': float(sel[ph == 2].float().mean())}
            for nm, C in (('actual', contrib), ('intrinsic', intrinsic)):
                for p in (1, 2):
                    msk = (ph == p) & (sel if nm == 'actual' else torch.ones_like(sel))
                    if msk.sum() < 50: row[f'{nm}_ph{p}'] = None; continue
                    v = C[msk].mean(0)
                    row[f'{nm}_ph{p}'] = float(v[QT].mean() - v[G1].mean())
            row['intrinsic_per_move'] = intrinsic.mean(0).cpu().tolist()
            # does the expert push toward the SPECIFIC correct move? (logit on the solver's move minus mean over moves)
            Y = torch.from_numpy(lab['move']).to(A.DEV)
            for p in (1, 2):
                msk = (ph == p) & sel
                row[f'correct_move_push_ph{p}'] = float(contrib[msk].gather(1, Y[msk][:, None]).mean()) if msk.sum() >= 50 else None
            dla[e] = row
    out['exp1'] = dla
    print("EXP1  push toward phase-1 moves (8 R/L/F/B quarter turns) minus push toward the 10 G1 moves, in logits:")
    print("      expert | routed: ph1 / ph2 | actual contribution on ph1 / ph2 tokens | intrinsic (all tokens) ph1 / ph2")
    f = lambda v: '   —  ' if v is None else f'{v:+6.2f}'
    for e, r in dla.items():
        print(f"      e{e}     | {r['selected_share_ph1']*100:4.0f}% / {r['selected_share_ph2']*100:4.0f}% | {f(r['actual_ph1'])} / {f(r['actual_ph2'])} | {f(r['intrinsic_ph1'])} / {f(r['intrinsic_ph2'])}"
              f" | push toward the solver's actual move: ph1 {f(r['correct_move_push_ph1'])} / ph2 {f(r['correct_move_push_ph2'])}")

def exp2(m, X, lab, out):
    from sklearn.linear_model import LogisticRegression
    h, xf, L = collect(m, X); moe = m.blocks[L].ffn; ph = lab['phase']
    with torch.no_grad():
        top1 = moe.router(h.to(A.DEV)).argmax(-1).cpu().numpy()
    share = lambda e, p: float((top1[ph == p] == e).mean())
    e1 = max(range(moe.E), key=lambda e: share(e, 1) - share(e, 2)); e2 = max(range(moe.E), key=lambda e: share(e, 2) - share(e, 1))
    Hn = h.numpy(); dmean = Hn[ph == 2].mean(0) - Hn[ph == 1].mean(0)
    tr = lab['solve'] % 10 < 7
    pr = LogisticRegression(max_iter=3000).fit(Hn[tr], (ph[tr] == 2)); dprobe = pr.coef_[0]
    probe_acc = float((pr.predict(Hn[~tr]) == (ph[~tr] == 2)).mean())
    Wr = moe.router.weight.detach().float().cpu().numpy(); wdiff = Wr[e2] - Wr[e1]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rng = np.random.RandomState(0); rand = [abs(cos(wdiff, rng.randn(len(wdiff)))) for _ in range(1000)]
    # how much of the router's e2-vs-e1 preference is carried by the phase direction alone
    u = dmean / np.linalg.norm(dmean); gap = Hn @ wdiff; proj = (Hn @ u) * (u @ wdiff)
    r2 = float(1 - ((gap - proj - (gap - proj).mean()) ** 2).sum() / ((gap - gap.mean()) ** 2).sum())
    out['exp2'] = dict(phase1_expert=e1, phase2_expert=e2, probe_acc=probe_acc, cos_wdiff_meandiff=cos(wdiff, dmean),
                       cos_wdiff_probe=cos(wdiff, dprobe), cos_meandiff_probe=cos(dmean, dprobe),
                       random_abs_cos_95pct=float(np.percentile(rand, 95)), r2_gap_explained_by_phase_direction=r2)
    o = out['exp2']
    print(f"EXP2  phase-1 expert e{e1}, phase-2 expert e{e2}; linear probe reads the phase from the router input with {probe_acc*100:.1f}% accuracy")
    print(f"      cosine(router weights e{e2} minus e{e1}, phase direction): mean-difference {o['cos_wdiff_meandiff']:+.3f}, probe {o['cos_wdiff_probe']:+.3f}"
          f"   (random direction: |cos| < {o['random_abs_cos_95pct']:.3f} in 95% of draws)")
    print(f"      share of the router's e{e2}-vs-e{e1} preference explained by the phase direction alone: R^2 = {r2:.2f}")

def exp3(m, X, lab, out, n=6000):
    """Steering along the phase direction at the last layer's router input (decision token only)."""
    h, xf, L = collect(m, X); moe = m.blocks[L].ffn; ph = lab['phase']; Hn = h.numpy()
    dmean = Hn[ph == 2].mean(0) - Hn[ph == 1].mean(0); scale = float(np.linalg.norm(dmean)); u = torch.tensor(dmean / scale, dtype=torch.float32)
    with torch.no_grad(): top1 = moe.router(h.to(A.DEV)).argmax(-1).cpu().numpy()
    e1 = max(range(moe.E), key=lambda e: (top1[ph == 1] == e).mean() - (top1[ph == 2] == e).mean())
    e2 = max(range(moe.E), key=lambda e: (top1[ph == 2] == e).mean() - (top1[ph == 1] == e).mean())
    rng = np.random.RandomState(0); r = torch.tensor(rng.randn(len(dmean)), dtype=torch.float32); r = r - (r @ u) * u; r = r / r.norm()
    Y = torch.from_numpy(lab['move']); state = {'vec': None, 'mode': None}
    def router_hook(mod, inp, o):
        if state['mode'] != 'router' or state['vec'] is None: return None
        o = o.clone(); B = o.shape[0] // 55; o[torch.arange(B, device=o.device) * 55] += (mod.weight @ state['vec'].to(o.device)).to(o.dtype); return o
    def norm_hook(mod, inp, o):
        if state['mode'] != 'full' or state['vec'] is None: return None
        o = o.clone(); o[:, 0] += state['vec'].to(o.device).to(o.dtype); return o
    h1 = moe.router.register_forward_hook(router_hook); h2 = m.blocks[L].n2.register_forward_hook(norm_hook)
    def measure(sel, vec, mode):
        state['vec'], state['mode'] = vec, mode; Xs = X[sel][:n]; ys = Y[sel][:n]; P = []; T = []
        with torch.no_grad():
            for i in range(0, len(Xs), 1024):
                lg, info, _ = m(Xs[i:i + 1024].to(A.DEV), collect=True)
                P.append(F.softmax(lg.float(), -1).cpu()); ti = info[L][1].reshape(-1, 55, moe.k)[:, 0].cpu(); T.append(ti)
        state['vec'] = None; P = torch.cat(P); T = torch.cat(T)
        return dict(ph2_expert_routed=float((T == e2).any(-1).float().mean()), ph1_expert_top1=float((T[:, 0] == e1).float().mean()),
                    g1_mass=float(P[:, G1].sum(-1).mean()), p_correct=float(P[torch.arange(len(ys)), ys].mean()))
    res = {}
    for name, sel, sign in (('phase-1 positions, pushed toward phase 2', torch.from_numpy(ph == 1), +1),
                            ('phase-2 positions, pushed toward phase 1', torch.from_numpy(ph == 2), -1)):
        rows = {}
        for a in (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            for mode, vecdir in (('router', u), ('full', u), ('full_random', r)):
                vec = sign * a * scale * vecdir if a > 0 else None
                rows[f'{mode}@{a}'] = measure(sel, vec, 'full' if mode.startswith('full') else 'router')
        res[name] = rows
        print(f"\nEXP3  {name}  (strength in units of the phase-1/phase-2 mean difference; phase-1 expert e{e1}, phase-2 expert e{e2})")
        print(f"      {'strength':>8} | {'ROUTER-ONLY: ph2 expert used / G1 mass / p(correct)':>52} | {'FULL: ph2 expert / G1 mass / p(correct)':>42} | RANDOM-dir full: G1 mass")
        for a in (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            ro, fu, ra = rows[f'router@{a}'], rows[f'full@{a}'], rows[f'full_random@{a}']
            print(f"      {a:8.2f} | {ro['ph2_expert_routed']*100:14.0f}% / {ro['g1_mass']*100:5.1f}% / {ro['p_correct']*100:5.1f}%{'':13}"
                  f" | {fu['ph2_expert_routed']*100:7.0f}% / {fu['g1_mass']*100:5.1f}% / {fu['p_correct']*100:5.1f}%{'':5} | {ra['g1_mass']*100:5.1f}%")
    h1.remove(); h2.remove()
    out['exp3'] = dict(phase1_expert=e1, phase2_expert=e2, scale=scale, results=res)

if __name__ == '__main__':
    tag = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else 'both'
    m, cfg = A.load(tag); X, lab = A.heldout(1200)
    path = f'{R}/out/prod/{tag}_mech.json'
    try: out = json.load(open(path))
    except Exception: out = {}
    if which in ('exp1', 'both'): exp1(m, X, lab, out)
    if which in ('exp2', 'both'): exp2(m, X, lab, out)
    if which == 'exp3': exp3(m, X, lab, out)
    json.dump(out, open(path, 'w'), indent=1, default=float)
