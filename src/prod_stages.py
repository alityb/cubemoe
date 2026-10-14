"""Stages B-E of the analysis of the production-style cube policy. Usage: python src/prod_stages.py <tag> <stage>
  B  attention: is it necessary, and does it ignore the 6 centre stickers (which never change)?
  C  internal features: can abstract cube properties be read from each layer, beyond the raw stickers?
  D  experts: pattern matching on stickers, or abstract cube properties? + single-expert knockouts
  E  the big picture: does the router split at the Kociemba phase boundary, and is the split used?
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from explore_cfop_analyze import corrected, cent
R = '/Users/alityb/projects/cubemoe'
MN = [f + s for f in 'URFDLB' for s in ('', "'", '2')]
G1 = [MN.index(x) for x in ('U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2')]
FB = [MN.index(x) for x in ('F', "F'", 'B', "B'")]; RL = [MN.index(x) for x in ('R', "R'", 'L', "L'")]
CENTRE_TOK = [1 + 9 * f + 4 for f in range(6)]                      # token 0 is CLS
KIND = np.array(['corner' if i % 9 in (0, 2, 6, 8) else 'edge' if i % 9 in (1, 3, 5, 7) else 'centre' for i in range(54)])

def acc(m, X, Y):
    return float((A.logits(m, X).argmax(-1).numpy() == Y).mean())

import os
SMOKE = os.environ.get('SMOKE') == '1'
def solve_rate(m, cubes, W=256):
    r = [A.beam_solve_path(m, c, W, 40) for c in cubes]; return sum(p is not None for p in r) / len(cubes)

def stage_B(m, X, lab, out):
    Y = lab['move']; cubes = A.random_cubes(3 if SMOKE else 40, 31337)
    rows = {}
    for mode in ('normal', 'zero', 'uniform'):
        with A.attention_mode(m, mode):
            rows[mode] = dict(acc=acc(m, X, Y), solve=solve_rate(m, cubes if mode == 'normal' else cubes[:max(1, len(cubes) // 2)]))
        print(f"  attention {mode:8s}: next-move accuracy {rows[mode]['acc']*100:5.1f}%   beam-256 solve {rows[mode]['solve']*100:5.1f}%", flush=True)
    per = {}
    for li in range(len(m.blocks)):
        with A.attention_mode(m, 'zero', layers={li}): per[li] = acc(m, X, Y)
    print("  accuracy with attention removed in ONE layer only: " + "  ".join(f"L{li} {v*100:.1f}%" for li, v in per.items()), flush=True)
    cap = A.capture(m, X[:6000]); share = {}
    for li, att in enumerate(cap['att']):
        a = att.mean(1)                                                 # (N, 55), average over heads
        st = a[:, 1:]; tot = a.sum(1)
        share[li] = {k: float(st[:, KIND == k].sum(1).mean() / tot.mean()) for k in ('centre', 'edge', 'corner')}
        share[li]['cls_self'] = float(a[:, 0].mean())
    print("  share of the decision token's attention going to centre / edge / corner stickers "
          "(if attention were spread evenly: 10.9% / 43.6% / 43.6%):")
    for li, s in share.items():
        print(f"    layer {li}: centres {s['centre']*100:4.1f}%  edges {s['edge']*100:4.1f}%  corners {s['corner']*100:4.1f}%  (itself {s['cls_self']*100:4.1f}%)")
    out['B'] = dict(modes=rows, per_layer_zero=per, att_share=share)

def stage_C(m, X, lab, out):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score
    cap = A.capture(m, X); sid = lab['solve']; tr = sid % 10 < 7; te = ~tr
    raw = F.one_hot(X, 6).reshape(len(X), -1).float().numpy()
    targets = {'in G1 (phase 2)': (lab['phase'] == 2).astype(int), '# misoriented edges': lab['fl'],
               '# misoriented corners': lab['tw'], '# slice edges out of slice': lab['so'],
               'moves to reach G1': lab['d1'], 'moves to solved': lab['rem']}
    feats = {'raw stickers (input)': raw}
    for li, r in enumerate(cap['res']): feats[f'layer {li}'] = r.numpy()
    tab = {}
    for tn, y in targets.items():
        tab[tn] = {}
        for fn, Z in feats.items():
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6; Zs = (Z - mu) / sd
            if tn.startswith('in G1'):
                p = LogisticRegression(max_iter=2000, C=1.0).fit(Zs[tr], y[tr]); v = float((p.predict(Zs[te]) == y[te]).mean())
            else:
                p = Ridge(alpha=10.0).fit(Zs[tr], y[tr]); v = float(r2_score(y[te], p.predict(Zs[te])))
            tab[tn][fn] = v
        print(f"  {tn:28s} " + "  ".join(f"{fn.replace('raw stickers (input)','raw')}: {v:.2f}" for fn, v in tab[tn].items()), flush=True)
    out['C'] = tab

def stage_D(m, X, lab, out):
    n = min(12000, len(X)); cap = A.capture(m, X[:n]); rng = np.random.RandomState(0)
    feats = {k: lab[k][:n] for k in ('phase', 'd1', 'fl', 'tw', 'so', 'move')}
    S0 = np.zeros(n, int); res = {}
    for li, route in enumerate(cap['route']):
        r = route.numpy()                                               # (n, 55) top-1 expert per token
        st = r[:, 1:].reshape(-1); pos = np.tile(np.arange(54), n); col = X[:n].numpy().reshape(-1)
        k = rng.choice(len(st), min(60000, len(st)), replace=False); Sk = np.zeros(len(k), int)
        lex = corrected(st[k], pos[k] * 6 + col[k], Sk, rng) / cent(st[k], Sk)
        E = r[:, 0]; H = cent(E, S0)
        abstract = {f: corrected(E, v, S0, rng) / H for f, v in feats.items()}
        single = max(corrected(E, X[:n, j].numpy(), S0, rng) / H for j in range(54) if KIND[j] != 'centre')
        res[li] = dict(sticker_tokens_by_own_position_colour=lex, cls_best_single_sticker=single, cls_abstract=abstract,
                       cls_routing_entropy_bits=float(H), reliable=bool(H > 0.5))
        print(f"  layer {li}: sticker tokens routed by their own (position, colour): {lex*100:5.1f}%   |   decision token: "
              f"best single sticker {single*100:4.1f}%, phase {abstract['phase']*100:4.1f}%, moves-to-G1 {abstract['d1']*100:4.1f}%, "
              f"bad edges {abstract['fl']*100:4.1f}%, bad corners {abstract['tw']*100:4.1f}%, next move {abstract['move']*100:4.1f}%"
              f"  [decision-routing entropy {H:.2f} bits{'' if H > 0.5 else ' -> TOO LOW, unreliable'}]", flush=True)
    out['D_routing'] = res
    # knockouts on the decision path (last layer) and a middle layer
    p0 = F.softmax(A.logits(m, X), -1); Y = torch.from_numpy(lab['move']); base = p0[torch.arange(len(Y)), Y]
    ko = {}
    for li in (len(m.blocks) // 2, len(m.blocks) - 1):
        E = m.blocks[li].ffn.E; ko[li] = {}
        for e in range(E):
            m.blocks[li].ffn._knockout = {e}
            pe = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; m.blocks[li].ffn._knockout = set()
            dd = (base - pe).numpy(); ph = lab['phase']; mv = lab['move']
            row = {'phase 1': float(dd[ph == 1].mean()), 'phase 2': float(dd[ph == 2].mean()),
                   'ph1 F/B (edge flips)': float(dd[(ph == 1) & np.isin(mv, FB)].mean()),
                   'ph1 R/L (corner twists)': float(dd[(ph == 1) & np.isin(mv, RL)].mean())}
            bt = np.array([max(dd[mv == t].mean(), 0) for t in range(18)]); q = bt / max(bt.sum(), 1e-12); q = q[q > 0]
            row['effective_moves'] = float(np.exp(-(q * np.log(q)).sum())) if bt.sum() > 1e-6 else None
            ko[li][e] = row
        print(f"  knockouts, layer {li} (drop in probability of the solver's move):")
        for e, rw in ko[li].items():
            print(f"    e{e}: phase1 {rw['phase 1']:+.3f}  phase2 {rw['phase 2']:+.3f}  | ph1 edge-flip moves {rw['ph1 F/B (edge flips)']:+.3f}  "
                  f"corner-twist moves {rw['ph1 R/L (corner twists)']:+.3f}  | effective #moves {rw['effective_moves'] or 0:.1f}", flush=True)
    out['D_knockout'] = ko

def stage_E(m, X, lab, out):
    cap = A.capture(m, X); L = len(m.blocks) - 1; E = cap['route'][L][:, 0].numpy()
    rng = np.random.RandomState(0); strat = lab['rem']; ph = lab['phase']
    H = cent(E, strat); c = corrected(E, ph, strat, rng) / H
    Esh = E.copy()
    for s in np.unique(strat):
        k = np.flatnonzero(strat == s); Esh[k] = Esh[rng.permutation(k)]
    csh = corrected(Esh, ph, strat, rng) / cent(Esh, strat)
    ceil = corrected(ph, ph, strat, rng) / max(cent(ph, strat), 1e-9)
    print(f"  last-layer routing of the decision token explained by PHASE (at fixed moves-to-solved): {c*100:.1f}% "
          f"(shuffled router {csh*100:.1f}%)", flush=True)
    nE = m.blocks[L].ffn.E
    p1 = [((E == e) & (ph == 1)).sum() / max((E == e).sum(), 1) if (E == e).sum() >= 50 else -1 for e in range(nE)]
    e1 = int(np.argmax(p1)); share = {e: float((E[ph == 1] == e).mean()) for e in range(nE)}
    print(f"  phase-1 expert: e{e1} (handles {share[e1]*100:.1f}% of phase-1 positions, {float((E[ph==2]==e1).mean())*100:.1f}% of phase-2)")
    sel = torch.from_numpy(ph == 2); Xs = X[sel]
    def g1mass(Z):
        p = F.softmax(A.logits(m, Z), -1); return float(p[:, G1].sum(1).mean())
    base = g1mass(Xs); loss = {}
    for e in range(nE):
        ov = torch.full((len(Xs), 55), -1, dtype=torch.long); ov[:, 0] = e
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xs), 2048):
                m.blocks[L].ffn._override = ov[i:i + 2048].reshape(-1).to(A.DEV)
                outs.append(F.softmax(m(Xs[i:i + 2048].to(A.DEV))[0].float(), -1).cpu())
        m.blocks[L].ffn._override = None
        loss[e] = base - float(torch.cat(outs)[:, G1].sum(1).mean())
    oth = np.median([v for e, v in loss.items() if e != e1])
    print(f"  CAUSAL: forcing phase-2 positions through the phase-1 expert destroys {loss[e1]*100:.1f}% of legal-move "
          f"probability; a typical other expert destroys {oth*100:.2f}%  (baseline legal mass {base*100:.1f}%)", flush=True)
    Y = torch.from_numpy(lab['move']); p0 = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; ko = {}
    for e in range(nE):
        m.blocks[L].ffn._knockout = {e}; pe = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; m.blocks[L].ffn._knockout = set()
        dd = (p0 - pe).numpy(); ko[e] = dict(phase1=float(dd[ph == 1].mean()), phase2=float(dd[ph == 2].mean()))
    o1 = np.median([ko[e]['phase1'] for e in ko if e != e1]); o2 = np.median([ko[e]['phase2'] for e in ko if e != e1])
    print(f"  KNOCKOUT: removing the phase-1 expert costs phase-1 positions {ko[e1]['phase1']*100:+.1f} pts and phase-2 "
          f"positions {ko[e1]['phase2']*100:+.1f} pts (typical other expert: {o1*100:+.1f} / {o2*100:+.1f})", flush=True)
    out['E'] = dict(phase_share=c, shuffled=csh, phase1_expert=e1, phase1_share=share[e1], phase2_share_of_e1=float((E[ph == 2] == e1).mean()),
                    loss_e1=loss[e1], median_other=float(oth), all_losses=loss, base=base, knockout=ko,
                    ko_e1_phase1=ko[e1]['phase1'], ko_e1_phase2=ko[e1]['phase2'], ko_other_phase1=float(o1), ko_other_phase2=float(o2))

if __name__ == '__main__':
    tag, stages = sys.argv[1], sys.argv[2]
    m, cfg = A.load(tag); X, lab = A.heldout(40 if SMOKE else 1200)
    path = f'{R}/out/prod/{tag}_analysis{"_smoke" if SMOKE else ""}.json'
    try: out = json.load(open(path))
    except Exception: out = {}
    for s in stages:
        print(f"\n=== stage {s} ===", flush=True); globals()[f'stage_{s}'](m, X, lab, out)
        json.dump(out, open(path, 'w'), indent=1, default=float)
