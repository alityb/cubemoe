"""Measures for prereg/PREREG_EXPERT_PAIRS.md (#10): how the two selected final-layer experts work together.
Usage: python src/pairs_analysis.py <tag> <data> [pos|neg]
   data: mixed250k (Kociemba) | cfop250k | cfop250k_g1pll        -> out/pairs/<tag>.json
   pos / neg: planted-backup control (prereg, "Controls")         -> out/pairs/<tag>_ctrl_<pos|neg>.json

Final layer, decision token only (the final layer's other tokens never reach the output). Same held-out positions as
every earlier analysis (1,200 solves, RandomState(0) solve-level split); halves A/B = even/odd held-out solve.
Removal of expert set Z from the final layer, three modes:
  reroute : router logits of Z set to -inf; next-ranked experts take the slots, weights renormalised (training rule;
            the knockout of all earlier analyses). PRIMARY.
  drop    : routing unchanged, outputs of Z removed, survivor keeps its trained weight.      (descriptive)
  renorm  : routing unchanged, outputs of Z removed, survivor weight set to 1.               (descriptive)
Two damage scales per position: p0 - p_Z of the teacher move, and margin0 - margin_Z, margin = teacher-move logit minus
the largest other logit (no softmax saturation)."""
import sys, os, json, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
from explore_cfop_analyze import corrected, cent
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('PAIRS_OUT', f'{R}/out/pairs')   # override only for smoke tests
NPERM = 50

def data_for(data, n_solves=1200):
    d = np.load(f'{R}/data/{data}.npz'); sl = d['sol_len']; te = np.random.RandomState(0).permutation(len(sl))[int(len(sl) * 0.9):][:n_solves]
    half = np.concatenate([np.full(int(sl[i]), k % 2) for k, i in enumerate(te)])
    if data == 'mixed250k':
        X, lab = A.heldout(n_solves, 'mixed250k'); return X, lab['phase'], lab['move'], lab['rem'], half, {1: 'phase1', 2: 'phase2'}
    import g1pll_analysis as G
    X, lab = G.heldout(data, n_solves); return X, lab['stage'], lab['move'], lab['rem'], half, {1: 'cross', 2: 'F2L', 3: 'OLL', 4: 'PLL'}

class Hook:
    """Wraps the final MoE's dispatch so that experts in Z contribute nothing (drop / renorm modes)."""
    def __init__(self, ffn): self.ffn, self.orig, self.Z, self.renorm = ffn, ffn.dispatch, None, False; ffn.dispatch = self
    def __call__(self, flat, topi, topv):
        if self.Z:
            dead = torch.isin(topi, torch.tensor(sorted(self.Z), device=topi.device)); w = topv.masked_fill(dead, 0.)
            if self.renorm: w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            topv = w
        return self.orig(flat, topi, topv)

class DormantRouter(nn.Module):
    """Planted-backup control: expert u is never selected unless expert e is knocked out (reroute)."""
    def __init__(self, ffn, e, u): super().__init__(); self.lin, self.e, self.u = ffn.router, e, u; self.ko = lambda: ffn._knockout
    def forward(self, x):
        z = self.lin(x)
        if self.e not in self.ko(): z = z.clone(); z[:, self.u] = float('-inf')
        return z

def plant(ffn, e, u, kind):
    with torch.no_grad():
        ffn.router.weight[u] = ffn.router.weight[e]                                   # u competes exactly like e
        if kind == 'pos':                                                             # ... and computes exactly like e
            for w in (ffn.w_gate, ffn.w_up, ffn.w_down): w[u] = w[e]
    ffn.router = DormantRouter(ffn, e, u)                                             # neg: u keeps its own weights

def main(tag, data, control=None):
    m, _ = A.load(tag); X, st, Y, S, half, NAMES = data_for(data, int(os.environ.get('PAIRS_SOLVES', 1200))); assert len(half) == len(Y)
    Yt = torch.from_numpy(Y); ar = torch.arange(len(Y)); ffn = m.blocks[-1].ffn; E = ffn.E
    parts = (('all', np.ones(len(Y), bool)), ('A', half == 0), ('B', half == 1))

    def run():
        z = A.logits(m, X).float(); p = F.softmax(z, -1)[ar, Yt]; zt = z[ar, Yt]
        zo = z.clone(); zo[ar, Yt] = -float('inf'); return p.numpy(), (zt - zo.max(-1).values).numpy()
    def route():
        cap = {}; h = ffn.register_forward_hook(lambda mod, i, o: cap.update(pr=o[1], ti=o[2])); top = []; pr = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                m(X[i:i + 2048].to(A.DEV)); n = cap['ti'].shape[0] // 55
                top.append(cap['ti'].reshape(n, 55, -1)[:, 0].cpu()); pr.append(cap['pr'].reshape(n, 55, -1)[:, 0].float().cpu())
        h.remove(); return torch.cat(top).numpy(), torch.cat(pr).numpy()
    hk = Hook(ffn)
    def removed(Z, mode):
        if mode == 'reroute': ffn._knockout = set(Z)
        else: hk.Z, hk.renorm = set(Z), mode == 'renorm'
        r = run(); ffn._knockout = set(); hk.Z = None; return r
    def agg(v): return {pt: {NAMES[s]: float(v[(st == s) & mk].mean()) for s in NAMES} | {'_all': float(v[mk].mean())} for pt, mk in parts}

    ctrl = None
    if control:                                   # pick e and u on the unmodified model, by fixed rules
        top0, _ = route(); p00, _ = run(); best = None
        for e in range(E):
            pe, _ = removed([e], 'reroute'); ds = {k: v for k, v in agg(p00 - pe)['all'].items() if k != '_all'}
            s = max(ds, key=ds.get)
            if ds[s] >= 0.03 and all(ds[t] <= 0.25 * ds[s] for t in ds if t != s) and (best is None or ds[s] > best[1]): best = (e, ds[s])
        e = best[0]; rate = np.array([(top0 == k).any(1).mean() for k in range(E)]); rate[e] = np.inf; u = int(rate.argmin())
        plant(ffn, e, u, control); ctrl = dict(kind=control, e=e, u=u, e_damage_before=best[1])

    top, probs = route(); p0, m0 = run(); sel = [(top == k).any(1) for k in range(E)]
    out = dict(tag=tag, data=data, control=ctrl, n_positions=int(len(Y)), stages=list(NAMES.values()),
               stage_share={NAMES[s]: float((st == s).mean()) for s in NAMES}, p0=agg(p0), margin0=agg(m0))
    # (1) stage information in the selected pair beyond the top-1 expert (permutation-corrected, NPERM permutations)
    pair = top[:, 0] * E + top[:, 1]; out['stage_info'] = {}
    for pt, mk in parts:
        rng = np.random.RandomState(0); Hs = cent(st[mk], S[mk])
        i1 = corrected(top[mk, 0], st[mk], S[mk], rng, NPERM); ip = corrected(pair[mk], st[mk], S[mk], rng, NPERM)
        out['stage_info'][pt] = dict(H=Hs, top1=i1, pair=ip, top2_only=corrected(top[mk, 1], st[mk], S[mk], rng, NPERM),
                                     increment=(ip - i1) / (Hs - i1) if Hs - i1 > 0 else None)
    # (2) single and pairwise damage, all modes, both scales; backup decomposition under reroute
    MODES = ('reroute', 'drop', 'renorm'); SETS = [(e,) for e in range(E)] + list(itertools.combinations(range(E), 2)); P, M = {}, {}
    for mo in MODES:
        for Z in SETS: P[mo, Z], M[mo, Z] = removed(list(Z), mo)
    key = lambda Z: ','.join(map(str, Z))
    out['single'] = {mo: {str(e): dict(p=agg(p0 - P[mo, (e,)]), margin=agg(m0 - M[mo, (e,)])) for e in range(E)} for mo in MODES}
    out['pair'] = {mo: {key(Z): dict(p=agg(p0 - P[mo, Z]), margin=agg(m0 - M[mo, Z])) for Z in SETS if len(Z) == 2} for mo in MODES}
    out['backup'] = {}                            # reroute, by position type: exclusive (one selected) vs co-selected
    for i, j in itertools.combinations(range(E), 2):
        Pi, Pj, Pij = P['reroute', (i,)], P['reroute', (j,)], P['reroute', (i, j)]
        excl = np.where(sel[i] & ~sel[j], Pi - Pij, 0.) + np.where(sel[j] & ~sel[i], Pj - Pij, 0.)   # partner unavailable as backup
        both = sel[i] & sel[j]; co = np.where(both, (p0 - Pij) - (p0 - Pi) - (p0 - Pj), 0.)
        out['backup'][key((i, j))] = dict(exclusive=agg(excl), coselected=agg(co), coselect_rate=agg(both.astype(float)))
    # (3) who substitutes under reroute, and selection rates
    out['substitute'] = {}
    for e in range(E):
        if sel[e].sum() == 0: continue
        pr = probs[sel[e]].copy(); pr[np.arange(sel[e].sum())[:, None], top[sel[e]]] = -1; v, c = np.unique(pr.argmax(1), return_counts=True)
        out['substitute'][str(e)] = {str(int(a)): float(b / sel[e].sum()) for a, b in zip(v, c)}
    out['select_rate_by_stage'] = {NAMES[s]: {str(e): float(sel[e][st == s].mean()) for e in range(E)} for s in NAMES}
    os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{tag}' + (f'_ctrl_{control}' if control else '') + '.json'
    json.dump(out, open(fn, 'w'), indent=1); print(f"{tag}{' control ' + control if control else ''}: wrote {fn} ({len(Y)} positions)", flush=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
