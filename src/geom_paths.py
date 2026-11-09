"""Measures for prereg/PREREG_GEOMETRY_PATHS.md (#12).
Usage: python src/geom_paths.py <tag> <data>      data: mixed250k (Kociemba) | cfop250k (CFOP)
       -> out/geom/<tag>__<data>.json
<data> may be the model's own training data or the other family's ("cross" runs: a Kociemba model's routing on
CFOP positions, and vice versa). Labels always come from <data>: phase (Kociemba) or stage (CFOP).
Decision token, every layer, the same 1,200 held-out solves as every earlier analysis; halves A/B = even/odd solve.

PART A, geometry vs selection, at each layer's router input h (the MoE's input, after the block's RMSNorm).
  eta2(M) = tr(M Bc M^T) / tr(M T M^T): the label's share of the within-S variance of the readout M h.
     T = within-S scatter; Bc = between-(S, label)-cell scatter minus its mean over NPERM within-S label permutations.
     S = moves remaining; 'uncond' = S constant (descriptive).
  eta2_h      = eta2(I): the label's share of the variance of h itself; its permutation distribution gives eta2_h_perm_p99.
  ceiling     = the best label share any linear readout reaches, CROSS-FITTED: the top generalised eigenvector of
                (Bc, T) on one half, scored on the other half (mean of both directions). Half parts reuse the
                all-positions ceiling of the same conditioning.
  eta2_z      = eta2(C W): the trained router (C centres the 8 logits; the softmax ignores a common shift).
  r           = eta2_z / ceiling: the router's label share as a fraction of the best possible linear readout.
  null 'pca'  (PRIMARY): routers with the trained router's singular values reading the top-7 principal subspace of h
                in a Haar-random orientation within it (NROT draws) = a router that reads only the dominant geometry.
  null 'haar' (descriptive): the trained router's singular values, Haar-random orientation in all of R^d.
PART B, expert paths: I(R) = [CE(label | S) - CE(label | S, R)] / H(label | S), bits, gradient-boosted classifiers
  (fixed iterations, no early stopping) fitted on one half of the solves and scored on the other (both directions).
  R = sorted top-2 pair of the decision token at one layer, at L0-4, or at all layers. Null: routing rows permuted
  within S x half, all layers. Also: entropy of the decision token's pair distribution per layer (to check that the
  cross runs are not degenerate)."""
import sys, os, json, itertools, warnings, numpy as np, torch, scipy.linalg as sl
warnings.filterwarnings('ignore', message='.*encountered in matmul')   # spurious on macOS Accelerate; results asserted finite
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
from pairs_analysis import data_for
from explore_cfop_analyze import cent
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('GEOM_OUT', f'{R}/out/geom')   # override only for smoke tests
NPERM, NROT, KPC = 50, 1000, 7

def capture(m, X, bs=2048):
    """Decision-token router input h and sorted top-2 experts at every layer."""
    L = len(m.blocks); H = [[] for _ in range(L)]; T = [[] for _ in range(L)]; hooks = []
    for l, b in enumerate(m.blocks):
        hooks.append(b.ffn.register_forward_pre_hook(lambda mod, inp, l=l: H[l].append(inp[0][:, 0].detach().float().cpu())))
        hooks.append(b.ffn.register_forward_hook(lambda mod, inp, o, l=l: T[l].append(o[2].reshape(-1, 55, o[2].shape[-1])[:, 0].cpu())))
    with torch.no_grad():
        for i in range(0, len(X), bs): m(X[i:i + bs].to(A.DEV))
    for h in hooks: h.remove()
    return [torch.cat(h).double().numpy() for h in H], [np.sort(torch.cat(t).numpy(), 1) for t in T]

def codes(x): return np.unique(x, return_inverse=True)[1]

def gsum(c, X):                                   # per-group sums of the rows of X (group codes c), via a sparse matmul
    import scipy.sparse as sp
    return np.asarray(sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(c.max() + 1, len(c))) @ X)

def within_perm(lab, groups, rng):
    lp = lab.copy()
    for v in np.unique(groups):
        ix = np.where(groups == v)[0]; lp[ix] = lab[rng.permutation(ix)]
    return lp

def scatters(h, S, lab, rng):
    """Within-S scatter T; corrected between-cell scatter Bc = B - mean(B_perm); permutation values of eta2_h."""
    s = codes(S); N = len(h); mu = gsum(s, h)
    hc = h - (mu / np.bincount(s)[:, None])[s]; T = hc.T @ hc / N
    def between(l):
        c = codes(s * (l.max() + 1) + l); sm = gsum(c, hc)
        return (sm.T / np.bincount(c)) @ sm / N
    B = between(lab); Bp = [between(within_perm(lab, s, rng)) for _ in range(NPERM)]; B0 = sum(Bp) / NPERM
    tT = np.trace(T); perm_eta = [float(np.trace(b - B0) / tT) for b in Bp]
    return T, B - B0, perm_eta

def eta(M, T, Bc): return float(np.trace(M @ Bc @ M.T) / np.trace(M @ T @ M.T))

def top_dir(T, Bc):
    w, V = sl.eigh((Bc + Bc.T) / 2, T + 1e-9 * np.trace(T) / len(T) * np.eye(len(T))); return V[:, -1]

def part_a(h, W, S, lab, half, rng, ceiling=None):
    T, Bc, perm_eta = scatters(h, S, lab, rng); E = W.shape[0]; Cc = np.eye(E) - 1. / E; M = Cc @ W
    if ceiling is None:                                    # cross-fitted best linear readout
        sc = {k: scatters(h[half == k], S[half == k], lab[half == k], rng)[:2] for k in (0, 1)}
        ceiling = float(np.mean([eta(top_dir(*sc[a])[None], *sc[1 - a]) for a in (0, 1)]))
    eh = float(np.trace(Bc) / np.trace(T)); ez = eta(M, T, Bc)
    U, sv, _ = np.linalg.svd(M); r_ = int((sv > 1e-8 * sv[0]).sum()); k = min(KPC, r_)
    Vp = np.linalg.eigh(T)[1][:, ::-1][:, :k]                                     # top-k principal directions of h
    null = {'pca': [], 'haar': []}
    for _ in range(NROT):
        q, rr = np.linalg.qr(rng.randn(k, k)); q = q * np.sign(np.diag(rr))
        null['pca'].append(eta(U[:, :k] @ np.diag(sv[:k]) @ (Vp @ q).T, T, Bc))
        q, rr = np.linalg.qr(rng.randn(W.shape[1], W.shape[1])); q = q * np.sign(np.diag(rr))
        null['haar'].append(eta(M @ q, T, Bc))
    out = dict(eta2_h=eh, eta2_h_perm_p99=float(np.percentile(perm_eta, 99)), eta2_z=ez, ceiling=ceiling,
               r=ez / ceiling if ceiling > 0 else None, router_rank=r_)
    for nm, v in null.items():
        v = np.array(v); assert np.isfinite(v).all()
        out[f'null_{nm}_median'] = float(np.median(v)); out[f'null_{nm}_p99'] = float(np.percentile(v, 99))
        out[f'pct_vs_{nm}'] = float((v < ez).mean())
    assert np.isfinite([eh, ez, ceiling]).all()
    return out

def part_b(top, S, lab, half, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    E = 8; pid = {p: i for i, p in enumerate(itertools.combinations(range(E), 2))}
    P = np.stack([np.array([pid[(a, b)] for a, b in t]) for t in top], 1)          # (N, layers) pair codes
    Lr = P.shape[1]; g = codes(S) * 2 + half; perm = np.arange(len(S))
    for v in np.unique(g):
        ix = np.where(g == v)[0]; perm[ix] = rng.permutation(ix)
    ent = []
    for l in range(Lr):
        p = np.bincount(P[:, l], minlength=len(pid)) / len(P); ent.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()))
    sets = {f'L{l}': [l] for l in range(Lr)} | {'L0-4': list(range(Lr - 1)), 'all': list(range(Lr)), 'null_all': list(range(Lr))}
    out = {'pair_entropy_bits': ent}
    for tr_h, name in ((0, 'fit_A_score_B'), (1, 'fit_B_score_A')):
        tr, te = half == tr_h, half != tr_h
        def ce(F, cat):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                                 categorical_features=cat, early_stopping=False, random_state=0)
            clf.fit(F[tr], lab[tr]); pr = clf.predict_proba(F[te]); k = np.searchsorted(clf.classes_, lab[te])
            assert (clf.classes_[k] == lab[te]).all()
            return float(-np.log2(np.clip(pr[np.arange(te.sum()), k], 1e-12, None)).mean())
        Sc = S[:, None].astype(float); base = ce(Sc, [False]); Hs = cent(lab[te], S[te]); r = {'H_label_given_S': Hs, 'CE_S': base}
        for k, ls in sets.items():
            Rf = P[perm][:, ls] if k == 'null_all' else P[:, ls]
            r[k] = (base - ce(np.concatenate([Sc, Rf], 1), [False] + [True] * len(ls))) / Hs
        out[name] = r
    return out

def main(tag, data):
    m, _ = A.load(tag); X, lab, _, S, half, NAMES = data_for(data, int(os.environ.get('GEOM_SOLVES', 1200)))
    assert len(half) == len(lab) == len(S)
    H, top = capture(m, X); rng = np.random.RandomState(0); Z = np.zeros_like(S)
    out = dict(tag=tag, data=data, n_positions=int(len(lab)), label_classes=sorted(NAMES.values()), partA={}, partB=None)
    for l, b in enumerate(m.blocks):
        W = b.ffn.router.weight.detach().float().cpu().double().numpy(); o = out['partA'][f'L{l}'] = {}
        o['given_S'] = part_a(H[l], W, S, lab, half, rng); o['uncond'] = part_a(H[l], W, Z, lab, half, rng)
        for hn, k in (('A', 0), ('B', 1)):
            mk = half == k
            o[f'given_S_half{hn}'] = part_a(H[l][mk], W, S[mk], lab[mk], half[mk], rng, ceiling=o['given_S']['ceiling'])
    out['partB'] = part_b(top, S, lab, half, rng)
    os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{tag}__{data}.json'; json.dump(out, open(fn, 'w'), indent=1)
    print(f'{tag} on {data}: wrote {fn} ({len(lab)} positions)', flush=True)

if __name__ == '__main__':
    main(*sys.argv[1:3])
