"""Measures for prereg/PREREG_PATHS_FRESH.md (#13): replication of #12 Part B (expert paths) on FRESH positions.
Usage: python src/paths_fresh.py <tag>      -> out/paths_fresh/<tag>.json
  tag = a trained model in out/prod, or random_s<seed>: an UNTRAINED model of the same architecture (6L, d256, 4 heads,
  mixtral preset) initialised with torch.manual_seed(seed) -- its routing path is a random hash of the cube state.
Positions: CFOP (cfop250k) held-out solves 1,200-2,399 of the RandomState(0) held-out ordering (never used before);
asserted disjoint from the training solves perm[:ntr]. Label: stage. Halves A/B = even/odd fresh solve.
I(R) = [CE(stage | S) - CE(stage | S, R)] / H(stage | S): gradient-boosted classifiers exactly as geom_paths.part_b
(200 iterations, 15 leaves, L2 = 1, no early stopping), fitted on one half of the solves, scored on the other.
Feature sets: each layer's sorted top-2 pair; L0-4; all six layers; 'additive' = all six layers with interaction
constraints (S may interact with each layer, layers may not interact with each other), compared with 'all_onehot', the
same features unconstrained. Both use one-hot pair columns and one copy of S per layer, because sklearn 1.7.2 does not
enforce multi-feature interaction constraints on categorical features, and a shared S feature would let the union rule
re-admit cross-layer interactions (both verified on synthetic XOR data);
null = routing rows permuted within S x half. Computed on all fresh positions and, descriptively, on the fresh positions
whose exact state never occurs in the training split ('nondup')."""
import sys, os, json, itertools, warnings, numpy as np, torch
warnings.filterwarnings('ignore', message='.*encountered in matmul')
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
from geom_paths import capture, codes
from explore_cfop_analyze import cent
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('PF_OUT', f'{R}/out/paths_fresh')   # override only for smoke tests

def fresh(first=1200, n=1200, data='cfop250k'):
    d = np.load(f'{R}/data/{data}.npz'); sl = d['sol_len']; N = len(sl); perm = np.random.RandomState(0).permutation(N)
    ntr = int(N * 0.9); te = perm[ntr:][first:first + n]                         # the training split is perm[:ntr]
    assert len(te) == n and not set(te.tolist()) & set(perm[:ntr].tolist())
    off = np.concatenate([[0], np.cumsum(sl.astype(np.int64))])
    rows = lambda ids: np.concatenate([np.arange(off[i], off[i + 1]) for i in ids])
    r = rows(te); t = d['pos'][r].astype(int); L = np.concatenate([np.full(int(sl[i]), int(sl[i])) for i in te])
    half = np.concatenate([np.full(int(sl[i]), k % 2) for k, i in enumerate(te)])
    states = d['states']; w = np.random.RandomState(1).randint(1, 2 ** 62, size=states.shape[1]).astype(np.int64)
    hsh = lambda X: (X.astype(np.int64) * w).sum(1)                              # 64-bit hash of a state (wraps; fine)
    tr_rows = rows(perm[:ntr]); train_h = np.concatenate([hsh(states[tr_rows[i:i + 2_000_000]]) for i in range(0, len(tr_rows), 2_000_000)])
    dup = np.isin(hsh(states[r]), train_h)
    return torch.from_numpy(states[r].astype(np.int64)), d['phase'][r].astype(int), L - t, half, dup

def model(tag):
    if not tag.startswith('random_s'): return A.load(tag)[0]
    from model_prod import ProdStateModel
    torch.manual_seed(int(tag[len('random_s'):])); return ProdStateModel(d=256, L=6, nh=4, preset='mixtral').to(A.DEV).eval()

def info(top, S, lab, half, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pid = {p: i for i, p in enumerate(itertools.combinations(range(8), 2))}
    P = np.stack([np.array([pid[(a, b)] for a, b in t]) for t in top], 1); Lr = P.shape[1]
    g = codes(S) * 2 + half; perm = np.arange(len(S))
    for v in np.unique(g):
        ix = np.where(g == v)[0]; perm[ix] = rng.permutation(ix)
    ent = []
    for l in range(Lr):
        p = np.bincount(P[:, l], minlength=len(pid)) / len(P); ent.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()))
    sets = {f'L{l}': list(range(l, l + 1)) for l in range(Lr)} | {'L0-4': list(range(Lr - 1)), 'all': list(range(Lr)),
                                                                  'null_all': list(range(Lr))}
    out = {'pair_entropy_bits': ent}
    for tr_h, name in ((0, 'fit_A_score_B'), (1, 'fit_B_score_A')):
        tr, te = half == tr_h, half != tr_h
        def ce(F, cat, cst=None):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                                 categorical_features=cat, early_stopping=False, random_state=0, interaction_cst=cst)
            clf.fit(F[tr], lab[tr]); pr = clf.predict_proba(F[te]); k = np.searchsorted(clf.classes_, lab[te])
            assert (clf.classes_[k] == lab[te]).all()
            return float(-np.log2(np.clip(pr[np.arange(te.sum()), k], 1e-12, None)).mean())
        Sc = S[:, None].astype(float); base = ce(Sc, [False]); Hs = cent(lab[te], S[te])
        r = {'H_label_given_S_bits': Hs, 'CE_S_bits': base}
        for k, ls in sets.items():
            Rf = P[perm][:, ls] if k == 'null_all' else P[:, ls]
            c = ce(np.concatenate([Sc, Rf], 1), [False] + [True] * len(ls)); r[k] = (base - c) / Hs; r[f'{k}_CE_bits'] = c
        OH = np.concatenate(sum([[Sc, np.eye(len(pid))[P[:, l]]] for l in range(Lr)], []), 1)     # [S, onehot(L0), S, onehot(L1), ...]
        w = 1 + len(pid); cst = [list(range(l * w, (l + 1) * w)) for l in range(Lr)]
        for k, c in (('all_onehot', None), ('additive', cst)):
            v = ce(OH, None, c); r[k] = (base - v) / Hs; r[f'{k}_CE_bits'] = v
        out[name] = r
    return out

def main(tag):
    m = model(tag); n = int(os.environ.get('PF_SOLVES', 1200)); X, st, S, half, dup = fresh(n=n)
    assert len(st) == len(S) == len(half) == len(dup) and set(np.unique(st)) == {1, 2, 3, 4}
    _, top = capture(m, X); nd = ~dup
    out = dict(tag=tag, data='cfop250k', solves='held-out 1200-2399', n_positions=int(len(st)), dup_fraction=float(dup.mean()),
               all=info(top, S, st, half, np.random.RandomState(0)),
               nondup=info([t[nd] for t in top], S[nd], st[nd], half[nd], np.random.RandomState(0)))
    os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{tag}.json'; json.dump(out, open(fn, 'w'), indent=1)
    print(f'{tag}: wrote {fn} ({len(st)} positions)', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
