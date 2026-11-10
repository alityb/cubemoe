"""Measures for prereg/PREREG_NOVEL_PATHS.md (#15): does the CFOP routing path carry stage on states never seen in training?
Usage: python src/novel_paths.py <tag>      -> $NP_OUT/<tag>.json
Positions: CFOP (cfop250k) held-out solves 4,800-7,199 of the RandomState(0) held-out ordering (never used before),
asserted disjoint from the training split; restricted to positions whose exact state never occurs in the training split
('novel'). On the #13 positions every OLL and PLL state recurs in training, so novel positions are cross and F2L
only (checked on solves 1,200-2,399 before writing the prereg): the label is cross vs F2L; S = moves remaining; halves A/B = even/odd solve. I(R) exactly as #13 (src/paths_fresh.info).
Routing path R = sorted top-2 pair of the decision token at each layer. For a C model, also the ROTATED control: at
each layer, the model's own router input h read by W q (trained router W, q a Haar-random orthogonal matrix,
RandomState(1000 + layer)): a router with the trained singular values reading the same representation in random
directions; its path is a hash of the trained representation. Per-pair biases are fitted (labels unused) so the
pair distribution matches the trained router's on the scored positions ('rotated', used by N3); the raw
rotation ('rotated_raw') is descriptive. Any novel OLL/PLL position is dropped and counted. Pair entropies reported."""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings('ignore', message='.*encountered in matmul')
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
from geom_paths import capture
from paths_fresh import fresh, model, info
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('NP_OUT', f'{R}/out/novel_paths')
FIRST, N = int(os.environ.get('NP_FIRST', 4800)), int(os.environ.get('NP_SOLVES', 2400))   # overrides only for checks on old solves

def top2(z): return np.sort(np.argsort(-z, 1)[:, :2], 1)

PAIRS = np.array([(a, c) for a in range(8) for c in range(a + 1, 8)])

def match_usage(z, target_pairs, iters=1000, lr=0.5):
    """Per-PAIR biases b (28) on the rotated pair scores z_a + z_b so that the frequency of each sorted top-2 pair under
    argmax(z_a + z_b + b) matches the trained router's pair frequencies on the same positions; labels are not used."""
    sc = z[:, PAIRS[:, 0]] + z[:, PAIRS[:, 1]]; code = {tuple(p): i for i, p in enumerate(PAIRS)}
    tgt = np.bincount([code[tuple(t)] for t in target_pairs], minlength=len(PAIRS)) / len(target_pairs)
    b = np.zeros(len(PAIRS)); eps = 1e-4; scale = sc.std()
    for _ in range(iters):
        f = np.bincount((sc + b).argmax(1), minlength=len(PAIRS)) / len(sc)
        b += lr * scale * (np.log(tgt + eps) - np.log(f + eps)) / 10
    return PAIRS[(sc + b).argmax(1)]

def rotated_top(m, H, top, keep):
    """Raw rotated paths (descriptive) and usage-matched rotated paths (N3), at the scored positions keep."""
    raw, matched = [], []
    for l, (b, h) in enumerate(zip(m.blocks, H)):
        W = b.ffn.router.weight.detach().float().cpu().double().numpy(); rs = np.random.RandomState(1000 + l)
        q, rr = np.linalg.qr(rs.randn(W.shape[1], W.shape[1])); q = q * np.sign(np.diag(rr))
        z = h[keep] @ (W @ q).T; assert np.isfinite(z).all(); raw.append(top2(z))
        matched.append(match_usage(z, top[l][keep]))
    return raw, matched

def main(tag):
    m = model(tag); X, st, S, half, dup = fresh(first=FIRST, n=N); nv0 = ~dup; nv = nv0 & (st <= 2)
    assert len(np.unique(st[nv])) == 2
    w = np.random.RandomState(1).randint(1, 2 ** 62, size=54).astype(np.int64)
    hs = (X.numpy().astype(np.int64) * w).sum(1); hA, hB = set(hs[nv & (half == 0)].tolist()), set(hs[nv & (half == 1)].tolist())
    cross_half = float(np.isin(hs[nv], list(hA & hB)).mean())
    H, top = capture(m, X)
    out = dict(tag=tag, solves=f'held-out {FIRST}-{FIRST + N - 1}', n_positions=int(len(st)), n_novel=int(nv.sum()),
               novel_stage_counts={int(k): int((st[nv0] == k).sum()) for k in (1, 2, 3, 4)},
               dropped_novel_OLL_PLL=int((nv0 & (st > 2)).sum()), novel_state_in_both_halves_fraction=cross_half,
               novel_F2L_fraction_of_F2L=float(nv[st == 2].mean()),
               novel=info([t[nv] for t in top], S[nv], st[nv], half[nv], np.random.RandomState(0)))
    if tag.startswith('cfop'):
        raw, mt = rotated_top(m, H, top, nv)
        out['rotated'] = info(mt, S[nv], st[nv], half[nv], np.random.RandomState(0))          # usage-matched (N3)
        out['rotated_raw'] = info(raw, S[nv], st[nv], half[nv], np.random.RandomState(0))     # descriptive
    os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{tag}.json'; json.dump(out, open(fn, 'w'), indent=1)
    print(f'{tag}: wrote {fn} ({int(nv.sum())} novel of {len(st)} positions)', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
