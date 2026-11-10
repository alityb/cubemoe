"""Path intervention (prereg/PREREG_PATH_SWAP.md, #14). Usage: python src/path_swap.py <tag> <data> <first> <n>
  data: cfop250k (label = stage) | mixed250k (label = Kociemba phase); positions = held-out solves first..first+n-1 of the
  RandomState(0) held-out ordering (asserted disjoint from the training split perm[:ntr]). -> $SWAP_OUT/<tag>__<data>.json
For each recipient position x (label s, moves remaining S) and each layer set G in {all = L0-5, L0-4, L5}:
  'cross' donor: a position from another solve with the same S and a DIFFERENT label;
  'same'  donor: a position from another solve with the same S and the SAME label;
  the two donors are matched on d = number of donor experts not in x's pair, summed over the layers of G (d >= 1, the
  same d for both; drawn at random among the values both pools offer; up to NCAND candidates per pool). Reported: the
  per-layer rate at which each donor's pair differs, and x's router mass on the donor pairs (cross vs same).
  Intervention: at every layer of G the decision token's two experts are replaced by the donor's (clean) pair, weighted
  by x's own router probabilities over those two experts, renormalised (as the model's own top-k renormalisation).
  Layers outside G route normally. Only the decision token (index 0) is intervened on.
Outcomes on x: changed = argmax move differs from the clean argmax; KL(clean || intervened) in bits; drop in log2 p of
  the solver's move; probe shift = change in P(donor's label) under a linear stage probe of the final decision-token
  representation (fitted on clean positions of the other half). For 'same' swaps the probe shift is measured on the
  label of the matched 'cross' donor, so cross - same isolates the label the route came from.
Primary contrast per set: D_changed = mean(changed | cross) - mean(changed | same), paired by recipient; 95% CI by a
  cluster bootstrap over solves (1000 resamples); also per half (A = even, B = odd solve)."""
import sys, os, json, itertools, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore', message='.*encountered in matmul')   # spurious on macOS Accelerate
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('SWAP_OUT', f'{R}/out/path_swap')
NREC, NCAND, NBOOT, T = int(os.environ.get('SWAP_NREC', 20000)), 64, 1000, 55
PID = {p: i for i, p in enumerate(itertools.combinations(range(8), 2))}; PAIRS = np.array(list(PID))

def positions(data, first, n):
    d = np.load(f'{R}/data/{data}.npz'); sl = d['sol_len']; N = len(sl); perm = np.random.RandomState(0).permutation(N)
    ntr = int(N * 0.9); te = perm[ntr:][first:first + n]
    assert len(te) == n and not set(te.tolist()) & set(perm[:ntr].tolist())
    off = np.concatenate([[0], np.cumsum(sl.astype(np.int64))])
    r = np.concatenate([np.arange(off[i], off[i + 1]) for i in te]); t = d['pos'][r].astype(int)
    L = np.concatenate([np.full(int(sl[i]), int(sl[i])) for i in te]); sid = np.concatenate([np.full(int(sl[i]), k) for k, i in enumerate(te)])
    return (torch.from_numpy(d['states'][r].astype(np.int64)), d['phase'][r].astype(int), d['moves'][r].astype(int), L - t, sid)

def patch(m):
    """Replace each MoE's forward (eval only) with one that can overwrite the decision token's pair: ffn._swap = pairs
    (B, 2) long tensor or None."""
    for b in m.blocks:
        f = b.ffn; f._swap = None
        def fwd(x, token_mask=None, f=f):
            B, Tt, d = x.shape; flat = x.reshape(-1, d); probs = F.softmax(f.router(flat).float(), -1)
            topv, topi = probs.topk(f.k, -1)
            if f._swap is not None:
                rows = torch.arange(B, device=x.device) * Tt; topi = topi.clone(); topi[rows] = f._swap
                topv = topv.clone(); topv[rows] = probs[rows].gather(1, f._swap)
            if f.norm_topk: topv = topv / topv.sum(-1, keepdim=True)
            return f.dispatch(flat, topi, topv).reshape(B, Tt, d), probs, topi, torch.zeros((), device=x.device)
        f.forward = fwd

def run(m, X, swap=None, bs=2048):
    """Logits, final decision-token representation, sorted top-2 pair codes and router probs per layer (decision token)."""
    Z, H, P, Pr = [], [], [], []; hk = m.norm.register_forward_hook(lambda mod, i, o: H.append(o.float().cpu()))
    with torch.no_grad():
        for i in range(0, len(X), bs):
            for l, b in enumerate(m.blocks): b.ffn._swap = None if swap is None or swap[l] is None else swap[l][i:i + bs].to(A.DEV)
            z, info, _ = m(X[i:i + bs].to(A.DEV), collect=True); Z.append(z.float().cpu())
            P.append(torch.stack([inf[1].reshape(-1, T, 2)[:, 0].sort(-1).values.cpu() for inf in info], 1))
            Pr.append(torch.stack([inf[0].reshape(-1, T, inf[0].shape[-1])[:, 0].float().cpu() for inf in info], 1))
    hk.remove()
    for b in m.blocks: b.ffn._swap = None
    P = torch.cat(P).numpy(); code = np.zeros(P.shape[:2], int)
    for (a, c), k in PID.items(): code[(P[..., 0] == a) & (P[..., 1] == c)] = k
    return torch.cat(Z), torch.cat(H).numpy(), code, torch.cat(Pr).numpy()

def donors(P, lab, S, sid, rec, G, rng):
    """Cross/same donors for recipients rec over layer set G, matched on d = number of differing EXPERTS summed over the
    layers of G (0-2 per layer: donor experts not in x's pair); -1 where no match."""
    cells = {}
    for i in range(len(S)): cells.setdefault((S[i], lab[i]), []).append(i)
    cells = {k: np.array(v) for k, v in cells.items()}; labs = np.unique(lab)
    cx, sm = np.full(len(rec), -1), np.full(len(rec), -1)
    for j, i in enumerate(rec):
        pools = {}
        for kind in ('cross', 'same'):
            c = [cells.get((S[i], l), np.zeros(0, int)) for l in labs if (l != lab[i]) == (kind == 'cross')]
            c = np.concatenate(c); c = c[sid[c] != sid[i]]
            if len(c) > NCAND: c = rng.choice(c, NCAND, replace=False)
            pools[kind] = (c, nexp(P[c][:, G], P[i, G]))
        common = np.intersect1d(pools['cross'][1], pools['same'][1]); common = common[common >= 1]
        if not len(common): continue
        dd = rng.choice(common)
        cx[j] = rng.choice(pools['cross'][0][pools['cross'][1] == dd]); sm[j] = rng.choice(pools['same'][0][pools['same'][1] == dd])
    return cx, sm

def nexp(Pc, Pi):
    """Pc (n, |G|) donor pair codes, Pi (|G|,) recipient's: number of donor experts not in the recipient's pair."""
    dp, rp = PAIRS[Pc], PAIRS[Pi]
    return (~(dp[..., :, None] == rp[None, :, None, :]).any(-1)).sum((1, 2))

def probe(H, lab, half):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cls = np.unique(lab); fits = {}
    for h in (0, 1):
        fits[h] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(H[half == h], lab[half == h])
        assert (fits[h].classes_ == cls).all()
    def p(Hx, halfx, labx):                                    # P(labx) under the probe fitted on the OTHER half
        out = np.zeros(len(Hx))
        for h in (0, 1):
            k = halfx == h
            if k.any(): out[k] = fits[1 - h].predict_proba(Hx[k])[np.arange(k.sum()), np.searchsorted(cls, labx[k])]
        return out
    acc = float(np.mean([(fits[1 - h].predict(H[half == h]) == lab[half == h]).mean() for h in (0, 1)]))
    return p, acc

def boot(v, cl, rng):
    u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}; s = []
    for _ in range(NBOOT):
        ix = np.concatenate([idx[c] for c in rng.choice(u, len(u))]); s.append(v[ix].mean())
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]

def boot_stat(stat, cl, rng):
    u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}
    s = [stat(np.concatenate([idx[c] for c in rng.choice(u, len(u))])) for _ in range(NBOOT)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]

def main(tag, data, first, n):
    first, n = int(first), int(n); m, _ = A.load(tag); m.eval()
    X, lab, mv, S, sid = positions(data, first, n); half = sid % 2; rng = np.random.RandomState(0)
    for b in m.blocks: b.ffn.forward = type(b.ffn).forward.__get__(b.ffn)        # unpatched reference
    Zref = A.logits(m, X[:4096]); patch(m)
    assert (Zref - run(m, X[:4096])[0]).abs().max().item() == 0.0                    # patched clean forward = original
    Z0, H0, P, Pr0 = run(m, X); lp0 = F.log_softmax(Z0, -1); a0 = Z0.argmax(-1); pr, pacc = probe(H0, lab, half)
    rec = np.sort(rng.choice(len(S), min(NREC, len(S)), replace=False)); L = P.shape[1]
    sets = {'all': list(range(L)), 'L0-4': list(range(L - 1)), 'L5': [L - 1]}
    out = dict(tag=tag, data=data, solves=f'held-out {first}-{first + n - 1}', n_positions=int(len(S)), n_recipients=int(len(rec)),
               clean_acc=float((a0.numpy() == mv).mean()), probe_acc=pacc, sets={})
    for name, G in sets.items():
        cx, sm = donors(P, lab, S, sid, rec, G, rng); ok = (cx >= 0) & (sm >= 0); r, cx, sm = rec[ok], cx[ok], sm[ok]
        lr = lambda dn: (P[dn][:, G] != P[r][:, G]).mean(0).tolist()                 # per-layer pair-difference rate
        mass = lambda dn: np.mean([Pr0[r, l][np.arange(len(r))[:, None], PAIRS[P[dn, l]]].sum(1) for l in G], 0)
        o = out['sets'][name] = dict(n_recipients=int(len(rec)), n_matched=int(ok.sum()),
                                     n_matched_by_label={int(k): int((lab[r] == k).sum()) for k in np.unique(lab)},
                                     d_mean=float(np.mean([nexp(P[cx[j:j + 1]][:, G], P[r[j], G])[0] for j in range(len(r))])),
                                     layers_differ={'cross': lr(cx), 'same': lr(sm)},
                                     mass_on_donor_pair={'cross': float(mass(cx).mean()), 'same': float(mass(sm).mean())})
        res = {}
        for kind, dn in (('cross', cx), ('same', sm)):
            sw = [torch.from_numpy(PAIRS[P[dn, l]]).long() if l in G else None for l in range(L)]
            Z1, H1, P1, _ = run(m, X[r], sw); lp1 = F.log_softmax(Z1, -1); p0 = lp0[r].exp()
            moved = (Z1 != Z0[r]).any(-1).numpy(); assert moved.mean() > 0.99, moved.mean()  # the swap changed the computation
            kl = ((p0 * (lp0[r] - lp1)).sum(-1) / np.log(2)).numpy()
            res[kind] = dict(changed=(Z1.argmax(-1) != a0[r]).numpy().astype(float), kl=kl,
                             dlogp=((lp0[r, mv[r]] - lp1[np.arange(len(r)), mv[r]]) / np.log(2)).numpy(),
                             shift=pr(H1, half[r], lab[cx]) - pr(H0[r], half[r], lab[cx]))
        dm = mass(cx) - mass(sm); vc = res['cross']['changed'] - res['same']['changed']
        icpt = lambda ix: float(np.linalg.lstsq(np.stack([np.ones(len(ix)), dm[ix]], 1), vc[ix], rcond=None)[0][0])  # intercept of D ~ mass gap
        o['changed_D_mass_adjusted'] = icpt(np.arange(len(dm))); o['changed_D_mass_adjusted_ci'] = boot_stat(icpt, sid[r], np.random.RandomState(2))
        for k in ('changed', 'kl', 'dlogp', 'shift'):
            v = res['cross'][k] - res['same'][k]
            o[k] = dict(cross=float(res['cross'][k].mean()), same=float(res['same'][k].mean()), D=float(v.mean()),
                        ci=boot(v, sid[r], np.random.RandomState(1)),
                        D_half={hn: float(v[half[r] == h].mean()) for hn, h in (('A', 0), ('B', 1))})
        print(f"{tag} {data} {name}: n={o['n_matched']} d={o['d_mean']:.2f} changed {o['changed']['cross']:.3f}/{o['changed']['same']:.3f} "
              f"D={o['changed']['D']:+.3f} {o['changed']['ci']}", flush=True)
    os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{tag}__{data}.json'; json.dump(out, open(fn, 'w'), indent=1)
    print(f'wrote {fn}', flush=True)

if __name__ == '__main__':
    main(*sys.argv[1:5])
