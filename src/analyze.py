"""STEP 3: routing analysis + all nulls + figures."""
import sys, os, json, argparse, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/alityb/projects/cubemoe/src')
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
ROOT = '/Users/alityb/projects/cubemoe'
E = 8

# ---------- fast discrete metrics ----------
def _ct(a, b):
    a = np.asarray(a).astype(np.int64); b = np.asarray(b).astype(np.int64)
    a = a - a.min(); b = b - b.min()
    A, B = a.max() + 1, b.max() + 1
    return np.bincount(a * B + b, minlength=A * B).reshape(A, B)

def mi_bits(a, b):
    ct = _ct(a, b); n = ct.sum()
    if n == 0: return 0.0
    p = ct / n; pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    q = pa @ pb; nz = (p > 0) & (q > 0)
    return float((p[nz] * np.log2(p[nz] / q[nz])).sum())

def bayes_acc(keys, label):
    ks = np.stack(keys, 1) if isinstance(keys, (list, tuple)) else keys[:, None]
    _, inv = np.unique(ks, axis=0, return_inverse=True)
    ct = _ct(inv, label)
    return float(ct.max(1).sum() / ct.sum())

def purity(expert, label):
    tot = 0.0; used = 0
    for e in range(E):
        m = expert == e
        if m.sum() == 0: continue
        used += 1; _, c = np.unique(label[m], return_counts=True); tot += c.max() / m.sum()
    return tot / max(used, 1)

def cond_mi(expert, label, strat, min_n=20):
    tot = 0.0; n = len(label)
    for s in np.unique(strat):
        m = strat == s
        if m.sum() < min_n or len(np.unique(label[m])) < 2: continue
        tot += (m.sum() / n) * mi_bits(expert[m], label[m])
    return float(tot)

def utilization(expert):
    f = np.bincount(np.asarray(expert).astype(int), minlength=E)[:E] / len(expert)
    ent = -(f[f > 0] * np.log2(f[f > 0])).sum()
    return float(np.minimum(f, 1.0 / E).sum()), float(ent), int((f > 0.005).sum())

def metrics(expert, label, strat):
    u, ent, nact = utilization(expert)
    r = dict(utilization=u, expert_entropy_bits=ent, experts_used=nact,
             purity_raw=purity(expert, label), nmi_raw=float(NMI(label, expert)),
             ari_raw=float(ARI(label, expert)), mi_raw_bits=mi_bits(expert, label))
    pv, nv, av, w = [], [], [], []
    for s in np.unique(strat):
        m = strat == s
        if m.sum() < 20: continue
        lab, ex = label[m], expert[m]; w.append(m.sum())
        pv.append(purity(ex, lab))
        if len(np.unique(lab)) < 2: nv.append(0.0); av.append(0.0)
        else: nv.append(NMI(lab, ex)); av.append(ARI(lab, ex))
    w = np.array(w, float); w /= w.sum()
    for k, v in (('purity', pv), ('nmi', nv), ('ari', av)):
        v = np.array(v); r[f'{k}_strat_mean'] = float(v.mean()); r[f'{k}_strat_wt'] = float((v * w).sum())
    r['cond_mi_bits'] = cond_mi(expert, label, strat)
    return r

# ---------- change point ----------
def changepoint(seq):
    n = len(seq)
    if n < 6: return None
    oh = np.zeros((n, E)); oh[np.arange(n), seq] = 1
    cum = np.vstack([np.zeros(E), np.cumsum(oh, 0)])
    c = np.arange(2, n - 1)
    P = cum[c] / c[:, None]; Q = (cum[n] - cum[c]) / (n - c)[:, None]
    tv = 0.5 * np.abs(P - Q).sum(1)
    j = int(tv.argmax()); return int(c[j]), float(tv[j])

def cp_test(expert, solve, seam, sollen=None, nperm=200, seed=0):
    """Regress fitted change-point on the true seam.
    The naive slope is confounded: a change point sits mid-sequence, and sequence length
    correlates with the seam, so even RANDOM routing scores slope ~0.5. We therefore also
    report the PARTIAL slope of seam controlling for sequence length, and use a
    length-stratified permutation null."""
    order = np.argsort(solve, kind='stable')
    sv, ex, sm = solve[order], expert[order], seam[order]
    sl = sollen[order] if sollen is not None else None
    bounds = np.flatnonzero(np.diff(sv)) + 1
    segs = np.split(ex, bounds); seams = np.split(sm, bounds)
    lens = np.split(sl, bounds) if sl is not None else None
    xs, ys, ns, tvs = [], [], [], []
    for i, (sg, smv) in enumerate(zip(segs, seams)):
        r = changepoint(sg)
        if r is None: continue
        xs.append(smv[0]); ys.append(r[0]); tvs.append(r[1])
        ns.append(lens[i][0] if lens is not None else len(sg))
    xs = np.array(xs, float); ys = np.array(ys, float); ns = np.array(ns, float)
    base = dict(n=int(len(xs)), seam_sd=float(xs.std()) if len(xs) else 0.0,
                mean_tv=float(np.mean(tvs)) if tvs else 0.0)
    if len(xs) < 20 or xs.std() < 1e-9:
        return dict(slope=float('nan'), r=float('nan'), p=float('nan'),
                    partial_slope=float('nan'), partial_p=float('nan'), **base)
    slope = float(np.polyfit(xs, ys, 1)[0]); rr = float(np.corrcoef(xs, ys)[0, 1])
    def partial(x, n_, y):
        A = np.stack([x, n_, np.ones_like(x)], 1)
        return float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    pslope = partial(xs, ns, ys)
    rng = np.random.RandomState(seed); c1 = c2 = 0
    keys = np.round(ns).astype(int)
    for _ in range(nperm):
        xp = xs.copy()
        for k in np.unique(keys):                       # length-stratified permutation
            m = keys == k
            if m.sum() > 1: xp[m] = rng.permutation(xs[m])
        if np.polyfit(xp, ys, 1)[0] >= slope: c1 += 1
        if partial(xp, ns, ys) >= pslope: c2 += 1
    return dict(slope=slope, r=rr, p=(c1 + 1) / (nperm + 1),
                partial_slope=pslope, partial_p=(c2 + 1) / (nperm + 1),
                pred_sd=float(ys.std()), _xs=xs.tolist(), _ys=ys.tolist(), **base)

# ---------- nulls ----------
def null_shuffled(expert, seed=0):
    return np.random.RandomState(seed).permutation(expert)

def null_position_router(strat):
    """Best-effort position-ONLY router: 8 quantile bins of the stratifier."""
    qs = np.quantile(strat, np.linspace(0, 1, E + 1)[1:-1])
    return np.searchsorted(qs, strat).astype(int)

def perm_p_condmi(expert, label, strat, obs, nperm=200, seed=0):
    rng = np.random.RandomState(seed); cnt = 0
    for _ in range(nperm):
        lab = label.copy()
        for s in np.unique(strat):
            m = strat == s
            if m.sum() > 1: lab[m] = rng.permutation(lab[m])
        if cond_mi(expert, lab, strat) >= obs: cnt += 1
    return (cnt + 1) / (nperm + 1)

def probe_phase(rs, label, strat, seed=0):
    X = rs.astype(np.float32); y = (label == 2).astype(int)
    n = len(y); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    ntr = int(n * 0.7); tr, te = idx[:ntr], idx[ntr:]
    lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(X[tr], y[tr])
    pred = lr.predict(X[te]); acc = float((pred == y[te]).mean())
    st = strat[te]; accs, ws = [], []
    for s in np.unique(st):
        m = st == s
        if m.sum() < 20: continue
        accs.append((pred[m] == y[te][m]).mean()); ws.append(m.sum())
    w = np.array(ws, float); w /= w.sum()
    return dict(probe_acc=acc, probe_acc_strat_wt=float((np.array(accs) * w).sum()))
