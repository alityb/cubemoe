"""Measures for prereg/PREREG_F2L_SPLIT.md (#11): do CFOP models divide the final layer's work at the F2L / last-layer
boundary rather than at CFOP's four stages?
Usage: python src/f2l_split.py <tag> [data=cfop250k]  -> out/f2l/<tag>.json
Final layer, decision token, the same 1,200 held-out solves as every earlier analysis (src/pairs_analysis.data_for).
Cuts: the three contiguous two-way splits of the solve, b_c = 1 if the stage is after cut c.
  'cross|rest' = {cross} vs {F2L,OLL,PLL};  'F2L|LL' = {cross,F2L} vs {OLL,PLL};  'OLL|PLL' = {cross,F2L,OLL} vs {PLL}"""
import sys, os, json, itertools, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
from pairs_analysis import data_for
from explore_cfop_analyze import corrected, cent
R = '/Users/alityb/projects/cubemoe'; NPERM = 50
CUTS = {'cross|rest': 1, 'F2L|LL': 2, 'OLL|PLL': 3}             # b = stage > cut

def main(tag, data='cfop250k'):
    m, _ = A.load(tag); X, st, Y, S, half, NAMES = data_for(data); ffn = m.blocks[-1].ffn; E = ffn.E
    assert len(half) == len(Y) and set(np.unique(st)) == {1, 2, 3, 4}
    cap = {}; h = ffn.register_forward_hook(lambda mod, i, o: cap.update(ti=o[2])); top = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            m(X[i:i + 2048].to(A.DEV)); n = cap['ti'].shape[0] // 55; top.append(cap['ti'].reshape(n, 55, -1)[:, 0].cpu())
    h.remove(); top = torch.cat(top).numpy(); pair = top[:, 0] * E + top[:, 1]
    parts = (('all', np.ones(len(Y), bool)), ('A', half == 0), ('B', half == 1))
    out = dict(tag=tag, data=data, n_positions=int(len(Y)), cut_info={}, top1_cut_info={})
    # (1) how much of each two-way cut the selected pair explains (fraction of H(cut | S), permutation-corrected)
    for c, k in CUTS.items():
        b = (st > k).astype(int); out['cut_info'][c] = {}; out['top1_cut_info'][c] = {}
        for pt, mk in parts:
            rng = np.random.RandomState(0); H = cent(b[mk], S[mk])
            out['cut_info'][c][pt] = corrected(pair[mk], b[mk], S[mk], rng, NPERM) / H
            out['top1_cut_info'][c][pt] = corrected(top[mk, 0], b[mk], S[mk], rng, NPERM) / H
    # (2) the "team": most co-selected pair on first-two-layer positions (cross + F2L), CROSS-FITTED: chosen on one
    #     half of the solves, measured on the other (team_A chosen on half A, measured on B, and vice versa)
    FL, LL = st <= 2, st >= 3; Yt = torch.from_numpy(Y); ar = torch.arange(len(Y))
    co = {(i, j): (np.sort(top, 1) == (i, j)).all(1) for i, j in itertools.combinations(range(E), 2)}
    def p():
        z = A.logits(m, X).float(); zo = z.clone(); zo[ar, Yt] = -float('inf')
        return F.softmax(z, -1)[ar, Yt].numpy(), (z[ar, Yt] - zo.max(-1).values).numpy()   # p(teacher move), margin
    p0, m0 = p(); out['crossfit'] = {}
    for pick, hm in (('A', half == 0), ('B', half == 1)):
        ev = ~hm                                              # evaluate on the other half
        team = max(co, key=lambda q: co[q][FL & hm].mean()); llp = max(co, key=lambda q: co[q][LL & hm].mean())
        rec = {}
        for name, pr in (('team', team), ('ll_pair', llp)):
            ffn._knockout = set(pr); pz, mz = p(); ffn._knockout = set(); d, dm = p0 - pz, m0 - mz
            for g in (FL & ev, LL & ev): assert g.sum() > 0
            rec[name] = dict(pair=list(pr),
                             margin_damage={NAMES[s]: float(dm[(st == s) & ev].mean()) for s in NAMES} | {'FL': float(dm[FL & ev].mean()), 'LL': float(dm[LL & ev].mean())},
                             rate={NAMES[s]: float(co[pr][(st == s) & ev].mean()) for s in NAMES} | {'FL': float(co[pr][FL & ev].mean()), 'LL': float(co[pr][LL & ev].mean())},
                             damage={NAMES[s]: float(d[(st == s) & ev].mean()) for s in NAMES} | {'FL': float(d[FL & ev].mean()), 'LL': float(d[LL & ev].mean())})
        rec['ll_pair_is_team'] = list(team) == list(llp)
        out['crossfit'][f'pick_{pick}_eval_{"B" if pick == "A" else "A"}'] = rec
    os.makedirs(f'{R}/out/f2l', exist_ok=True); json.dump(out, open(f'{R}/out/f2l/{tag}.json', 'w'), indent=1)
    ci = out['cut_info']; msg = f"{tag}: pair-explained cut info " + " ".join(f"{c} {ci[c]['all']:.3f}" for c in CUTS)
    for k, r in out['crossfit'].items():
        t, l = r['team'], r['ll_pair']
        msg += (f" | {k}: team {tuple(t['pair'])} cross {t['rate']['cross']:.2f} FL {t['rate']['FL']:.2f} KO FL {t['damage']['FL']*100:.1f} "
                f"LL {t['damage']['LL']*100:.1f}; LL pair {tuple(l['pair'])} LL {l['rate']['LL']:.2f}")
    print(msg, flush=True)

if __name__ == '__main__':
    main(*sys.argv[1:])
