"""Measures for prereg/PREREG_G1_PLL.md (#9): binary-stage routing shares and per-stage knockouts, final layer.
Usage: python src/g1pll_analysis.py <tag> <data>   e.g. cfop_g1pll_6L256_s0 cfop250k_g1pll  |  cfop_mixtral_6L256_s0 cfop250k
-> out/g1pll/<tag>.json"""
import sys, os, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from explore_cfop_analyze import corrected, cent
R = '/Users/alityb/projects/cubemoe'; STG = {1: 'cross', 2: 'F2L', 3: 'OLL', 4: 'PLL'}

def heldout(data, n_solves=1200):
    d = dict(np.load(f'{R}/data/{data}.npz')); n = len(d['sol_len']); te = np.random.RandomState(0).permutation(n)[int(n * 0.9):][:n_solves]
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))]); rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in te])
    sl = np.concatenate([np.full(int(d['sol_len'][i]), int(d['sol_len'][i])) for i in te]); t = d['pos'][rows].astype(int)
    return torch.from_numpy(d['states'][rows].astype(np.int64)), dict(stage=d['phase'][rows].astype(int), move=d['moves'][rows].astype(int), rem=sl - t)

def main(tag, data):
    m, _ = A.load(tag); X, lab = heldout(data); L = len(m.blocks) - 1; rng = np.random.RandomState(0)
    E = A.capture(m, X)['route'][L][:, 0].numpy(); S = lab['rem']; H = cent(E, S); st = lab['stage']
    out = dict(tag=tag, data=data, n_positions=len(st),
               y_PLL=corrected(E, (st == 4).astype(int), S, rng) / H, y_OLL=corrected(E, (st == 3).astype(int), S, rng) / H,
               y_stage=corrected(E, st, S, rng) / H)
    Y = torch.from_numpy(lab['move']); p0 = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; ko = {}
    for e in range(m.blocks[L].ffn.E):
        m.blocks[L].ffn._knockout = {e}; pe = F.softmax(A.logits(m, X), -1)[torch.arange(len(Y)), Y]; m.blocks[L].ffn._knockout = set()
        dd = (p0 - pe).numpy(); ko[e] = {STG[s]: float(dd[st == s].mean()) for s in STG}
    out['knockout'] = ko
    out['pll_dedicated'] = [e for e, r in ko.items() if r['PLL'] >= 0.03 and all(r[s] <= 0.25 * r['PLL'] for s in ('cross', 'F2L', 'OLL'))]
    os.makedirs(f'{R}/out/g1pll', exist_ok=True); json.dump(out, open(f'{R}/out/g1pll/{tag}.json', 'w'), indent=1)
    print(f"{tag} [{data}]: y_PLL {out['y_PLL']*100:.1f}% | y_OLL {out['y_OLL']*100:.1f}% | y_stage {out['y_stage']*100:.1f}% | "
          f"PLL-dedicated experts {out['pll_dedicated']}", flush=True)
    for e, r in ko.items():
        if max(r.values()) > 0.02: print(f"   e{e}: " + "  ".join(f"{s} {v*100:+.1f}" for s, v in r.items()), flush=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
