"""Is the side choice the policy's or the search's? At the first state of every last-layer algorithm on the beam-256
path, compare the policy's probability of the move actually played (first move of the chosen side's version) with the
first move of the same algorithm from the training solver's side (R side). Plus top-1 agreement over LL moves.
Usage: python src/cfop_side_choice.py  -> out/prod/cfop_side_choice.json"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import prod_analysis as A
from cfop_ll import states, ids, inv, strip_auf, U_MOVES
from cfop_verify import flags, first
from cfop_alphabet import NAMES, rot
R = '/Users/alityb/projects/cubemoe'
ORIG = {}
for a in NAMES:
    for b in (a, inv(a)):
        for k in range(4): ORIG[ids(rot(b, k))] = (ids(b), k)

def alg_spans(S, moves):
    _, f, _ = flags(S); tf = first(f); n = len(moves); t = tf; out = []
    while t < n:
        if f[t] and f[t + 1]: t += 1; continue
        u = t + 1
        while u <= n and not f[u]: u += 1
        out.append((t, u)); t = u
    return tf, out

res = {}
for s in range(5):
    tag = f'cfop_mixtral_6L256_s{s}'; m, _ = A.load(tag); P = json.load(open(f'{R}/out/prod/{tag}_paths.json'))
    pos, played, rside, kk, llX, llY = [], [], [], [], [], []
    for p in P:
        mv = p['model']; S = states(p['start'], mv); tf, spans = alg_spans(S, mv)
        llX += [S[t] for t in range(tf, len(mv))]; llY += mv[tf:]
        for t, u in spans:
            core = strip_auf(mv[t:u]); lead = 0
            while lead < len(mv[t:u]) and mv[t + lead] in U_MOVES: lead += 1
            if core in ORIG:
                b, k = ORIG[core]; pos.append(S[t + lead]); played.append(core[0]); rside.append(b[0]); kk.append(k)
    with torch.no_grad():
        pr = F.softmax(A.logits(m, torch.from_numpy(np.stack(pos))), -1).numpy()
        pl = F.softmax(A.logits(m, torch.from_numpy(np.stack(llX))), -1).numpy()
    played, rside, kk = map(np.array, (played, rside, kk)); o = kk > 0; i = np.arange(len(played))
    top1 = pr.argmax(-1) == played
    res[tag] = dict(n_algs=len(played), other_side=float(o.mean()),
                    top1_at_alg_start_other_side=float(top1[o].mean()), top1_at_alg_start_R_side=float(top1[~o].mean()),
                    p_played_other=float(pr[i, played][o].mean()), p_Rside_first_move_other=float(pr[i, rside][o].mean()),
                    ll_moves_top1=float((pl.argmax(-1) == np.array(llY)).mean()))
    r = res[tag]
    print(f"{tag}: {r['n_algs']} base-alg executions, {r['other_side']*100:.0f}% from another side | at their first move the policy's "
          f"top-1 IS the played move {r['top1_at_alg_start_other_side']*100:.0f}% (R-side ones {r['top1_at_alg_start_R_side']*100:.0f}%); "
          f"p(played) {r['p_played_other']:.2f} vs p(R-side first move) {r['p_Rside_first_move_other']:.2f} | all LL moves top-1 {r['ll_moves_top1']*100:.0f}%", flush=True)
json.dump(res, open(f'{R}/out/prod/cfop_side_choice.json', 'w'), indent=1)
