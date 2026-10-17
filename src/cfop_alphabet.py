"""The model's last-layer 'alphabet': split the part after F2L into algorithms (maximal runs of F2L-broken states,
bounded by F2L-intact checkpoints; U-layer turns between them are AUFs) and identify each against the training
solver's 7 base algorithms (and inverses).
Usage: python src/cfop_alphabet.py <tag> [<tag> ...]"""
import sys, json, numpy as np
from collections import Counter
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
from cfop_ll import states, ids, inv, strip_auf, MN
from cfop_verify import flags, first
from cfop_canon import OLL_BASE, PLL_BASE
R = '/Users/alityb/projects/cubemoe'
NAMES = dict(zip(OLL_BASE, ['Sune', 'Anti-Sune', "F R U R' U' F'"])) | dict(zip(PLL_BASE, ['T-perm', 'Ua-perm', 'Aa-perm', 'J-perm']))
SIDES = ['F', 'R', 'B', 'L']                              # a y rotation relabels side faces cyclically; U/D unchanged
def rot(alg, k): return ' '.join((SIDES[(SIDES.index(m[0]) + k) % 4] if m[0] in SIDES else m[0]) + m[1:] for m in alg.split())
LIB = {}
for a, nm in NAMES.items():
    for k in range(4):
        tag = '' if k == 0 else ' [other side]'
        LIB.setdefault(ids(rot(a, k)), nm + tag); LIB.setdefault(ids(rot(inv(a), k)), nm + ' (inverse)' + tag)

def algs(start, moves):
    S = states(start, moves); _, f, _ = flags(S); tf = first(f); n = len(moves); out = []; t = tf
    while t < n:
        if f[t] and (t + 1 > n or f[t + 1]): t += 1; continue          # AUF / move that keeps F2L intact
        u = t + 1
        while u <= n and not f[u]: u += 1                                # F2L broken until u
        out.append(tuple(moves[t:u])); t = u
    return out

def main(tags):
    tot = {'solver': Counter(), 'model': Counter()}; nalg = {'solver': [], 'model': []}; lens = {'solver': [], 'model': []}
    for tag in tags:
        for p in json.load(open(f'{R}/out/prod/{tag}_paths.json')):
            for who in ('solver', 'model'):
                A = algs(p['start'], p[who]); nalg[who].append(len(A))
                for a in A:
                    lens[who].append(len(a)); tot[who][LIB.get(strip_auf(a), 'not a base alg')] += 1
    for who in ('solver', 'model'):
        n = sum(tot[who].values())
        print(f"{who}: {np.mean(nalg[who]):.2f} algorithms per last layer (dist {dict(sorted(Counter(nalg[who]).items()))}), "
              f"mean alg length {np.mean(lens[who]):.1f}")
        for k, c in tot[who].most_common(): print(f"   {c / n * 100:5.1f}%  {k}")
    novel = Counter()
    for tag in tags:
        for p in json.load(open(f'{R}/out/prod/{tag}_paths.json')):
            for a in algs(p['start'], p['model']):
                if strip_auf(a) not in LIB: novel[' '.join(MN[x] for x in strip_auf(a))] += 1
    print("most common model sequences that are not a base algorithm from any side:")
    for k, c in novel.most_common(12): print(f"   {c:4d}x  {k}")

if __name__ == '__main__':
    main(sys.argv[1:])
