"""Apply prereg/PREREG_RL_ROUTING.md mechanically: gains vs the re-evaluated base, H1-H3 per seed, verdict.
Usage: python src/rl_report.py <seed> [<seed> ...]   (reads out/rl/eval_*.json and out/rl/chosen_lr.json)"""
import sys, json, numpy as np
R = '/Users/alityb/projects/cubemoe'; OUT = f'{R}/out/rl'
LR = json.load(open(f'{OUT}/chosen_lr.json'))
ARMS = ('router', 'lora', 'experts', 'full')

def gains(seed):
    tag = f'cfop_mixtral_6L256_s{seed}'; base = json.load(open(f'{OUT}/eval_{tag}_base.json')); out = {}
    for arm in ARMS:
        e = json.load(open(f"{OUT}/eval_{tag}_{arm}_lr{LR[arm]:g}_s0.json"))
        out[arm] = {s: {b: e[s][b]['sampled'] - base[s][b]['sampled'] for b in ('EASY', 'MID', 'HARD', 'ALL')} for s in ('held', 'random')}
        out[arm]['greedy'] = {s: e[s]['ALL']['greedy'] - base[s]['ALL']['greedy'] for s in ('held', 'random')}
    hard_set = 'held' if base['held']['HARD']['n'] >= 50 else 'random'
    return out, base, hard_set

def judge(g, hard_set):
    fe, fh = g['full']['held']['EASY'], g['full'][hard_set]['HARD']
    h1 = None if fe < 0.03 else g['router']['held']['EASY'] >= 0.7 * fe
    h2 = None if fe < 0.03 else g['router']['held']['EASY'] - g['lora']['held']['EASY'] >= 0.25 * fe
    h3 = None if fh < 0.02 else g['router'][hard_set]['HARD'] <= 0.5 * fh
    return dict(H1=h1, H2=h2, H3=h3)

seeds = [int(s) for s in sys.argv[1:]]; per = {}
for s in seeds:
    g, base, hs = gains(s); per[s] = dict(gains=g, judge=judge(g, hs), hard_set=hs)
    print(f"seed {s} (HARD bin from '{hs}'; base n EASY/MID/HARD held = {base['held']['EASY']['n']}/{base['held']['MID']['n']}/{base['held']['HARD']['n']})")
    print(f"  {'arm':8s} " + "  ".join(f"{s_}:{b:4s}" for s_ in ('held', 'random') for b in ('EASY', 'MID', 'HARD', 'ALL')) + "  greedy held/random")
    for arm in ARMS:
        print(f"  {arm:8s} " + "  ".join(f"{g[arm][s_][b]*100:+9.1f}" for s_ in ('held', 'random') for b in ('EASY', 'MID', 'HARD', 'ALL'))
              + f"  {g[arm]['greedy']['held']*100:+.1f}/{g[arm]['greedy']['random']*100:+.1f}")
    print(f"  H1/H2/H3: {per[s]['judge']}  (None = undecidable)")
if len(seeds) >= 3:
    hs = per[seeds[0]]['hard_set']
    mean = {arm: {st: {b: float(np.mean([per[s]['gains'][arm][st][b] for s in seeds])) for b in ('EASY', 'MID', 'HARD', 'ALL')}
                  for st in ('held', 'random')} for arm in ARMS}
    mj = judge(mean, hs)
    for h in ('H1', 'H2', 'H3'):
        v = [per[s]['judge'][h] for s in seeds]; ok = sum(x is True for x in v) >= 2 and mj[h] is True
        print(f"{h}: holds in {sum(x is True for x in v)}/{len(v)} seeds (undecidable in {sum(x is None for x in v)}); on the mean: {mj[h]} -> "
              f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
else:
    print("fewer than 3 confirmatory seeds: preliminary, no verdict")
json.dump({str(k): v for k, v in per.items()}, open(f'{OUT}/report_{"_".join(map(str, seeds))}.json', 'w'), indent=1, default=str)
