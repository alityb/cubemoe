"""Pilot for the RL routing study: BASE-MODEL properties only (no RL outcomes), on the pilot seed (4), so the
pre-registration can fix difficulty bins, group size, temperature and budget before any RL run.

  - greedy and sampled (temperature 1.0, 0.7) solve rates on held-out scrambles, budget 90
  - per-scramble pass@k (k <= 64) distribution -> how many scrambles are 'triggerable' vs 'unreached'
  - for GRPO groups of G = 8 / 16: fraction of groups with mixed rewards (the only groups that give signal)
  - rollout throughput on this machine
Usage: python src/rl_pilot.py [tag] -> out/rl/pilot_<tag>.json"""
import sys, os, json, time, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from rl_common import rollout, scrambles
R = '/Users/alityb/projects/cubemoe'; os.makedirs(f'{R}/out/rl', exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else 'cfop_mixtral_6L256_s4'
m, _ = A.load(tag); out = {}
X, _ = scrambles('test', 1000)
t0 = time.time(); g = rollout(m, X, 90, temp=0); out['greedy'] = float(g['solved'].float().mean())
print(f"{tag}: greedy {out['greedy']*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
for temp in (1.0, 0.7):
    t0 = time.time(); r = rollout(m, X, 90, temp=temp, G=64); dt = time.time() - t0
    s = r['solved'].view(1000, 64).float(); p = s.mean(1).numpy(); steps = int(r['length'].sum())
    res = dict(pass1=float(p.mean()), frac_pass64_zero=float((p == 0).mean()), frac_pass64_pos=float((p > 0).mean()),
               frac_greedyfail_but_pos=float(((p > 0) & ~g['solved'].numpy()).mean()),
               pass_hist={f'{lo}-{hi}': float(((p > lo) & (p <= hi)).mean()) for lo, hi in ((0, .05), (.05, .25), (.25, .5), (.5, .75), (.75, 1.0))},
               states_per_s=steps / dt, solved_len=float(r['length'][r['solved']].float().mean()))
    for G in (8, 16):
        sg = s[:, :G]; res[f'mixed_groups_G{G}'] = float(((sg.sum(1) > 0) & (sg.sum(1) < G)).float().mean())
    out[f'temp{temp}'] = res
    print(f"  temp {temp}: pass@1 {res['pass1']*100:.1f}% | pass@64 = 0 for {res['frac_pass64_zero']*100:.0f}% of scrambles | "
          f"greedy fails but pass@64 > 0: {res['frac_greedyfail_but_pos']*100:.0f}% | mixed groups G8 {res['mixed_groups_G8']*100:.0f}% "
          f"G16 {res['mixed_groups_G16']*100:.0f}% | {res['states_per_s']:.0f} states/s | {dt:.0f}s", flush=True)
    print(f"     per-scramble pass rate histogram: {res['pass_hist']}", flush=True)
json.dump(out, open(f'{R}/out/rl/pilot_{tag}.json', 'w'), indent=1)
