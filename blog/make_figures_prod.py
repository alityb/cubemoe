"""Figures 1-2 for blog/phasesplit.md, from the production-style Kociemba models. Every number is computed here or read
from out/prod/*.json; nothing is typed in by hand. Style matches blog/make_figures.py."""
import sys, json, numpy as np, torch
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src', '/Users/alityb/projects/cubemoe/probe']
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import prod_analysis as A, cube_fast as C, cube
from qa import facelets_to_cubie
R = '/Users/alityb/projects/cubemoe'; F = f'{R}/blog/figures'
SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'; S1='#2a78d6'; S2='#eb6834'
plt.rcParams.update({'font.family':['Helvetica Neue','Arial','sans-serif'],'font.size':11,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,
    'xtick.color':MUTED,'ytick.color':MUTED,'axes.facecolor':SURF,'figure.facecolor':SURF,'savefig.facecolor':SURF,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.8,'axes.axisbelow':True,'axes.titlesize':12.5,
    'axes.titleweight':'semibold','axes.titlecolor':INK,'axes.titlelocation':'left'})
def save(fig, name): fig.savefig(f'{F}/{name}.png', dpi=200, bbox_inches='tight'); plt.close(fig); print('wrote', name)
def in_g1(s):
    cp, co, ep, eo = facelets_to_cubie(''.join('URFDLB'[x] for x in s))
    return all(x == 0 for x in co) and all(x == 0 for x in eo) and all(ep[p] >= 8 for p in range(8, 12))
def top1(m, X):
    cap = A.capture(m, torch.as_tensor(np.asarray(X)).long()); return cap['route'][len(m.blocks) - 1][:, 0].numpy()
E2 = [json.load(open(f'{R}/out/prod/mixtral_6L256_s{s}_mech.json'))['exp2']['phase2_expert'] for s in range(5)]
E1 = [json.load(open(f'{R}/out/prod/mixtral_6L256_s{s}_mech.json'))['exp2']['phase1_expert'] for s in range(5)]

# ---------- Fig 1a: one random cube solved by seed 0 ----------
m, _ = A.load('mixtral_6L256_s0'); start = A.random_cubes(1, 2026)[0]
path = A.beam_solve_path(m, start, 256, 40); S = [start.numpy()]
for mv in path: S.append(C.apply(S[-1][None], mv)[0])
S = np.stack(S[:-1]); ex = top1(m, S); g = next(t for t in range(len(S)) if in_g1(S[t]))
MN = [f + s for f in 'URFDLB' for s in ('', "'", '2')]
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
a.axvspan(-0.5, g - 0.5, color='#f0efec', zorder=0, lw=0); a.axvline(g - 0.5, color=INK2, lw=1, ls=(0, (3, 3)))
cols = [S2 if e == E2[0] else S1 for e in ex]
a.scatter(range(len(ex)), ex, c=cols, s=46, zorder=3, edgecolors=SURF, linewidths=1.2)
for t, mv in enumerate(path): a.text(t, -1.05, MN[mv], ha='center', va='top', fontsize=7.2, color=MUTED, rotation=90)
a.set_ylim(-2.4, 7.6); a.set_yticks(range(8)); a.set_yticklabels([f'e{i}' for i in range(8)]); a.set_xticks([])
a.set_xlabel('move played (model, beam 256)', labelpad=22); a.set_ylabel('final-layer expert (top-1)')
a.text((g - 1) / 2, 7.25, 'phase 1', ha='center', color=INK2, fontsize=10.5); a.text(g + (len(ex) - g) / 2 - 0.5, 7.25, 'in $G_1$', ha='center', color=INK2, fontsize=10.5)
a.set_title(f'a  One random cube, seed 0 ({len(path)} moves)')

# ---------- Fig 1b: phase-2 expert use around the G1 boundary, held-out solver positions, all seeds ----------
X, lab = A.heldout(600); ph = lab['phase']; sid = lab['solve']
seam = {}; 
for i, (s_, p_) in enumerate(zip(sid, ph)):
    if p_ == 2 and s_ not in seam: seam[s_] = i
first = {}
for i, s_ in enumerate(sid): first.setdefault(s_, i)
rel = np.array([i - seam[s_] if s_ in seam else 999 for i, s_ in enumerate(sid)])
ks = np.arange(-8, 9)
for s in range(5):
    m, _ = A.load(f'mixtral_6L256_s{s}'); e = top1(m, X)
    y = [100 * (e[rel == k] == E2[s]).mean() for k in ks]
    print(f'fig1b seed {s}: phase-2 expert on phase-1 steps -8..-1: {min(y[:8]):.1f}-{max(y[:8]):.1f}%, on steps 0..8: {min(y[8:]):.1f}-{max(y[8:]):.1f}%')
    b.plot(ks[ks < 0], np.array(y)[ks < 0], color=S1, lw=1.6, alpha=0.85, marker='o', ms=3.5)
    b.plot(ks[ks >= 0], np.array(y)[ks >= 0], color=S2, lw=1.6, alpha=0.85, marker='o', ms=3.5)
b.axvline(-0.5, color=INK2, lw=1, ls=(0, (3, 3))); b.axvspan(-8.5, -0.5, color='#f0efec', zorder=0, lw=0)
b.set_xlim(-8.5, 8.5); b.set_xticks(range(-8, 9, 2)); b.set_ylim(-3, 103); b.set_xlabel('moves relative to entering $G_1$ (solver positions, held out)')
b.set_ylabel('decision tokens sent to the\nphase-2 expert (%)'); b.set_title('b  All five seeds')
save(fig, 'prod_fig1_routing')

# ---------- Fig 2: knockout dissociation ----------
fig, ax = plt.subplots(figsize=(5.6, 5.0))
for s in range(5):
    ko = json.load(open(f'{R}/out/prod/mixtral_6L256_s{s}_analysis.json'))['E']['knockout']
    for e, v in ko.items():
        x, y = 100 * v['phase1'], 100 * v['phase2']; big = max(x, y) > 2
        ax.scatter(x, y, s=46 if big else 18, color=(S2 if y > x else S1) if big else MUTED, alpha=0.9 if big else 0.6, edgecolors=SURF, linewidths=1, zorder=3)
ax.plot([0, 32], [0, 32], color=AXIS, lw=0.8, ls=(0, (3, 3)))
ax.set_xlim(-1, 32); ax.set_ylim(-1, 32); ax.set_xlabel('drop in p(solver\'s move) on phase-1 positions (points)')
ax.set_ylabel('drop on phase-2 positions (points)'); ax.set_title('Removing one final-layer expert')
ax.text(16, 1.6, 'phase-1 experts', color=S1, fontsize=10.5); ax.text(1.2, 27.5, 'phase-2 experts', color=S2, fontsize=10.5)
ax.text(14, 22, '40 experts, 5 seeds\ngrey: < 2 points either way', color=MUTED, fontsize=9)
save(fig, 'prod_fig2_knockout')
print('fig1a: path', len(path), 'moves; G1 entered before move', g, '; experts used', sorted(set(ex.tolist())), 'ph2 expert', E2[0])
