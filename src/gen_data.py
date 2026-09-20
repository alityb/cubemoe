"""STEP 1: generate forced-seam and naive-Kociemba datasets."""
import sys, os, time, random, argparse
import numpy as np
sys.path.insert(0, '/Users/alityb/projects/cubemoe/probe')

MOVE_NAMES = [f+s for f in 'URFDLB' for s in ('',"'",'2')]
MOVE_ID = {m: i for i, m in enumerate(MOVE_NAMES)}
COLOR_ID = {c: i for i, c in enumerate('URFDLB')}

def _worker(args):
    kind, n, seed, lo, hi = args
    import cube
    rng = random.Random(seed)
    out = []
    if kind == 'mixed':
        sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
        import solver2
        while len(out) < n:
            st = solver2.sample_state(rng, near_frac=0.5, jmax=12)
            r = solver2.solve_full(st)
            if r is None: continue
            mv, sm = r
            out.append((st, mv, sm))
    elif kind == 'forced':
        import phase1, phase2
        tries = 0
        while len(out) < n and tries < n * 40:
            tries += 1
            s = cube.random_state(rng)
            L = rng.randint(lo, hi)
            m1 = phase1.solve_phase1(s, L, rng)
            if m1 is None: continue
            g = s
            for m in m1: g = cube.apply_move(g, m)
            m2 = phase2.solve_phase2(g)
            if m2 is None: continue
            out.append((s, m1 + m2, len(m1)))
    else:
        import kociemba
        while len(out) < n:
            s = cube.random_state(rng)
            mv = kociemba.solve(cube.to_facelets(s)).split()
            cur = s; g1 = []
            for m in mv:
                g1.append(cube.in_G1(cur)); cur = cube.apply_move(cur, m)
            g1.append(cube.in_G1(cur))
            if True not in g1: continue
            out.append((s, mv, g1.index(True)))
    # expand
    rec = []
    for s, mv, seam in out:
        cur = s; sts = []
        for m in mv:
            sts.append([COLOR_ID[c] for c in cube.to_facelets(cur)])
            cur = cube.apply_move(cur, m)
        rec.append((sts, [MOVE_ID[m] for m in mv], seam))
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True); ap.add_argument('--n', type=int, default=25000)
    ap.add_argument('--lo', type=int, default=10); ap.add_argument('--hi', type=int, default=20)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed_base', type=int, default=1000)
    a = ap.parse_args()
    from multiprocessing import Pool
    per = a.n // a.workers + 1
    t0 = time.time()
    with Pool(a.workers) as p:
        chunks = p.map(_worker, [(a.kind, per, a.seed_base + i, a.lo, a.hi) for i in range(a.workers)])
    recs = [r for c in chunks for r in c][:a.n]
    dt = time.time() - t0
    print(f"[{a.kind}] {len(recs)} solves in {dt:.1f}s ({len(recs)/dt:.1f}/s wall, {a.workers} workers)")

    states, moves, pos, remaining, phase, solve_id = [], [], [], [], [], []
    seam, sol_len, init_state = [], [], []
    for i, (sts, mvs, sm) in enumerate(recs):
        n = len(mvs)
        states.extend(sts); moves.extend(mvs)
        pos.extend(range(n)); remaining.extend(range(n, 0, -1))
        phase.extend([1 if t < sm else 2 for t in range(n)]); solve_id.extend([i] * n)
        seam.append(sm); sol_len.append(n); init_state.append(sts[0])
    d = dict(states=np.array(states, np.uint8), moves=np.array(moves, np.uint8),
             pos=np.array(pos, np.uint8), remaining=np.array(remaining, np.uint8),
             phase=np.array(phase, np.uint8), solve_id=np.array(solve_id, np.int32),
             seam=np.array(seam, np.uint8), sol_len=np.array(sol_len, np.uint8),
             init_state=np.array(init_state, np.uint8))
    path = os.environ.get('PHASESPLIT_OUT', f"/Users/alityb/projects/cubemoe/data/{a.kind}.npz")
    np.savez_compressed(path, **d)
    sd = float(np.std(d['seam']))
    # fraction of step-index cells containing both phases (row-weighted)
    both = 0
    for t in np.unique(d['pos']):
        m = d['pos'] == t
        if len(np.unique(d['phase'][m])) == 2: both += m.sum()
    print(f"  steps={len(moves)}  seam mean={d['seam'].mean():.2f} sd={sd:.2f} range={d['seam'].min()}-{d['seam'].max()}")
    print(f"  sol_len mean={d['sol_len'].mean():.2f}  phase1 frac={np.mean(d['phase']==1):.3f}")
    print(f"  rows in step-index cells containing BOTH phases: {both/len(moves):.3f}")
    print(f"  saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")

if __name__ == '__main__': main()
