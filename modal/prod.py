"""Production-style (OLMoE-like) MoE cube policy, trained on Modal to actually SOLVE.

  probe  : time N training steps end-to-end on a given GPU (costs cents) before any long run
  train  : resumable training run; checkpoints + log to /vol/out/prod/<tag>.{pt,json}
           periodic gate: held-out next-move accuracy, greedy and beam-search solve rate on random cubes
  solve  : beam-search solve rates for a saved checkpoint

Data: /vol/data/mixed.npz (the 250k-solve Kociemba set, sha256 80ea6303...), split by solve exactly like
src/train.py. Augmentation: the 8 whole-cube rotations that keep the U-D axis (src/cube_fast.py), which
map Kociemba solutions to Kociemba-style solutions with the phase boundary at the same step (verified
1600/1600 locally).
"""
import modal, pathlib, time
ROOT = pathlib.Path(__file__).parent.parent
app = modal.App("phasesplit-prod")
vol = modal.Volume.from_name("phasesplit-vol", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0", "numpy")
       .add_local_dir(str(ROOT / "src"), "/repo/src")
       .add_local_dir(str(ROOT / "probe"), "/repo/probe"))

def _setup():
    import sys; sys.path[:0] = ["/repo/src", "/repo/probe"]

class Data:
    def __init__(self, dev, data="mixed", rot="all"):
        import numpy as np, torch, cube_fast as C
        d = dict(np.load(f"/vol/data/{data}.npz")); self.data = data
        n = len(d["sol_len"]); perm = np.random.RandomState(0).permutation(n); ntr = int(n * 0.9)
        off = np.concatenate([[0], np.cumsum(d["sol_len"].astype(np.int64))])
        def rows(ids): return np.concatenate([np.arange(off[i], off[i + 1]) for i in ids])
        tr, te = rows(perm[:ntr]), rows(perm[ntr:])
        self.Xtr = torch.from_numpy(d["states"][tr].astype(np.int64)).to(dev); self.Ytr = torch.from_numpy(d["moves"][tr].astype(np.int64)).to(dev)
        self.Xte = torch.from_numpy(d["states"][te].astype(np.int64)).to(dev); self.Yte = torch.from_numpy(d["moves"][te].astype(np.int64)).to(dev)
        self.held_init = d["init_state"][perm[ntr:]][:600]
        self.rots = list(range(8)) if rot == "all" else list(C.KEEP_D)   # CFOP: only rotations keeping D on the bottom
        self.RP = torch.from_numpy(C.ROT_PERM).to(dev); self.RC = torch.from_numpy(C.ROT_RECOLOR.astype(np.int64)).to(dev)
        self.RM = torch.from_numpy(C.ROT_MOVE).to(dev)
    def batch(self, bs, gen):
        import torch
        i = torch.randint(0, len(self.Xtr), (bs,), device=self.Xtr.device, generator=gen)
        r = torch.tensor(self.rots, device=self.Xtr.device)[torch.randint(0, len(self.rots), (bs,), device=self.Xtr.device, generator=gen)]
        x = self.Xtr[i]; y = self.Ytr[i]
        x = self.RC[r[:, None], x.gather(1, self.RP[r])]       # rotate stickers, relabel colours
        return x, self.RM[r, y]

def random_cubes(n, seed, dev):
    """Uniformly random reachable cubes via probe/cube.random_state, encoded as stickers."""
    import random, numpy as np, torch, cube, cube_fast as C
    rng = random.Random(seed); CID = {c: i for i, c in enumerate("URFDLB")}
    return torch.tensor(np.stack([[CID[c] for c in cube.to_facelets(cube.random_state(rng))] for _ in range(n)]), device=dev)



def gate(model, D, dev, n_greedy=200, n_beam=40, widths=(256,), budget=40):
    import torch, numpy as np
    model.eval(); out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        idx = torch.arange(0, min(len(D.Xte), 200000), device=dev); acc = []
        for i in range(0, len(idx), 8192):
            lg, _, _ = model(D.Xte[idx[i:i + 8192]]); acc.append((lg.argmax(-1) == D.Yte[idx[i:i + 8192]]).float())
        out["heldout_acc"] = float(torch.cat(acc).mean())
    import cube_fast as C
    MP = torch.from_numpy(C.MOVE_PERM).to(dev); SOLVED = torch.from_numpy(C.SOLVED.astype("int64")).to(dev)
    if "cfop" in D.data: cubes = torch.from_numpy(D.held_init[:n_greedy].astype("int64")).to(dev)   # CFOP: held-out CFOP scrambles
    else: cubes = random_cubes(n_greedy, 123, dev)
    cur = cubes.clone(); done = torch.zeros(len(cur), dtype=torch.bool, device=dev)
    for t in range(budget):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a = model(cur)[0].argmax(-1)
        cur = torch.where(done[:, None], cur, cur.gather(1, MP[a])); done |= (cur == SOLVED).all(-1)
    out["greedy_solve_random" if "cfop" not in D.data else "greedy_solve_heldout"] = float(done.float().mean())
    from solve_fast import beam_solve
    for w in widths:
        r = [beam_solve(model, c, w, budget, dev) for c in cubes[:n_beam]]
        ok = [x for x in r if x is not None]
        out[f"beam{w}_solve_random" if "cfop" not in D.data else f"beam{w}_solve_heldout"] = len(ok) / n_beam; out[f"beam{w}_avg_moves"] = float(np.mean(ok)) if ok else None
    model.train(); return out

def _train(tag, preset, d, L, nh, bs, total_states, lr, seed, gate_every, max_hours, data="mixed", rot="all", budget=40):
    import os, json, math, torch, torch.nn.functional as F
    _setup()
    from model_prod import ProdStateModel
    dev = "cuda"; torch.manual_seed(seed); os.makedirs("/vol/out/prod", exist_ok=True)
    D = Data(dev, data, rot); model = ProdStateModel(d=d, L=L, nh=nh, preset=preset).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps = total_states // bs; warm = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    ck = f"/vol/out/prod/{tag}.pt"; logf = f"/vol/out/prod/{tag}.json"; step = 0; log = dict(cfg=dict(tag=tag, preset=preset, d=d, L=L, nh=nh, bs=bs, total_states=total_states, lr=lr, seed=seed, data=data, rot=rot, budget=budget), gates=[])
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["sd"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); step = s["step"]; log = s["log"]
        print(f"resumed {tag} at step {step}/{steps}", flush=True)
    gen = torch.Generator(device=dev).manual_seed(seed * 1000 + step)
    t0 = time.time(); last_save = time.time(); model.train()
    def save():
        torch.save(dict(sd=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, log=log), ck + ".tmp")
        os.replace(ck + ".tmp", ck); json.dump(log, open(logf, "w"), indent=1); vol.commit()
    while step < steps:
        x, y = D.batch(bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _, aux = model(x)
            loss = F.cross_entropy(lg.float(), y) + aux
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 200 == 0:
            el = time.time() - t0
            print(f"{tag} step {step}/{steps} loss {loss.item():.4f} {bs*200/max(el,1e-9):,.0f} states/s", flush=True); t0 = time.time()
        if step % gate_every == 0 or step == steps:
            g = gate(model, D, dev, widths=(256,) if step < steps else (256, 1024), budget=budget); g["step"] = step; g["states_seen"] = step * bs
            log["gates"].append(g); print(f"GATE {tag} {g}", flush=True); save(); last_save = time.time()
        elif time.time() - last_save > 600: save(); last_save = time.time()
        if (time.time() - log.setdefault("_start", time.time())) / 3600 > max_hours: print("time budget reached; saving and exiting", flush=True); break
    save(); return log["gates"][-1] if log["gates"] else None

@app.function(image=img, volumes={"/vol": vol}, gpu="A10G", timeout=86400)
def train_a10g(tag: str, preset: str = "olmoe", d: int = 256, layers: int = 6, nh: int = 4, bs: int = 1024,
               total_states: int = 60_000_000, lr: float = 1e-3, seed: int = 0, gate_every: int = 5000, max_hours: float = 20.0):
    return _train(tag, preset, d, layers, nh, bs, total_states, lr, seed, gate_every, max_hours)

@app.function(image=img, volumes={"/vol": vol}, gpu="L40S", timeout=86400)
def train_l40s(tag: str, preset: str = "olmoe", d: int = 256, layers: int = 6, nh: int = 4, bs: int = 1024,
               total_states: int = 60_000_000, lr: float = 1e-3, seed: int = 0, gate_every: int = 5000, max_hours: float = 20.0,
               data: str = "mixed", rot: str = "all", budget: int = 40):
    return _train(tag, preset, d, layers, nh, bs, total_states, lr, seed, gate_every, max_hours, data, rot, budget)

def _probe(preset, d, L, bs, steps):
    import torch, torch.nn.functional as F
    _setup()
    from model_prod import ProdStateModel
    dev = "cuda"; D = Data(dev); m = ProdStateModel(d=d, L=L, nh=4, preset=preset).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3); gen = torch.Generator(device=dev).manual_seed(0)
    for i in range(steps):
        if i == 10: torch.cuda.synchronize(); t0 = time.time()
        x, y = D.batch(bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _, aux = m(x); loss = F.cross_entropy(lg.float(), y) + aux
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); sps = bs * (steps - 10) / (time.time() - t0)
    tg = time.time(); g = gate(m, D, dev, n_greedy=50, n_beam=3, widths=(256,)); gate_s = time.time() - tg
    return dict(gpu=torch.cuda.get_device_name(0), preset=preset, d=d, L=L, bs=bs, states_per_s=round(sps),
                hours_per_60M=round(60e6 / sps / 3600, 2), gate_seconds=round(gate_s, 1), gate_untrained=g)

@app.function(image=img, volumes={"/vol": vol}, gpu="A10G", timeout=1800)
def probe_a10g(preset: str = "olmoe", d: int = 256, L: int = 6, bs: int = 1024, steps: int = 60):
    return _probe(preset, d, L, bs, steps)

@app.function(image=img, volumes={"/vol": vol}, gpu="L40S", timeout=1800)
def probe_l40s(preset: str = "olmoe", d: int = 256, L: int = 6, bs: int = 1024, steps: int = 60):
    return _probe(preset, d, L, bs, steps)

@app.local_entrypoint()
def probe():
    calls = [probe_a10g.spawn(), probe_l40s.spawn(), probe_a10g.spawn(preset="mixtral"), probe_l40s.spawn(preset="mixtral")]
    for c in calls: print(c.get())

@app.function(image=img, volumes={"/vol": vol}, gpu="L40S", timeout=7200)
def evaluate(tag: str, n: int = 500, widths: str = "64,256,1024", seed: int = 2026, budget: int = 40, data: str = "mixed"):
    """Large solve test on FRESH random cubes (seed differs from the training gate's seed 123)."""
    import torch, numpy as np, time, json
    _setup()
    from model_prod import ProdStateModel
    from solve_fast import beam_solve
    s = torch.load(f"/vol/out/prod/{tag}.pt", map_location="cuda", weights_only=False); c = s["log"]["cfg"]
    m = ProdStateModel(d=c["d"], L=c["L"], nh=c["nh"], preset=c["preset"]).cuda(); m.load_state_dict(s["sd"]); m.eval()
    if "cfop" in data:
        D = Data("cuda", data, "keepD"); cubes = torch.from_numpy(D.held_init[:n].astype("int64")).cuda()
    else: cubes = random_cubes(n, seed, "cuda")
    out = {"tag": tag, "n": n, "seed": seed, "budget": budget, "data": data, "test_set": "held-out CFOP scrambles" if "cfop" in data else "random cubes"}
    for w in [int(x) for x in widths.split(",")]:
        t0 = time.time(); r = [beam_solve(m, cb, w, budget, "cuda") for cb in cubes]; ok = [x for x in r if x is not None]
        out[f"beam{w}"] = dict(solved=len(ok), rate=len(ok) / n, avg_moves=float(np.mean(ok)) if ok else None,
                               max_moves=int(max(ok)) if ok else None, sec_per_cube=(time.time() - t0) / n)
        print(f"{tag} beam {w}: solved {len(ok)}/{n}, avg {out[f'beam{w}']['avg_moves']:.2f} moves, {out[f'beam{w}']['sec_per_cube']:.2f} s/cube", flush=True)
    json.dump(out, open(f"/vol/out/prod/{tag}_eval.json", "w"), indent=1); vol.commit(); return out

@app.function(image=img, volumes={"/vol": vol}, gpu="L40S", timeout=3600)
def solve_paths(tag: str, starts: list, width: int = 256, budget: int = 90):
    """Beam-search solution paths (move lists) for given start states; same search as src/cfop_paths.py."""
    import torch
    _setup(); import prod_analysis as A; A.DEV = "cuda"
    from model_prod import ProdStateModel
    s = torch.load(f"/vol/out/prod/{tag}.pt", map_location="cuda", weights_only=False); c = s["log"]["cfg"]
    m = ProdStateModel(d=c["d"], L=c["L"], nh=c["nh"], preset=c["preset"]).cuda(); m.load_state_dict(s["sd"]); m.eval()
    return [A.beam_solve_path(m, torch.tensor(x), width, budget) for x in starts]

@app.local_entrypoint()
def cfop_paths(tags: str = "cfop_mixtral_6L256_s1,cfop_mixtral_6L256_s2,cfop_mixtral_6L256_s3,cfop_mixtral_6L256_s4", n: int = 100):
    """Same scrambles and output format as src/cfop_paths.py -> out/prod/<tag>_paths.json (local)."""
    import numpy as np, json
    d = dict(np.load(ROOT / "data/cfop250k.npz")); N = len(d["sol_len"])
    te = np.random.RandomState(0).permutation(N)[int(N * 0.9):][:n]; off = np.concatenate([[0], np.cumsum(d["sol_len"].astype(np.int64))])
    starts = [d["init_state"][i].astype(int).tolist() for i in te]
    calls = {t: solve_paths.spawn(t, starts) for t in tags.split(",")}
    for t, call in calls.items():
        P = call.get()
        out = [dict(solve=int(i), start=s, model=p, solver=[int(x) for x in d["moves"][off[i]:off[i + 1]]]) for i, s, p in zip(te, starts, P)]
        json.dump(out, open(ROOT / f"out/prod/{t}_paths.json", "w"))
        print(t, "solved", sum(p is not None for p in P), "/", len(P), "mean len", np.mean([len(p) for p in P if p]))

@app.local_entrypoint()
def cfop_random_greedy(n_random: int = 200, n_held: int = 100):
    """CFOP policies on UNIFORMLY RANDOM cubes (beam 256/1024, greedy) and greedy on held-out CFOP scrambles.
    -> out/prod/<tag>_{random_w256,random_w1024,random_greedy,held_greedy}_paths.json (local)"""
    import numpy as np, json, sys
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "probe")]
    from prod_analysis import random_cubes
    rnd = random_cubes(n_random, 2026).tolist()
    d = dict(np.load(ROOT / "data/cfop250k.npz")); N = len(d["sol_len"])
    te = np.random.RandomState(0).permutation(N)[int(N * 0.9):][:n_held]; off = np.concatenate([[0], np.cumsum(d["sol_len"].astype(np.int64))])
    held = [d["init_state"][i].astype(int).tolist() for i in te]
    runs = {"random_w256": (rnd, 256, 120), "random_w1024": (rnd, 1024, 120), "random_greedy": (rnd, 1, 120), "held_greedy": (held, 1, 90)}
    calls = {(t, r): solve_paths.spawn(t, s, w, b) for t in [f"cfop_mixtral_6L256_s{k}" for k in range(5)] for r, (s, w, b) in runs.items()}
    for (t, r), call in calls.items():
        P = call.get(); S = runs[r][0]
        sol = [[int(x) for x in d["moves"][off[i]:off[i + 1]]] for i in te] if r.startswith("held") else [None] * len(S)
        json.dump([dict(start=s, model=p, solver=q) for s, p, q in zip(S, P, sol)], open(ROOT / f"out/prod/{t}_{r}_paths.json", "w"))
        ok = [len(p) for p in P if p]
        print(f"{t} {r}: solved {len(ok)}/{len(P)}, mean {np.mean(ok) if ok else float('nan'):.1f} moves", flush=True)
