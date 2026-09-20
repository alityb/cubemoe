"""AMENDMENT 2 — SCALE REPLICATION on CUDA. Same prereg, no new criteria.
6L/d256 (train.py's cuda branch), 250k solves, seeds 20-24 per arm + hash twins."""
import modal, pathlib
ROOT = pathlib.Path(__file__).parent.parent
app = modal.App("phasesplit-scale")
vol = modal.Volume.from_name("phasesplit-vol", create_if_missing=True)

img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0","numpy","scikit-learn","scipy","kociemba")
       .add_local_dir(str(ROOT/"src"), "/repo/src")
       .add_local_dir(str(ROOT/"probe"), "/repo/probe"))

def _prep():
    """Recreate the repo layout the scripts expect, rooted at /repo."""
    import os, sys
    os.makedirs("/repo/data", exist_ok=True); os.makedirs("/repo/out", exist_ok=True)
    for a,b in [("/vol/data","/repo/data"),("/vol/out","/repo/out")]:
        os.makedirs(a, exist_ok=True)
        if not os.path.islink(b):
            import shutil
            if os.path.isdir(b): shutil.rmtree(b)
            os.symlink(a,b)
    sys.path[:0] = ["/repo/src","/repo/probe"]

def _patch_paths():
    """The scripts hardcode /Users/alityb/projects/cubemoe; make that path resolve to /repo."""
    import os
    os.makedirs("/Users/alityb/projects", exist_ok=True)
    tgt="/Users/alityb/projects/cubemoe"
    if not os.path.exists(tgt): os.symlink("/repo", tgt)

@app.function(image=img, volumes={"/vol": vol}, cpu=16, timeout=7200)
def gen_shard(i: int, n: int):
    import subprocess, os
    _prep(); _patch_paths()
    out=f"/vol/data/shard_{i}.npz"
    if os.path.exists(out): return f"shard {i} exists"
    env=dict(os.environ, PHASESPLIT_OUT=out, PYTHONPATH="/repo/src:/repo/probe")
    r=subprocess.run(["python","/repo/src/gen_data.py","--kind","mixed","--n",str(n),
                      "--workers","16","--seed_base",str(20000+i*100)],
                     cwd="/repo", env=env, capture_output=True, text=True)
    vol.commit()
    return (r.stdout or "")[-600:] + (r.stderr or "")[-400:]

@app.function(image=img, volumes={"/vol": vol}, cpu=8, timeout=3600)
def merge(nshards: int):
    import numpy as np, os
    _prep(); _patch_paths()
    parts=[dict(np.load(f"/vol/data/shard_{i}.npz")) for i in range(nshards)]
    out={}; off=0
    for k in ("states","moves","pos","remaining","phase"):
        out[k]=np.concatenate([p[k] for p in parts])
    sid=[]; 
    for p in parts:
        sid.append(p["solve_id"].astype(np.int64)+off); off+=len(p["seam"])
    out["solve_id"]=np.concatenate(sid).astype(np.int32)
    for k in ("seam","sol_len"): out[k]=np.concatenate([p[k] for p in parts])
    out["init_state"]=np.concatenate([p["init_state"] for p in parts])
    np.savez_compressed("/vol/data/mixed.npz", **out); vol.commit()
    return dict(solves=int(len(out["seam"])), steps=int(len(out["moves"])),
                seam_sd=float(out["seam"].std()))

@app.function(image=img, volumes={"/vol": vol}, cpu=8, timeout=7200)
def qa():
    import subprocess, os
    _prep(); _patch_paths()
    r=subprocess.run(["python","/repo/src/qa.py","--kind","mixed","--nsample","1200"],
                     cwd="/repo", env=dict(os.environ, PYTHONPATH="/repo/src:/repo/probe"),
                     capture_output=True, text=True)
    vol.commit()
    return dict(rc=r.returncode, out=(r.stdout or "")[-3000:], err=(r.stderr or "")[-800:])

@app.function(image=img, volumes={"/vol": vol}, gpu="A100-40GB", timeout=14400)
def train(model: str, seed: int, epochs: int):
    import subprocess, os
    _prep(); _patch_paths()
    tag=f"{'moe_mixed' if model=='moe' else ('state_mixed' if model=='state' else 'hash_mixed')}_s{seed}"
    if os.path.exists(f"/vol/out/{tag}.json"): return f"SKIP {tag}"
    r=subprocess.run(["python","/repo/src/train.py","--model",model,"--data","mixed",
                      "--epochs",str(epochs),"--seed",str(seed)],
                     cwd="/repo", env=dict(os.environ, PYTHONPATH="/repo/src:/repo/probe"),
                     capture_output=True, text=True)
    vol.commit()
    return (r.stdout or "")[-500:] + (r.stderr or "")[-500:]

@app.function(image=img, volumes={"/vol": vol}, gpu="A100-40GB", timeout=14400)
def confirm(tag: str, is_state: bool, max_solves: int):
    import subprocess, os
    _prep(); _patch_paths()
    cmd=["python","/repo/src/confirm.py","--tag",tag,"--data","mixed","--max_solves",str(max_solves)]
    if is_state: cmd.append("--state")
    r=subprocess.run(cmd, cwd="/repo", env=dict(os.environ, PYTHONPATH="/repo/src:/repo/probe"),
                     capture_output=True, text=True)
    vol.commit()
    return (r.stdout or "")[-800:] + (r.stderr or "")[-600:]

@app.function(image=img, volumes={"/vol": vol}, cpu=4, timeout=3600)
def gates():
    import subprocess, os
    _prep(); _patch_paths()
    r=subprocess.run(["python","/repo/src/confirm_gates.py"], cwd="/repo",
                     env=dict(os.environ, PYTHONPATH="/repo/src:/repo/probe"),
                     capture_output=True, text=True)
    return (r.stdout or "")+(r.stderr or "")[-500:]

@app.local_entrypoint()
def data(nshards: int = 10, per: int = 25000):
    print(f"generating {nshards} x {per} = {nshards*per} solves ...")
    for r in gen_shard.starmap([(i, per) for i in range(nshards)]):
        print("  ", (r or "").strip().splitlines()[-1] if r else "")
    print("merge:", merge.remote(nshards))
    q = qa.remote()
    print("QA rc:", q["rc"])
    print(q["out"])

@app.function(image=img, volumes={"/vol": vol}, cpu=8, timeout=3600)
def prefix_diag():
    """Is the 5-prefix gate failure real stereotypy, or pigeonhole at 10x n?
    Scale-controlled: subsample 25k solves from the 250k set and recompute."""
    import numpy as np, collections
    d=dict(np.load("/vol/data/mixed.npz"))
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    n=len(d['seam'])
    def rate(idx,k=5):
        c=collections.Counter()
        for i in idx:
            sm=int(d['seam'][i])
            if sm>=k: c[tuple(d['moves'][off[i]:off[i]+k].tolist())]+=1
        tot=sum(c.values())
        return (1-len(c)/max(tot,1)), len(c), tot
    rng=np.random.RandomState(0); out={}
    out['full_250k']=rate(range(n))
    for trial in range(3):
        sub=rng.choice(n,25000,replace=False)
        out[f'subsample_25k_{trial}']=rate(sub)
    for m in (50000,100000):
        out[f'subsample_{m//1000}k']=rate(rng.choice(n,m,replace=False))
    return {k:(round(v[0],4),v[1],v[2]) for k,v in out.items()}

@app.local_entrypoint()
def diag():
    r = prefix_diag.remote()
    print("\n=== 5-PREFIX REPEAT RATE vs SAMPLE SIZE (scale-controlled) ===")
    print(f"{'set':<22}{'repeat_rate':>12}{'distinct':>10}{'total':>9}")
    for k,v in r.items():
        print(f"{k:<22}{v[0]:>12.4f}{v[1]:>10}{v[2]:>9}")
    print("\nlocal 25k reference (confirmatory set): 0.029")

@app.local_entrypoint()
def full(max_a: int = 1200, max_c: int = 800):
    q = qa.remote()
    print("QA rc:", q["rc"]); print(q["out"][-1800:])
    if q["rc"] != 0:
        print("QA FAILED — refusing to train."); return
    jobs = [("moe", s, 10) for s in range(20,25)] + \
           [("hash", s, 10) for s in range(20,25)] + \
           [("state", s, 3) for s in range(20,25)]
    for r in train.starmap(jobs):
        print("  ", (r or "").strip().splitlines()[-1] if r else "")
    cj = [(f"moe_mixed_s{s}", False, max_a) for s in range(20,25)] + \
         [(f"state_mixed_s{s}", True, max_c) for s in range(20,25)]
    for r in confirm.starmap(cj):
        print("  ", (r or "").strip().splitlines()[-1] if r else "")
    print(gates.remote())

@app.function(image=img, volumes={"/vol": vol}, gpu="A100-40GB", timeout=1800)
def throughput_probe(steps: int = 150):
    """Measure arm-A and arm-C tok/s at 6L/d256 on the 250k set. Mimics train.py's loop
    exactly (same batch sizes, same ops) but runs only `steps` steps. Costs cents."""
    import time, numpy as np, torch, torch.nn.functional as F
    _prep(); _patch_paths()
    import sys; sys.path[:0]=["/repo/src","/repo/probe"]
    from model import SeqModel, StateModel, lb_loss, N_MOVE
    from train import build_seq
    d=dict(np.load("/repo/data/mixed.npz"))
    n=len(d['seam']); perm=np.random.RandomState(0).permutation(n); tr=perm[:int(n*0.9)]
    dev='cuda'; D,NL=256,6; out={}
    # ---- arm A ----
    maxlen=54+int(d['sol_len'].max())+1
    X,Y,P=build_seq(d,tr[:4000],maxlen)
    m=SeqModel(D,NL,4,maxlen,'moe',8,2,'learned').to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4); bs=128
    ntok=0; torch.cuda.synchronize(); t0=time.time()
    for i in range(steps):
        sl=slice((i*bs)%3000,(i*bs)%3000+bs)
        x=torch.from_numpy(X[sl]).to(dev); y=torch.from_numpy(Y[sl]).to(dev); pm=torch.from_numpy(P[sl]).to(dev)
        lg,info=m(x,pm,collect=True)
        loss=F.cross_entropy(lg.reshape(-1,N_MOVE),y.reshape(-1),ignore_index=-100)
        v=(y.reshape(-1)!=-100); ntok+=int(v.sum())
        loss=loss+0.01*sum(lb_loss(p,t,8,v) for p,t in info)/len(info)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); out['armA_tok_s']=ntok/(time.time()-t0)
    del m,opt; torch.cuda.empty_cache()
    # ---- arm C ----
    off=np.concatenate([[0],np.cumsum(d['sol_len'].astype(np.int64))])
    rows=np.concatenate([np.arange(off[i],off[i+1]) for i in tr[:3000]])
    st=d['states'][rows].astype(np.int64); yy=d['moves'][rows].astype(np.int64)
    m=StateModel(D,NL,4,'moe',8,2).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4); bs=512
    ntok=0; torch.cuda.synchronize(); t0=time.time()
    for i in range(steps):
        sl=slice((i*bs)%(len(st)-bs),(i*bs)%(len(st)-bs)+bs)
        x=torch.from_numpy(st[sl]).to(dev); y=torch.from_numpy(yy[sl]).to(dev)
        lg,info=m(x,collect=True)
        loss=F.cross_entropy(lg,y)
        v=torch.ones(x.shape[0]*55,dtype=torch.bool,device=dev); ntok+=x.shape[0]*55
        loss=loss+0.01*sum(lb_loss(p,t,8,v) for p,t in info)/len(info)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); out['armC_tok_s']=ntok/(time.time()-t0)
    # ---- project ----
    ntr=int(n*0.9); mv=len(d['moves'])/n
    tokA=ntr*mv*10; tokC=int(len(d['moves'])*0.9)*55*3
    out['armA_hours']=tokA/out['armA_tok_s']/3600
    out['armC_hours']=tokC/out['armC_tok_s']/3600
    out['ratio_C_over_A']=out['armC_tok_s']/out['armA_tok_s']
    return out

@app.local_entrypoint()
def probe():
    r=throughput_probe.remote()
    P=2.10
    print("\n=== MEASURED at 6L/d256, A100-40GB, 250k set ===")
    print(f"  arm A: {r['armA_tok_s']:>10,.0f} tok/s -> {r['armA_hours']:.2f} h/model = ${r['armA_hours']*P:.2f}")
    print(f"  arm C: {r['armC_tok_s']:>10,.0f} tok/s -> {r['armC_hours']:.2f} h/model = ${r['armC_hours']*P:.2f}")
    print(f"  arm C is {r['ratio_C_over_A']:.1f}x faster per token than arm A (local was 6.8x)")
    print(f"\n  REMAINING WORK: 5 armA + 4 hash + 5 armC")
    print(f"    arm A + hash : ${9*r['armA_hours']*P:.2f}")
    print(f"    arm C        : ${5*r['armC_hours']*P:.2f}")
    print(f"    TOTAL        : ${(9*r['armA_hours']+5*r['armC_hours'])*P:.2f}   (credits left $13.69)")

@app.local_entrypoint()
def armA(max_a: int = 1200):
    """Arm A + hash twins only. Low concurrency so each finished model banks to the volume."""
    jobs=[("moe",s,10) for s in range(20,25)]+[("hash",s,10) for s in range(20,25)]
    for i in range(0,len(jobs),3):
        grp=jobs[i:i+3]
        print(f"--- group {i//3+1}: {[f'{m}_s{s}' for m,s,_ in grp]}")
        for r in train.starmap(grp):
            print("   ", (r or "").strip().splitlines()[-1] if r else "")
    cj=[(f"moe_mixed_s{s}", False, max_a) for s in range(20,25)]
    for i in range(0,len(cj),3):
        for r in confirm.starmap(cj[i:i+3]):
            print("   ", (r or "").strip().splitlines()[-1] if r else "")
    print(gates.remote())

@app.local_entrypoint()
def armC(max_c: int = 800):
    """Arm C only (for the second workspace). QA-gates the uploaded dataset first."""
    q = qa.remote()
    print("QA rc:", q["rc"]); print(q["out"][-1400:])
    if q["rc"] != 0:
        print("QA FAILED — refusing to train."); return
    jobs=[("state",s,3) for s in range(20,25)]
    for i in range(0,len(jobs),2):
        grp=jobs[i:i+2]
        print(f"--- group {i//2+1}: {[f'state_s{s}' for _,s,_ in grp]}")
        for r in train.starmap(grp):
            print("   ", (r or "").strip().splitlines()[-1] if r else "")
    cj=[(f"state_mixed_s{s}", True, max_c) for s in range(20,25)]
    for i in range(0,len(cj),2):
        for r in confirm.starmap(cj[i:i+2]):
            print("   ", (r or "").strip().splitlines()[-1] if r else "")
