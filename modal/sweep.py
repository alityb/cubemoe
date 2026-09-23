"""H_slack routing sweep — runs on a separate workspace. Same pre-registration."""
import modal, pathlib
ROOT = pathlib.Path(__file__).parent.parent
app = modal.App("phasesplit-sweep")
vol = modal.Volume.from_name("phasesplit-vol", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0","numpy","scikit-learn","scipy","kociemba")
       .add_local_dir(str(ROOT/"src"), "/repo/src")
       .add_local_dir(str(ROOT/"probe"), "/repo/probe"))

def _prep():
    import os, sys, shutil
    os.makedirs("/vol/data",exist_ok=True); os.makedirs("/vol/out",exist_ok=True)
    for a,b in [("/vol/data","/repo/data"),("/vol/out","/repo/out")]:
        if not os.path.islink(b):
            if os.path.isdir(b): shutil.rmtree(b)
            os.makedirs("/repo",exist_ok=True); os.symlink(a,b)
    os.makedirs("/Users/alityb/projects",exist_ok=True)
    if not os.path.exists("/Users/alityb/projects/cubemoe"): os.symlink("/repo","/Users/alityb/projects/cubemoe")
    sys.path[:0]=["/repo/src","/repo/probe"]

@app.function(image=img, volumes={"/vol": vol}, cpu=4, timeout=1800)
def verify_data():
    import hashlib
    _prep()
    h=hashlib.sha256(open("/vol/data/mixed.npz","rb").read()).hexdigest()
    return dict(sha256=h[:16], matches_qa_passed_file=(h[:16]=="80ea630325a31bea"))

@app.function(image=img, volumes={"/vol": vol}, gpu="A10G", timeout=1800)
def probe_a10g(steps: int = 120):
    return _probe(steps)

@app.function(image=img, volumes={"/vol": vol}, gpu="L4", timeout=1800)
def probe_l4(steps: int = 120):
    return _probe(steps)

def _probe(steps):
    import time, numpy as np, torch, torch.nn.functional as F
    _prep()
    from model import SeqModel, lb_loss, N_MOVE
    from train import build_seq
    d=dict(np.load("/repo/data/mixed.npz"))
    n=len(d['seam']); tr=np.random.RandomState(0).permutation(n)[:4000]
    maxlen=54+int(d['sol_len'].max())+1
    X,Y,P=build_seq(d,tr,maxlen)
    m=SeqModel(256,6,4,maxlen,'moe',8,2,'learned').cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4); bs=128; ntok=0
    torch.cuda.synchronize(); t0=time.time()
    for i in range(steps):
        sl=slice((i*bs)%3000,(i*bs)%3000+bs)
        x=torch.from_numpy(X[sl]).cuda(); y=torch.from_numpy(Y[sl]).cuda(); pm=torch.from_numpy(P[sl]).cuda()
        lg,info=m(x,pm,collect=True)
        loss=F.cross_entropy(lg.reshape(-1,N_MOVE),y.reshape(-1),ignore_index=-100)
        v=(y.reshape(-1)!=-100); ntok+=int(v.sum())
        loss=loss+0.01*sum(lb_loss(p,t,8,v) for p,t in info)/len(info)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    tok_s=ntok/(time.time()-t0)
    ntr=int(n*0.9); mv=len(d['moves'])/n; tokA=ntr*mv*10
    return dict(tok_s=tok_s, hours=tokA/tok_s/3600, gpu=torch.cuda.get_device_name(0))

@app.local_entrypoint()
def plan():
    v=verify_data.remote()
    print("data sha256:", v)
    if not v["matches_qa_passed_file"]:
        print("!! uploaded file differs from the QA-passed 250k set — STOP"); return
    for name,fn,price in [("A10G",probe_a10g,1.10),("L4",probe_l4,0.80)]:
        r=fn.remote()
        print(f"  {name:<5} {r['gpu']:<22} {r['tok_s']:>9,.0f} tok/s  {r['hours']:.2f} h/model  ${r['hours']*price:.2f}/model  -> 20 models ${r['hours']*price*20:.2f}")
    print(f"  {'A100':<5} {'NVIDIA A100-40GB':<22} {17183:>9,} tok/s  0.75 h/model  $1.57/model  -> 20 models $31.40  (measured earlier)")

VARIANTS = {
  "V1_top1_e8_local":  dict(model="moe",  experts=8, topk=1, lbl_scope="local"),
  "V2_top1_e2_local":  dict(model="moe",  experts=2, topk=1, lbl_scope="local"),
  "V4_top2_e8_large":  dict(model="moe",  experts=8, topk=2, lbl_scope="large"),
  "H1_hash_e8":        dict(model="hash", experts=8, topk=1, lbl_scope="off"),
  "H2_hash_e2":        dict(model="hash", experts=2, topk=1, lbl_scope="off"),
  # Amendment 4: single-axis arms
  "V6_top1_e8_large":  dict(model="moe",  experts=8, topk=1, lbl_scope="large"),
  "V7_top2_e8_local":  dict(model="moe",  experts=8, topk=2, lbl_scope="local"),
  "H1_hash_e8_t1":     dict(model="hash", experts=8, topk=1, lbl_scope="off"),
}

@app.function(image=img, volumes={"/vol": vol}, gpu="A10G", timeout=14400)
def train_variant(vname: str, seed: int):
    import subprocess, os
    _prep()
    v=VARIANTS[vname]; tag=f"{vname}_s{seed}"
    if os.path.exists(f"/vol/out/{tag}.json"): return f"SKIP {tag}"
    cmd=["python","/repo/src/train.py","--model",v["model"],"--data","mixed","--epochs","10",
         "--seed",str(seed),"--experts",str(v["experts"]),"--topk",str(v["topk"]),
         "--lbl_scope",v["lbl_scope"],"--tag",tag]
    r=subprocess.run(cmd,cwd="/repo",env=dict(os.environ,PYTHONPATH="/repo/src:/repo/probe"),
                     capture_output=True,text=True)
    vol.commit()
    return (r.stdout or "")[-400:]+(r.stderr or "")[-400:]

@app.local_entrypoint()
def sweep():
    jobs=[(v,s) for v in VARIANTS for s in range(30,35)]
    print(f"CORE+ : {len(VARIANTS)} variants x 5 seeds = {len(jobs)} models on A10G (~${len(jobs)*0.99:.2f})")
    for i in range(0,len(jobs),4):
        grp=jobs[i:i+4]
        print(f"--- group {i//4+1}/{(len(jobs)+3)//4}: {[f'{v}_s{s}' for v,s in grp]}", flush=True)
        for r in train_variant.starmap(grp):
            print("   ", (r or "").strip().splitlines()[-1] if r else "", flush=True)
    print("SWEEP TRAINING DONE")

@app.local_entrypoint()
def finish_v1():
    """Option 1: complete V1 (seed 34 only). Resumable — skips anything already banked."""
    import time
    t0=time.time()
    r=train_variant.remote("V1_top1_e8_local", 34)
    print((r or "").strip().splitlines()[-1] if r else "")
    print(f"wall {time.time()-t0:.0f}s")


@app.local_entrypoint()
def run_arm(arm: str = "V6_top1_e8_large", seeds: str = "30,31,32,33,34"):
    """Amendment 4 single-axis arm. Resumable; banks each model as it finishes."""
    import time
    S=[int(x) for x in seeds.split(",")]
    print(f"{arm}: seeds {S}  (~${len(S)*1.10:.2f} at post-fix rate)", flush=True)
    t0=time.time()
    for i in range(0,len(S),3):
        grp=[(arm,s) for s in S[i:i+3]]
        print(f"--- group {i//3+1}: {[f'{a}_s{s}' for a,s in grp]}", flush=True)
        for r in train_variant.starmap(grp):
            print("   ", (r or "").strip().splitlines()[-1] if r else "", flush=True)
    print(f"{arm} DONE in {(time.time()-t0)/3600:.2f}h")


@app.local_entrypoint()
def run_arm_once(arm: str = "V6_top1_e8_large", seeds: str = "33,34"):
    """Single dispatch: all seeds submitted at once so a client death cannot stall
    dispatch between groups (the failure mode that killed the grouped runner twice)."""
    S=[int(x) for x in seeds.split(",")]
    jobs=[(arm,s) for s in S]
    print(f"{arm}: submitting {len(jobs)} models in ONE starmap -> {S}", flush=True)
    for r in train_variant.starmap(jobs):
        print("   ", (r or "").strip().splitlines()[-1] if r else "", flush=True)
    print(f"{arm} DISPATCH COMPLETE")
