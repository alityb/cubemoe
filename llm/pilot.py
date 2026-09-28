"""EXPLORATORY pilot (Granite). Stage 'base': full-context and 2-token-context probabilities of every
target token + routing at every layer. Stage 'ko': single-expert knockout and size-matched random
noise at chosen layers."""
import sys, json, time, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/Users/alityb/projects/cubemoe/llm')
import harness
from transformers import AutoModelForCausalLM, AutoTokenizer
R='/Users/alityb/projects/cubemoe/llm'
ap=argparse.ArgumentParser(); ap.add_argument('--model',default='ibm-granite/granite-3.1-1b-a400m-base')
ap.add_argument('--stage'); ap.add_argument('--layers',default=''); ap.add_argument('--n',type=int,default=16)
ap.add_argument('--T',type=int,default=512); a=ap.parse_args()
short=a.model.split('/')[-1]; dev='mps'
tok=AutoTokenizer.from_pretrained(a.model); m=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16).to(dev).eval()
Rs=harness.routers(m); NL=len(Rs)
def chunks():
    C=json.load(open(f'{R}/corpus.json')); out=[]
    ids=tok(C['prose'][:400000]).input_ids
    for i in range(a.n): out.append(('prose',ids[i*a.T:(i+1)*a.T]))
    for s in C['code']:
        t=tok(s).input_ids
        if len(t)>=a.T and sum(1 for d,_ in out if d=='code')<a.n: out.append(('code',t[:a.T]))
    for s in C['license']:
        t=tok(s).input_ids
        if len(t)>=a.T and sum(1 for d,_ in out if d=='license')<a.n//2: out.append(('license',t[:a.T]))
    return out
CH=chunks(); X=torch.tensor([c for _,c in CH]); dom=np.array([d for d,_ in CH])
def logp_full():
    out=[]
    with torch.no_grad():
        for i in range(0,len(X),4):
            lg=m(X[i:i+4].to(dev)).logits.float()
            out.append(torch.log_softmax(lg[:,:-1],-1).gather(-1,X[i:i+4,1:,None].to(dev))[...,0].cpu())
    return torch.cat(out).numpy()          # (n_chunks, T-1): log p(x_{t+1} | x_<=t)
if a.stage=='base':
    t0=time.time()
    for r in Rs: r._record=[]
    lp=logp_full()
    route=np.stack([torch.cat(r._record).numpy().reshape(len(X),a.T,-1)[:,:-1] for r in Rs])   # (L, chunks, T-1, k)
    harness.reset(m)
    # 2-token context: predict x_{t+1} from (x_{t-1}, x_t) only
    W=torch.stack([X[:,:-2],X[:,1:-1]],-1).reshape(-1,2); tg=X[:,2:].reshape(-1)
    loc=[]
    with torch.no_grad():
        for i in range(0,len(W),2048):
            lg=m(W[i:i+2048].to(dev)).logits[:,-1].float()
            loc.append(torch.log_softmax(lg,-1).gather(-1,tg[i:i+2048,None].to(dev))[:,0].cpu())
    lloc=np.full(lp.shape,np.nan); lloc[:,1:]=torch.cat(loc).numpy().reshape(len(X),-1)
    np.savez_compressed(f'{R}/base_{short}.npz',lp=lp,lloc=lloc,route=route.astype(np.int16),
                        tok=X[:,:-1].numpy(),tgt=X[:,1:].numpy(),dom=dom)
    pf=np.exp(lp); pl=np.exp(lloc); ok=pf>0.5
    ctx=ok&(pl<0.1); lcl=ok&(pl>0.5)
    print(f"{short}: {len(X)} chunks x {a.T} tokens, {time.time()-t0:.0f}s")
    for d in ('prose','code','license'):
        s=dom==d
        print(f"  {d:8s} predicted well (p>0.5): {ok[s].mean():.2f}   of those: needs-context {ctx[s].sum()/ok[s].sum():.2f}  local {lcl[s].sum()/ok[s].sum():.2f}")
    print(f"  groups: needs-context n={ctx.sum()} (mean p_full {pf[ctx].mean():.2f})   local n={lcl.sum()} (mean p_full {pf[lcl].mean():.2f})")
elif a.stage=='ko':
    B=dict(np.load(f'{R}/base_{short}.npz')); lp0=B['lp']
    blocks=harness.moe_blocks(m); res={}
    for L in [int(x) for x in a.layers.split(',')]:
        state={'mode':None}
        def hook(mod,inp,out):
            if state['mode']=='store': state['y']=out.detach()
            elif state['mode']=='cmp': state['dn']=(out.detach()-state['y']).float().norm(dim=-1,keepdim=True)
            elif state['mode']=='noise':
                n=torch.randn_like(out.float()); n=n/n.norm(dim=-1,keepdim=True)*state['dn']
                return (state['y'].float()+n).to(out.dtype)
            return None
        h=blocks[L].register_forward_hook(hook); E=Rs[L].num_experts; t0=time.time()
        dko=np.zeros((E,)+lp0.shape,np.float32); dno=np.zeros_like(dko)
        for i in range(0,len(X),4):
            xb=X[i:i+4].to(dev); tb=X[i:i+4,1:,None].to(dev)
            with torch.no_grad():
                state['mode']='store'; m(xb); y0=state['y']
                for e in range(E):
                    Rs[L]._knockout={e}; state['mode']='cmp'; state['y']=y0
                    lg=m(xb).logits.float(); lk=torch.log_softmax(lg[:,:-1],-1).gather(-1,tb)[...,0].cpu().numpy()
                    Rs[L]._knockout=set(); state['mode']='noise'; torch.manual_seed(e)
                    lg=m(xb).logits.float(); ln=torch.log_softmax(lg[:,:-1],-1).gather(-1,tb)[...,0].cpu().numpy()
                    dko[e,i:i+4]=lp0[i:i+4]-lk; dno[e,i:i+4]=lp0[i:i+4]-ln
        h.remove(); state['mode']=None
        np.savez_compressed(f'{R}/ko_{short}_L{L}.npz',dko=dko,dno=dno)
        print(f"  layer {L}: {E} experts done in {time.time()-t0:.0f}s",flush=True)
