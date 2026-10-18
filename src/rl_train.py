"""GRPO on a CFOP cube policy with only one part of the model trainable (prereg/PREREG_RL_ROUTING.md).

Arms (everything else frozen):
  router   the 6 router matrices (12,288 parameters)
  experts  all expert weights (33.0M); routers frozen
  full     every parameter (34.6M)
  lora     parameter-matched control: rank-4 LoRA on each layer's attention output projection (12,288 parameters)

One on-policy update per batch: B training scrambles x G samples at temperature `temp`, reward 1 if solved within
`budget` moves, advantage = (r - group mean) / (group std + 1e-6), loss = -sum_t A * log pi(a_t | s_t) / (#steps).
No KL term, no auxiliary load-balancing loss (the router is free to move).
Usage: python src/rl_train.py <tag> <arm> <lr> [steps] [seed]  -> out/rl/<tag>_<arm>_lr<lr>_s<seed>.{pt,json}"""
import sys, os, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
import prod_analysis as A
from rl_common import rollout, scrambles, amp
R = '/Users/alityb/projects/cubemoe'; DEV = A.DEV

class LoRA(nn.Module):
    def __init__(self, base, r=4, alpha=8.0):
        super().__init__(); self.base = base; d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in)); self.B = nn.Parameter(torch.zeros(d_out, r)); self.s = alpha / r
    def forward(self, x): return self.base(x) + (x @ self.A.T @ self.B.T) * self.s

def setup(m, arm):
    for p in m.parameters(): p.requires_grad_(False)
    if arm == 'lora':
        for b in m.blocks: b.attn.o = LoRA(b.attn.o).to(next(m.parameters()).device)
    for n, p in m.named_parameters():
        if arm == 'full' or (arm == 'router' and '.router.' in n) or (arm == 'experts' and any(k in n for k in ('w_gate', 'w_up', 'w_down'))) \
                or (arm == 'lora' and ('.o.A' in n or '.o.B' in n)):
            p.requires_grad_(True)
    return [p for p in m.parameters() if p.requires_grad]

def train(tag, arm, lr, steps=300, seed=0, B=64, G=8, temp=1.0, budget=90, eval_every=50, out_dir=f'{R}/out/rl', log_every=10):
    torch.manual_seed(seed); np.random.seed(seed)
    m, cfg = A.load(tag); m.eval(); params = setup(m, arm); npar = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    name = f'{tag}_{arm}_lr{lr:g}_s{seed}'; log = dict(tag=tag, arm=arm, lr=lr, seed=seed, B=B, G=G, temp=temp, budget=budget, n_params=npar, steps=[])
    Xev, _ = scrambles('test', 256)
    def evaluate():
        g = rollout(m, Xev, budget, temp=0); s = rollout(m, Xev, budget, temp=temp, G=4)
        return dict(greedy=float(g['solved'].float().mean()), sampled=float(s['solved'].float().mean()))
    log['eval0'] = evaluate(); print(f"{name}: {npar} trainable params | start {log['eval0']}", flush=True)
    t0 = time.time()
    for it in range(1, steps + 1):
        X, _ = scrambles('train', B, seed=seed * 100000 + it)
        r = rollout(m, X, budget, temp=temp, G=G, keep=True); rew = r['solved'].float().view(B, G)
        adv = ((rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-6)).view(-1)
        ep_adv = adv[r['episode']]; use = ep_adv != 0; nsteps = len(r['actions'])
        opt.zero_grad(set_to_none=True); used = int(use.sum())
        if used:
            S, Aa, W = r['states'][use], r['actions'][use], ep_adv[use]
            for i in range(0, used, 1024):   # chunked backward; gradients summed, so chunk size does not change the update
                with amp(DEV): lg = m(S[i:i + 1024].to(DEV))[0].float()
                lp = F.log_softmax(lg, -1).gather(1, Aa[i:i + 1024].to(DEV)[:, None]).squeeze(1)
                (-(W[i:i + 1024].to(DEV) * lp).sum() / nsteps).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        row = dict(step=it, reward=float(rew.mean()), mixed_groups=float(((rew.sum(1) > 0) & (rew.sum(1) < G)).float().mean()),
                   steps_used=used, sec=time.time() - t0)
        if it % eval_every == 0 or it == steps: row['eval'] = evaluate()
        log['steps'].append(row)
        if it % log_every == 0 or 'eval' in row:
            print(f"  [{it}] reward {row['reward']:.3f} mixed {row['mixed_groups']:.2f} {row['sec']:.0f}s" + (f" eval {row['eval']}" if 'eval' in row else ''), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(sd={k: v for k, v in m.state_dict().items()}, arm=arm, cfg=cfg), f'{out_dir}/{name}.pt')
    json.dump(log, open(f'{out_dir}/{name}.json', 'w'), indent=1)
    return log

if __name__ == '__main__':
    a = sys.argv; train(a[1], a[2], float(a[3]), int(a[4]) if len(a) > 4 else 300, int(a[5]) if len(a) > 5 else 0)
