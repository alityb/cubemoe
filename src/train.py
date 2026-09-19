import sys, os, time, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/Users/alityb/projects/cubemoe/src')
from model import SeqModel, StateModel, lb_loss, device_auto, N_COLOR, N_MOVE

ROOT = '/Users/alityb/projects/cubemoe'

def load(kind):
    d = dict(np.load(f'{ROOT}/data/{kind}.npz'))
    n = len(d['seam']); rng = np.random.RandomState(0); perm = rng.permutation(n)
    ntr = int(n * 0.9)
    return d, perm[:ntr], perm[ntr:]

def build_seq(d, ids_sel, maxlen):
    """[54 stickers][moves] ; targets at pos 53..53+n-1"""
    X = np.zeros((len(ids_sel), maxlen), np.int64)
    Y = np.full((len(ids_sel), maxlen), -100, np.int64)
    P = np.ones((len(ids_sel), maxlen), bool)
    off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
    for r, i in enumerate(ids_sel):
        st = d['init_state'][i].astype(np.int64)
        mv = d['moves'][off[i]:off[i+1]].astype(np.int64)
        n = len(mv); L = 54 + n
        X[r, :54] = st; X[r, 54:L] = mv + N_COLOR
        Y[r, 53:53+n] = mv
        P[r, :L] = False
    return X, Y, P

def run(args):
    dev = device_auto()
    small = dev != 'cuda'
    D, NL = (128, 4) if small else (256, 6)
    print(f"device={dev}  -> d={D} layers={NL}" + ("  [REDUCED: no CUDA available]" if small else ""))
    d, tr, te = load(args.data)
    maxlen = 54 + int(d['sol_len'].max()) + 1
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.model in ('moe', 'dense', 'hash'):
        kind = 'dense' if args.model == 'dense' else 'moe'
        router = 'hash' if args.model == 'hash' else 'learned'
        m = SeqModel(D, NL, 4, maxlen, kind, 8, 2, router).to(dev)
        Xtr, Ytr, Ptr = build_seq(d, tr, maxlen); Xte, Yte, Pte = build_seq(d, te, maxlen)
        bs = 128
    else:  # state / state-dense
        kind = 'dense' if args.model == 'state_dense' else 'moe'
        m = StateModel(D, NL, 4, kind, 8, 2).to(dev)
        off = np.concatenate([[0], np.cumsum(d['sol_len'].astype(np.int64))])
        mtr = np.concatenate([np.arange(off[i], off[i+1]) for i in tr])
        mte = np.concatenate([np.arange(off[i], off[i+1]) for i in te])
        Xtr, Ytr = d['states'][mtr].astype(np.int64), d['moves'][mtr].astype(np.int64)
        Xte, Yte = d['states'][mte].astype(np.int64), d['moves'][mte].astype(np.int64)
        bs = 512
    np_ = sum(p.numel() for p in m.parameters())
    print(f"model={args.model} params={np_/1e6:.2f}M  train={len(Xtr)} test={len(Xte)}")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(steps, 1), pct_start=0.1)

    t0 = time.time(); ntok = 0; step = 0; acc_hist = []
    for ep in range(args.epochs):
        m.train(); order = np.random.permutation(len(Xtr))
        for b in range(len(Xtr) // bs):
            sel = order[b*bs:(b+1)*bs]
            if args.model in ('moe', 'dense', 'hash'):
                x = torch.from_numpy(Xtr[sel]).to(dev); y = torch.from_numpy(Ytr[sel]).to(dev)
                pm = torch.from_numpy(Ptr[sel]).to(dev)
                logits, info = m(x, pm, collect=(kind == 'moe'))
                loss = F.cross_entropy(logits.reshape(-1, N_MOVE), y.reshape(-1), ignore_index=-100)
                valid = (y.reshape(-1) != -100)
                ntok += int(valid.sum())
            else:
                x = torch.from_numpy(Xtr[sel]).to(dev); y = torch.from_numpy(Ytr[sel]).to(dev)
                logits, info = m(x, collect=(kind == 'moe'))
                loss = F.cross_entropy(logits, y)
                valid = torch.ones(x.shape[0] * 55, dtype=torch.bool, device=dev)
                ntok += x.shape[0] * 55
            if kind == 'moe' and args.model != 'hash':
                aux = sum(lb_loss(p, t, 8, valid) for p, t in info) / len(info)
                loss = loss + args.lbl * aux
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); step += 1
        # eval
        m.eval(); cor = tot = 0
        with torch.no_grad():
            for b in range(0, len(Xte), bs):
                if args.model in ('moe', 'dense', 'hash'):
                    x = torch.from_numpy(Xte[b:b+bs]).to(dev); y = torch.from_numpy(Yte[b:b+bs]).to(dev)
                    lg, _ = m(x, torch.from_numpy(Pte[b:b+bs]).to(dev))
                    v = y != -100
                    cor += int((lg.argmax(-1)[v] == y[v]).sum()); tot += int(v.sum())
                else:
                    x = torch.from_numpy(Xte[b:b+bs]).to(dev); y = torch.from_numpy(Yte[b:b+bs]).to(dev)
                    lg, _ = m(x)
                    cor += int((lg.argmax(-1) == y).sum()); tot += len(y)
        acc = cor / tot; acc_hist.append(acc)
        print(f"  ep{ep+1}/{args.epochs} loss={loss.item():.4f} test_acc={acc:.4f} tok/s={ntok/(time.time()-t0):.0f}")
    maj = float(np.bincount(Yte[Yte != -100].ravel(), minlength=N_MOVE).max() / (Yte != -100).sum())
    wall = time.time() - t0
    tag = f"{args.model}_{args.data}" + (f"_s{args.seed}" if args.seed else "")
    torch.save({'sd': m.state_dict(), 'cfg': dict(D=D, NL=NL, maxlen=maxlen, model=args.model, data=args.data)},
               f'{ROOT}/out/{tag}.pt')
    res = dict(tag=tag, acc=acc, majority=maj, params=np_, tok_s=ntok/wall, wall_s=wall, device=dev,
               d=D, layers=NL, acc_hist=acc_hist, epochs=args.epochs, batch=bs)
    json.dump(res, open(f'{ROOT}/out/{tag}.json', 'w'), indent=1)
    print(f"DONE {tag}: test_acc={acc:.4f} majority={maj:.4f} tok/s={ntok/wall:.0f} wall={wall:.0f}s")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--data', default='forced')
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lbl', type=float, default=0.01); ap.add_argument('--seed', type=int, default=0)
    run(ap.parse_args())
