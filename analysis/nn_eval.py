"""Neural-network evaluation: same splits, subsets and windows as eval2.py, but the models read the raw
post-fault trajectories (angle relative to the centre of inertia, speed deviation) instead of hand-built features.

Models: 1D-CNN, GRU, small Transformer encoder. Fixed hyper-parameters, no tuning on test folds.
Usage: python nn_eval.py [--quick] [--windows 0 0.1] [--repeats 2] [--models cnn gru transformer]
"""
import sys, pathlib, argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, warnings
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as f
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--windows", type=float, nargs="+", default=[0.0, 0.1])
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--models", nargs="+", default=["cnn", "gru", "transformer"])
ap.add_argument("--out", default="results/nn_results.csv")
args = ap.parse_args()

d = f.load_results()
d = d[d.label_status != "uncertain"].reset_index(drop=True)
buses = sorted(d.bus.unique())
M = f.M


def sequences(df, W):
    """Return padded array (n, T, 20), lengths, kept row indices."""
    seqs, idx = [], []
    for i, r in df.iterrows():
        if r.no_postfault_signal == 1:
            continue
        z = np.load(f"{r.trajdir}/{r.sid}.npz"); dl, om = z["delta"], z["omega"]
        i1 = int(round((r.t_clear + W - 0.9) / 0.01))
        if i1 >= len(dl) or np.isnan(dl[i1]).any():
            continue
        coi = (dl * M).sum(1, keepdims=True) / M.sum()
        a = (dl - coi)[10:i1 + 1]
        w = (om - 1.0)[10:i1 + 1]
        seqs.append(np.concatenate([a, w], 1)); idx.append(i)
    L = np.array([len(s) for s in seqs]); T = L.max()
    X = np.zeros((len(seqs), T, 20), np.float32)
    for k, s in enumerate(seqs):
        X[k, :len(s)] = s
    return X, L, np.array(idx)


class CNN(nn.Module):
    def __init__(self, c=20):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(128, 1)

    def forward(self, x, L):
        h = self.net(x.transpose(1, 2)).transpose(1, 2)               # n,T,64
        m = (torch.arange(x.shape[1])[None] < L[:, None]).unsqueeze(-1)
        mx = h.masked_fill(~m, -1e9).max(1).values
        mean = (h * m).sum(1) / L[:, None]
        return self.head(torch.cat([mx, mean], 1)).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, c=20):
        super().__init__()
        self.rnn = nn.GRU(c, 64, batch_first=True); self.head = nn.Linear(64, 1)

    def forward(self, x, L):
        p = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(p)
        return self.head(h[-1]).squeeze(-1)


class Transformer(nn.Module):
    def __init__(self, c=20, dm=64):
        super().__init__()
        self.inp = nn.Linear(c, dm); self.pos = nn.Parameter(torch.zeros(1, 80, dm))
        layer = nn.TransformerEncoderLayer(dm, 4, 128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2); self.head = nn.Linear(dm, 1)

    def forward(self, x, L):
        T = x.shape[1]
        pad = torch.arange(T)[None] >= L[:, None]
        h = self.enc(self.inp(x) + self.pos[:, :T], src_key_padding_mask=pad)
        m = (~pad).unsqueeze(-1)
        return self.head((h * m).sum(1) / L[:, None]).squeeze(-1)


MODELS = {"cnn": CNN, "gru": GRU, "transformer": Transformer}


def fit_predict(name, Xtr, Ltr, ytr, Xte, Lte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Xtr.reshape(-1, 20)[(np.arange(Xtr.shape[1])[None] < Ltr[:, None]).reshape(-1)].mean(0)
    sd = Xtr.reshape(-1, 20)[(np.arange(Xtr.shape[1])[None] < Ltr[:, None]).reshape(-1)].std(0) + 1e-6
    def prep(X, L):
        Z = (X - mu) / sd
        Z[np.arange(X.shape[1])[None] >= L[:, None]] = 0
        return torch.tensor(Z, dtype=torch.float32), torch.tensor(L)
    xt, lt = prep(Xtr.copy(), Ltr); xe, le = prep(Xte.copy(), Lte)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = MODELS[name]()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(yt)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for b in range(0, n, 128):
            ii = perm[b:b + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[ii], lt[ii]), yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe, le)).numpy()


def split_random(y, g, seed):
    yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def split_bus(y, g, seed):
    order = np.random.default_rng(seed).permutation(buses)
    fold_of = {b: i % 5 for i, b in enumerate(order)}
    fi = np.array([fold_of[b] for b in g])
    for k in range(5):
        yield np.where(fi != k)[0], np.where(fi == k)[0]


def auc_bacc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), balanced_accuracy_score(y, p > .5)


rng = np.random.default_rng(0)


def boot(y, p, g, mask, by_bus, B=500):
    yy, pp, gg = y[mask], p[mask], g[mask]
    pt = auc_bacc(yy, pp); out = []
    if by_bus:
        ub = np.unique(gg); idx_by = {b: np.where(gg == b)[0] for b in ub}
        for _ in range(B):
            ii = np.concatenate([idx_by[b] for b in rng.choice(ub, len(ub))])
            out.append(auc_bacc(yy[ii], pp[ii]))
    else:
        for _ in range(B):
            ii = rng.integers(0, len(yy), len(yy)); out.append(auc_bacc(yy[ii], pp[ii]))
    o = np.array(out)
    return pt, np.nanpercentile(o, 2.5, axis=0), np.nanpercentile(o, 97.5, axis=0)


rows, t0 = [], time.time()
for W in args.windows:
    X, L, idx = sequences(d, W); sub = d.loc[idx].reset_index(drop=True)
    y = sub.label_final.values.astype(int); g = sub.bus.values
    hard = sub.hard.values == 1; n1 = sub.night.values == 1
    print(f"W={W}: n={len(y)}, T={X.shape[1]}", flush=True)
    for name in args.models:
        for sp_name, sp, by_bus in (("random", split_random, False), ("held-out bus", split_bus, True)):
            P = np.zeros(len(y))
            for seed in range(args.repeats):
                cf = pathlib.Path(f"nn_cache/{name}_{int(W*1000)}_{sp_name.replace(' ', '')}_{seed}.npy")
                if cf.exists() and not args.quick:
                    P += np.load(cf) / args.repeats
                    continue
                p = np.full(len(y), np.nan)
                for k, (tr, te) in enumerate(sp(y, g, seed)):
                    ff = cf.with_name(cf.stem + f"_fold{k}.npy")
                    if ff.exists() and not args.quick:
                        p[te] = np.load(ff); continue
                    p[te] = fit_predict(name, X[tr], L[tr], y[tr], X[te], L[te], seed)
                    if not args.quick:
                        ff.parent.mkdir(exist_ok=True); np.save(ff, p[te])
                    if args.quick:
                        break
                if not args.quick:
                    cf.parent.mkdir(exist_ok=True); np.save(cf, p)
                P += p / args.repeats
                print(f"  {name} {sp_name} repeat {seed} done, {time.time() - t0:.0f}s", flush=True)
            ok = ~np.isnan(P)
            for sname, mask in (("night 1 only (uniform)", n1), ("hard (35-65%)", hard), ("all", np.ones(len(y), bool))):
                m = mask & ok
                if m.sum() < 30:
                    continue
                pt, lo, hi = boot(y, P, g, m, by_bus)
                rows.append(dict(model=name, window_ms=int(W * 1000), split=sp_name, subset=sname, n=int(m.sum()),
                                 auc=pt[0], auc_lo=lo[0], auc_hi=hi[0], bacc=pt[1], bacc_lo=lo[1], bacc_hi=hi[1]))
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(pd.DataFrame(rows).tail(3).round(3).to_string(index=False), flush=True)
print("done", time.time() - t0)
