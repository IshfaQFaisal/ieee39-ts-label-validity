"""Second evaluation pass: confidence intervals, usual-benchmark baseline (night 1 only),
train-on-uniform / test-on-hard, and per-bus held-out results.

Cluster bootstrap: random-split metrics resample scenarios; held-out-bus metrics resample buses
(the bus is the unit of generalisation there).
"""
import sys, json, numpy as np, pandas as pd, warnings
import pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as f
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings("ignore")

d = f.load_results()
d = d[d.label_status != "uncertain"].reset_index(drop=True)
buses = sorted(d.bus.unique())
REPEATS = 5
rng = np.random.default_rng(0)


def meta(df, with_bus):
    X = np.column_stack([df.duration, df.load_scale, np.log10(df.xf), df.gen_bus])
    if with_bus:
        X = np.column_stack([X, np.stack([(df.bus == b).astype(float) for b in buses], 1)])
    return X


def split_random(y, g, seed):
    yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def split_bus(y, g, seed):
    order = np.random.default_rng(seed).permutation(buses)
    fold_of = {b: i % 5 for i, b in enumerate(order)}
    fi = np.array([fold_of[b] for b in g])
    for k in range(5):
        yield np.where(fi != k)[0], np.where(fi == k)[0]


def oof(X, y, g, splitter):
    P = np.zeros(len(y))
    for seed in range(REPEATS):
        p = np.full(len(y), np.nan)
        for tr, te in splitter(y, g, seed):
            m = HGB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        P += p / REPEATS
    return P


def auc_bacc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), balanced_accuracy_score(y, p > .5)


def boot(y, p, g, mask, by_bus, B=500):
    yy, pp, gg = y[mask], p[mask], g[mask]
    pt = auc_bacc(yy, pp)
    out = []
    if by_bus:
        ub = np.unique(gg); idx_by = {b: np.where(gg == b)[0] for b in ub}
        for _ in range(B):
            pick = rng.choice(ub, len(ub))
            ii = np.concatenate([idx_by[b] for b in pick])
            out.append(auc_bacc(yy[ii], pp[ii]))
    else:
        for _ in range(B):
            ii = rng.integers(0, len(yy), len(yy))
            out.append(auc_bacc(yy[ii], pp[ii]))
    o = np.array(out)
    lo = np.nanpercentile(o, 2.5, axis=0); hi = np.nanpercentile(o, 97.5, axis=0)
    return pt, lo, hi


rows, preds = [], {}
settings = [("metadata + bus id", lambda s: meta(s, True), None),
            ("metadata, no bus id", lambda s: meta(s, False), None),
            ("trajectory, 0 ms", None, 0.0),
            ("trajectory, 100 ms", None, 0.1)]
for name, mk, W in settings:
    if W is None:
        sub = d.reset_index(drop=True); X = mk(sub)
    else:
        F, S, idx = f.build(d, W); sub = d.loc[idx].reset_index(drop=True); X = F
    y = sub.label_final.values.astype(int); g = sub.bus.values
    hard = sub.hard.values == 1; n1 = sub.night.values == 1
    for sp_name, sp, by_bus in (("random", split_random, False), ("held-out bus", split_bus, True)):
        p = oof(X, y, g, sp)
        preds[(name, sp_name)] = (sub[["sid", "bus", "night", "hard", "label_final"]].copy().assign(p=p))
        for sname, mask in (("night 1 only (uniform)", n1), ("night 2 (targeted)", ~n1), ("hard (35-65%)", hard),
                            ("all", np.ones(len(y), bool))):
            pt, lo, hi = boot(y, p, g, mask, by_bus)
            rows.append(dict(features=name, split=sp_name, subset=sname, n=int(mask.sum()),
                             auc=pt[0], auc_lo=lo[0], auc_hi=hi[0], bacc=pt[1], bacc_lo=lo[1], bacc_hi=hi[1]))
out = pd.DataFrame(rows)
out.to_csv(str(HERE.parent / "results") + "/eval2_results.csv", index=False)

# ---- train on night 1 (uniform), test on night 2 hard subset
res = []
for name, mk, W in settings:
    if W is None:
        sub = d.reset_index(drop=True); X = mk(sub)
    else:
        F, S, idx = f.build(d, W); sub = d.loc[idx].reset_index(drop=True); X = F
    y = sub.label_final.values.astype(int); n1 = sub.night.values == 1
    hard = sub.hard.values == 1
    ps = []
    for seed in range(REPEATS):
        m = HGB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[n1], y[n1])
        ps.append(m.predict_proba(X[~n1])[:, 1])
    p = np.mean(ps, 0)
    ytest = y[~n1]; htest = hard[~n1]
    # within-night-1 random CV (the usual benchmark)
    X1, y1, g1 = X[n1], y[n1], sub.bus.values[n1]
    p1 = oof(X1, y1, g1, split_random)
    res.append(dict(features=name,
                    night1_random_cv_auc=roc_auc_score(y1, p1), night1_random_cv_bacc=balanced_accuracy_score(y1, p1 > .5),
                    train_n1_test_n2_all_auc=roc_auc_score(ytest, p), train_n1_test_n2_all_bacc=balanced_accuracy_score(ytest, p > .5),
                    train_n1_test_n2_hard_auc=roc_auc_score(ytest[htest], p[htest]),
                    train_n1_test_n2_hard_bacc=balanced_accuracy_score(ytest[htest], p[htest] > .5),
                    n_test_hard=int(htest.sum())))
pd.DataFrame(res).to_csv(str(HERE.parent / "results") + "/eval2_shift.csv", index=False)

# ---- per-bus held-out results for the trajectory model at clearing
pb = preds[("trajectory, 0 ms", "held-out bus")]
rows = []
for b, gdf in pb.groupby("bus"):
    h = gdf[gdf.hard == 1]
    rows.append(dict(bus=b, n=len(gdf), unstable=float(gdf.label_final.mean()),
                     auc=auc_bacc(gdf.label_final.values, gdf.p.values)[0],
                     bacc=auc_bacc(gdf.label_final.values, gdf.p.values)[1],
                     n_hard=len(h), bacc_hard=auc_bacc(h.label_final.values, h.p.values)[1] if len(h) > 20 else np.nan))
pd.DataFrame(rows).to_csv(str(HERE.parent / "results") + "/eval2_per_bus.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(out.round(3).to_string(index=False))
print(); print(pd.DataFrame(res).round(3).T.to_string())
print(); print(pd.DataFrame(rows).round(2).to_string(index=False))
