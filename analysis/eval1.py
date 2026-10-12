"""First full evaluation: random vs held-out-bus splits, all rows vs near-boundary (hard) subset.

Pool = night 1 (uniform sampling) + night 2 (targeted near the stability boundary).
Labels: 0.5 ms rerun label where one exists, otherwise the 1 ms label.
"hard" = scenarios the boundary model gave a 35-65% chance of being unstable.
"""
import sys, numpy as np, pandas as pd, warnings
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


def cv_pred(X, y, g, splitter, seed):
    p = np.full(len(y), np.nan)
    for tr, te in splitter(y, g, seed):
        m = HGB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def metrics(y, p, mask):
    return roc_auc_score(y[mask], p[mask]), balanced_accuracy_score(y[mask], p[mask] > .5)


rows = []


def run(name, X, sub):
    y = sub.label_final.values.astype(int)
    g = sub.bus.values
    hard = sub.hard.values == 1
    for sp_name, sp in (("random", split_random), ("held-out bus", split_bus)):
        res = {"all": [], "hard": []}
        for seed in range(REPEATS):
            p = cv_pred(X, y, g, sp, seed)
            res["all"].append(metrics(y, p, np.ones(len(y), bool)))
            res["hard"].append(metrics(y, p, hard))
        for subset in ("all", "hard"):
            a = np.array(res[subset])
            rows.append(dict(features=name, split=sp_name, subset=subset,
                             n=int(len(y) if subset == "all" else hard.sum()),
                             auc=a[:, 0].mean(), auc_sd=a[:, 0].std(),
                             bacc=a[:, 1].mean(), bacc_sd=a[:, 1].std()))


run("metadata + bus id", meta(d, True), d)
run("metadata, no bus id", meta(d, False), d)
for W in (0.0, 0.05, 0.1):
    F, S, idx = f.build(d, W)
    sub = d.loc[idx].reset_index(drop=True)
    run(f"trajectory, {int(W * 1000)} ms after clearing", F, sub)
    y = sub.label_final.values.astype(int)
    hard = sub.hard.values == 1
    for name, m in (("all", np.ones(len(y), bool)), ("hard", hard)):
        rows.append(dict(features=f"rule: angle spread at {int(W * 1000)} ms", split="none", subset=name,
                         n=int(m.sum()), auc=roc_auc_score(y[m], S[m]), auc_sd=0, bacc=np.nan, bacc_sd=0))

out = pd.DataFrame(rows)
out.to_csv(str(HERE.parent / "results") + "/eval1_results.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
