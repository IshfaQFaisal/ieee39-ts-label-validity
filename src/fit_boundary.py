"""Fit the boundary model: a logistic regression of P(unstable) on the uniform night-1 scenarios.

logit = b0 + intercept[bus] + c_dur*duration + c_load*(load_scale-1) + c_logxf*(log10(xf)+2.5)

Continuous inputs are standardised before fitting (L2 regularisation, C=10), then the coefficients are
converted back to the original units. This approximately reproduces the released src/boundary.json
(which was fitted earlier; its exact fitting script was not kept). The released file is what was used
to sample night 2 and define the near-boundary subset, so this script writes to a separate file and
never overwrites it.

Usage (from the repo root):  python src/fit_boundary.py
"""
import json, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
released = json.load(open(ROOT / "src" / "boundary.json"))
buses = released["buses"]

d = pd.read_csv(ROOT / "data" / "night1" / "results.csv")
d = d[d.label >= 0]                                   # drop scenarios without a label

onehot = np.stack([(d.bus == k).astype(float) for k in buses], 1)
cont = np.column_stack([d.duration, d.load_scale - 1, np.log10(d.xf) + 2.5])
sc = StandardScaler().fit(cont)
X = np.column_stack([onehot, sc.transform(cont)])

m = LogisticRegression(C=10, max_iter=20000).fit(X, d.label)
coef = m.coef_[0]
w = coef[-3:] / sc.scale_                             # back to original units
b0 = float(m.intercept_[0] - (coef[-3:] * sc.mean_ / sc.scale_).sum())

out = dict(buses=buses, intercept=[float(c) for c in coef[:len(buses)]], b0=b0,
           c_dur=float(w[0]), c_load=float(w[1]), c_logxf=float(w[2]),
           note=f"refit on night-1 (n={len(d)}), standardised inputs, C=10")
json.dump(out, open(ROOT / "src" / "boundary_refit.json", "w"), indent=1)

print(f"n = {len(d)}")
print(f"{'':10s}{'refit':>9s}{'released':>10s}")
for k in ("b0", "c_dur", "c_load", "c_logxf"):
    print(f"{k:10s}{out[k]:9.2f}{released[k]:10.2f}")

def prob(r, b):
    z = b["b0"] + np.array(b["intercept"])[[buses.index(x) for x in r.bus]] + b["c_dur"] * r.duration \
        + b["c_load"] * (r.load_scale - 1) + b["c_logxf"] * (np.log10(r.xf) + 2.5)
    return 1 / (1 + np.exp(-z))
pr, pf = prob(d, released), prob(d, out)
print("corr of probabilities (released vs refit):", round(float(np.corrcoef(pr, pf)[0, 1]), 4))
print("same side of 0.5:", round(float(((pr > .5) == (pf > .5)).mean()), 4))
band = (pr >= .35) & (pr <= .65)
print("scenarios in the 35-65% band, released/refit:", int(band.sum()), int(((pf >= .35) & (pf <= .65)).sum()))
