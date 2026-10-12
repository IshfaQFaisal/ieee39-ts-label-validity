"""Build one feature table from night 1 (uniform) and night 2 (targeted near-boundary)."""
import json, pathlib, numpy as np, pandas as pd

M = np.array([87.4, 50.7, 60.4, 67.2, 56.2, 75.6, 54.1, 47.2, 116.2, 1199.])   # inertia of the 10 generators
ROOT = pathlib.Path(__file__).resolve().parent.parent
N1, N2 = str(ROOT / "data" / "night1"), str(ROOT / "data" / "night2")
BD = json.load(open(ROOT / "src" / "boundary.json"))

def p_boundary(bus, load, dur, xf):
    z = (BD["b0"] + BD["intercept"][BD["buses"].index(int(bus))] + BD["c_dur"] * dur
         + BD["c_load"] * (load - 1.0) + BD["c_logxf"] * (np.log10(xf) + 2.5))
    return 1 / (1 + np.exp(-z))

def load_results():
    a = pd.read_csv(f"{N1}/results.csv"); a["night"] = 1; a["trajdir"] = f"{N1}/traj"
    b = pd.read_csv(f"{N2}/results.csv"); b["night"] = 2; b["trajdir"] = f"{N2}/traj"
    d = pd.concat([a, b], ignore_index=True)
    d = d[d.label >= 0].copy()
    # final label: use the 0.5 ms rerun where we have it (all solver failures in both nights)
    d["label_final"] = d.label
    d["label_status"] = "1ms"
    for folder in (N1, N2):
        try:
            r = pd.read_csv(f"{folder}/rerun_failures.csv")
        except FileNotFoundError:
            continue
        m = dict(zip(r.sid, r["label_0.5ms"]))
        hit = d.sid.isin(m)
        d.loc[hit, "label_final"] = d.loc[hit, "sid"].map(m)
        d.loc[hit, "label_status"] = "rerun_0.5ms"
    # low-xf angle-separation recheck at 0.5 ms (night 2)
    try:
        r = pd.read_csv(f"{N2}/recheck_angle_lowxf.csv")
        m = dict(zip(r.sid, r["label_0.5ms"])); hit = d.sid.isin(m) & (d.label_status == "1ms")
        d.loc[hit, "label_final"] = d.loc[hit, "sid"].map(m); d.loc[hit, "label_status"] = "rerun_0.5ms"
    except FileNotFoundError:
        pass
    # any solver failure without a rerun stays flagged as uncertain
    d.loc[(d.cause == "solver_failure") & (d.label_status == "1ms"), "label_status"] = "uncertain"
    d["p_model"] = [p_boundary(r.bus, r.load_scale, r.duration, r.xf) for r in d.itertuples()]
    d["hard"] = ((d.p_model >= 0.35) & (d.p_model <= 0.65)).astype(int)
    return d.reset_index(drop=True)

def traj_features(path, t_clear, W):
    z = np.load(path); dl, om = z["delta"], z["omega"]
    i1 = int(round((t_clear + W - 0.9) / 0.01))
    if i1 >= len(dl) or np.isnan(dl[i1]).any():
        return None
    coi = (dl * M).sum(1, keepdims=True) / M.sum()
    a = (dl - coi)[10:i1 + 1]          # angles relative to inertia-weighted centre, from fault inception
    w = (om - 1.0)[10:i1 + 1]
    sp = dl[10:i1 + 1].max(1) - dl[10:i1 + 1].min(1)
    f = np.concatenate([a[-1], w[-1], a.max(0), a.min(0), w.max(0), w.min(0),
                        [sp[-1], sp.max(), sp[-1] - sp[max(-6, -len(sp))], np.abs(w).max()]])
    return f, sp[-1]

def build(d, W):
    F, S, idx = [], [], []
    for i, r in d.iterrows():
        if r.no_postfault_signal == 1:
            continue
        o = traj_features(f"{r.trajdir}/{r.sid}.npz", r.t_clear, W)
        if o is None:
            continue
        F.append(o[0]); S.append(o[1]); idx.append(i)
    return np.array(F), np.array(S), np.array(idx)
