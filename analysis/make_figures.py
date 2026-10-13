"""Figures for the paper. Run from the repository root: python analysis/make_figures.py"""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import features as f

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300})

# ---------- Figure 1: accuracy vs evaluation design
g = pd.read_csv(ROOT / "results" / "eval2_results.csv")
n = pd.read_csv(ROOT / "results" / "nn_results.csv")
def pick_g(feat, split, subset):
    r = g[(g.features == feat) & (g.split == split) & (g.subset == subset)].iloc[0]; return r.bacc, r.bacc_lo, r.bacc_hi
def pick_n(model, split, subset):
    r = n[(n.model == model) & (n.window_ms == 100) & (n.split == split) & (n.subset == subset)].iloc[0]; return r.bacc, r.bacc_lo, r.bacc_hi
settings = [("Uniform data\nrandom split", "random", "night 1 only (uniform)"),
            ("Uniform data\nheld-out buses", "held-out bus", "night 1 only (uniform)"),
            ("Near-boundary\nrandom split", "random", "hard (35-65%)"),
            ("Near-boundary\nheld-out buses", "held-out bus", "hard (35-65%)")]
series = [("Scenario parameters only (no dynamics), gradient boosting", GRAY, "s", lambda sp, su: pick_g("metadata + bus id", sp, su)),
          ("Trajectory features, gradient boosting", BLUE, "o", lambda sp, su: pick_g("trajectory, 100 ms", sp, su)),
          ("Raw trajectories, 1D-CNN", ORANGE, "^", lambda sp, su: pick_n("cnn", sp, su)),
          ("Raw trajectories, GRU", AQUA, "D", lambda sp, su: pick_n("gru", sp, su))]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
off = np.linspace(-0.27, 0.27, len(series)); rows = []
for k, (lab, col, mk, fn) in enumerate(series):
    xs, ys, lo, hi = [], [], [], []
    for i, (_, sp, su) in enumerate(settings):
        b, l, h = fn(sp, su); xs.append(i + off[k]); ys.append(b); lo.append(b - l); hi.append(h - b)
        rows.append(dict(model=lab, setting=settings[i][0].replace("\n", " "), bacc=b, lo=l, hi=h))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ecolor=col, mec="white", mew=0.8, ms=7, capsize=2.5, lw=1.4, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in settings])
ax.set_ylim(0.4, 1.0); ax.set_ylabel("Balanced accuracy (95% CI)")
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":"); ax.text(3.45, 0.505, "chance", ha="right", va="bottom", color=MUTED, fontsize=8)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5, frameon=False)
ax.set_title("Reported accuracy depends on how the test set is built", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig1_accuracy_by_design.png"); fig.savefig(ROOT / "figures" / "fig1_accuracy_by_design.pdf")
pd.DataFrame(rows).to_csv(ROOT / "figures" / "fig1_data.csv", index=False)

# ---------- Figure 2: per-bus held-out results on near-boundary scenarios
pb = pd.read_csv(ROOT / "results" / "eval2_per_bus.csv"); pb = pb[pb.bacc_hard.notna()].sort_values("bacc_hard")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(range(len(pb)), pb.bacc_hard, color=BLUE, width=0.72)
ax.set_xticks(range(len(pb))); ax.set_xticklabels(pb.bus.astype(int), fontsize=7.5)
ax.set_xlabel("Fault bus (held out from training)"); ax.set_ylabel("Balanced accuracy,\nnear-boundary scenarios")
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":"); ax.set_ylim(0.4, 1.0); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for i, v in enumerate(pb.bacc_hard):
    if v < 0.65: ax.text(i, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center", fontsize=7, color=INK)
ax.set_title("Near-boundary accuracy at unseen buses varies from 0.5 to 1.0", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig2_per_bus.png"); fig.savefig(ROOT / "figures" / "fig2_per_bus.pdf")
pb.to_csv(ROOT / "figures" / "fig2_data.csv", index=False)

# ---------- Figure 3: solver-failure labels re-simulated at a finer step
d = f.load_results()
sf = d[(d.cause == "solver_failure") & (d.label_status == "rerun_0.5ms")].copy()
sf["still_unstable"] = (sf.label_final == 1)
t = sf.groupby("bus").agg(unstable=("still_unstable", "sum"), total=("still_unstable", "size")).reset_index()
t["stable"] = t.total - t.unstable; t = t.sort_values("total", ascending=False)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(t))
ax.bar(x, t.unstable, color=BLUE, width=0.7, label="Still unstable at 0.5 ms")
ax.bar(x, t.stable, bottom=t.unstable, color=ORANGE, width=0.7, label="Stable at 0.5 ms (label was wrong)", edgecolor="white", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(t.bus.astype(int)); ax.set_xlabel("Fault bus"); ax.set_ylabel("Scenarios labelled\nunstable by solver failure")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
tot, wrong = int(t.total.sum()), int(t.stable.sum())
ax.set_title(f"{wrong} of {tot} solver-failure labels ({100*wrong/tot:.0f}%) turn out stable at a finer time step", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig3_label_flips.png"); fig.savefig(ROOT / "figures" / "fig3_label_flips.pdf")
t.to_csv(ROOT / "figures" / "fig3_data.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False)); print(t.to_string(index=False)); print(tot, wrong)
