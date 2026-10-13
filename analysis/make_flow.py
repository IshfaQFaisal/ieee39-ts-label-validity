"""Pipeline (methodology) figure. Run from the repository root: python analysis/make_flow.py"""
import pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
INK, MUTED = "#0b0b0b", "#52514e"
NEUTRAL_F, NEUTRAL_E = "#f1f0ed", "#8a8984"
A_F, A_E = "#dbe8f8", "#2a78d6"      # label validity
B_F, B_E = "#fbe3d8", "#eb6834"      # evaluation validity
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
W, H = 21.5, 15.5

def box(x, y, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    ax.text(x + W / 2, y + H - 2.3, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(x + W / 2, y + H / 2 - 2.0, body, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.35)

def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.2, color=MUTED, connectionstyle=f"arc3,rad={rad}"))

xs = [1.5, 26.5, 51.5, 76.5]
y1, y2 = 36, 8
box(xs[0], y1, "1  Sample faults", "7,000 scenarios: bus, load\n0.8-1.3, duration, impedance\n3,000 uniform +\n4,000 near the boundary", NEUTRAL_F, NEUTRAL_E)
box(xs[1], y1, "2  Simulate", "ANDES time-domain,\nIEEE 39-bus, 10 generators\n1 ms step, 3 s window", NEUTRAL_F, NEUTRAL_E)
box(xs[2], y1, "3  Label", "Unstable if rotor-angle\nspread > 180 degrees\nor solver fails", NEUTRAL_F, NEUTRAL_E)
box(xs[3], y1, "4  Audit labels", "Re-run uncertain cases at\n0.5 and 0.25 ms; keep the\nfiner label (22 of 7,000\nchanged)", A_F, A_E)
box(xs[3], y2, "5  Model inputs", "Parameters only | trajectory\nfeatures | raw trajectories\n(decide at clearing, +100 ms)", NEUTRAL_F, NEUTRAL_E)
box(xs[2], y2, "6  Train models", "Gradient boosting,\n1D-CNN, GRU\n(fixed settings)", NEUTRAL_F, NEUTRAL_E)
box(xs[1], y2, "7  Split", "Random vs held-out-bus\nfive-fold splits,\nseveral repeats", B_F, B_E)
box(xs[0], y2, "8  Score", "Uniform vs near-boundary\nsubsets; balanced accuracy,\ncluster-bootstrap 95% CI", B_F, B_E)

for i in range(3):
    arrow((xs[i] + W + 0.5, y1 + H / 2), (xs[i + 1] - 0.5, y1 + H / 2))
arrow((xs[3] + W / 2, y1 - 0.5), (xs[3] + W / 2, y2 + H + 0.5))
for i in (3, 2, 1):
    arrow((xs[i] - 0.5, y2 + H / 2), (xs[i - 1] + W + 0.5, y2 + H / 2))

# feedback: boundary model from the first 3,000 guides sampling of the next 4,000
ax.add_patch(FancyArrowPatch((xs[2] + W / 2, y1 + H + 0.8), (xs[0] + W / 2, y1 + H + 0.8), arrowstyle="-|>", mutation_scale=9, lw=1.0,
                             color=NEUTRAL_E, ls=(0, (4, 3)), connectionstyle="arc3,rad=0.12"))
ax.text(xs[1] + W / 2 + 1, y1 + H + 7.6, "boundary model fitted on the uniform set guides the near-boundary samples", ha="center", fontsize=6.3, color=MUTED)

# legend
ax.add_patch(FancyBboxPatch((1.5, 1.2), 3, 2.2, boxstyle="round,pad=0.1,rounding_size=0.6", fc=A_F, ec=A_E, lw=1.2))
ax.text(5.6, 2.3, "Label validity", fontsize=7, va="center", color=INK)
ax.add_patch(FancyBboxPatch((22.5, 1.2), 3, 2.2, boxstyle="round,pad=0.1,rounding_size=0.6", fc=B_F, ec=B_E, lw=1.2))
ax.text(26.6, 2.3, "Evaluation validity", fontsize=7, va="center", color=INK)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig0_pipeline.png"); fig.savefig(ROOT / "figures" / "fig0_pipeline.pdf")
