"""pipeline_diagram.py — L8/L9/L10 pipeline diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# ── colours ──────────────────────────────────────────────────────────────────
C_TRAIN   = "#4C72B0"
C_CORPUS  = "#DD8452"
C_REPLAY  = "#55A868"
C_HIT     = "#8172B2"
C_ARROW   = "#444444"
C_WARM    = "#C44E52"

def box(ax, x, y, w, h, color, label, sublabel="", fontsize=11):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="white",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.28,
                sublabel, ha="center", va="center",
                fontsize=9, color="white", alpha=0.88, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16),
                zorder=2)

def hit_label(ax, x, y, text, color=C_HIT):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color,
                      edgecolor="white", linewidth=1.2, zorder=5),
            zorder=5)

# ── column x-centres and row heights ─────────────────────────────────────────
#   columns: L7-seed | L8 | L9 | L10
BW, BH = 2.2, 0.85      # box width / height
COL  = [1.0, 3.8, 7.2, 10.6]   # left edge of each level column
ROW  = [5.2, 3.6, 2.0]          # Train / Corpus / Replay  top-y

LEVELS = [
    {"label": "L8",  "ops": "70 ops", "col": COL[1]},
    {"label": "L9",  "ops": "83 ops", "col": COL[2]},
    {"label": "L10", "ops": "87 ops", "col": COL[3]},
]

HITS = [
    {"text": "L7 hits\n66.3%",  "x": COL[0] + BW/2, "y": ROW[0] + BH/2},
    {"text": "L8 hits\n70.6%",  "x": COL[1] + BW/2 + 0.05, "y": 4.50},
    {"text": "L9 hits\n71.9%",  "x": COL[2] + BW/2 + 0.05, "y": 4.50},
]

REPLAY_THRESH = ["≥ 69%", "≥ 71%", "71.94%"]

# ── draw boxes ────────────────────────────────────────────────────────────────
for i, lv in enumerate(LEVELS):
    cx = lv["col"]
    # Train
    box(ax, cx, ROW[0], BW, BH, C_TRAIN,
        f"Train PPO {lv['label']}", lv["ops"])
    # Corpus
    box(ax, cx, ROW[1], BW, BH, C_CORPUS,
        "Build Corpus", "new-bin programs")
    # Replay
    box(ax, cx, ROW[2], BW, BH, C_REPLAY,
        "Replay Corpus", REPLAY_THRESH[i])

# ── vertical arrows inside each column (Train→Corpus→Replay) ─────────────────
for lv in LEVELS:
    cx = lv["col"] + BW/2
    arrow(ax, cx, ROW[0],       cx, ROW[1] + BH + 0.05)
    arrow(ax, cx, ROW[1],       cx, ROW[2] + BH + 0.05)

# ── L7 hits box ───────────────────────────────────────────────────────────────
hit_label(ax, COL[0] + BW/2, ROW[0] + BH/2, "L7 hits\n66.3%")

# ── warm-start arrows (hits → next Train) ─────────────────────────────────────
# L7 → L8 Train
arrow(ax,
      COL[0] + BW + 0.05,  ROW[0] + BH/2,
      COL[1] - 0.05,        ROW[0] + BH/2,
      color=C_WARM, lw=2.2)

# L8 hits bubble + arrow → L9 Train
hit_label(ax, COL[1] + BW/2, 4.62, "L8 hits\n70.6%")
arrow(ax,
      COL[1] + BW + 0.05,  ROW[0] + BH/2,
      COL[2] - 0.05,        ROW[0] + BH/2,
      color=C_WARM, lw=2.2)

# L9 hits bubble + arrow → L10 Train
hit_label(ax, COL[2] + BW/2, 4.62, "L9 hits\n71.9%")
arrow(ax,
      COL[2] + BW + 0.05,  ROW[0] + BH/2,
      COL[3] - 0.05,        ROW[0] + BH/2,
      color=C_WARM, lw=2.2)

# L10 final coverage bubble above Build Corpus
hit_label(ax, COL[3] + BW/2, 4.62, "L10 hits\n71.94%")

# ── row labels (left margin) ──────────────────────────────────────────────────
for row_y, label in zip(ROW, ["Train PPO", "Build Corpus", "Replay Corpus"]):
    ax.text(0.35, row_y + BH/2, label,
            ha="center", va="center", fontsize=8.5,
            color="#555555", rotation=90, style="italic")

# ── title + legend ────────────────────────────────────────────────────────────
ax.set_title("L8 → L9 → L10  ISA Completion Pipeline",
             fontsize=14, fontweight="bold", pad=10, color="#222222")

legend_patches = [
    mpatches.Patch(color=C_TRAIN,  label="Train PPO"),
    mpatches.Patch(color=C_CORPUS, label="Build Corpus"),
    mpatches.Patch(color=C_REPLAY, label="Replay / Regression"),
    mpatches.Patch(color=C_WARM,   label="Warm-start hits"),
]
ax.legend(handles=legend_patches, loc="lower center",
          ncol=4, fontsize=9, framealpha=0.9,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("pipeline_l8_l9_l10.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved → pipeline_l8_l9_l10.png")
plt.show()
