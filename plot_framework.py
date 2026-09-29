import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16, 3.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 3.5)
ax.axis("off")

# 六个方框：位置、标题、内容
boxes = [
    (0.3,  "Design space",     "4 thicknesses\n40–180 nm"),
    (2.9,  "Physics engine",   "TMM forward model\nR(λ), 400–800 nm"),
    (5.5,  "Dataset",          "5000 spectra\n4000/500/500"),
    (8.1,  "MLP surrogate",    "d1...d4 → spectrum\n4-128-128-64-41"),
    (10.7, "Fast screening",   "10,000 candidates\ndesign_seed = seed + 1"),
    (13.3, "Physics check",    "Top candidates\nTMM verification"),
]

box_w = 2.3
box_h = 2.0
y0 = 0.8

for x, title, content in boxes:
    rect = FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="#2c3e50",
        facecolor="#eaf2fb"
    )
    ax.add_patch(rect)
    ax.text(x + box_w / 2, y0 + box_h - 0.35, title,
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + box_w / 2, y0 + box_h / 2 - 0.25, content,
            ha="center", va="center", fontsize=9)

# 箭头
for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + box_w
    x_end = boxes[i + 1][0]
    arrow = FancyArrowPatch(
        (x_start + 0.02, y0 + box_h / 2),
        (x_end - 0.02, y0 + box_h / 2),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color="#2c3e50"
    )
    ax.add_patch(arrow)

# 底部注释
ax.text(8.0, 0.2,
        "The MLP acts as a surrogate model; all final designs are re-evaluated by the physical TMM model.",
        ha="center", va="center", fontsize=9, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("results/fig1_framework.png", dpi=200, bbox_inches="tight")
plt.close()

print("已保存 Figure 1: results/fig1_framework.png")