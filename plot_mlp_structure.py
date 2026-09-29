import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis("off")

# 五个方框：位置、标题、内容
boxes = [
    (0.3,  "Input",   "4 thicknesses"),
    (3.3,  "Dense",   "128 + ReLU"),
    (6.3,  "Dense",   "128 + ReLU"),
    (9.3,  "Dense",   "64 + ReLU"),
    (12.3, "Output",  "41 reflectance\nvalues"),
]

box_w = 2.4
box_h = 2.0
y0 = 1.5

for x, title, content in boxes:
    rect = FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="#2c3e50",
        facecolor="#eaf2fb"
    )
    ax.add_patch(rect)
    ax.text(x + box_w / 2, y0 + box_h - 0.4, title,
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + box_w / 2, y0 + box_h / 2 - 0.3, content,
            ha="center", va="center", fontsize=10)

# 箭头
for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + box_w
    x_end = boxes[i + 1][0]
    arrow = FancyArrowPatch(
        (x_start + 0.02, y0 + box_h / 2),
        (x_end - 0.02, y0 + box_h / 2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.5,
        color="#2c3e50"
    )
    ax.add_patch(arrow)

# 底部标注
ax.text(8.0, 0.5,
        "Input: [d1, d2, d3, d4]                                          "
        "Output: R(400…800 nm), 41 points",
        ha="center", va="center", fontsize=10, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("results/fig3_mlp_structure.png", dpi=200, bbox_inches="tight")
plt.close()

print("已保存 Figure 3: results/fig3_mlp_structure.png")