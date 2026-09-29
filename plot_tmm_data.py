import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(18, 6))
ax.set_xlim(0, 18)
ax.set_ylim(0, 6)
ax.axis("off")

# ============================================================
# 左边：四层膜结构 Air / H / L / H / L / Glass
# ============================================================
x0 = 0.4
y0 = 0.8
layer_h = 0.7
layer_w = 1.8

layers = [
    ("Air",   "#ffffff", "black"),
    ("H",     "#a6c8e0", "black"),
    ("L",     "#f4d9a0", "black"),
    ("H",     "#a6c8e0", "black"),
    ("L",     "#f4d9a0", "black"),
    ("Glass", "#c8c8c8", "black"),
]

for i, (name, color, edge) in enumerate(layers):
    y = y0 + (len(layers) - 1 - i) * layer_h
    rect = Rectangle((x0, y), layer_w, layer_h,
                     facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x0 + layer_w / 2, y + layer_h / 2, name,
            ha="center", va="center", fontsize=11, fontweight="bold")

ax.text(x0 + layer_w / 2, y0 - 0.35,
        "Air / H / L / H / L / Glass",
        ha="center", va="center", fontsize=10, style="italic")

# ============================================================
# 中间：Input vector → TMM → Output spectrum
# ============================================================
mid_x = 3.0
mid_y = 2.2
box_w = 2.0
box_h = 1.6

# Input vector
box1 = FancyBboxPatch((mid_x, mid_y), box_w, box_h,
                      boxstyle="round,pad=0.08",
                      linewidth=1.5, edgecolor="#2c3e50",
                      facecolor="#eaf2fb")
ax.add_patch(box1)
ax.text(mid_x + box_w / 2, mid_y + box_h - 0.35, "Input vector",
        ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(mid_x + box_w / 2, mid_y + box_h / 2 - 0.25,
        "d = [d1, d2, d3, d4]",
        ha="center", va="center", fontsize=9)

# TMM
box2 = FancyBboxPatch((mid_x + box_w + 0.5, mid_y), box_w, box_h,
                      boxstyle="round,pad=0.08",
                      linewidth=1.5, edgecolor="#2c3e50",
                      facecolor="#eaf2fb")
ax.add_patch(box2)
ax.text(mid_x + box_w + 0.5 + box_w / 2, mid_y + box_h - 0.35, "TMM",
        ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(mid_x + box_w + 0.5 + box_w / 2, mid_y + box_h / 2 - 0.25,
        "2×2 matrices",
        ha="center", va="center", fontsize=9)

# Output spectrum
box3 = FancyBboxPatch((mid_x + 2 * box_w + 1.0, mid_y), box_w, box_h,
                      boxstyle="round,pad=0.08",
                      linewidth=1.5, edgecolor="#2c3e50",
                      facecolor="#eaf2fb")
ax.add_patch(box3)
ax.text(mid_x + 2 * box_w + 1.0 + box_w / 2, mid_y + box_h - 0.35,
        "Output spectrum",
        ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(mid_x + 2 * box_w + 1.0 + box_w / 2, mid_y + box_h / 2 - 0.25,
        "R(λ), 41 points",
        ha="center", va="center", fontsize=9)

# 箭头
for i in range(2):
    x_start = mid_x + (i + 1) * box_w + i * 0.5
    x_end = x_start + 0.5
    arrow = FancyArrowPatch(
        (x_start + 0.02, mid_y + box_h / 2),
        (x_end - 0.02, mid_y + box_h / 2),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=1.5, color="#2c3e50"
    )
    ax.add_patch(arrow)

# ============================================================
# 右边：Dataset 说明框
# ============================================================
ds_x = 11.5
ds_y = 1.2
ds_w = 6.0
ds_h = 3.6

ds_box = FancyBboxPatch((ds_x, ds_y), ds_w, ds_h,
                        boxstyle="round,pad=0.1",
                        linewidth=1.5, edgecolor="#2c3e50",
                        facecolor="#f7f9fc")
ax.add_patch(ds_box)

ax.text(ds_x + ds_w / 2, ds_y + ds_h - 0.35, "Dataset",
        ha="center", va="center", fontsize=12, fontweight="bold")

ax.text(ds_x + 0.4, ds_y + ds_h - 1.0,
        "X: 5000 × 4 thickness matrix",
        ha="left", va="center", fontsize=10)
ax.text(ds_x + 0.4, ds_y + ds_h - 1.6,
        "Y: 5000 × 41 reflectance matrix",
        ha="left", va="center", fontsize=10)
ax.text(ds_x + 0.4, ds_y + ds_h - 2.2,
        "4000 train / 500 validation / 500 test",
        ha="left", va="center", fontsize=10)
ax.text(ds_x + 0.4, ds_y + ds_h - 2.8,
        "(fixed split for all experiments)",
        ha="left", va="center", fontsize=9, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("results/fig2_tmm_data.png", dpi=200, bbox_inches="tight")
plt.close()

print("已保存 Figure 2: results/fig2_tmm_data.png")