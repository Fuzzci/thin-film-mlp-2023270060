import numpy as np
import matplotlib.pyplot as plt
import torch

from config import WAVELENGTHS, TARGET_WAVELENGTH, DESIGN_SEED
from model import MLP

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

target_idx = WAVELENGTHS.index(TARGET_WAVELENGTH)

# ========== 加载筛选结果 ==========
X_cand = np.load("results/screen_X_cand.npy")
Y_pred = np.load("results/screen_Y_pred.npy")
top10_idx = np.load("results/screen_top10_idx.npy")
tmm_spectra = np.load("results/screen_tmm_spectra.npy")
mlp_target = np.load("results/screen_mlp_target.npy")
tmm_target = np.load("results/screen_tmm_target.npy")

# 按 TMM 排序选 Top5
top5_local = np.argsort(tmm_target)[::-1][:5]

# ========== Figure 7: Top5 光谱对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 左图：Top1 的 MLP vs TMM 光谱
ax = axes[0]
global_i = top10_idx[top5_local[0]]
ax.plot(WAVELENGTHS, Y_pred[global_i], "r--", linewidth=2, label="MLP prediction")
ax.plot(WAVELENGTHS, tmm_spectra[top5_local[0]], "b-", linewidth=2, label="TMM verification")
ax.axvline(TARGET_WAVELENGTH, color="green", linestyle=":", linewidth=2,
           label=f"Target = {TARGET_WAVELENGTH} nm")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("Top-1 design: MLP vs TMM")
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：Top5 的 TMM 光谱
ax = axes[1]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
for rank, local_i in enumerate(top5_local, 1):
    ax.plot(WAVELENGTHS, tmm_spectra[local_i], color=colors[rank-1],
            linewidth=2, label=f"Rank {rank}")
ax.axvline(TARGET_WAVELENGTH, color="green", linestyle=":", linewidth=2,
           label=f"Target = {TARGET_WAVELENGTH} nm")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("Top-5 designs (TMM verified)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig7_design.png", dpi=200)
plt.close()
print("已保存 Figure 7: results/fig7_design.png")

# ========== 打印 Table 1 数据 ==========
print("\n===== Table 1. Top five designs =====")
print(f"{'Rank':<5}{'d1(nm)':>9}{'d2(nm)':>9}{'d3(nm)':>9}{'d4(nm)':>9}"
      f"{'MLP@740':>11}{'TMM@740':>11}")
for rank, local_i in enumerate(top5_local, 1):
    global_i = top10_idx[local_i]
    d = X_cand[global_i]
    print(f"{rank:<5}{d[0]:>9.2f}{d[1]:>9.2f}{d[2]:>9.2f}{d[3]:>9.2f}"
          f"{mlp_target[global_i]:>11.4f}{tmm_target[local_i]:>11.4f}")