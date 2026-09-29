import numpy as np
import matplotlib.pyplot as plt
import torch

from config import WAVELENGTHS, SEED
from model import MLP

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 加载测试集和模型 ==========
X_test = np.load("results/X_test.npy").astype(np.float32)
Y_test = np.load("results/Y_test.npy").astype(np.float32)
Y_pred = np.load("results/pred_test.npy").astype(np.float32)

# ========== 计算每个测试样本的 MSE ==========
mse_per_sample = np.mean((Y_test - Y_pred) ** 2, axis=1)

# ========== 找误差最大的 3 个样本 ==========
worst_idx = np.argsort(mse_per_sample)[::-1][:3]

print("===== 测试集中误差最大的 3 个样本 =====")
for i, idx in enumerate(worst_idx, 1):
    print(f"样本 {idx} | MSE = {mse_per_sample[idx]:.6f}")

# ========== 画 Figure 8 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, idx in enumerate(worst_idx):
    ax = axes[i]
    ax.plot(WAVELENGTHS, Y_test[idx], "b-", linewidth=2, label="TMM (ground truth)")
    ax.plot(WAVELENGTHS, Y_pred[idx], "r--", linewidth=2, label="MLP prediction")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Sample {idx}, MSE = {mse_per_sample[idx]:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig8_failure.png", dpi=200)
plt.close()
print("\n已保存 Figure 8: results/fig8_failure.png")

# ========== 打印这几个样本的膜厚 ==========
print("\n===== 对应膜厚 =====")
for idx in worst_idx:
    d = X_test[idx]
    print(f"样本 {idx}: d = [{d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}, {d[3]:.2f}] nm, "
          f"MSE = {mse_per_sample[idx]:.6f}")