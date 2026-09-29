import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from config import WAVELENGTHS, SEED, MLP_HIDDEN
from model import MLP

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("results", exist_ok=True)

# ========== 1. Loss 曲线 ==========
train_losses = np.load("results/train_losses.npy")
val_losses = np.load("results/val_losses.npy")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.yscale("log")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig4_loss.png", dpi=200)
plt.close()
print("已保存 Figure 4: results/fig4_loss.png")

# ========== 2. 测试样本光谱对比 ==========
X_test = np.load("results/X_test.npy").astype(np.float32)
Y_test = np.load("results/Y_test.npy").astype(np.float32)

# 加载模型
model = MLP()
model.load_state_dict(torch.load("results/mlp_full.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    X_test_t = torch.tensor(X_test)
    Y_pred = model(X_test_t).numpy()

# 随机选 3 个测试样本，固定随机种子
np.random.seed(SEED)
indices = np.random.choice(len(X_test), size=3, replace=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, idx in enumerate(indices):
    ax = axes[i]
    ax.plot(WAVELENGTHS, Y_test[idx], "b-", label="TMM", linewidth=2)
    ax.plot(WAVELENGTHS, Y_pred[idx], "r--", label="MLP", linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Test Sample {idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig5_spectra.png", dpi=200)
plt.close()
print("已保存 Figure 5: results/fig5_spectra.png")

# ========== 3. 打印单个样本的误差 ==========
for idx in indices:
    mse = np.mean((Y_test[idx] - Y_pred[idx]) ** 2)
    print(f"样本 {idx}: MSE = {mse:.6f}")