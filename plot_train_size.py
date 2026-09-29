import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sizes = [500, 1000, 2000, 4000]
mse = np.load("results/train_size_mse.npy")
mae = np.load("results/train_size_mae.npy")

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(sizes, mse, "o-", color="tab:blue", label="MSE", linewidth=2, markersize=8)
ax1.set_xlabel("Training samples")
ax1.set_ylabel("Test MSE", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(sizes, mae, "s--", color="tab:red", label="MAE", linewidth=2, markersize=8)
ax2.set_ylabel("Test MAE", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("Effect of training-set size on test error")
fig.tight_layout()
plt.savefig("results/fig6_train_size.png", dpi=200)
plt.close()

print("已保存 Figure 6: results/fig6_train_size.png")
for s, m, a in zip(sizes, mse, mae):
    print(f"{s:5d} 样本 | MSE = {m:.6f} | MAE = {a:.6f}")