import numpy as np
import torch
import random

from config import (
    SEED, DESIGN_SEED, TARGET_WAVELENGTH,
    WAVELENGTHS, D_MIN, D_MAX
)
from model import MLP
from tmm import reflectance_spectrum

# ========== 固定随机种子 ==========
random.seed(DESIGN_SEED)
np.random.seed(DESIGN_SEED)
torch.manual_seed(DESIGN_SEED)

# ========== 找到 target 波长在数组中的索引 ==========
target_idx = WAVELENGTHS.index(TARGET_WAVELENGTH)
print(f"目标波长: {TARGET_WAVELENGTH} nm, 索引: {target_idx}")

# ========== 用 design_seed 生成 10000 组新候选 ==========
N_CANDIDATES = 10000
X_cand = np.random.uniform(D_MIN, D_MAX, size=(N_CANDIDATES, 4)).astype(np.float32)

# ========== 加载训练好的 MLP ==========
model = MLP()
model.load_state_dict(torch.load("results/mlp_full.pth", map_location="cpu"))
model.eval()

# ========== 用 MLP 预测所有候选的光谱 ==========
with torch.no_grad():
    X_cand_t = torch.tensor(X_cand)
    Y_pred_cand = model(X_cand_t).numpy()

# ========== 按 target 波长处的 MLP 预测反射率排序 ==========
mlp_target_values = Y_pred_cand[:, target_idx]
top10_idx = np.argsort(mlp_target_values)[::-1][:10]

print("\n===== MLP 预测 Top10 =====")
for rank, idx in enumerate(top10_idx, 1):
    d = X_cand[idx]
    print(f"Rank {rank:2d} | d = [{d[0]:6.2f}, {d[1]:6.2f}, {d[2]:6.2f}, {d[3]:6.2f}] "
          f"| MLP R@740nm = {mlp_target_values[idx]:.4f}")

# ========== 用 TMM 重新验证 Top10 ==========
print("\n===== TMM 验证 Top10 =====")
tmm_target_values = []
tmm_spectra = []

for idx in top10_idx:
    d = X_cand[idx]
    R = reflectance_spectrum(d)
    tmm_spectra.append(R)
    tmm_target_values.append(R[target_idx])
    print(f"d = [{d[0]:6.2f}, {d[1]:6.2f}, {d[2]:6.2f}, {d[3]:6.2f}] "
          f"| MLP = {mlp_target_values[idx]:.4f} | TMM = {R[target_idx]:.4f}")

# ========== 按 TMM 结果选 Top5 ==========
tmm_target_values = np.array(tmm_target_values)
top5_local = np.argsort(tmm_target_values)[::-1][:5]

print("\n===== 最终 Top5（按 TMM 排序） =====")
print(f"{'Rank':<5}{'d1':>8}{'d2':>8}{'d3':>8}{'d4':>8}{'MLP@740':>12}{'TMM@740':>12}")
for rank, local_i in enumerate(top5_local, 1):
    global_i = top10_idx[local_i]
    d = X_cand[global_i]
    print(f"{rank:<5}{d[0]:>8.2f}{d[1]:>8.2f}{d[2]:>8.2f}{d[3]:>8.2f}"
          f"{mlp_target_values[global_i]:>12.4f}{tmm_target_values[local_i]:>12.4f}")

# ========== 保存结果 ==========
np.save("results/screen_X_cand.npy", X_cand)
np.save("results/screen_Y_pred.npy", Y_pred_cand)
np.save("results/screen_top10_idx.npy", top10_idx)
np.save("results/screen_tmm_spectra.npy", np.array(tmm_spectra))
np.save("results/screen_mlp_target.npy", mlp_target_values)
np.save("results/screen_tmm_target.npy", tmm_target_values)

print("\n结果已保存到 results/")