import numpy as np
import random
import os

from config import (
    SEED, WAVELENGTHS,
    D_MIN, D_MAX,
    N_TOTAL, N_TRAIN, N_VAL, N_TEST
)
from tmm import reflectance_spectrum

# ========== 固定随机种子 ==========
random.seed(SEED)
np.random.seed(SEED)

# ========== 生成 5000 组随机膜厚 ==========
# 每层在 40 到 180 nm 之间均匀随机
X = np.random.uniform(D_MIN, D_MAX, size=(N_TOTAL, 4))

# ========== 计算每组膜厚的反射光谱 ==========
Y = np.zeros((N_TOTAL, len(WAVELENGTHS)))

for i in range(N_TOTAL):
    Y[i] = reflectance_spectrum(X[i])
    if (i + 1) % 500 == 0:
        print(f"已计算 {i + 1} / {N_TOTAL} 组")

print("数据生成完成！")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ========== 固定划分 4000 / 500 / 500 ==========
# 先做一次固定随机排列
perm = np.random.permutation(N_TOTAL)

X = X[perm]
Y = Y[perm]

X_train = X[:N_TRAIN]
Y_train = Y[:N_TRAIN]

X_val = X[N_TRAIN:N_TRAIN + N_VAL]
Y_val = Y[N_TRAIN:N_TRAIN + N_VAL]

X_test = X[N_TRAIN + N_VAL:]
Y_test = Y[N_TRAIN + N_VAL:]

print("训练集:", X_train.shape, Y_train.shape)
print("验证集:", X_val.shape, Y_val.shape)
print("测试集:", X_test.shape, Y_test.shape)

# ========== 保存到 results 文件夹 ==========
os.makedirs("results", exist_ok=True)

np.save("results/X_train.npy", X_train)
np.save("results/Y_train.npy", Y_train)
np.save("results/X_val.npy", X_val)
np.save("results/Y_val.npy", Y_val)
np.save("results/X_test.npy", X_test)
np.save("results/Y_test.npy", Y_test)

print("数据已保存到 results/ 文件夹")