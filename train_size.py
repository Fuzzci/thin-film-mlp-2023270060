import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os

from config import SEED
from model import MLP

# ========== 固定随机种子 ==========
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ========== 加载数据 ==========
X_train = np.load("results/X_train.npy").astype(np.float32)
Y_train = np.load("results/Y_train.npy").astype(np.float32)
X_val   = np.load("results/X_val.npy").astype(np.float32)
Y_val   = np.load("results/Y_val.npy").astype(np.float32)
X_test  = np.load("results/X_test.npy").astype(np.float32)
Y_test  = np.load("results/Y_test.npy").astype(np.float32)

X_val_t  = torch.tensor(X_val)
Y_val_t  = torch.tensor(Y_val)
X_test_t = torch.tensor(X_test)
Y_test_t = torch.tensor(Y_test)

# ========== 不同训练样本数 ==========
sizes = [500, 1000, 2000, 4000]
test_mse_list = []
test_mae_list = []

EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3

for size in sizes:
    print(f"\n===== 训练样本数: {size} =====")

    # 每次重新设置种子，保证对比公平
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_sub = torch.tensor(X_train[:size])
    Y_sub = torch.tensor(Y_train[:size])

    train_loader = DataLoader(
        TensorDataset(X_sub, Y_sub),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t.to(device))
                val_loss = criterion(val_pred, Y_val_t.to(device)).item()
            print(f"Epoch {epoch+1:4d} | Val MSE = {val_loss:.6f}")

    # 测试
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t.to(device))
        mse = criterion(pred_test, Y_test_t.to(device)).item()
        mae = torch.mean(torch.abs(pred_test - Y_test_t.to(device))).item()

    test_mse_list.append(mse)
    test_mae_list.append(mae)

    print(f"训练样本数 {size}: 测试 MSE = {mse:.6f}, MAE = {mae:.6f}")

# ========== 保存结果 ==========
np.save("results/train_size_mse.npy", np.array(test_mse_list))
np.save("results/train_size_mae.npy", np.array(test_mae_list))

print("\n===== 汇总 =====")
for s, mse, mae in zip(sizes, test_mse_list, test_mae_list):
    print(f"{s:5d} 样本 | MSE = {mse:.6f} | MAE = {mae:.6f}")

print("结果已保存到 results/")