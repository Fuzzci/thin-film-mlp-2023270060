import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os

from config import SEED, MLP_HIDDEN
from model import MLP

# ========== 固定随机种子 ==========
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== 检查是否有 GPU ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ========== 加载数据 ==========
X_train = np.load("results/X_train.npy").astype(np.float32)
Y_train = np.load("results/Y_train.npy").astype(np.float32)
X_val   = np.load("results/X_val.npy").astype(np.float32)
Y_val   = np.load("results/Y_val.npy").astype(np.float32)
X_test  = np.load("results/X_test.npy").astype(np.float32)
Y_test  = np.load("results/Y_test.npy").astype(np.float32)

# ========== 转成 PyTorch 张量 ==========
X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train)
X_val_t   = torch.tensor(X_val)
Y_val_t   = torch.tensor(Y_val)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, Y_val_t),
    batch_size=64, shuffle=False
)

# ========== 模型、损失、优化器 ==========
model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 500
train_losses = []
val_losses = []

# ========== 训练 ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    # 验证
    model.eval()
    total_val = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total_val += criterion(pred, yb).item() * xb.size(0)

    val_loss = total_val / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:4d} | Train MSE = {train_loss:.6f} | Val MSE = {val_loss:.6f}")

# ========== 测试 ==========
model.eval()
with torch.no_grad():
    pred_test = model(X_test_t.to(device))
    test_mse = criterion(pred_test, Y_test_t.to(device)).item()
    test_mae = torch.mean(torch.abs(pred_test - Y_test_t.to(device))).item()

print("\n===== 最终结果 =====")
print(f"测试集 MSE: {test_mse:.6f}")
print(f"测试集 MAE: {test_mae:.6f}")

# ========== 保存模型和 loss 曲线 ==========
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/mlp_full.pth")
np.save("results/train_losses.npy", np.array(train_losses))
np.save("results/val_losses.npy", np.array(val_losses))
np.save("results/pred_test.npy", pred_test.cpu().numpy())
np.save("results/Y_test.npy", Y_test)

print("模型和 loss 曲线已保存到 results/")