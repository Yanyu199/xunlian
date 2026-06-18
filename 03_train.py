# tem_model_factory/03_train.py
import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from config import *
from core.net import TEM_Seq2Seq_Net


# ==========================================
# 修复 3: 定制的相对百分比误差损失函数 (MAPE变体)
# ==========================================
class TEMRelativeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        """
        考虑到序列输出是交替的: [res1, thick1, res2, thick2, ..., res_n]
        我们统一计算相对百分比误差
        """
        eps = 1e-8
        # 计算绝对相对误差: |目标 - 预测| / 目标
        # 使用 torch.mean(..., dim=1) 对每一个样本求所有参数的平均相对误差
        rel_error = torch.mean(torch.abs(targets - outputs) / (targets + eps), dim=1)

        # 在整个 Batch 上求平均
        return torch.mean(rel_error)


# ==========================================
# 修复 5: 学习率调度策略 (对齐原作者的阶梯衰减)
# ==========================================
def lambda_lr(epoch):
    if epoch <= 70:
        return 1.0
    elif epoch <= 150:
        return 0.5
    elif epoch <= 180:
        return 0.3
    elif epoch <= 200:
        return 0.1
    else:
        return 0.05


def train_pipeline():
    print("=== 开始运行 03: 深度学习 Seq2Seq 反演训练 ===")

    # 1. 加载数据
    X_raw = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_raw = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    batch_count = X_raw.shape[0]

    # ==========================================
    # 修复 4: 标签 (Y) 交织处理与偏移对齐
    # ==========================================
    # 1) Y_raw 是 [res1..res5, thick1..thick4]
    # 我们需要将其交织(interleave)为 Seq2Seq 需要的 [res1, thick1, res2, thick2..., res5]
    Y_interleaved = np.zeros_like(Y_raw)
    for i in range(LAYER_NUM - 1):
        Y_interleaved[:, i * 2] = Y_raw[:, i]  # 填入电阻率
        Y_interleaved[:, i * 2 + 1] = Y_raw[:, LAYER_NUM + i]  # 填入厚度
    Y_interleaved[:, -1] = Y_raw[:, LAYER_NUM - 1]  # 填入最后一个半空间电阻率

    # 2) 对 X 和 Y 均进行对数变换 (物理数据跨度大必须对数化)
    X_log = np.log10(np.abs(X_raw) + 1e-12)
    Y_log = np.log10(Y_interleaved)

    # 3) MinMax 归一化提取参数
    x_max, x_min = np.max(X_log, axis=0), np.min(X_log, axis=0)
    y_max, y_min = np.max(Y_log, axis=0), np.min(Y_log, axis=0)

    X_scaled = (X_log - x_min) / (x_max - x_min + 1e-8)
    Y_scaled = (Y_log - y_min) / (y_max - y_min + 1e-8)

    # 4) 极度关键：给 Y 加上 10.0，与 net.py 中的 relu(x + 10.0) 激活层在数值空间对齐
    Y_scaled = Y_scaled + 10.0

    # 保存归一化字典供后续测试预测使用
    scaler_dict = {
        "x_max": x_max.tolist(), "x_min": x_min.tolist(),
        "y_max": y_max.tolist(), "y_min": y_min.tolist()
    }
    with open(SCALER_SAVE_PATH, "w") as f:
        json.dump(scaler_dict, f)
    print(f"✅ 归一化参数已保存至: {SCALER_SAVE_PATH}")

    # 2. 构造 Dataloader
    tensor_x = torch.Tensor(X_scaled).to(DEVICE)
    tensor_y = torch.Tensor(Y_scaled).to(DEVICE)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. 初始化模型 (修复 6：传入 layer_num 精确控制解码步长)
    model = TEM_Seq2Seq_Net(layer_num=LAYER_NUM).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = TEMRelativeLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    print(f"🚀 开始训练网络... 总 Epochs={EPOCHS}, Device={DEVICE}")
    best_loss = float('inf')

    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_x)

            # 计算定制的相对百分比误差
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪防爆 (Seq2Seq / LSTM 常用技巧)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        # 调度器在每个 Epoch 结束步进
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        avg_loss = total_loss / len(loader)

        # 仅定期打印日志，减少刷屏
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:03d}/{EPOCHS}], LR: {current_lr:.5f}, Loss(MAPE): {avg_loss:.5f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"🎉 训练完美结束，最优模型(Loss={best_loss:.5f})已保存至 {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_pipeline()