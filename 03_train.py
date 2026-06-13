import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from config import *
from core.net import TEM_Seq2Seq_Net  # 导入你原本写的网络


def train_pipeline():
    # 1. 读取数据
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    # 2. 数据归一化 (极度重要：保存参数供后端使用)
    x_max, x_min = np.max(X, axis=0), np.min(X, axis=0)
    X_scaled = (X - x_min) / (x_max - x_min + 1e-8)

    scaler_dict = {"x_max": x_max.tolist(), "x_min": x_min.tolist()}
    with open(SCALER_SAVE_PATH, "w") as f:
        json.dump(scaler_dict, f)
    print("归一化参数已保存至:", SCALER_SAVE_PATH)

    # 3. 构造 Dataloader
    tensor_x = torch.Tensor(X_scaled).to(DEVICE)
    tensor_y = torch.Tensor(Y).to(DEVICE)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 初始化模型
    model = TEM_Seq2Seq_Net(input_dim=TIME_CHANNELS, output_dim=LAYER_NUM * 2 - 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 5. 训练循环
    best_loss = float('inf')
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"训练结束，最优模型已保存至 {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_pipeline()