import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import *
from core.net import TEM_Seq2Seq_Net


class TEMRelativeLoss(nn.Module):
    def forward(self, outputs, targets):
        eps = 1e-8
        return torch.mean(torch.mean(torch.abs(targets - outputs) / (torch.abs(targets) + eps), dim=1))


def lambda_lr(epoch):
    if epoch <= 70:
        return 1.0
    if epoch <= 150:
        return 0.5
    if epoch <= 180:
        return 0.3
    if epoch <= 200:
        return 0.1
    return 0.05


def interleave_targets(y_raw):
    y_interleaved = np.zeros_like(y_raw)
    for i in range(LAYER_NUM - 1):
        y_interleaved[:, i * 2] = y_raw[:, i]
        y_interleaved[:, i * 2 + 1] = y_raw[:, LAYER_NUM + i]
    y_interleaved[:, -1] = y_raw[:, LAYER_NUM - 1]
    return y_interleaved


def train_pipeline():
    print("=== Step 03: Seq2Seq inversion training ===")
    x_path = os.path.join(DATA_DIR, "X_train.npy")
    y_path = os.path.join(DATA_DIR, "Y_train.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("未找到 X_train.npy 或 Y_train.npy，请先运行 02_data_generator.py")

    x_raw = np.load(x_path)
    y_raw = np.load(y_path)
    if len(x_raw) < 2:
        raise ValueError("训练样本数量至少需要 2 条。")
    time_path = os.path.join(DATA_DIR, "time_gates.npy")
    if os.path.exists(time_path):
        target_times = np.load(time_path)
    else:
        target_times = np.logspace(np.log10(TIME_MIN), np.log10(TIME_MAX), x_raw.shape[1])

    y_interleaved = interleave_targets(y_raw)
    x_log = np.log10(np.abs(x_raw) + 1e-12)
    y_log = np.log10(np.maximum(y_interleaved, 1e-12))

    x_max, x_min = np.max(x_log, axis=0), np.min(x_log, axis=0)
    y_max, y_min = np.max(y_log, axis=0), np.min(y_log, axis=0)
    x_scaled = (x_log - x_min) / (x_max - x_min + 1e-8)
    y_scaled = (y_log - y_min) / (y_max - y_min + 1e-8) + 10.0

    scaler = {
        "x_max": x_max.tolist(),
        "x_min": x_min.tolist(),
        "y_max": y_max.tolist(),
        "y_min": y_min.tolist(),
        "space": "log10",
        "layer_num": LAYER_NUM,
        "time_channels": int(x_raw.shape[1]),
        "time_min": float(target_times[0]),
        "time_max": float(target_times[-1]),
        "target_times": target_times.astype(float).tolist(),
    }
    with open(SCALER_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)

    tensor_x = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    tensor_y = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(tensor_x, tensor_y)

    valid_len = max(1, int(len(dataset) * VALID_PORTION))
    train_len = len(dataset) - valid_len
    if train_len < 1:
        train_len, valid_len = len(dataset), 0

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    if valid_len:
        train_set, valid_set = random_split(dataset, [train_len, valid_len], generator=generator)
    else:
        train_set, valid_set = dataset, None

    train_loader = DataLoader(train_set, batch_size=min(BATCH_SIZE, train_len), shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=min(BATCH_SIZE, max(1, valid_len)), shuffle=False) if valid_set else None

    model = TEM_Seq2Seq_Net(layer_num=LAYER_NUM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    criterion = TEMRelativeLoss()
    best_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(loss.item())

        train_loss = train_total / max(len(train_loader), 1)
        valid_loss = None
        if valid_loader is not None:
            model.eval()
            valid_total = 0.0
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    valid_total += float(criterion(model(batch_x), batch_y).item())
            valid_loss = valid_total / max(len(valid_loader), 1)

        scheduler.step()
        monitor = valid_loss if valid_loss is not None else train_loss
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})

        if monitor < best_loss:
            best_loss = monitor
            torch.save({
                "model_state": model.state_dict(),
                "metadata": {
                    "layer_num": LAYER_NUM,
                    "time_channels": int(x_raw.shape[1]),
                    "time_min": float(target_times[0]),
                    "time_max": float(target_times[-1]),
                    "best_loss": float(best_loss),
                    "target_space": "log10",
                },
            }, MODEL_SAVE_PATH)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            valid_text = f", valid={valid_loss:.5f}" if valid_loss is not None else ""
            print(f"epoch {epoch + 1:03d}/{EPOCHS}: train={train_loss:.5f}{valid_text}")

    with open(TRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. best_loss={best_loss:.5f}, model={MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_pipeline()
