import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from config import LAYER_NUM, MODEL_SAVE_PATH, SCALER_SAVE_PATH, TIME_CHANNELS, TIME_MAX, TIME_MIN
from core.net import TEM_Seq2Seq_Net
from core.real_data import TEMDataset, quality_control, resample_log_time


def default_model_times() -> np.ndarray:
    return np.logspace(np.log10(TIME_MIN), np.log10(TIME_MAX), TIME_CHANNELS)


def load_scaler(path: str = SCALER_SAVE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到归一化参数文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    scaler: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            scaler[key] = np.asarray(value, dtype=float)
        else:
            scaler[key] = value
    return scaler


def _load_state_dict(path: str) -> Tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint.get("metadata", {})
    return checkpoint, {}


def load_model(path: str = MODEL_SAVE_PATH, layer_num: int = LAYER_NUM) -> Tuple[TEM_Seq2Seq_Net, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型文件: {path}")
    state_dict, metadata = _load_state_dict(path)
    model_layer_num = int(layer_num or metadata.get("layer_num") or LAYER_NUM)
    model = TEM_Seq2Seq_Net(layer_num=model_layer_num)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def _scaler_layer_num(scaler: Dict[str, Any]) -> int:
    return int(scaler.get("layer_num") or LAYER_NUM)


def _model_times_from_scaler(scaler: Dict[str, Any]) -> np.ndarray:
    if "target_times" in scaler:
        return np.asarray(scaler["target_times"], dtype=float)
    if {"time_min", "time_max", "time_channels"}.issubset(scaler):
        return np.logspace(
            np.log10(float(scaler["time_min"])),
            np.log10(float(scaler["time_max"])),
            int(scaler["time_channels"]),
        )
    return default_model_times()


def _scaler_looks_log_space(scaler: Dict[str, Any]) -> bool:
    if scaler.get("space") == "log10":
        return True
    return bool(np.nanmedian(scaler["x_max"]) < 0 and np.nanmedian(scaler["x_min"]) < 0)


def scale_inputs(responses: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    if responses.shape[1] != len(scaler["x_min"]):
        raise ValueError(
            f"模型需要 {len(scaler['x_min'])} 个时间道，但当前重采样后为 {responses.shape[1]} 个。"
        )
    if _scaler_looks_log_space(scaler):
        x_values = np.log10(np.abs(responses) + 1e-12)
    else:
        x_values = np.abs(responses)
    return (x_values - scaler["x_min"]) / (scaler["x_max"] - scaler["x_min"] + 1e-8)


def inverse_outputs(outputs: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    y_scaled = outputs - 10.0
    y_values = y_scaled * (scaler["y_max"] - scaler["y_min"] + 1e-8) + scaler["y_min"]
    if np.nanmedian(scaler["y_max"]) > 1000:
        return np.maximum(y_values, 0.0)
    return 10 ** y_values


def deinterleave_params(values: np.ndarray, layer_num: int = LAYER_NUM) -> List[dict]:
    rows = []
    for row in values:
        layers = []
        for i in range(layer_num):
            rho = float(row[i * 2]) if i < layer_num - 1 else float(row[-1])
            thickness = float(row[i * 2 + 1]) if i < layer_num - 1 else None
            layers.append({"layer": i + 1, "resistivity": rho, "thickness": thickness})
        rows.append({"layers": layers})
    return rows


def predict_dataset(dataset: TEMDataset, model_path: str = MODEL_SAVE_PATH, scaler_path: str = SCALER_SAVE_PATH) -> dict:
    scaler = load_scaler(scaler_path)
    target_times = _model_times_from_scaler(scaler)
    responses = resample_log_time(dataset.times, dataset.responses, target_times)
    x_scaled = scale_inputs(responses, scaler)

    layer_num = _scaler_layer_num(scaler)
    model, metadata = load_model(model_path, layer_num=layer_num)
    with torch.no_grad():
        preds = model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
    physical = inverse_outputs(preds, scaler)

    warnings = []
    if not _scaler_looks_log_space(scaler):
        warnings.append("当前 scaler 似乎来自旧版原始空间归一化，建议重新训练以获得更可靠结果。")
    if metadata.get("time_channels") and int(metadata["time_channels"]) != len(target_times):
        warnings.append("模型元数据中的时间道数与 scaler 不一致，请确认模型和归一化文件来自同一次训练。")
    if metadata.get("layer_num") and int(metadata["layer_num"]) != layer_num:
        warnings.append("模型元数据中的层数与 scaler 不一致，请确认模型和归一化文件来自同一次训练。")

    qc = quality_control(dataset)
    result_rows = deinterleave_params(physical, layer_num=layer_num)
    for idx, row in enumerate(result_rows):
        row["point"] = dataset.point_names[idx]
        row["qc_status"] = qc["points"][idx]["status"]

    return {
        "qc": qc,
        "warnings": warnings,
        "model": {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "layer_num": layer_num,
            "time_channels": int(len(target_times)),
            "metadata": metadata,
        },
        "target_times": target_times.tolist(),
        "results": result_rows,
    }
