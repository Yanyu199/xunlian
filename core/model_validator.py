from typing import Any, Dict, List

import numpy as np
import torch

from config import MODEL_SAVE_PATH, SCALER_SAVE_PATH
from core.predictor import (
    _model_times_from_scaler,
    _scaler_layer_num,
    _scaler_looks_log_space,
    deinterleave_params,
    inverse_outputs,
    load_model,
    load_scaler,
    scale_inputs,
)
from core.real_data import TEMDataset, quality_control, resample_log_time


def _add_check(checks: List[Dict[str, Any]], name: str, status: str, message: str, weight: int) -> None:
    checks.append({
        "name": name,
        "status": status,
        "message": message,
        "weight": weight,
    })


def _status_from_score(score: float, failed: bool, warning_count: int) -> str:
    if failed or score < 55:
        return "fail"
    if score < 80 or warning_count:
        return "warning"
    return "pass"


def _prediction_statistics(values: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(values)
    positive = values[finite & (values > 0)]
    stats: Dict[str, Any] = {
        "shape": [int(v) for v in values.shape],
        "finite_ratio": float(np.mean(finite)) if values.size else 0.0,
        "min": float(np.nanmin(values)) if values.size else None,
        "max": float(np.nanmax(values)) if values.size else None,
        "median": float(np.nanmedian(values)) if values.size else None,
    }
    if positive.size:
        stats["positive_min"] = float(np.min(positive))
        stats["positive_max"] = float(np.max(positive))
        stats["positive_dynamic_range"] = float(np.max(positive) / max(np.min(positive), 1e-30))
    else:
        stats["positive_min"] = None
        stats["positive_max"] = None
        stats["positive_dynamic_range"] = None
    return stats


def validate_model_with_dataset(
    dataset: TEMDataset,
    model_path: str = MODEL_SAVE_PATH,
    scaler_path: str = SCALER_SAVE_PATH,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    score = 100.0

    try:
        scaler = load_scaler(scaler_path)
        _add_check(checks, "scaler", "pass", "归一化文件可以读取。", 10)
    except Exception as exc:
        _add_check(checks, "scaler", "fail", f"归一化文件读取失败: {exc}", 40)
        return {
            "status": "fail",
            "score": 0,
            "checks": checks,
            "warnings": [str(exc)],
            "model": {"model_path": model_path, "scaler_path": scaler_path},
        }

    required_keys = {"x_min", "x_max", "y_min", "y_max"}
    missing_keys = sorted(required_keys - set(scaler))
    if missing_keys:
        score -= 35
        _add_check(checks, "scaler_keys", "fail", f"归一化文件缺少字段: {', '.join(missing_keys)}", 35)
        return {
            "status": "fail",
            "score": max(0, round(score, 1)),
            "checks": checks,
            "warnings": ["scaler 字段不完整，无法验证模型。"],
            "model": {"model_path": model_path, "scaler_path": scaler_path},
        }
    _add_check(checks, "scaler_keys", "pass", "归一化字段完整。", 10)

    layer_num = _scaler_layer_num(scaler)
    target_times = _model_times_from_scaler(scaler)

    try:
        model, metadata = load_model(model_path, layer_num=layer_num)
        _add_check(checks, "model_load", "pass", "模型文件可以加载，结构与当前代码匹配。", 20)
    except Exception as exc:
        _add_check(checks, "model_load", "fail", f"模型加载失败或结构不匹配: {exc}", 45)
        return {
            "status": "fail",
            "score": 0,
            "checks": checks,
            "warnings": [str(exc)],
            "model": {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "layer_num": layer_num,
                "time_channels": int(len(target_times)),
            },
        }

    warnings: List[str] = []
    if metadata.get("time_channels") and int(metadata["time_channels"]) != len(target_times):
        score -= 12
        warnings.append("模型元数据中的时间道数量与 scaler 不一致。")
        _add_check(checks, "metadata_time", "warning", warnings[-1], 12)
    else:
        _add_check(checks, "metadata_time", "pass", "模型时间道与 scaler 匹配。", 8)

    if metadata.get("layer_num") and int(metadata["layer_num"]) != layer_num:
        score -= 12
        warnings.append("模型元数据中的地层层数与 scaler 不一致。")
        _add_check(checks, "metadata_layers", "warning", warnings[-1], 12)
    else:
        _add_check(checks, "metadata_layers", "pass", "模型层数与 scaler 匹配。", 8)

    qc = quality_control(dataset)
    if qc["status"] == "warning":
        score -= 10
        warnings.append("上传数据存在质控告警，验证结果需要结合原始曲线判断。")
        _add_check(checks, "data_qc", "warning", "上传数据存在质控告警。", 10)
    else:
        _add_check(checks, "data_qc", "pass", "上传数据质控通过。", 10)

    try:
        responses = resample_log_time(dataset.times, dataset.responses, target_times)
        x_scaled = scale_inputs(responses, scaler)
    except Exception as exc:
        _add_check(checks, "preprocess", "fail", f"数据预处理失败: {exc}", 35)
        return {
            "status": "fail",
            "score": max(0, round(score - 35, 1)),
            "checks": checks,
            "warnings": warnings + [str(exc)],
            "qc": qc,
            "model": {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "layer_num": layer_num,
                "time_channels": int(len(target_times)),
                "metadata": metadata,
            },
        }

    scaled_min = float(np.nanmin(x_scaled))
    scaled_max = float(np.nanmax(x_scaled))
    outside_ratio = float(np.mean((x_scaled < -0.2) | (x_scaled > 1.2)))
    if outside_ratio > 0.2:
        score -= 18
        warnings.append("实测数据与训练归一化范围偏离较多，模型可能在外推。")
        _add_check(checks, "input_range", "warning", f"{outside_ratio:.1%} 的输入点超出训练范围。", 18)
    elif outside_ratio > 0.05:
        score -= 8
        warnings.append("少量实测数据超出训练归一化范围。")
        _add_check(checks, "input_range", "warning", f"{outside_ratio:.1%} 的输入点超出训练范围。", 8)
    else:
        _add_check(checks, "input_range", "pass", "实测数据基本落在训练归一化范围内。", 12)

    try:
        with torch.no_grad():
            raw_preds = model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
        physical = inverse_outputs(raw_preds, scaler)
        _add_check(checks, "inference", "pass", "模型可以完成一次反演推理。", 20)
    except Exception as exc:
        _add_check(checks, "inference", "fail", f"模型推理失败: {exc}", 45)
        return {
            "status": "fail",
            "score": 0,
            "checks": checks,
            "warnings": warnings + [str(exc)],
            "qc": qc,
            "model": {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "layer_num": layer_num,
                "time_channels": int(len(target_times)),
                "metadata": metadata,
            },
        }

    if not np.all(np.isfinite(physical)):
        score -= 30
        warnings.append("反演输出中存在 NaN 或 Inf。")
        _add_check(checks, "output_finite", "fail", "反演输出中存在 NaN 或 Inf。", 30)
    else:
        _add_check(checks, "output_finite", "pass", "反演输出全部为有限数值。", 12)

    non_positive_ratio = float(np.mean(physical <= 0))
    if non_positive_ratio > 0:
        score -= min(25, 100 * non_positive_ratio)
        warnings.append("反演输出中存在非正值，电阻率/厚度结果不合理。")
        _add_check(checks, "output_positive", "warning", f"{non_positive_ratio:.1%} 的输出为非正值。", 15)
    else:
        _add_check(checks, "output_positive", "pass", "反演输出均为正值。", 10)

    stats = _prediction_statistics(physical)
    dynamic_range = stats.get("positive_dynamic_range")
    if dynamic_range is not None and dynamic_range > 1e8:
        score -= 10
        warnings.append("反演输出动态范围过大，可能存在不稳定尖点。")
        _add_check(checks, "output_range", "warning", "反演输出动态范围过大。", 10)
    elif dynamic_range is not None and dynamic_range < 1.05 and physical.shape[0] > 1:
        score -= 8
        warnings.append("不同测点输出几乎没有变化，模型可能欠敏感或输入列选择不正确。")
        _add_check(checks, "output_range", "warning", "不同测点输出变化过小。", 8)
    else:
        _add_check(checks, "output_range", "pass", "反演输出范围未发现明显异常。", 8)

    layer_results = deinterleave_params(physical[: min(5, physical.shape[0])], layer_num=layer_num)
    for idx, row in enumerate(layer_results):
        row["point"] = dataset.point_names[idx]

    failed = any(item["status"] == "fail" for item in checks)
    warning_count = sum(1 for item in checks if item["status"] == "warning")
    final_score = max(0.0, min(100.0, score))

    return {
        "status": _status_from_score(final_score, failed, warning_count),
        "score": round(final_score, 1),
        "summary": {
            "input_scaled_min": scaled_min,
            "input_scaled_max": scaled_max,
            "input_outside_ratio": outside_ratio,
            "prediction_stats": stats,
            "scaler_space": "log10" if _scaler_looks_log_space(scaler) else "linear/raw",
        },
        "checks": checks,
        "warnings": warnings,
        "qc": qc,
        "model": {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "layer_num": layer_num,
            "time_channels": int(len(target_times)),
            "metadata": metadata,
        },
        "preview_results": layer_results,
    }
