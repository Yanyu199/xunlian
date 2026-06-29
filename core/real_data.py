import io
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class TEMDataset:
    times: np.ndarray
    responses: np.ndarray
    point_names: List[str]
    metadata: dict = field(default_factory=dict)


def _decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _split_line(line: str) -> List[str]:
    return [x for x in re.split(r"[\s,\t,;]+", line.strip()) if x]


def _read_numeric_rows(text: str):
    rows: List[List[float]] = []
    header: Optional[List[str]] = None
    for raw_line in io.StringIO(text):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _split_line(line)
        if len(parts) < 2:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            header = parts
    if not rows:
        raise ValueError("未读取到有效数值数据。")
    return np.asarray(rows, dtype=float), header


def _normalise_time_units(raw_times: np.ndarray):
    raw_times = raw_times.astype(float)
    max_time = float(np.nanmax(raw_times))
    if max_time > 10:
        return raw_times * 1e-6, "microsecond"
    if max_time > 0.1:
        return raw_times * 1e-3, "millisecond"
    return raw_times, "second"


def _format_numeric_id(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _unique_rows(arr: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    if not columns:
        return np.empty((0, 0))
    return np.unique(arr[:, columns], axis=0)


def _regular_group_count(arr: np.ndarray, group_columns: Sequence[int], time_col: int) -> bool:
    groups = _unique_rows(arr, group_columns)
    times = np.unique(arr[:, time_col])
    if len(groups) < 1 or len(times) < 5:
        return False
    if arr.shape[0] != len(groups) * len(times):
        return False
    counts = {}
    for row in arr:
        key = tuple(row[list(group_columns)])
        counts[key] = counts.get(key, 0) + 1
    return all(count == len(times) for count in counts.values())


def _looks_like_engineering_z_dat(arr: np.ndarray) -> bool:
    if arr.ndim != 2 or arr.shape[1] < 7:
        return False
    return _regular_group_count(arr, (0, 1, 2), 3)


def _parse_engineering_z_dat(arr: np.ndarray) -> TEMDataset:
    # Engineering Z.dat format observed from site exports:
    # point_id, line_id, channel_id, time(us), z_response, auxiliary, invalid_sentinel
    needed = arr[:, [0, 1, 2, 3, 4]]
    if not np.all(np.isfinite(needed)):
        raise ValueError("工程 Z.dat 的测点、时间或响应列中存在 NaN/Inf。")

    raw_times = np.unique(arr[:, 3])
    times, unit = _normalise_time_units(raw_times)
    order = np.argsort(times)
    times = times[order]
    raw_times = raw_times[order]

    group_cols = (0, 1, 2)
    groups = _unique_rows(arr, group_cols)
    groups = groups[np.lexsort((groups[:, 2], groups[:, 1], groups[:, 0]))]
    c2_constant = len(np.unique(arr[:, 1])) == 1
    c3_constant = len(np.unique(arr[:, 2])) == 1

    responses = []
    point_names = []
    point_keys = []
    for group in groups:
        mask = np.ones(arr.shape[0], dtype=bool)
        for col, value in zip(group_cols, group):
            mask &= arr[:, col] == value
        rows = arr[mask]
        lookup = {float(row[3]): float(row[4]) for row in rows}
        trace = [lookup.get(float(t), np.nan) for t in raw_times]
        responses.append(trace)

        if c2_constant and c3_constant:
            name = _format_numeric_id(group[0])
        else:
            name = "-".join(_format_numeric_id(v) for v in group)
        point_names.append(name)
        point_keys.append([float(v) for v in group])

    responses = np.asarray(responses, dtype=float)
    if not np.all(np.isfinite(responses)):
        raise ValueError("工程 Z.dat 中存在缺失时间道或非有限响应值。")

    sentinel_values = np.unique(arr[:, 6]) if arr.shape[1] >= 7 else np.asarray([])
    sentinel = float(sentinel_values[0]) if len(sentinel_values) == 1 else None
    return TEMDataset(
        times=times,
        responses=responses,
        point_names=point_names,
        metadata={
            "format": "engineering_z_dat",
            "format_label": "工程 Z.dat 长表",
            "point_id_columns": [1, 2, 3],
            "time_source_column": 4,
            "response_source_column": 5,
            "auxiliary_column": 6,
            "sentinel_column": 7,
            "sentinel_value": sentinel,
            "time_unit_inferred": unit,
            "raw_time_min": float(np.min(raw_times)),
            "raw_time_max": float(np.max(raw_times)),
            "point_keys": point_keys[:20],
            "raw_column_count": int(arr.shape[1]),
        },
    )


def _looks_like_long_export(arr: np.ndarray) -> bool:
    if arr.shape[1] < 5:
        return False
    return _regular_group_count(arr, (0,), 3)


def _parse_long_export(arr: np.ndarray) -> TEMDataset:
    # Observed Z.dat format:
    # point_id, const, const, time, response, auxiliary, invalid_sentinel
    needed = arr[:, [0, 3, 4]]
    if not np.all(np.isfinite(needed)):
        raise ValueError("长表数据的测点、时间或响应列中存在 NaN/Inf。")

    point_ids = np.unique(arr[:, 0])
    raw_times = np.unique(arr[:, 3])
    times, unit = _normalise_time_units(raw_times)
    order = np.argsort(times)
    times = times[order]
    raw_times = raw_times[order]

    responses = []
    point_names = []
    for point_id in point_ids:
        rows = arr[arr[:, 0] == point_id]
        lookup = {float(row[3]): float(row[4]) for row in rows}
        trace = [lookup.get(float(t), np.nan) for t in raw_times]
        responses.append(trace)
        point_names.append(str(int(point_id)) if float(point_id).is_integer() else str(point_id))

    responses = np.asarray(responses, dtype=float)
    if not np.all(np.isfinite(responses)):
        raise ValueError("长表数据中存在缺失时间道或非有限响应值。")

    return TEMDataset(
        times=times,
        responses=responses,
        point_names=point_names,
        metadata={
            "format": "long_z_export",
            "time_source_column": 4,
            "response_source_column": 5,
            "time_unit_inferred": unit,
            "raw_time_min": float(np.min(raw_times)),
            "raw_time_max": float(np.max(raw_times)),
            "ignored_columns": [2, 3, 6, 7] if arr.shape[1] >= 7 else [],
        },
    )


def _parse_wide_table(arr: np.ndarray, header: Optional[List[str]]) -> TEMDataset:
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("数据格式不正确：需要 time + 至少一个测点响应列。")
    if not np.all(np.isfinite(arr)):
        raise ValueError("数据中存在 NaN 或 Inf，请先检查原始文件。")

    times, unit = _normalise_time_units(arr[:, 0])
    responses = arr[:, 1:].T.astype(float)
    if np.any(times <= 0):
        raise ValueError("时间道必须全部为正数。")

    order = np.argsort(times)
    times = times[order]
    responses = responses[:, order]

    if header and len(header) == arr.shape[1]:
        point_names = header[1:]
    else:
        point_names = [f"P{i + 1}" for i in range(responses.shape[0])]

    return TEMDataset(
        times=times,
        responses=responses,
        point_names=point_names,
        metadata={"format": "wide_time_table", "time_unit_inferred": unit},
    )


def parse_real_tem_text(text: str) -> TEMDataset:
    arr, header = _read_numeric_rows(text)
    if arr.ndim != 2:
        raise ValueError("数据格式不正确：未形成二维数值表。")
    if _looks_like_engineering_z_dat(arr):
        return _parse_engineering_z_dat(arr)
    if _looks_like_long_export(arr):
        return _parse_long_export(arr)
    if np.any(~np.isfinite(arr)):
        raise ValueError("数据中存在 NaN 或 Inf，请先检查原始文件。")
    return _parse_wide_table(arr, header)


def parse_real_tem_bytes(content: bytes) -> TEMDataset:
    return parse_real_tem_text(_decode_bytes(content))


def suggest_training_params(dataset: TEMDataset) -> dict:
    point_count, time_count = dataset.responses.shape
    time_min = float(dataset.times[0])
    time_max = float(dataset.times[-1])
    inferred_layer_num = int(np.clip(round(point_count / 10), 2, 8))
    if dataset.metadata.get("format") == "engineering_z_dat":
        sample_size = int(max(300, min(5000, point_count * 20)))
    else:
        sample_size = int(max(50, min(5000, point_count * 10)))
    return {
        "layer_num": inferred_layer_num,
        "sample_size": sample_size,
        "time_channels": int(time_count),
        "time_min": time_min,
        "time_max": time_max,
        "batch_size": int(min(128, max(1, sample_size))),
        "forward_batch_size": int(min(50, max(1, sample_size))),
        "prior_sim_samples": int(min(100, max(20, point_count))),
        "use_prior": True,
    }


def quality_control(dataset: TEMDataset) -> dict:
    responses = dataset.responses
    times = dataset.times
    point_reports = []

    for idx, row in enumerate(responses):
        finite = np.isfinite(row)
        abs_row = np.abs(row)
        nonzero = abs_row[abs_row > 0]
        dynamic_range = float(np.max(nonzero) / np.min(nonzero)) if nonzero.size else 0.0
        log_resp = np.log10(abs_row + 1e-30)
        jumps = np.abs(np.diff(log_resp))
        max_jump = float(np.max(jumps)) if jumps.size else 0.0

        warnings = []
        if not np.all(finite):
            warnings.append("存在非有限值")
        zero_count = int(np.sum(row == 0))
        if zero_count:
            warnings.append(f"存在 {zero_count} 个零值")
        if max_jump > 1.2:
            warnings.append("相邻时间道跳变较大")
        if dynamic_range < 10:
            warnings.append("动态范围偏小，可能信噪比较低")

        point_reports.append({
            "name": dataset.point_names[idx],
            "negative_count": int(np.sum(row < 0)),
            "zero_count": zero_count,
            "max_log_jump": max_jump,
            "dynamic_range": dynamic_range,
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
        })

    global_warnings = []
    if not np.all(np.diff(times) > 0):
        global_warnings.append("时间道不是严格递增")
    if responses.shape[1] < 5:
        global_warnings.append("时间道数量过少，训练可靠性较低")
    if np.nanmax(np.abs(responses)) <= 0:
        global_warnings.append("响应值全为零或无效")

    abs_responses = np.abs(responses)
    preview_points = min(5, responses.shape[0])
    preview_times = min(12, responses.shape[1])
    preview = {
        "point_names": dataset.point_names[:preview_points],
        "times": times[:preview_times].astype(float).tolist(),
        "responses": responses[:preview_points, :preview_times].astype(float).tolist(),
    }

    return {
        "point_count": int(responses.shape[0]),
        "time_count": int(responses.shape[1]),
        "time_min": float(times[0]),
        "time_max": float(times[-1]),
        "response_min": float(np.nanmin(responses)),
        "response_max": float(np.nanmax(responses)),
        "abs_response_min": float(np.nanmin(abs_responses)),
        "abs_response_max": float(np.nanmax(abs_responses)),
        "metadata": dataset.metadata,
        "preview": preview,
        "suggested_params": suggest_training_params(dataset),
        "status": "warning" if global_warnings or any(p["status"] == "warning" for p in point_reports) else "ok",
        "warnings": global_warnings,
        "points": point_reports,
    }


def resample_log_time(times: np.ndarray, responses: np.ndarray, target_times: Iterable[float]) -> np.ndarray:
    target = np.asarray(list(target_times), dtype=float)
    if times.shape == target.shape and np.allclose(times, target):
        return responses.copy()

    log_t = np.log10(times)
    log_target = np.log10(target)
    out = []
    for row in responses:
        dominant_sign = -1.0 if np.sum(row < 0) > np.sum(row > 0) else 1.0
        log_y = np.log10(np.abs(row) + 1e-30)
        interp = np.interp(log_target, log_t, log_y, left=log_y[0], right=log_y[-1])
        out.append(dominant_sign * (10 ** interp))
    return np.asarray(out, dtype=float)
