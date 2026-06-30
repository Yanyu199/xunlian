import json
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import (
    CODEXDATA_DIR,
    MODEL_SAVE_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    SCALER_SAVE_PATH,
    TRAIN_HISTORY_PATH,
    TRAINING_JOBS_DIR,
)
from core.forward_tem import TEMForwardModeler, forward_backend_status
from core.net import TEM_Seq2Seq_Net
from core.real_data import parse_real_tem_bytes, quality_control, resample_log_time


@dataclass
class TrainingParams:
    layer_num: int = 5
    sample_size: int = 50
    time_channels: int = 30
    time_min: float = 1e-5
    time_max: float = 1e-2
    use_prior: bool = True
    r_min: float = 10.0
    r_max: float = 1000.0
    prior_r_min_low: float = 1.0
    prior_r_min_high: float = 50.0
    prior_r_max_low: float = 10.0
    prior_r_max_high: float = 300.0
    prior_init_points: int = 4
    prior_iter: int = 2
    prior_sim_samples: int = 20
    thickness_min: float = 10.0
    thickness_max: float = 100.0
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    valid_portion: float = 0.2
    device: str = "auto"
    use_amp: bool = True
    torch_threads: int = 0
    forward_batch_size: int = 50
    tx_size_key: str = "4"
    random_seed: int = RANDOM_SEED
    stall_seconds: int = 60


class TEMRelativeLoss(nn.Module):
    def forward(self, outputs, targets):
        eps = 1e-8
        return torch.mean(torch.mean(torch.abs(targets - outputs) / (torch.abs(targets) + eps), dim=1))


class TrainingJob:
    def __init__(self, job_id: str, filename: str, params: TrainingParams):
        self.job_id = job_id
        self.filename = filename
        self.params = params
        self.status = "queued"
        self.stage = "等待开始"
        self.stage_index = 0
        self.stage_count = 6
        self.stage_progress = 0.0
        self.total_progress = 0.0
        self.message = "任务已创建"
        self.logs: List[dict] = []
        self.warnings: List[str] = []
        self.error: Optional[str] = None
        self.result: Optional[dict] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.last_update = time.time()
        self.lock = threading.Lock()

    def log(self, message: str, level: str = "info", key: bool = False, stage: Optional[str] = None):
        with self.lock:
            self.logs.append({
                "time": time.strftime("%H:%M:%S"),
                "stage": stage or self.stage,
                "message": message,
                "level": level,
                "key": key,
            })
            self.logs = self.logs[-300:]
            self.last_update = time.time()

    def update(self, stage: str, stage_index: int, stage_progress: float, total_progress: float, message: str, level: str = "info", key: bool = False):
        with self.lock:
            self.status = "running"
            self.stage = stage
            self.stage_index = stage_index
            self.stage_progress = float(max(0.0, min(100.0, stage_progress)))
            self.total_progress = float(max(0.0, min(100.0, total_progress)))
            self.message = message
            self.last_update = time.time()
        self.log(message, level=level, key=key, stage=stage)

    def snapshot(self):
        with self.lock:
            now = time.time()
            started_at = self.started_at or self.created_at
            elapsed = now - started_at
            eta = None
            if self.status == "running" and self.total_progress > 0:
                eta = max(0.0, elapsed * (100.0 - self.total_progress) / self.total_progress)
            stalled = self.status == "running" and (now - self.last_update) > self.params.stall_seconds
            return {
                "job_id": self.job_id,
                "filename": self.filename,
                "status": self.status,
                "stage": self.stage,
                "stage_index": self.stage_index,
                "stage_count": self.stage_count,
                "stage_progress": self.stage_progress,
                "total_progress": self.total_progress,
                "message": self.message,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "stalled": stalled,
                "last_update_seconds_ago": now - self.last_update,
                "warnings": self.warnings,
                "error": self.error,
                "result": self.result,
                "logs": list(self.logs),
                "params": asdict(self.params),
            }


JOBS: Dict[str, TrainingJob] = {}
JOBS_LOCK = threading.Lock()


def gpu_status() -> dict:
    available = torch.cuda.is_available()
    info = {"cuda_available": available, "device_count": torch.cuda.device_count() if available else 0}
    if available:
        idx = torch.cuda.current_device()
        info.update({
            "current_device": idx,
            "device_name": torch.cuda.get_device_name(idx),
        })
    return info


def default_training_params() -> dict:
    data = asdict(TrainingParams())
    data["gpu"] = gpu_status()
    return data


def parse_training_params(raw: Optional[str]) -> TrainingParams:
    defaults = asdict(TrainingParams())
    if raw:
        data = json.loads(raw)
        defaults.update({k: v for k, v in data.items() if k in defaults})
    params = TrainingParams(**defaults)
    validate_params(params)
    return params


def validate_params(params: TrainingParams):
    if params.layer_num < 2:
        raise ValueError("layer_num 至少为 2。")
    if params.sample_size < 2:
        raise ValueError("sample_size 至少为 2。")
    if params.time_channels < 5:
        raise ValueError("time_channels 至少为 5。")
    if params.time_min <= 0 or params.time_max <= params.time_min:
        raise ValueError("time_min/time_max 设置不合理。")
    if params.r_min <= 0 or params.r_max <= params.r_min:
        raise ValueError("r_min/r_max 设置不合理。")
    if params.prior_r_min_low <= 0 or params.prior_r_min_high <= params.prior_r_min_low:
        raise ValueError("prior_r_min_low/prior_r_min_high 设置不合理。")
    if params.prior_r_max_low <= 0 or params.prior_r_max_high <= params.prior_r_max_low:
        raise ValueError("prior_r_max_low/prior_r_max_high 设置不合理。")
    if params.prior_init_points < 0 or params.prior_iter < 0:
        raise ValueError("prior_init_points/prior_iter 不能为负数。")
    if params.prior_sim_samples < 2:
        raise ValueError("prior_sim_samples 至少为 2。")
    if params.thickness_min <= 0 or params.thickness_max <= params.thickness_min:
        raise ValueError("thickness_min/thickness_max 设置不合理。")
    if params.epochs < 1:
        raise ValueError("epochs 至少为 1。")
    if params.batch_size < 1:
        raise ValueError("batch_size 至少为 1。")
    if params.forward_batch_size < 1:
        raise ValueError("forward_batch_size 至少为 1。")
    if not (0 <= params.valid_portion < 0.9):
        raise ValueError("valid_portion 必须在 [0, 0.9) 内。")
    if params.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device 必须是 auto、cpu 或 cuda。")


def _safe_filename(filename: str) -> str:
    base = os.path.basename((filename or "").replace("\\", "/")) or "uploaded_z_data.dat"
    base = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", base).strip("._")
    return base or "uploaded_z_data.dat"


def create_training_job(content: bytes, filename: str, params: TrainingParams) -> str:
    job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    safe_name = _safe_filename(filename)
    job = TrainingJob(job_id, safe_name, params)
    job_dir = os.path.join(TRAINING_JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, safe_name), "wb") as f:
        f.write(content)
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_training_job, args=(job, content, job_dir), daemon=True).start()
    return job_id


def get_training_job(job_id: str) -> Optional[dict]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return job.snapshot() if job else None


def _target_times(params: TrainingParams) -> np.ndarray:
    return np.logspace(np.log10(params.time_min), np.log10(params.time_max), params.time_channels)


def _extract_features(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    n_samples, n_times = data.shape
    n_windows = 5
    edges = np.linspace(0, n_times, n_windows + 1, dtype=int)
    feats = np.zeros((n_samples, n_windows + 3))
    for i in range(n_samples):
        trace = np.abs(data[i]) + 1e-30
        energies = []
        for w in range(n_windows):
            seg = trace[edges[w]:edges[w + 1]]
            energies.append(float(np.sqrt(np.sum(seg ** 2))))
        start_idx = max(1, int(0.02 * n_times))
        try:
            slope, _ = np.polyfit(np.log(times[start_idx:]), np.log(trace[start_idx:]), 1)
        except Exception:
            slope = 0.0
        feats[i] = energies + [float(slope), float(np.max(trace)), float(np.mean(trace[int(0.7 * n_times):]))]
    return feats


def _mmd_score(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.spatial.distance import pdist, squareform
    z = np.vstack((x, y))
    dists = pdist(z, "euclidean")
    sigma = max(float(np.median(dists)) if len(dists) else 1.0, 1e-6)
    gamma = 1.0 / (2 * sigma ** 2)
    xx = np.exp(-gamma * squareform(pdist(x, "sqeuclidean")))
    yy = np.exp(-gamma * squareform(pdist(y, "sqeuclidean")))
    xy = np.exp(-gamma * squareform(pdist(z, "sqeuclidean"))[:len(x), len(x):])
    n, m = len(x), len(y)
    return max(float((np.sum(xx) - np.trace(xx)) / max(n * (n - 1), 1) + (np.sum(yy) - np.trace(yy)) / max(m * (m - 1), 1) - 2 * np.sum(xy) / max(n * m, 1)), 0.0)


def _auc_score(x_sim: np.ndarray, x_real: np.ndarray, seed: int) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    x = np.vstack((x_sim, x_real))
    y = np.hstack((np.zeros(len(x_sim)), np.ones(len(x_real))))
    min_class = int(min(np.sum(y == 0), np.sum(y == 1)))
    folds = min(5, min_class)
    if folds < 2:
        return 0.5
    probs = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y):
        clf = RandomForestClassifier(n_estimators=50, random_state=seed)
        clf.fit(x[train_idx], y[train_idx])
        class_probs = clf.predict_proba(x[test_idx])
        if class_probs.shape[1] < 2:
            return 0.5
        probs[test_idx] = class_probs[:, 1]
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        return 0.5
    return auc if auc >= 0.5 else 1.0 - auc


def _estimate_prior(job: TrainingJob, real_data: np.ndarray, times: np.ndarray, modeler: TEMForwardModeler, rng: np.random.Generator) -> tuple:
    p = job.params
    total = max(1, p.prior_init_points + p.prior_iter)
    if not p.use_prior or (p.prior_init_points + p.prior_iter) <= 0:
        job.update("先验范围", 2, 100, 20, "关键环节：跳过先验搜索，直接使用前端设置的电阻率范围。", key=True)
        return p.r_min, p.r_max

    job.update("先验范围", 2, 1, 10.2, "关键环节：开始用真实 z 数据约束电阻率范围。", key=True)
    feat_real = _extract_features(real_data, times)
    all_params, all_scores = [], []

    def evaluate(r_min, r_max):
        rhos = 10 ** rng.uniform(np.log10(r_min), np.log10(r_max), (p.prior_sim_samples, p.layer_num))
        thicks = 10 ** rng.uniform(np.log10(p.thickness_min), np.log10(p.thickness_max), (p.prior_sim_samples, p.layer_num - 1))
        sim = np.abs(modeler.forward_batch(rhos, thicks, times))
        feat_sim = _extract_features(sim, times)
        stack = np.vstack((feat_real, feat_sim))
        mean, std = np.mean(stack, axis=0), np.std(stack, axis=0) + 1e-9
        x_real, x_sim = (feat_real - mean) / std, (feat_sim - mean) / std
        mmd, auc = _mmd_score(x_sim, x_real), _auc_score(x_sim, x_real, p.random_seed)
        return -(mmd + 0.2 * abs(auc - 0.5)), mmd, auc

    if p.prior_init_points <= 0:
        score, mmd, auc = evaluate(p.r_min, p.r_max)
        all_params.append([p.r_min, p.r_max])
        all_scores.append(score)
        job.update("先验范围", 2, 5, 11, f"使用前端初始范围作为先验起点：[{p.r_min:.2f}, {p.r_max:.2f}]，MMD={mmd:.4f}，AUC={auc:.4f}")

    for i in range(p.prior_init_points):
        r_min = rng.uniform(p.prior_r_min_low, p.prior_r_min_high)
        r_max = rng.uniform(max(r_min + 1, p.prior_r_max_low), p.prior_r_max_high)
        score, mmd, auc = evaluate(r_min, r_max)
        all_params.append([r_min, r_max])
        all_scores.append(score)
        done = i + 1
        progress = done / total * 100
        job.update("先验范围", 2, progress, 10 + progress * 0.2, f"随机搜索 {done}/{total}：范围 [{r_min:.2f}, {r_max:.2f}]，MMD={mmd:.4f}，AUC={auc:.4f}")

    for i in range(p.prior_iter):
        best = int(np.argmax(all_scores))
        best_min, best_max = all_params[best]
        r_min = float(np.clip(best_min + rng.normal(0, 5), p.prior_r_min_low, p.prior_r_min_high))
        r_max = float(np.clip(best_max + rng.normal(0, 20), r_min + 1, p.prior_r_max_high))
        score, mmd, auc = evaluate(r_min, r_max)
        all_params.append([r_min, r_max])
        all_scores.append(score)
        done = p.prior_init_points + i + 1
        progress = done / total * 100
        job.update("先验范围", 2, progress, 10 + progress * 0.2, f"局部细化 {done}/{total}：范围 [{r_min:.2f}, {r_max:.2f}]，MMD={mmd:.4f}，AUC={auc:.4f}")

    best = int(np.argmax(all_scores))
    final_min, final_max = all_params[best]
    return float(final_min), float(final_max)


def _generate_samples(job: TrainingJob, r_min: float, r_max: float, times: np.ndarray, modeler: TEMForwardModeler, rng: np.random.Generator):
    p = job.params
    job.update("正演样本", 3, 1, 30.3, "关键环节：开始生成训练样本参数。", key=True)
    rhos = 10 ** rng.uniform(np.log10(r_min), np.log10(r_max), (p.sample_size, p.layer_num))
    thicks = 10 ** rng.uniform(np.log10(p.thickness_min), np.log10(p.thickness_max), (p.sample_size, p.layer_num - 1))
    batch_size = min(max(1, p.forward_batch_size), p.sample_size)
    batches = []
    total_batches = int(np.ceil(p.sample_size / batch_size))
    for idx, start in enumerate(range(0, p.sample_size, batch_size)):
        end = min(start + batch_size, p.sample_size)
        batches.append(modeler.forward_batch(rhos[start:end], thicks[start:end], times))
        progress = (idx + 1) / total_batches * 100
        job.update("正演样本", 3, progress, 30 + progress * 0.3, f"正演批次 {idx + 1}/{total_batches}：已完成 {end}/{p.sample_size} 个样本")
    return np.concatenate(batches, axis=0), np.hstack((rhos, thicks))


def _interleave_targets(y_raw: np.ndarray, layer_num: int) -> np.ndarray:
    y = np.zeros_like(y_raw)
    for i in range(layer_num - 1):
        y[:, i * 2] = y_raw[:, i]
        y[:, i * 2 + 1] = y_raw[:, layer_num + i]
    y[:, -1] = y_raw[:, layer_num - 1]
    return y


def _resolve_device(job: TrainingJob) -> torch.device:
    requested = job.params.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        job.warnings.append("请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU，已切换到 CPU。")
        requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        job.update("模型训练", 5, 1, 60.4, f"关键环节：PyTorch 训练启用 GPU：{name}。", key=True)
    else:
        job.update("模型训练", 5, 1, 60.4, "关键环节：PyTorch 训练使用 CPU。若服务器有 NVIDIA GPU，请确认 CUDA 版 PyTorch 可用。", key=True, level="warning")
    return device


def _align_simulated_inputs_to_real(x_raw: np.ndarray, real_resampled: np.ndarray):
    job.update("璁粌鍑嗗", 4, 45, 57, "Key step: align simulated response to real Z log-amplitude domain, then fit MinMax scaler.", key=True)
    x_log, x_min, x_max, input_alignment = _align_simulated_inputs_to_real(x_raw, real_resampled)
    real_log = np.log10(np.abs(real_resampled) + 1e-12)

    sim_mean = np.mean(x_log, axis=0)
    sim_std = np.std(x_log, axis=0) + 1e-8
    real_mean = np.mean(real_log, axis=0)
    real_std_raw = np.std(real_log, axis=0)
    real_std = np.maximum(real_std_raw, 0.25 * sim_std) + 1e-8

    x_aligned = (x_log - sim_mean) / sim_std * real_std + real_mean
    combined = np.vstack((x_aligned, real_log))
    x_min = np.min(combined, axis=0)
    x_max = np.max(combined, axis=0)

    diagnostics = {
        "method": "per_time_log_standardize_simulated_to_real",
        "scaler_fit_source": "aligned_simulated_plus_real",
        "sim_log_mean": sim_mean.tolist(),
        "sim_log_std": sim_std.tolist(),
        "real_log_mean": real_mean.tolist(),
        "real_log_std": real_std.tolist(),
        "real_log_std_raw": real_std_raw.tolist(),
        "aligned_sim_log_min": np.min(x_aligned, axis=0).tolist(),
        "aligned_sim_log_max": np.max(x_aligned, axis=0).tolist(),
        "real_log_min": np.min(real_log, axis=0).tolist(),
        "real_log_max": np.max(real_log, axis=0).tolist(),
    }
    return x_aligned, x_min, x_max, diagnostics


def _train_model(job: TrainingJob, x_raw: np.ndarray, y_raw: np.ndarray, target_times: np.ndarray, real_resampled: np.ndarray):
    p = job.params
    if p.torch_threads and p.torch_threads > 0:
        torch.set_num_threads(p.torch_threads)
        job.log(f"已设置 PyTorch CPU 线程数：{p.torch_threads}", key=True)

    job.update("训练准备", 4, 10, 56, "关键环节：整理标签顺序 [电阻率1, 厚度1, ..., 半空间电阻率]。", key=True)
    y_interleaved = _interleave_targets(y_raw, p.layer_num)
    x_log = np.log10(np.abs(x_raw) + 1e-12)
    y_log = np.log10(np.maximum(y_interleaved, 1e-12))

    job.update("训练准备", 4, 45, 57, "关键环节：对输入响应和目标参数执行 log10 + MinMax 归一化。", key=True)
    y_max, y_min = np.max(y_log, axis=0), np.min(y_log, axis=0)
    x_scaled = (x_log - x_min) / (x_max - x_min + 1e-8)
    y_scaled = (y_log - y_min) / (y_max - y_min + 1e-8) + 10.0

    device = _resolve_device(job)
    dataset = TensorDataset(torch.tensor(x_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
    valid_len = max(1, int(len(dataset) * p.valid_portion)) if len(dataset) > 2 else 0
    train_len = len(dataset) - valid_len
    generator = torch.Generator().manual_seed(p.random_seed)
    if valid_len:
        train_set, valid_set = random_split(dataset, [train_len, valid_len], generator=generator)
    else:
        train_set, valid_set = dataset, None

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=min(p.batch_size, max(1, train_len)), shuffle=True, drop_last=False, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=min(p.batch_size, max(1, valid_len)), shuffle=False, pin_memory=pin_memory) if valid_set else None
    job.update("训练准备", 4, 100, 60, f"训练集 {train_len} 条，验证集 {valid_len} 条，batch={p.batch_size}。", key=True)

    model = TEM_Seq2Seq_Net(layer_num=p.layer_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate)
    criterion = TEMRelativeLoss()
    use_amp = p.use_amp and device.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        job.log("关键环节：已启用 CUDA 混合精度 AMP，加快训练并降低显存占用。", key=True)

    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(p.epochs):
        model.train()
        train_total = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(batch_x), batch_y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_total += float(loss.detach().cpu().item())
        train_loss = train_total / max(len(train_loader), 1)

        valid_loss = None
        if valid_loader is not None:
            model.eval()
            valid_total = 0.0
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        valid_total += float(criterion(model(batch_x), batch_y).detach().cpu().item())
            valid_loss = valid_total / max(len(valid_loader), 1)

        monitor = valid_loss if valid_loss is not None else train_loss
        improved = monitor < best_loss
        if improved:
            best_loss = monitor
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss, "best_loss": best_loss})
        progress = (epoch + 1) / p.epochs * 100
        msg = f"epoch {epoch + 1}/{p.epochs}：训练损失={train_loss:.5f}"
        if valid_loss is not None:
            msg += f"，验证损失={valid_loss:.5f}"
        if improved:
            msg += "，关键环节：刷新最佳模型"
        job.update("模型训练", 5, progress, 60 + progress * 0.35, msg, key=improved)

    scaler = {
        "x_max": x_max.tolist(),
        "x_min": x_min.tolist(),
        "y_max": y_max.tolist(),
        "y_min": y_min.tolist(),
        "space": "log10",
        "layer_num": p.layer_num,
        "time_channels": p.time_channels,
        "time_min": float(target_times[0]),
        "time_max": float(target_times[-1]),
        "target_times": target_times.astype(float).tolist(),
        "input_alignment": input_alignment,
    }
    return best_state, best_loss, scaler, history, str(device)


def _run_training_job(job: TrainingJob, content: bytes, job_dir: str):
    job.started_at = time.time()
    try:
        p = job.params
        rng = np.random.default_rng(p.random_seed)
        job.update("数据读取", 1, 5, 1, "关键环节：开始解析原始 z 数据文件。", key=True)
        dataset = parse_real_tem_bytes(content)
        qc = quality_control(dataset)
        job.update("数据读取", 1, 35, 4, f"识别格式：{qc.get('metadata', {}).get('format', 'unknown')}；测点={qc['point_count']}，原始时间道={qc['time_count']}。", key=True)
        target_times = _target_times(p)
        real_resampled = np.abs(resample_log_time(dataset.times, dataset.responses, target_times))
        job.update("数据读取", 1, 100, 10, f"关键环节：已重采样到 {p.time_channels} 个训练时间道，时间范围 {p.time_min:.3e}-{p.time_max:.3e}s。", key=True)

        forward_status = forward_backend_status()
        if forward_status["gpu_accelerated"]:
            backend_msg = f"初始化正演器。关键环节：正演样本生成启用 GPU/CuPy：{forward_status.get('device_name', 'CUDA')}。"
            backend_level = "info"
        else:
            backend_msg = f"初始化正演器。注意：正演样本生成当前使用 CPU/NumPy。原因：{forward_status.get('fallback_reason') or '未启用 CuPy'}"
            backend_level = "warning"
        job.update("先验范围", 2, 0, 10, backend_msg, key=True, level=backend_level)
        modeler = TEMForwardModeler(tx_size_key=p.tx_size_key, center_z=0.0)
        r_min, r_max = _estimate_prior(job, real_resampled, target_times, modeler, rng)
        job.update("先验范围", 2, 100, 30, f"关键环节：最终电阻率范围 [{r_min:.2f}, {r_max:.2f}] Ω·m。", key=True)

        x_data, y_data = _generate_samples(job, r_min, r_max, target_times, modeler, rng)
        np.save(os.path.join(job_dir, "X_train.npy"), x_data)
        np.save(os.path.join(job_dir, "Y_train.npy"), y_data)
        np.save(os.path.join(job_dir, "time_gates.npy"), target_times)
        job.update("正演样本", 3, 100, 55, f"关键环节：样本落盘完成，X{x_data.shape}, Y{y_data.shape}。", key=True)

        best_state, best_loss, scaler, history, used_device = _train_model(job, x_data, y_data, target_times, real_resampled)
        if best_state is None:
            raise RuntimeError("训练未产生有效模型。")

        job.update("保存结果", 6, 20, 96, "关键环节：正在保存任务模型，并同步为当前激活模型。", key=True)
        result_dir = os.path.join(OUTPUT_DIR, job.job_id)
        os.makedirs(result_dir, exist_ok=True)
        model_path = os.path.join(result_dir, "best_tem_model.pt")
        scaler_path = os.path.join(result_dir, "data_scaler.json")
        history_path = os.path.join(result_dir, "train_history.json")
        uploaded_data_path = os.path.join(result_dir, job.filename)
        summary_path = os.path.join(result_dir, "training_summary.json")
        checkpoint = {
            "model_state": best_state,
            "metadata": {
                "job_id": job.job_id,
                "layer_num": p.layer_num,
                "time_channels": p.time_channels,
                "time_min": float(target_times[0]),
                "time_max": float(target_times[-1]),
                "best_loss": float(best_loss),
                "target_space": "log10",
                "resistivity_range": [r_min, r_max],
                "used_device": used_device,
                "params": asdict(p),
            },
        }
        torch.save(checkpoint, model_path)
        torch.save(checkpoint, MODEL_SAVE_PATH)
        for path in (scaler_path, SCALER_SAVE_PATH):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(scaler, f, indent=2)
        for path in (history_path, TRAIN_HISTORY_PATH):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        with open(uploaded_data_path, "wb") as f:
            f.write(content)

        summary = {
            "job_id": job.job_id,
            "filename": job.filename,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "history_path": history_path,
            "uploaded_data_path": uploaded_data_path,
            "active_model_path": MODEL_SAVE_PATH,
            "active_scaler_path": SCALER_SAVE_PATH,
            "active_history_path": TRAIN_HISTORY_PATH,
            "best_loss": float(best_loss),
            "used_device": used_device,
            "params": asdict(p),
            "qc": qc,
            "resistivity_range": [r_min, r_max],
            "input_alignment": scaler.get("input_alignment"),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with job.lock:
            job.status = "completed"
            job.stage = "完成"
            job.stage_index = 6
            job.stage_progress = 100
            job.total_progress = 100
            job.message = "训练完成"
            job.finished_at = time.time()
            job.last_update = time.time()
            job.result = {
                "result_dir": result_dir,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "history_path": history_path,
                "summary_path": summary_path,
                "uploaded_data_path": uploaded_data_path,
                "active_model_path": MODEL_SAVE_PATH,
                "active_scaler_path": SCALER_SAVE_PATH,
                "active_history_path": TRAIN_HISTORY_PATH,
                "job_dir": job_dir,
                "best_loss": float(best_loss),
                "used_device": used_device,
                "qc": qc,
                "resistivity_range": [r_min, r_max],
                "codexdata_dir": CODEXDATA_DIR,
            }
        job.log(f"关键环节：模型已保存到 {model_path}，并激活到 {MODEL_SAVE_PATH}", key=True)
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
            job.message = "训练失败"
            job.finished_at = time.time()
            job.last_update = time.time()
        job.log(traceback.format_exc(), level="error", key=True)
