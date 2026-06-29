import json
import os

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from config import DATA_DIR, LAYER_NUM, OUTPUT_DIR, RANDOM_SEED, THICKNESS_MAX, THICKNESS_MIN
from core.forward_tem import TEMForwardModeler
from core.real_data import parse_real_tem_bytes, quality_control


def parse_dbzdt_txt(file_path):
    with open(file_path, "rb") as f:
        dataset = parse_real_tem_bytes(f.read())
    return dataset.times, np.abs(dataset.responses), quality_control(dataset)


def extract_features(data, t_vec):
    n_samples, n_times = data.shape
    n_windows = 5
    edges = np.linspace(0, n_times, n_windows + 1, dtype=int)
    feats = np.zeros((n_samples, n_windows + 3))

    for i in range(n_samples):
        trace = np.abs(data[i, :]) + 1e-30
        window_energy = []
        for w in range(n_windows):
            start, end = edges[w], edges[w + 1]
            segment = trace[start:end] if end > start else trace[start:start + 1]
            window_energy.append(np.sqrt(np.sum(segment ** 2)))

        start_idx = max(1, int(0.02 * n_times))
        try:
            slope, _ = np.polyfit(np.log(t_vec[start_idx:]), np.log(trace[start_idx:]), 1)
        except Exception:
            slope = 0.0

        feats[i, :] = window_energy + [slope, float(np.max(trace)), float(np.mean(trace[int(0.7 * n_times):]))]
    return feats


def compute_mmd(x, y):
    z = np.vstack((x, y))
    dists = pdist(z, "euclidean")
    sigma = max(float(np.median(dists)) if len(dists) else 1.0, 1e-6)
    gamma = 1.0 / (2 * sigma ** 2)

    xx = np.exp(-gamma * squareform(pdist(x, "sqeuclidean")))
    yy = np.exp(-gamma * squareform(pdist(y, "sqeuclidean")))
    xy = np.exp(-gamma * squareform(pdist(z, "sqeuclidean"))[:len(x), len(x):])

    n, m = len(x), len(y)
    sum_xx = (np.sum(xx) - np.trace(xx)) / max(n * (n - 1), 1)
    sum_yy = (np.sum(yy) - np.trace(yy)) / max(m * (m - 1), 1)
    sum_xy = np.sum(xy) / max(n * m, 1)
    return max(float(sum_xx + sum_yy - 2 * sum_xy), 0.0)


def classifier_auc(x_sim, x_real):
    x = np.vstack((x_sim, x_real))
    y = np.hstack((np.zeros(len(x_sim)), np.ones(len(x_real))))
    folds = min(5, len(y))
    if folds < 2 or len(np.unique(y)) < 2:
        return 0.5

    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    probs = np.zeros(len(y))
    for train_idx, test_idx in kf.split(x):
        clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
        clf.fit(x[train_idx], y[train_idx])
        probs[test_idx] = clf.predict_proba(x[test_idx])[:, 1]

    auc = roc_auc_score(y, probs)
    return float(auc if auc >= 0.5 else 1.0 - auc)


def target_func(r_min, r_max, times, feat_real, modeler, rng, n_sim=100):
    if r_max <= r_min:
        return -1e6, float("inf"), 1.0

    rhos = 10 ** rng.uniform(np.log10(r_min), np.log10(r_max), (n_sim, LAYER_NUM))
    thicknesses = 10 ** rng.uniform(np.log10(THICKNESS_MIN), np.log10(THICKNESS_MAX), (n_sim, LAYER_NUM - 1))
    sim_data = np.abs(modeler.forward_batch(rhos, thicknesses, times))
    feat_sim = extract_features(sim_data, times)

    all_stack = np.vstack((feat_real, feat_sim))
    mean = np.mean(all_stack, axis=0)
    std = np.std(all_stack, axis=0) + 1e-9
    x_real = (feat_real - mean) / std
    x_sim = (feat_sim - mean) / std

    mmd_val = compute_mmd(x_sim, x_real)
    auc_val = classifier_auc(x_sim, x_real)
    score = mmd_val + 0.2 * abs(auc_val - 0.5)
    return -score, mmd_val, auc_val


def analyze_prior_info():
    print("=== Step 01: prior range analysis ===")
    txt_path = os.path.join(DATA_DIR, "dBzdt.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"未找到真实施工数据文件: {txt_path}")

    times, real_data, qc = parse_dbzdt_txt(txt_path)
    print(f"Loaded {qc['point_count']} points and {qc['time_count']} time gates. QC={qc['status']}")
    feat_real = extract_features(real_data, times)

    rng = np.random.default_rng(RANDOM_SEED)
    modeler = TEMForwardModeler()
    init_points, n_iter = 30, 15
    r_min_bounds = [1.0, 50.0]
    r_max_bounds = [10.0, 300.0]
    all_params, all_scores = [], []

    for i in range(init_points):
        r_min = rng.uniform(*r_min_bounds)
        r_max = rng.uniform(max(r_min + 10, r_max_bounds[0]), r_max_bounds[1])
        score, mmd, auc = target_func(r_min, r_max, times, feat_real, modeler, rng)
        all_params.append([r_min, r_max])
        all_scores.append(score)
        print(f"[random {i + 1}/{init_points}] r_min={r_min:.1f}, r_max={r_max:.1f}, MMD={mmd:.4f}, AUC={auc:.4f}")

    for i in range(n_iter):
        best_idx = int(np.argmax(all_scores))
        best_r_min, best_r_max = all_params[best_idx]
        r_min = np.clip(best_r_min + rng.normal(0, 5), r_min_bounds[0], r_min_bounds[1])
        r_max = np.clip(best_r_max + rng.normal(0, 20), r_min + 10, r_max_bounds[1])
        score, mmd, auc = target_func(r_min, r_max, times, feat_real, modeler, rng)
        all_params.append([r_min, r_max])
        all_scores.append(score)
        print(f"[local {i + 1}/{n_iter}] r_min={r_min:.1f}, r_max={r_max:.1f}, score={-score:.4f}")

    best_idx = int(np.argmax(all_scores))
    final_r_min, final_r_max = all_params[best_idx]
    prior_config = {
        "r_min": float(final_r_min),
        "r_max": float(final_r_max),
        "time_channels": int(len(times)),
        "qc": qc,
    }
    with open(os.path.join(OUTPUT_DIR, "prior_bounds.json"), "w", encoding="utf-8") as f:
        json.dump(prior_config, f, indent=2, ensure_ascii=False)
    print(f"Recommended resistivity range: [{final_r_min:.1f}, {final_r_max:.1f}] ohm*m")


if __name__ == "__main__":
    analyze_prior_info()
